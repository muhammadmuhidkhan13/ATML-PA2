from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW

from common.data import (
    load_yaml,
    prompt_messages,
    read_jsonl,
    repo_path,
)
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import (
    append_jsonl,
    save_json,
    set_seed,
    wall_timer,
)
from common.metrics import (
    masked_mean,
    mean_response_length,
    sample_entropy,
    sampled_kl,
)
from common.models import (
    count_parameters,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def prepare_ppo_continuation(
    config_path: str,
):
    """Load the supplied midpoint policy, critic, and reward model."""

    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    policy = load_policy(
        cfg,
        adapter_path=cfg[
            "paths"
        ]["ppo_midpoint_policy"],
        trainable=True,
    )

    value_model = load_value_model(
        cfg,
        cfg["paths"][
            "ppo_midpoint_value"
        ],
        train_mode=cfg.get(
            "value_train_mode",
            "head_only",
        ),
    )

    reward_model, reward_tokenizer = (
        load_reward_model(cfg)
    )

    prompt_rows = read_jsonl(
        cfg["paths"][
            "rl_prompt_train"
        ]
    )

    if not prompt_rows:
        raise ValueError(
            "The PPO prompt dataset is empty."
        )

    policy_parameters = (
        trainable_parameters(policy)
    )

    if not policy_parameters:
        raise RuntimeError(
            "The PPO policy has no trainable parameters."
        )

    value_parameters = (
        trainable_parameters(
            value_model
        )
    )

    if not value_parameters:
        raise RuntimeError(
            "The PPO critic has no trainable parameters."
        )

    policy_optimizer = AdamW(
        policy_parameters,
        lr=float(
            cfg[
                "policy_learning_rate"
            ]
        ),
        weight_decay=0.0,
    )

    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(
                cfg[
                    "value_lora_learning_rate"
                ]
            ),
            head_lr=float(
                cfg[
                    "value_head_learning_rate"
                ]
            ),
        ),
        weight_decay=0.0,
        eps=1.0e-5,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": (
            reward_tokenizer
        ),
        "prompt_rows": prompt_rows,
        "policy_parameters": (
            policy_parameters
        ),
        "value_parameters": (
            value_parameters
        ),
        "policy_optimizer": (
            policy_optimizer
        ),
        "value_optimizer": (
            value_optimizer
        ),
    }


def select_prompt_rows(
    prompt_rows: list[dict],
    update_index: int,
    prompts_per_update: int,
):
    """Select fixed prompt rows deterministically.

    Matched PPO forks must use identical prompt IDs. Therefore,
    selection follows dataset order instead of depending on model
    behavior or hyperparameters.
    """

    start = (
        update_index
        * prompts_per_update
    )

    selected = []

    for offset in range(
        prompts_per_update
    ):
        row_index = (
            start + offset
        ) % len(prompt_rows)

        selected.append(
            prompt_rows[row_index]
        )

    return selected


def response_values(
    value_model,
    sequences,
    attention_mask,
    prompt_width,
    response_steps,
):
    """Return critic values aligned with generated response tokens.

    The value at sequence position t is aligned with the action token
    predicted immediately after that position, matching the language
    model log-probability alignment.
    """

    all_values = token_values(
        value_model,
        input_ids=sequences,
        attention_mask=attention_mask,
    )

    start = prompt_width - 1
    end = start + response_steps

    return all_values[:, start:end]


def _mean(values):
    if not values:
        return None

    return float(
        sum(values) / len(values)
    )


def _maximum(values):
    if not values:
        return None

    return float(max(values))


def run_ppo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    run_name: str = "standard",
):
    """Continue PPO training from the supplied midpoint checkpoints."""

    bundle = prepare_ppo_continuation(
        config_path
    )

    cfg = bundle["cfg"]

    if updates is not None:
        cfg["updates"] = int(
            updates
        )

    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(
            clip_epsilon
        )

    if kl_beta is not None:
        cfg["kl_beta"] = float(
            kl_beta
        )

    total_updates = int(
        cfg["updates"]
    )

    prompts_per_update = int(
        cfg["prompts_per_update"]
    )

    ppo_epochs = int(
        cfg["ppo_epochs"]
    )

    clip_value = float(
        cfg["clip_epsilon"]
    )

    selected_kl_beta = float(
        cfg["kl_beta"]
    )

    gamma = float(
        cfg["gamma"]
    )

    gae_lambda = float(
        cfg["gae_lambda"]
    )

    value_coefficient = float(
        cfg["value_coef"]
    )

    missing_eos_penalty = float(
        cfg["missing_eos_penalty"]
    )

    max_grad_norm = float(
        cfg["max_grad_norm"]
    )

    if total_updates < 1:
        raise ValueError(
            "PPO updates must be at least 1."
        )

    if prompts_per_update < 1:
        raise ValueError(
            "prompts_per_update must be at least 1."
        )

    if ppo_epochs < 1:
        raise ValueError(
            "ppo_epochs must be at least 1."
        )

    if clip_value < 0:
        raise ValueError(
            "clip_epsilon must be nonnegative."
        )

    if selected_kl_beta < 0:
        raise ValueError(
            "kl_beta must be nonnegative."
        )

    output_spec = (
        output
        or cfg["output"]
    )

    output_path = repo_path(
        output_spec
    )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    value_output_path = (
        output_path
        / "value_model"
    )

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        results_dir
        / f"{run_name}_train_log.jsonl"
    )

    generations_path = (
        results_dir
        / f"{run_name}_generations.jsonl"
    )

    summary_path = (
        results_dir
        / f"{run_name}_train_summary.json"
    )

    # Clear outputs belonging to an older run with the same name.
    log_path.write_text(
        "",
        encoding="utf-8",
    )

    generations_path.write_text(
        "",
        encoding="utf-8",
    )

    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    value_model = bundle[
        "value_model"
    ]
    reward_model = bundle[
        "reward_model"
    ]
    reward_tokenizer = bundle[
        "reward_tokenizer"
    ]
    prompt_rows = bundle[
        "prompt_rows"
    ]
    policy_parameters = bundle[
        "policy_parameters"
    ]
    value_parameters = bundle[
        "value_parameters"
    ]
    policy_optimizer = bundle[
        "policy_optimizer"
    ]
    value_optimizer = bundle[
        "value_optimizer"
    ]

    policy_device = next(
        policy.parameters()
    ).device

    value_device = next(
        value_model.parameters()
    ).device

    generation_cfg = cfg.get(
        "generation",
        {},
    )

    policy_total, policy_trainable = (
        count_parameters(policy)
    )

    value_total, value_trainable = (
        count_parameters(
            value_model
        )
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(
            policy_device
        )

    timer = wall_timer()

    update_records = []
    total_generated_tokens = 0

    # Evaluation mode disables dropout but does not disable gradients.
    # This keeps new-policy log-probabilities comparable with the
    # old-policy log-probabilities saved from the rollout.
    policy.eval()

    for update_index in range(
        total_updates
    ):
        selected_rows = (
            select_prompt_rows(
                prompt_rows,
                update_index,
                prompts_per_update,
            )
        )

        prompts = [
            prompt_messages(row)
            for row in selected_rows
        ]

        generated = batch_generate(
            model=policy,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_length=int(
                cfg[
                    "max_prompt_length"
                ]
            ),
            max_new_tokens=int(
                cfg[
                    "max_response_length"
                ]
            ),
            temperature=float(
                generation_cfg.get(
                    "temperature",
                    0.7,
                )
            ),
            top_p=float(
                generation_cfg.get(
                    "top_p",
                    0.9,
                )
            ),
            do_sample=bool(
                generation_cfg.get(
                    "do_sample",
                    True,
                )
            ),
        )

        sequences = generated[
            "sequences"
        ].to(policy_device)

        attention_mask = generated[
            "attention_mask"
        ].to(policy_device)

        response_ids = generated[
            "response_ids"
        ].to(policy_device)

        response_mask = generated[
            "response_mask"
        ].to(policy_device)

        prompt_width = int(
            generated[
                "prompt_width"
            ]
        )

        response_steps = int(
            response_ids.shape[1]
        )

        response_lengths = [
            int(length)
            for length in generated[
                "response_lengths"
            ]
        ]

        total_generated_tokens += sum(
            response_lengths
        )

        # The rollout policy becomes pi_old for this PPO update.
        with torch.no_grad():
            old_logp, _ = (
                response_token_logprobs(
                    policy,
                    sequences,
                    attention_mask,
                    prompt_width,
                    response_ids,
                )
            )

            # Disabling the policy adapter exposes the frozen
            # reference policy.
            with reference_mode(
                policy
            ):
                reference_logp, _ = (
                    response_token_logprobs(
                        policy,
                        sequences,
                        attention_mask,
                        prompt_width,
                        response_ids,
                    )
                )

        old_logp = old_logp.detach()
        reference_logp = (
            reference_logp.detach()
        )

        raw_task_reward = (
            score_reward_pairs(
                rm_model=reward_model,
                rm_tokenizer=(
                    reward_tokenizer
                ),
                prompts=prompts,
                responses=generated[
                    "responses"
                ],
                max_length=int(
                    cfg[
                        "reward_max_length"
                    ]
                ),
            )
            .float()
            .to(policy_device)
        )

        terminated_tensor = torch.tensor(
            generated[
                "terminated_with_eos"
            ],
            dtype=torch.bool,
            device=policy_device,
        )

        missing_eos = (
            ~terminated_tensor
        ).float()

        adjusted_task_reward = (
            raw_task_reward
            - missing_eos_penalty
            * missing_eos
        )

        token_rewards = shaped_rewards(
            task_reward=(
                adjusted_task_reward
            ),
            policy_logp=old_logp,
            ref_logp=reference_logp,
            response_mask=response_mask,
            beta_kl=selected_kl_beta,
        )

        # Obtain the critic baseline for the rollout without updating it.
        value_model.eval()

        with torch.no_grad():
            old_values = response_values(
                value_model=value_model,
                sequences=sequences.to(
                    value_device
                ),
                attention_mask=(
                    attention_mask.to(
                        value_device
                    )
                ),
                prompt_width=prompt_width,
                response_steps=(
                    response_steps
                ),
            ).to(policy_device)

        old_values = (
            old_values.detach()
            * response_mask
        )

        advantages, returns = (
            compute_gae(
                rewards=token_rewards,
                values=old_values,
                mask=response_mask,
                gamma=gamma,
                lam=gae_lambda,
            )
        )

        advantages = (
            advantages.detach()
        )

        returns = returns.detach()

        normalized_advantages = (
            normalize_advantages(
                advantages,
                response_mask,
            ).detach()
        )

        rollout_kl = sampled_kl(
            old_logp,
            reference_logp,
            response_mask,
        )

        rollout_entropy = sample_entropy(
            old_logp,
            response_mask,
        )

        rollout_reward_mean = float(
            raw_task_reward.mean().item()
        )

        adjusted_reward_mean = float(
            adjusted_task_reward.mean().item()
        )

        response_length_mean = (
            mean_response_length(
                response_mask
            )
        )

        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        epoch_clip_fractions = []
        epoch_ratio_means = []
        epoch_policy_grad_norms = []
        epoch_value_grad_norms = []

        for ppo_epoch in range(
            ppo_epochs
        ):
            # -------------------------
            # Policy update
            # -------------------------
            policy.eval()

            policy_optimizer.zero_grad(
                set_to_none=True
            )

            new_logp, _ = (
                response_token_logprobs(
                    policy,
                    sequences,
                    attention_mask,
                    prompt_width,
                    response_ids,
                )
            )

            policy_loss, ratio, (
                clip_fraction
            ) = ppo_policy_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                advantage=(
                    normalized_advantages
                ),
                mask=response_mask,
                eps=clip_value,
            )

            entropy = sample_entropy(
                new_logp,
                response_mask,
            )

            ratio_mean = masked_mean(
                ratio,
                response_mask,
            )

            policy_loss.backward()

            policy_grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    policy_parameters,
                    max_grad_norm,
                )
            )

            policy_optimizer.step()

            # -------------------------
            # Critic update
            # -------------------------
            value_model.train()

            value_optimizer.zero_grad(
                set_to_none=True
            )

            predicted_values = (
                response_values(
                    value_model=(
                        value_model
                    ),
                    sequences=(
                        sequences.to(
                            value_device
                        )
                    ),
                    attention_mask=(
                        attention_mask.to(
                            value_device
                        )
                    ),
                    prompt_width=(
                        prompt_width
                    ),
                    response_steps=(
                        response_steps
                    ),
                )
            )

            critic_mask = (
                response_mask.to(
                    value_device
                )
            )

            critic_returns = returns.to(
                value_device
            )

            value_loss = value_mse_loss(
                predicted_values=(
                    predicted_values
                ),
                returns=critic_returns,
                mask=critic_mask,
            )

            weighted_value_loss = (
                value_coefficient
                * value_loss
            )

            weighted_value_loss.backward()

            value_grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    value_parameters,
                    max_grad_norm,
                )
            )

            value_optimizer.step()

            epoch_policy_losses.append(
                float(
                    policy_loss
                    .detach()
                    .item()
                )
            )

            epoch_value_losses.append(
                float(
                    value_loss
                    .detach()
                    .item()
                )
            )

            epoch_entropies.append(
                float(
                    entropy
                    .detach()
                    .item()
                )
            )

            epoch_clip_fractions.append(
                float(
                    clip_fraction
                    .detach()
                    .item()
                )
            )

            epoch_ratio_means.append(
                float(
                    ratio_mean
                    .detach()
                    .item()
                )
            )

            epoch_policy_grad_norms.append(
                float(
                    policy_grad_norm
                    .detach()
                    .item()
                )
            )

            epoch_value_grad_norms.append(
                float(
                    value_grad_norm
                    .detach()
                    .item()
                )
            )

        record = {
            "run_name": run_name,
            "update": update_index + 1,
            "prompt_ids": [
                row.get(
                    "prompt_id",
                    row.get(
                        "source_index",
                        update_index
                        * prompts_per_update
                        + row_offset,
                    ),
                )
                for row_offset, row
                in enumerate(
                    selected_rows
                )
            ],
            "num_prompts": len(
                selected_rows
            ),
            "generated_tokens": int(
                sum(response_lengths)
            ),
            "total_generated_tokens": int(
                total_generated_tokens
            ),
            "learned_reward": (
                rollout_reward_mean
            ),
            "adjusted_task_reward": (
                adjusted_reward_mean
            ),
            "sampled_kl": float(
                rollout_kl
                .detach()
                .item()
            ),
            "policy_loss": _mean(
                epoch_policy_losses
            ),
            "value_loss": _mean(
                epoch_value_losses
            ),
            "rollout_entropy": float(
                rollout_entropy
                .detach()
                .item()
            ),
            "entropy": _mean(
                epoch_entropies
            ),
            "clip_fraction": _mean(
                epoch_clip_fractions
            ),
            "ratio_mean": _mean(
                epoch_ratio_means
            ),
            "policy_gradient_norm": (
                _mean(
                    epoch_policy_grad_norms
                )
            ),
            "value_gradient_norm": (
                _mean(
                    epoch_value_grad_norms
                )
            ),
            "gradient_norm": _mean(
                epoch_policy_grad_norms
            ),
            "response_length": float(
                response_length_mean
            ),
            "terminated_with_eos_rate": (
                float(
                    terminated_tensor
                    .float()
                    .mean()
                    .item()
                )
            ),
            "truncation_rate": float(
                torch.tensor(
                    generated[
                        "truncated"
                    ],
                    dtype=torch.float32,
                ).mean().item()
            ),
            "missing_eos_rate": float(
                missing_eos
                .mean()
                .item()
            ),
            "clip_epsilon": (
                clip_value
            ),
            "kl_beta": (
                selected_kl_beta
            ),
            "ppo_epochs": (
                ppo_epochs
            ),
            "elapsed_seconds": float(
                timer()
            ),
        }

        append_jsonl(
            log_path,
            record,
        )

        update_records.append(
            record
        )

        for row_index, (
            source_row,
            response,
        ) in enumerate(
            zip(
                selected_rows,
                generated[
                    "responses"
                ],
            )
        ):
            generation_record = {
                "run_name": run_name,
                "update": (
                    update_index + 1
                ),
                "prompt_id": (
                    source_row.get(
                        "prompt_id"
                    )
                ),
                "source_index": (
                    source_row.get(
                        "source_index"
                    )
                ),
                "prompt_messages": (
                    prompts[row_index]
                ),
                "response": response,
                "learned_reward": float(
                    raw_task_reward[
                        row_index
                    ].item()
                ),
                "adjusted_task_reward": float(
                    adjusted_task_reward[
                        row_index
                    ].item()
                ),
                "response_length": int(
                    response_lengths[
                        row_index
                    ]
                ),
                "terminated_with_eos": bool(
                    generated[
                        "terminated_with_eos"
                    ][row_index]
                ),
                "truncated": bool(
                    generated[
                        "truncated"
                    ][row_index]
                ),
                "sampled_kl": float(
                    (
                        (
                            old_logp[
                                row_index
                            ]
                            - reference_logp[
                                row_index
                            ]
                        )
                        * response_mask[
                            row_index
                        ]
                    ).sum().item()
                    / max(
                        1.0,
                        float(
                            response_mask[
                                row_index
                            ].sum().item()
                        ),
                    )
                ),
            }

            append_jsonl(
                generations_path,
                generation_record,
            )

        print(
            f"[{run_name}] "
            f"update={record['update']}/"
            f"{total_updates} "
            f"reward="
            f"{record['learned_reward']:.4f} "
            f"kl="
            f"{record['sampled_kl']:.4f} "
            f"policy_loss="
            f"{record['policy_loss']:.4f} "
            f"value_loss="
            f"{record['value_loss']:.4f} "
            f"clip_fraction="
            f"{record['clip_fraction']:.4f} "
            f"length="
            f"{record['response_length']:.1f}",
            flush=True,
        )

        # Release tensors belonging only to this rollout before the
        # next generation. This is important on a 6 GiB GPU.
        del sequences
        del attention_mask
        del response_ids
        del response_mask
        del old_logp
        del reference_logp
        del old_values
        del advantages
        del returns
        del normalized_advantages
        del token_rewards
        del raw_task_reward
        del adjusted_task_reward

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy.save_pretrained(
        output_path
    )

    tokenizer.save_pretrained(
        output_path
    )

    value_output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    value_model.save_pretrained(
        value_output_path
    )

    elapsed_seconds = float(
        timer()
    )

    peak_vram_gib = None

    if torch.cuda.is_available():
        peak_vram_gib = float(
            torch.cuda.max_memory_allocated(
                policy_device
            )
            / (1024 ** 3)
        )

    summary = {
        "run_name": run_name,
        "config": config_path,
        "seed": int(cfg["seed"]),
        "policy_start_checkpoint": (
            cfg["paths"][
                "ppo_midpoint_policy"
            ]
        ),
        "value_start_checkpoint": (
            cfg["paths"][
                "ppo_midpoint_value"
            ]
        ),
        "policy_output": str(
            output_spec
        ),
        "value_output": str(
            value_output_path.relative_to(
                repo_path(".")
            )
        ),
        "updates": total_updates,
        "prompts_per_update": (
            prompts_per_update
        ),
        "ppo_epochs": ppo_epochs,
        "clip_epsilon": clip_value,
        "kl_beta": selected_kl_beta,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "value_coefficient": (
            value_coefficient
        ),
        "missing_eos_penalty": (
            missing_eos_penalty
        ),
        "max_prompt_length": int(
            cfg["max_prompt_length"]
        ),
        "max_response_length": int(
            cfg["max_response_length"]
        ),
        "policy_learning_rate": float(
            cfg[
                "policy_learning_rate"
            ]
        ),
        "value_lora_learning_rate": (
            float(
                cfg[
                    "value_lora_learning_rate"
                ]
            )
        ),
        "value_head_learning_rate": (
            float(
                cfg[
                    "value_head_learning_rate"
                ]
            )
        ),
        "policy_total_parameters": (
            policy_total
        ),
        "policy_trainable_parameters": (
            policy_trainable
        ),
        "value_total_parameters": (
            value_total
        ),
        "value_trainable_parameters": (
            value_trainable
        ),
        "total_generated_tokens": (
            total_generated_tokens
        ),
        "mean_learned_reward": _mean(
            [
                row["learned_reward"]
                for row in update_records
            ]
        ),
        "mean_adjusted_task_reward": (
            _mean(
                [
                    row[
                        "adjusted_task_reward"
                    ]
                    for row
                    in update_records
                ]
            )
        ),
        "mean_sampled_kl": _mean(
            [
                row["sampled_kl"]
                for row in update_records
            ]
        ),
        "mean_policy_loss": _mean(
            [
                row["policy_loss"]
                for row in update_records
            ]
        ),
        "mean_value_loss": _mean(
            [
                row["value_loss"]
                for row in update_records
            ]
        ),
        "mean_entropy": _mean(
            [
                row["entropy"]
                for row in update_records
            ]
        ),
        "mean_clip_fraction": _mean(
            [
                row[
                    "clip_fraction"
                ]
                for row in update_records
            ]
        ),
        "mean_gradient_norm": _mean(
            [
                row[
                    "gradient_norm"
                ]
                for row in update_records
            ]
        ),
        "max_gradient_norm": _maximum(
            [
                row[
                    "gradient_norm"
                ]
                for row in update_records
            ]
        ),
        "mean_value_gradient_norm": (
            _mean(
                [
                    row[
                        "value_gradient_norm"
                    ]
                    for row
                    in update_records
                ]
            )
        ),
        "mean_response_length": _mean(
            [
                row[
                    "response_length"
                ]
                for row in update_records
            ]
        ),
        "mean_missing_eos_rate": _mean(
            [
                row[
                    "missing_eos_rate"
                ]
                for row in update_records
            ]
        ),
        "wall_clock_seconds": (
            elapsed_seconds
        ),
        "peak_vram_gib": (
            peak_vram_gib
        ),
        "train_log": str(
            log_path.relative_to(
                repo_path(".")
            )
        ),
        "generations": str(
            generations_path.relative_to(
                repo_path(".")
            )
        ),
    }

    save_json(
        summary_path,
        summary,
    )

    print(
        f"Saved PPO policy adapter to "
        f"{output_path}",
        flush=True,
    )

    print(
        f"Saved PPO value adapter to "
        f"{value_output_path}",
        flush=True,
    )

    print(
        f"Saved PPO training log to "
        f"{log_path}",
        flush=True,
    )

    print(
        f"Saved PPO generations to "
        f"{generations_path}",
        flush=True,
    )

    print(
        f"Saved PPO summary to "
        f"{summary_path}",
        flush=True,
    )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Continue PPO training from the supplied "
            "midpoint policy and critic checkpoints."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/ppo.yaml",
    )

    parser.add_argument(
        "--output",
    )

    parser.add_argument(
        "--updates",
        type=int,
    )

    parser.add_argument(
        "--clip-epsilon",
        type=float,
    )

    parser.add_argument(
        "--kl-beta",
        type=float,
    )

    parser.add_argument(
        "--run-name",
        default="standard",
    )

    args = parser.parse_args()

    run_ppo(
        config_path=args.config,
        output=args.output,
        updates=args.updates,
        clip_epsilon=(
            args.clip_epsilon
        ),
        kl_beta=args.kl_beta,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()