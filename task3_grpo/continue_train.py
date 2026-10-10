from __future__ import annotations

import argparse
from statistics import mean

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
from common.models import (
    count_parameters,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import (
    grpo_policy_loss,
    group_relative_advantages,
    mask_truncated_sequences,
)


def _safe_mean(values):
    if not values:
        return 0.0
    return float(mean(values))


def _prompt_id(row, fallback_index):
    value = row.get("prompt_id")

    if value is None:
        return f"row_{fallback_index}"

    return str(value)


def _source_index(row, fallback_index):
    value = row.get("source_index")

    if value is None:
        return int(fallback_index)

    return int(value)


def prepare_grpo_continuation(
    config_path: str,
):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"][
            "grpo_midpoint_policy"
        ],
        trainable=True,
    )

    (
        reward_model,
        reward_tokenizer,
    ) = load_reward_model(cfg)

    prompts = read_jsonl(
        cfg["paths"]["rl_prompt_train"]
    )

    if not prompts:
        raise ValueError(
            "The GRPO training prompt pool is empty."
        )

    parameters = trainable_parameters(policy)

    if not parameters:
        raise RuntimeError(
            "The GRPO policy has no trainable parameters."
        )

    optimizer = AdamW(
        parameters,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(
            cfg.get("weight_decay", 0.0)
        ),
        eps=float(
            cfg.get("optimizer_epsilon", 1.0e-8)
        ),
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "parameters": parameters,
        "optimizer": optimizer,
    }


def _generate_one_completion(
    policy,
    tokenizer,
    messages,
    cfg,
):
    generated = batch_generate(
        model=policy,
        tokenizer=tokenizer,
        prompts=[messages],
        max_prompt_length=int(
            cfg["max_prompt_length"]
        ),
        max_new_tokens=int(
            cfg["max_completion_length"]
        ),
        temperature=float(
            cfg.get(
                "temperature",
                cfg.get(
                    "generation_temperature",
                    0.7,
                ),
            )
        ),
        top_p=float(
            cfg.get(
                "top_p",
                cfg.get(
                    "generation_top_p",
                    0.9,
                ),
            )
        ),
        do_sample=bool(
            cfg.get("do_sample", True)
        ),
    )

    return {
        "sequences": (
            generated["sequences"].clone()
        ),
        "attention_mask": (
            generated["attention_mask"].clone()
        ),
        "prompt_width": int(
            generated["prompt_width"]
        ),
        "response_ids": (
            generated["response_ids"].clone()
        ),
        "response_mask": (
            generated["response_mask"].clone()
        ),
        "response": str(
            generated["responses"][0]
        ),
        "terminated_with_eos": bool(
            generated[
                "terminated_with_eos"
            ][0]
        ),
        "truncated": bool(
            generated["truncated"][0]
        ),
        "response_length": int(
            generated["response_lengths"][0]
        ),
    }


def _score_one_response(
    reward_model,
    reward_tokenizer,
    messages,
    response,
    cfg,
):
    reward = score_reward_pairs(
        rm_model=reward_model,
        rm_tokenizer=reward_tokenizer,
        prompts=[messages],
        responses=[response],
        max_length=int(
            cfg.get("reward_max_length", 1024)
        ),
    )

    return float(reward[0].item())


def _collect_old_and_reference_logps(
    policy,
    rollout,
):
    was_training = policy.training
    policy.eval()

    with torch.no_grad():
        old_logp, _ = response_token_logprobs(
            model=policy,
            sequences=rollout["sequences"],
            attention_mask=rollout[
                "attention_mask"
            ],
            prompt_width=rollout[
                "prompt_width"
            ],
            response_ids=rollout[
                "response_ids"
            ],
        )

        with reference_mode(policy):
            ref_logp, _ = (
                response_token_logprobs(
                    model=policy,
                    sequences=rollout[
                        "sequences"
                    ],
                    attention_mask=rollout[
                        "attention_mask"
                    ],
                    prompt_width=rollout[
                        "prompt_width"
                    ],
                    response_ids=rollout[
                        "response_ids"
                    ],
                )
            )

    if was_training:
        policy.train()

    rollout["old_logp"] = old_logp.detach()
    rollout["ref_logp"] = ref_logp.detach()


def _effective_response_mask(
    rollout,
    cfg,
):
    mask = rollout["response_mask"]

    if bool(
        cfg.get(
            "mask_truncated_completions",
            True,
        )
    ):
        mask = mask_truncated_sequences(
            mask,
            [rollout["truncated"]],
        )

    return mask


def _weighted_token_mean(
    weighted_sum,
    token_count,
):
    if token_count <= 0:
        return 0.0

    return float(weighted_sum / token_count)


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
):
    bundle = prepare_grpo_continuation(
        config_path
    )

    cfg = bundle["cfg"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle[
        "reward_tokenizer"
    ]
    prompt_rows = bundle["prompt_rows"]
    parameters = bundle["parameters"]
    optimizer = bundle["optimizer"]

    selected_updates = int(
        cfg["updates"]
        if updates is None
        else updates
    )

    if selected_updates < 1:
        raise ValueError(
            "The number of GRPO updates must be "
            "at least one."
        )

    if loss_type not in {
        "grpo",
        "dr_grpo",
    }:
        raise ValueError(
            f"Unknown loss_type={loss_type!r}"
        )

    num_generations = int(
        cfg["num_generations"]
    )

    if num_generations < 2:
        raise ValueError(
            "GRPO requires at least two "
            "generations per prompt."
        )

    prompts_per_update = int(
        cfg.get("prompts_per_update", 1)
    )

    if prompts_per_update < 1:
        raise ValueError(
            "prompts_per_update must be at least one."
        )

    policy_epochs = int(
        cfg.get("policy_epochs", 1)
    )

    if policy_epochs < 1:
        raise ValueError(
            "policy_epochs must be at least one."
        )

    clip_epsilon = float(
        cfg["clip_epsilon"]
    )
    kl_beta = float(
        cfg["kl_beta"]
    )
    max_grad_norm = float(
        cfg["max_grad_norm"]
    )

    out_spec = output or cfg["output"]
    out = repo_path(out_spec)

    results_dir = repo_path(
        cfg["results_dir"]
    )
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_log_path = (
        results_dir
        / f"{run_name}_train_log.jsonl"
    )

    train_generations_path = (
        results_dir
        / f"{run_name}_train_generations.jsonl"
    )

    train_summary_path = (
        results_dir
        / f"{run_name}_train_summary.json"
    )

    train_log_path.write_text(
        "",
        encoding="utf-8",
    )

    train_generations_path.write_text(
        "",
        encoding="utf-8",
    )

    device = next(
        policy.parameters()
    ).device

    total_parameters, trainable_count = (
        count_parameters(policy)
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(
            device
        )

    timer = wall_timer()

    total_generated_tokens = 0
    optimizer_updates = 0

    run_rewards = []
    run_group_stds = []
    run_uninformative = []
    run_losses = []
    run_policy_terms = []
    run_kl_penalties = []
    run_sampled_kls = []
    run_entropies = []
    run_clip_fractions = []
    run_ratio_means = []
    run_grad_norms = []
    run_response_lengths = []
    run_truncation_rates = []
    run_eos_rates = []

    prompt_cursor = 0

    for update_index in range(
        selected_updates
    ):
        update_rollouts = []
        update_group_stds = []
        update_uninformative = []

        for local_group_index in range(
            prompts_per_update
        ):
            prompt_row_index = (
                prompt_cursor
                % len(prompt_rows)
            )
            prompt_cursor += 1

            prompt_row = prompt_rows[
                prompt_row_index
            ]

            messages = prompt_messages(
                prompt_row
            )

            current_prompt_id = _prompt_id(
                prompt_row,
                prompt_row_index,
            )

            current_source_index = (
                _source_index(
                    prompt_row,
                    prompt_row_index,
                )
            )

            group_rollouts = []

            for generation_index in range(
                num_generations
            ):
                rollout = (
                    _generate_one_completion(
                        policy=policy,
                        tokenizer=tokenizer,
                        messages=messages,
                        cfg=cfg,
                    )
                )

                reward = _score_one_response(
                    reward_model=reward_model,
                    reward_tokenizer=(
                        reward_tokenizer
                    ),
                    messages=messages,
                    response=rollout[
                        "response"
                    ],
                    cfg=cfg,
                )

                rollout.update(
                    {
                        "group_index": (
                            local_group_index
                        ),
                        "generation_index": (
                            generation_index
                        ),
                        "prompt_row_index": (
                            prompt_row_index
                        ),
                        "prompt_id": (
                            current_prompt_id
                        ),
                        "source_index": (
                            current_source_index
                        ),
                        "messages": messages,
                        "reward": reward,
                    }
                )

                _collect_old_and_reference_logps(
                    policy,
                    rollout,
                )

                rollout[
                    "effective_response_mask"
                ] = _effective_response_mask(
                    rollout,
                    cfg,
                )

                group_rollouts.append(
                    rollout
                )

                total_generated_tokens += (
                    rollout["response_length"]
                )

            group_rewards = torch.tensor(
                [
                    item["reward"]
                    for item in group_rollouts
                ],
                device=device,
                dtype=torch.float32,
            )

            group_std = float(
                group_rewards.std(
                    unbiased=False
                ).item()
            )

            uninformative = float(
                group_std <= 1.0e-6
            )

            update_group_stds.append(
                group_std
            )

            update_uninformative.append(
                uninformative
            )

            update_rollouts.extend(
                group_rollouts
            )

        reward_tensor = torch.tensor(
            [
                item["reward"]
                for item in update_rollouts
            ],
            device=device,
            dtype=torch.float32,
        )

        group_id_tensor = torch.tensor(
            [
                item["group_index"]
                for item in update_rollouts
            ],
            device=device,
            dtype=torch.long,
        )

        advantages = (
            group_relative_advantages(
                rewards=reward_tensor,
                group_ids=group_id_tensor,
            )
        )

        for rollout_index, rollout in enumerate(
            update_rollouts
        ):
            rollout["advantage"] = float(
                advantages[
                    rollout_index
                ].item()
            )

            append_jsonl(
                train_generations_path,
                {
                    "run_name": run_name,
                    "update": update_index + 1,
                    "loss_type": loss_type,
                    "prompt_id": rollout[
                        "prompt_id"
                    ],
                    "source_index": rollout[
                        "source_index"
                    ],
                    "group_index": rollout[
                        "group_index"
                    ],
                    "generation_index": rollout[
                        "generation_index"
                    ],
                    "prompt_messages": rollout[
                        "messages"
                    ],
                    "response": rollout[
                        "response"
                    ],
                    "reward": rollout[
                        "reward"
                    ],
                    "advantage": rollout[
                        "advantage"
                    ],
                    "response_length": rollout[
                        "response_length"
                    ],
                    "terminated_with_eos": (
                        rollout[
                            "terminated_with_eos"
                        ]
                    ),
                    "truncated": rollout[
                        "truncated"
                    ],
                    "used_for_policy_loss": bool(
                        rollout[
                            "effective_response_mask"
                        ].sum().item()
                        > 0
                    ),
                },
            )

        epoch_losses = []
        epoch_policy_terms = []
        epoch_kl_penalties = []
        epoch_grad_norms = []

        sampled_kl_weighted_sum = 0.0
        entropy_weighted_sum = 0.0
        clip_weighted_sum = 0.0
        ratio_weighted_sum = 0.0
        valid_token_total = 0.0

        for epoch_index in range(
            policy_epochs
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            for rollout_index, rollout in enumerate(
                update_rollouts
            ):
                new_logp, _ = (
                    response_token_logprobs(
                        model=policy,
                        sequences=rollout[
                            "sequences"
                        ],
                        attention_mask=rollout[
                            "attention_mask"
                        ],
                        prompt_width=rollout[
                            "prompt_width"
                        ],
                        response_ids=rollout[
                            "response_ids"
                        ],
                    )
                )

                loss, diagnostics = (
                    grpo_policy_loss(
                        new_logp=new_logp,
                        old_logp=rollout[
                            "old_logp"
                        ],
                        seq_adv=advantages[
                            rollout_index:
                            rollout_index + 1
                        ],
                        token_mask=rollout[
                            "effective_response_mask"
                        ],
                        ref_logp=rollout[
                            "ref_logp"
                        ],
                        eps=clip_epsilon,
                        beta=kl_beta,
                        loss_type=loss_type,
                        max_completion_length=int(
                            cfg[
                                "max_completion_length"
                            ]
                        ),
                    )
                )

                scaled_loss = (
                    loss
                    / float(
                        len(update_rollouts)
                    )
                )

                scaled_loss.backward()

                epoch_losses.append(
                    float(
                        loss.detach().item()
                    )
                )

                epoch_policy_terms.append(
                    float(
                        diagnostics[
                            "policy_term"
                        ].item()
                    )
                )

                epoch_kl_penalties.append(
                    float(
                        diagnostics[
                            "kl_penalty"
                        ].item()
                    )
                )

                if (
                    epoch_index
                    == policy_epochs - 1
                ):
                    valid_tokens = float(
                        rollout[
                            "effective_response_mask"
                        ].sum().item()
                    )

                    valid_token_total += (
                        valid_tokens
                    )

                    sampled_kl_weighted_sum += (
                        float(
                            diagnostics[
                                "sampled_kl"
                            ].item()
                        )
                        * valid_tokens
                    )

                    entropy_weighted_sum += (
                        float(
                            diagnostics[
                                "sample_entropy"
                            ].item()
                        )
                        * valid_tokens
                    )

                    clip_weighted_sum += (
                        float(
                            diagnostics[
                                "clip_fraction"
                            ].item()
                        )
                        * valid_tokens
                    )

                    ratio_weighted_sum += (
                        float(
                            diagnostics[
                                "ratio_mean"
                            ].item()
                        )
                        * valid_tokens
                    )

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    max_grad_norm,
                )
            )

            optimizer.step()
            optimizer_updates += 1

            epoch_grad_norms.append(
                float(grad_norm.detach().item())
            )

        update_rewards = [
            float(item["reward"])
            for item in update_rollouts
        ]

        update_lengths = [
            float(item["response_length"])
            for item in update_rollouts
        ]

        update_truncated = [
            float(item["truncated"])
            for item in update_rollouts
        ]

        update_terminated = [
            float(
                item["terminated_with_eos"]
            )
            for item in update_rollouts
        ]

        mean_reward = _safe_mean(
            update_rewards
        )

        mean_group_std = _safe_mean(
            update_group_stds
        )

        uninformative_fraction = _safe_mean(
            update_uninformative
        )

        mean_loss = _safe_mean(
            epoch_losses
        )

        mean_policy_term = _safe_mean(
            epoch_policy_terms
        )

        mean_kl_penalty = _safe_mean(
            epoch_kl_penalties
        )

        mean_sampled_kl = (
            _weighted_token_mean(
                sampled_kl_weighted_sum,
                valid_token_total,
            )
        )

        mean_entropy = _weighted_token_mean(
            entropy_weighted_sum,
            valid_token_total,
        )

        mean_clip_fraction = (
            _weighted_token_mean(
                clip_weighted_sum,
                valid_token_total,
            )
        )

        mean_ratio = _weighted_token_mean(
            ratio_weighted_sum,
            valid_token_total,
        )

        mean_grad_norm = _safe_mean(
            epoch_grad_norms
        )

        mean_response_length = _safe_mean(
            update_lengths
        )

        truncation_rate = _safe_mean(
            update_truncated
        )

        eos_rate = _safe_mean(
            update_terminated
        )

        record = {
            "run_name": run_name,
            "update": update_index + 1,
            "updates": selected_updates,
            "loss_type": loss_type,
            "prompt_ids": list(
                dict.fromkeys(
                    item["prompt_id"]
                    for item in update_rollouts
                )
            ),
            "num_prompt_groups": (
                prompts_per_update
            ),
            "num_generations_per_prompt": (
                num_generations
            ),
            "num_completions": len(
                update_rollouts
            ),
            "generated_tokens": int(
                sum(update_lengths)
            ),
            "total_generated_tokens": int(
                total_generated_tokens
            ),
            "mean_reward": mean_reward,
            "mean_group_reward_std": (
                mean_group_std
            ),
            "uninformative_group_fraction": (
                uninformative_fraction
            ),
            "informative_group_rate": (
                1.0
                - uninformative_fraction
            ),
            "loss": mean_loss,
            "policy_loss": (
                mean_policy_term
            ),
            "kl_penalty": mean_kl_penalty,
            "sampled_kl": mean_sampled_kl,
            "entropy": mean_entropy,
            "clip_fraction": (
                mean_clip_fraction
            ),
            "ratio_mean": mean_ratio,
            "gradient_norm": mean_grad_norm,
            "max_epoch_gradient_norm": (
                max(epoch_grad_norms)
                if epoch_grad_norms
                else 0.0
            ),
            "response_length": (
                mean_response_length
            ),
            "terminated_with_eos_rate": (
                eos_rate
            ),
            "truncation_rate": (
                truncation_rate
            ),
            "valid_policy_tokens": int(
                valid_token_total
            ),
            "clip_epsilon": (
                clip_epsilon
            ),
            "kl_beta": kl_beta,
            "policy_epochs": (
                policy_epochs
            ),
            "elapsed_seconds": float(
                timer()
            ),
        }

        append_jsonl(
            train_log_path,
            record,
        )

        print(
            f"[{run_name}] "
            f"update={record['update']}/"
            f"{selected_updates} "
            f"reward={record['mean_reward']:.4f} "
            f"group_std="
            f"{record['mean_group_reward_std']:.4f} "
            f"uninformative="
            f"{record['uninformative_group_fraction']:.4f} "
            f"kl={record['sampled_kl']:.4f} "
            f"policy_loss="
            f"{record['policy_loss']:.4f} "
            f"clip_fraction="
            f"{record['clip_fraction']:.4f} "
            f"length="
            f"{record['response_length']:.1f}",
            flush=True,
        )

        run_rewards.append(
            mean_reward
        )
        run_group_stds.append(
            mean_group_std
        )
        run_uninformative.append(
            uninformative_fraction
        )
        run_losses.append(
            mean_loss
        )
        run_policy_terms.append(
            mean_policy_term
        )
        run_kl_penalties.append(
            mean_kl_penalty
        )
        run_sampled_kls.append(
            mean_sampled_kl
        )
        run_entropies.append(
            mean_entropy
        )
        run_clip_fractions.append(
            mean_clip_fraction
        )
        run_ratio_means.append(
            mean_ratio
        )
        run_grad_norms.append(
            mean_grad_norm
        )
        run_response_lengths.append(
            mean_response_length
        )
        run_truncation_rates.append(
            truncation_rate
        )
        run_eos_rates.append(
            eos_rate
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy.save_pretrained(out)
    tokenizer.save_pretrained(out)

    elapsed_seconds = float(timer())

    peak_vram_gib = None

    if torch.cuda.is_available():
        peak_vram_gib = float(
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024 ** 3)
        )

    summary = {
        "run_name": run_name,
        "config": config_path,
        "seed": int(cfg["seed"]),
        "policy_start_checkpoint": str(
            cfg["paths"][
                "grpo_midpoint_policy"
            ]
        ),
        "policy_output": str(out_spec),
        "loss_type": loss_type,
        "updates": selected_updates,
        "optimizer_updates": (
            optimizer_updates
        ),
        "prompts_per_update": (
            prompts_per_update
        ),
        "num_generations": (
            num_generations
        ),
        "policy_epochs": policy_epochs,
        "clip_epsilon": clip_epsilon,
        "kl_beta": kl_beta,
        "max_prompt_length": int(
            cfg["max_prompt_length"]
        ),
        "max_completion_length": int(
            cfg["max_completion_length"]
        ),
        "mask_truncated_completions": bool(
            cfg.get(
                "mask_truncated_completions",
                True,
            )
        ),
        "learning_rate": float(
            cfg["learning_rate"]
        ),
        "max_grad_norm": (
            max_grad_norm
        ),
        "policy_total_parameters": (
            total_parameters
        ),
        "policy_trainable_parameters": (
            trainable_count
        ),
        "total_generated_tokens": int(
            total_generated_tokens
        ),
        "mean_reward": _safe_mean(
            run_rewards
        ),
        "mean_group_reward_std": (
            _safe_mean(run_group_stds)
        ),
        "mean_uninformative_group_fraction": (
            _safe_mean(
                run_uninformative
            )
        ),
        "mean_informative_group_rate": (
            1.0
            - _safe_mean(
                run_uninformative
            )
        ),
        "mean_loss": _safe_mean(
            run_losses
        ),
        "mean_policy_loss": _safe_mean(
            run_policy_terms
        ),
        "mean_kl_penalty": _safe_mean(
            run_kl_penalties
        ),
        "mean_sampled_kl": _safe_mean(
            run_sampled_kls
        ),
        "mean_entropy": _safe_mean(
            run_entropies
        ),
        "mean_clip_fraction": (
            _safe_mean(
                run_clip_fractions
            )
        ),
        "mean_ratio": _safe_mean(
            run_ratio_means
        ),
        "mean_gradient_norm": (
            _safe_mean(
                run_grad_norms
            )
        ),
        "max_gradient_norm": (
            max(run_grad_norms)
            if run_grad_norms
            else 0.0
        ),
        "mean_response_length": (
            _safe_mean(
                run_response_lengths
            )
        ),
        "mean_truncation_rate": (
            _safe_mean(
                run_truncation_rates
            )
        ),
        "mean_terminated_with_eos_rate": (
            _safe_mean(
                run_eos_rates
            )
        ),
        "wall_clock_seconds": (
            elapsed_seconds
        ),
        "peak_vram_gib": (
            peak_vram_gib
        ),
        "train_log": str(
            train_log_path.relative_to(
                repo_path(".")
            )
        ),
        "train_generations": str(
            train_generations_path.relative_to(
                repo_path(".")
            )
        ),
    }

    save_json(
        train_summary_path,
        summary,
    )

    print(
        f"Saved GRPO policy adapter to {out}"
    )
    print(
        "Saved GRPO training log to "
        f"{train_log_path}"
    )
    print(
        "Saved GRPO training generations to "
        f"{train_generations_path}"
    )
    print(
        "Saved GRPO training summary to "
        f"{train_summary_path}"
    )

    return summary


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Continue GRPO training from the supplied "
            "midpoint policy."
        )
    )

    ap.add_argument(
        "--config",
        default="configs/grpo.yaml",
    )

    ap.add_argument("--output")

    ap.add_argument(
        "--updates",
        type=int,
    )

    ap.add_argument(
        "--loss-type",
        choices=[
            "grpo",
            "dr_grpo",
        ],
        default="grpo",
    )

    ap.add_argument(
        "--run-name",
        default="standard",
    )

    args = ap.parse_args()

    run_grpo(
        config_path=args.config,
        output=args.output,
        updates=args.updates,
        loss_type=args.loss_type,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()