from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import numpy as np
import torch

from common.data import (
    load_yaml,
    prompt_messages,
    read_jsonl,
    repo_path,
)
from common.logging_utils import (
    save_json,
    set_seed,
    wall_timer,
)
from common.metrics import (
    masked_mean,
)
from common.models import (
    load_policy,
    load_tokenizer,
)
from common.generation import (
    response_token_logprobs,
)
from task2_ppo.continue_train import (
    run_ppo,
)
from task2_ppo.evaluate import (
    run_evaluation,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
)


def load_cached_rollouts(path):
    rows = torch.load(
        repo_path(path),
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Expected a non-empty list in the "
            "supplied PPO rollout cache."
        )

    normalized = []

    for row in rows:
        row = dict(row)

        if (
            "old_logprobs" not in row
            and "old_policy_logprobs" in row
        ):
            row["old_logprobs"] = row[
                "old_policy_logprobs"
            ]

        if (
            "ref_logprobs" not in row
            and "reference_logprobs" in row
        ):
            row["ref_logprobs"] = row[
                "reference_logprobs"
            ]

        normalized.append(row)

    required = {
        "source_index",
        "prompt_id",
        "response",
        "response_tokens",
        "old_logprobs",
        "ref_logprobs",
        "values",
        "effective_terminal_reward",
        "terminated_with_eos",
    }

    missing = (
        required
        - set(normalized[0])
    )

    if missing:
        raise ValueError(
            "Unexpected PPO cache schema. "
            f"Missing fields: {sorted(missing)}"
        )

    return normalized


def _mean(values):
    if not values:
        return None

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=float,
            )
        )
    )


def _std(values):
    if not values:
        return None

    return float(
        np.std(
            np.asarray(
                values,
                dtype=float,
            )
        )
    )


def _safe_correlation(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if (
        len(a) < 2
        or np.std(a) == 0
        or np.std(b) == 0
    ):
        return None

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def _explained_variance(
    predictions,
    targets,
):
    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    targets = np.asarray(
        targets,
        dtype=float,
    )

    target_variance = np.var(
        targets
    )

    if target_variance == 0:
        return None

    residual_variance = np.var(
        targets - predictions
    )

    return float(
        1.0
        - residual_variance
        / target_variance
    )


def _epsilon_tag(epsilon):
    text = f"{float(epsilon):g}"

    return text.replace(
        ".",
        "p",
    )


def build_prompt_lookup(
    prompt_rows,
):
    by_source_index = {}
    by_prompt_id = {}

    for row in prompt_rows:
        source_index = row.get(
            "source_index"
        )

        prompt_id = row.get(
            "prompt_id"
        )

        if source_index is not None:
            by_source_index[
                source_index
            ] = row

        if prompt_id is not None:
            by_prompt_id[
                prompt_id
            ] = row

    return (
        by_source_index,
        by_prompt_id,
    )


def find_prompt_row(
    cached_row,
    by_source_index,
    by_prompt_id,
):
    source_index = cached_row.get(
        "source_index"
    )

    prompt_id = cached_row.get(
        "prompt_id"
    )

    if source_index in by_source_index:
        row = by_source_index[
            source_index
        ]

        if (
            prompt_id is None
            or row.get("prompt_id")
            == prompt_id
        ):
            return row

    if prompt_id in by_prompt_id:
        return by_prompt_id[
            prompt_id
        ]

    raise KeyError(
        "Could not find cached rollout prompt "
        f"source_index={source_index}, "
        f"prompt_id={prompt_id!r} "
        "in the PPO training prompt pool."
    )


def reconstruct_cached_sequence(
    tokenizer,
    source_row,
    cached_row,
    max_prompt_length,
):
    messages = prompt_messages(
        source_row
    )

    rendered_prompt = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    prompt_ids = tokenizer(
        rendered_prompt,
        truncation=True,
        max_length=max_prompt_length,
    )["input_ids"]

    response_ids = tokenizer(
        cached_row["response"],
        add_special_tokens=False,
    )["input_ids"]

    if cached_row[
        "terminated_with_eos"
    ]:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "The cached response terminated with EOS, "
                "but the tokenizer has no EOS token."
            )

        response_ids = (
            response_ids
            + [tokenizer.eos_token_id]
        )

    expected_tokens = int(
        cached_row[
            "response_tokens"
        ]
    )

    if len(response_ids) != expected_tokens:
        raise ValueError(
            "Cached response did not reconstruct to the "
            "recorded number of tokens: "
            f"prompt_id={cached_row.get('prompt_id')}, "
            f"expected={expected_tokens}, "
            f"reconstructed={len(response_ids)}."
        )

    full_ids = (
        prompt_ids
        + response_ids
    )

    return {
        "messages": messages,
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "full_ids": full_ids,
    }


@torch.inference_mode()
def rescore_cached_response(
    policy,
    full_ids,
    response_ids,
    prompt_width,
):
    device = next(
        policy.parameters()
    ).device

    sequence_tensor = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(
        sequence_tensor
    )

    response_tensor = torch.tensor(
        [response_ids],
        dtype=torch.long,
        device=device,
    )

    new_logprobs, _ = (
        response_token_logprobs(
            policy,
            sequence_tensor,
            attention_mask,
            prompt_width,
            response_tensor,
        )
    )

    return (
        new_logprobs[
            0
        ].detach().cpu()
    )


def reconstruct_cached_statistics(
    cfg,
    cached_rows,
    policy,
    tokenizer,
):
    prompt_rows = read_jsonl(
        cfg["paths"][
            "rl_prompt_eval"
        ]
    )

    (
        by_source_index,
        by_prompt_id,
    ) = build_prompt_lookup(
        prompt_rows
    )

    old_logprob_parts = []
    new_logprob_parts = []
    ref_logprob_parts = []
    value_parts = []
    advantage_parts = []
    return_parts = []
    rollout_details = []

    for cache_index, cached_row in enumerate(
        cached_rows
    ):
        source_row = find_prompt_row(
            cached_row,
            by_source_index,
            by_prompt_id,
        )

        reconstructed = (
            reconstruct_cached_sequence(
                tokenizer=tokenizer,
                source_row=source_row,
                cached_row=cached_row,
                max_prompt_length=int(
                    cfg[
                        "max_prompt_length"
                    ]
                ),
            )
        )

        old_logprobs = cached_row[
            "old_logprobs"
        ].detach().float().cpu()

        ref_logprobs = cached_row[
            "ref_logprobs"
        ].detach().float().cpu()

        values = cached_row[
            "values"
        ].detach().float().cpu()

        expected_tokens = int(
            cached_row[
                "response_tokens"
            ]
        )

        for field_name, tensor in (
            (
                "old_logprobs",
                old_logprobs,
            ),
            (
                "ref_logprobs",
                ref_logprobs,
            ),
            (
                "values",
                values,
            ),
        ):
            if tensor.numel() != expected_tokens:
                raise ValueError(
                    f"{field_name} length mismatch for "
                    f"prompt_id="
                    f"{cached_row.get('prompt_id')}: "
                    f"expected={expected_tokens}, "
                    f"observed={tensor.numel()}."
                )

        new_logprobs = (
            rescore_cached_response(
                policy=policy,
                full_ids=reconstructed[
                    "full_ids"
                ],
                response_ids=reconstructed[
                    "response_ids"
                ],
                prompt_width=len(
                    reconstructed[
                        "prompt_ids"
                    ]
                ),
            )
        )

        if (
            new_logprobs.numel()
            != expected_tokens
        ):
            raise ValueError(
                "Re-scored midpoint log-probability "
                "length mismatch for "
                f"prompt_id="
                f"{cached_row.get('prompt_id')}: "
                f"expected={expected_tokens}, "
                f"observed="
                f"{new_logprobs.numel()}."
            )

        mask = torch.ones(
            1,
            expected_tokens,
            dtype=torch.float32,
        )

        terminal_reward = torch.tensor(
            [
                float(
                    cached_row[
                        "effective_terminal_reward"
                    ]
                )
            ],
            dtype=torch.float32,
        )

        rewards = shaped_rewards(
            task_reward=terminal_reward,
            policy_logp=(
                old_logprobs.unsqueeze(0)
            ),
            ref_logp=(
                ref_logprobs.unsqueeze(0)
            ),
            response_mask=mask,
            beta_kl=float(
                cfg["kl_beta"]
            ),
        )

        advantages, returns = (
            compute_gae(
                rewards=rewards,
                values=(
                    values.unsqueeze(0)
                ),
                mask=mask,
                gamma=float(
                    cfg["gamma"]
                ),
                lam=float(
                    cfg["gae_lambda"]
                ),
            )
        )

        old_logprob_parts.append(
            old_logprobs
        )

        new_logprob_parts.append(
            new_logprobs
        )

        ref_logprob_parts.append(
            ref_logprobs
        )

        value_parts.append(
            values
        )

        advantage_parts.append(
            advantages[0]
        )

        return_parts.append(
            returns[0]
        )

        rollout_details.append(
            {
                "cache_index": (
                    cache_index
                ),
                "source_index": (
                    cached_row[
                        "source_index"
                    ]
                ),
                "prompt_id": (
                    cached_row[
                        "prompt_id"
                    ]
                ),
                "response": (
                    cached_row[
                        "response"
                    ]
                ),
                "response_tokens": (
                    expected_tokens
                ),
                "raw_terminal_reward": (
                    cached_row.get(
                        "raw_terminal_reward"
                    )
                ),
                "effective_terminal_reward": (
                    float(
                        cached_row[
                            "effective_terminal_reward"
                        ]
                    )
                ),
                "terminated_with_eos": (
                    bool(
                        cached_row[
                            "terminated_with_eos"
                        ]
                    )
                ),
                "clipped_at_max": (
                    bool(
                        cached_row.get(
                            "clipped_at_max",
                            False,
                        )
                    )
                ),
                "old_policy_logprob_mean": (
                    float(
                        old_logprobs.mean().item()
                    )
                ),
                "midpoint_policy_logprob_mean": (
                    float(
                        new_logprobs.mean().item()
                    )
                ),
                "reference_logprob_mean": (
                    float(
                        ref_logprobs.mean().item()
                    )
                ),
                "sampled_kl": float(
                    (
                        old_logprobs
                        - ref_logprobs
                    ).mean().item()
                ),
                "advantage_mean": float(
                    advantages[
                        0
                    ].mean().item()
                ),
                "return_mean": float(
                    returns[
                        0
                    ].mean().item()
                ),
                "value_mean": float(
                    values.mean().item()
                ),
            }
        )

        print(
            "Re-scored cached rollout "
            f"{cache_index + 1}/"
            f"{len(cached_rows)}",
            flush=True,
        )

    old_logprobs = torch.cat(
        old_logprob_parts
    )

    new_logprobs = torch.cat(
        new_logprob_parts
    )

    ref_logprobs = torch.cat(
        ref_logprob_parts
    )

    values = torch.cat(
        value_parts
    )

    advantages = torch.cat(
        advantage_parts
    )

    returns = torch.cat(
        return_parts
    )

    mask = torch.ones_like(
        advantages,
        dtype=torch.float32,
    )

    normalized_advantages = (
        normalize_advantages(
            advantages.unsqueeze(0),
            mask.unsqueeze(0),
        )[0]
    )

    return {
        "old_logprobs": (
            old_logprobs
        ),
        "new_logprobs": (
            new_logprobs
        ),
        "ref_logprobs": (
            ref_logprobs
        ),
        "values": values,
        "advantages": advantages,
        "normalized_advantages": (
            normalized_advantages
        ),
        "returns": returns,
        "mask": mask,
        "rollout_details": (
            rollout_details
        ),
    }


def analyze_cached_geometry(
    config_path,
    epsilons,
):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    cached_rows = (
        load_cached_rollouts(
            cfg["cached_rollouts"]
        )
    )

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    policy = load_policy(
        cfg,
        adapter_path=cfg[
            "paths"
        ]["ppo_midpoint_policy"],
        trainable=False,
    )

    policy.eval()

    reconstructed = (
        reconstruct_cached_statistics(
            cfg=cfg,
            cached_rows=cached_rows,
            policy=policy,
            tokenizer=tokenizer,
        )
    )

    old_logprobs = reconstructed[
        "old_logprobs"
    ].unsqueeze(0)

    new_logprobs = reconstructed[
        "new_logprobs"
    ].unsqueeze(0)

    advantages = reconstructed[
        "normalized_advantages"
    ].unsqueeze(0)

    mask = reconstructed[
        "mask"
    ].unsqueeze(0)

    ratio = torch.exp(
        new_logprobs
        - old_logprobs
    )

    unclipped_surrogate = (
        ratio * advantages
    )

    geometry_rows = []

    for epsilon in epsilons:
        epsilon = float(epsilon)

        policy_loss, _, clip_fraction = (
            ppo_policy_loss(
                new_logp=(
                    new_logprobs
                ),
                old_logp=(
                    old_logprobs
                ),
                advantage=advantages,
                mask=mask,
                eps=epsilon,
            )
        )

        clipped_ratio = ratio.clamp(
            1.0 - epsilon,
            1.0 + epsilon,
        )

        clipped_surrogate = (
            clipped_ratio
            * advantages
        )

        selected_surrogate = (
            torch.minimum(
                unclipped_surrogate,
                clipped_surrogate,
            )
        )

        affected = (
            selected_surrogate
            < unclipped_surrogate
        ).float()

        affected_fraction = (
            masked_mean(
                affected,
                mask,
            )
        )

        geometry_rows.append(
            {
                "epsilon": epsilon,
                "valid_tokens": int(
                    mask.sum().item()
                ),
                "ratio_mean": float(
                    masked_mean(
                        ratio,
                        mask,
                    ).item()
                ),
                "ratio_std": float(
                    ratio[
                        mask.bool()
                    ].std(
                        unbiased=False
                    ).item()
                ),
                "ratio_min": float(
                    ratio[
                        mask.bool()
                    ].min().item()
                ),
                "ratio_max": float(
                    ratio[
                        mask.bool()
                    ].max().item()
                ),
                "clip_fraction": float(
                    clip_fraction.item()
                ),
                "affected_token_fraction": (
                    float(
                        affected_fraction.item()
                    )
                ),
                "unclipped_surrogate_mean": (
                    float(
                        masked_mean(
                            unclipped_surrogate,
                            mask,
                        ).item()
                    )
                ),
                "clipped_candidate_mean": (
                    float(
                        masked_mean(
                            clipped_surrogate,
                            mask,
                        ).item()
                    )
                ),
                "selected_surrogate_mean": (
                    float(
                        masked_mean(
                            selected_surrogate,
                            mask,
                        ).item()
                    )
                ),
                "policy_loss": float(
                    policy_loss.item()
                ),
            }
        )

    values = reconstructed[
        "values"
    ].numpy()

    returns = reconstructed[
        "returns"
    ].numpy()

    cache_summary = {
        "config": config_path,
        "cache": cfg[
            "cached_rollouts"
        ],
        "num_rollouts": len(
            cached_rows
        ),
        "valid_tokens": int(
            reconstructed[
                "mask"
            ].sum().item()
        ),
        "midpoint_policy_checkpoint": (
            cfg["paths"][
                "ppo_midpoint_policy"
            ]
        ),
        "kl_beta": float(
            cfg["kl_beta"]
        ),
        "gamma": float(
            cfg["gamma"]
        ),
        "gae_lambda": float(
            cfg["gae_lambda"]
        ),
        "geometry": geometry_rows,
        "critic": {
            "value_mean": float(
                np.mean(values)
            ),
            "value_std": float(
                np.std(values)
            ),
            "return_mean": float(
                np.mean(returns)
            ),
            "return_std": float(
                np.std(returns)
            ),
            "value_return_correlation": (
                _safe_correlation(
                    values,
                    returns,
                )
            ),
            "return_explained_variance": (
                _explained_variance(
                    values,
                    returns,
                )
            ),
        },
        "rollout_details": (
            reconstructed[
                "rollout_details"
            ]
        ),
    }

    del policy

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cache_summary


def save_csv(
    path: Path,
    rows: list[dict],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def run_matched_forks(
    config_path,
    epsilons,
    fork_updates,
    eval_max_examples=None,
    skip_reward=False,
    skip_evaluation=False,
):
    cfg = load_yaml(config_path)

    comparisons = []

    for condition_index, epsilon in enumerate(
        epsilons
    ):
        epsilon = float(epsilon)
        tag = _epsilon_tag(epsilon)

        run_name = (
            f"clip_{tag}"
        )

        output_path = (
            "outputs/task2_ppo/"
            "clipping/"
            f"{run_name}"
        )

        print(
            f"\n[{condition_index + 1}/"
            f"{len(epsilons)}] "
            f"Training matched PPO fork "
            f"with epsilon={epsilon}",
            flush=True,
        )

        training_summary = run_ppo(
            config_path=config_path,
            output=output_path,
            updates=int(
                fork_updates
            ),
            clip_epsilon=epsilon,
            kl_beta=float(
                cfg["kl_beta"]
            ),
            run_name=run_name,
        )

        evaluation_metrics = None

        if not skip_evaluation:
            print(
                f"[{condition_index + 1}/"
                f"{len(epsilons)}] "
                f"Evaluating epsilon="
                f"{epsilon}",
                flush=True,
            )

            evaluation_metrics = (
                run_evaluation(
                    config_path=(
                        config_path
                    ),
                    adapter=output_path,
                    name=(
                        f"{run_name}_eval"
                    ),
                    max_examples=(
                        eval_max_examples
                    ),
                    skip_reward=(
                        skip_reward
                    ),
                    batch_size=1,
                )
            )

        comparison = {
            "epsilon": epsilon,
            "run_name": run_name,
            "updates": int(
                fork_updates
            ),
            "training_mean_reward": (
                training_summary.get(
                    "mean_learned_reward"
                )
            ),
            "training_mean_kl": (
                training_summary.get(
                    "mean_sampled_kl"
                )
            ),
            "training_mean_policy_loss": (
                training_summary.get(
                    "mean_policy_loss"
                )
            ),
            "training_mean_value_loss": (
                training_summary.get(
                    "mean_value_loss"
                )
            ),
            "training_mean_clip_fraction": (
                training_summary.get(
                    "mean_clip_fraction"
                )
            ),
            "training_mean_gradient_norm": (
                training_summary.get(
                    "mean_gradient_norm"
                )
            ),
            "training_max_gradient_norm": (
                training_summary.get(
                    "max_gradient_norm"
                )
            ),
            "training_mean_length": (
                training_summary.get(
                    "mean_response_length"
                )
            ),
            "heldout_reward": (
                None
                if evaluation_metrics is None
                else evaluation_metrics.get(
                    "reward_model_score_mean"
                )
            ),
            "heldout_kl": (
                None
                if evaluation_metrics is None
                else evaluation_metrics.get(
                    "sampled_kl_token_mean"
                )
            ),
            "heldout_entropy": (
                None
                if evaluation_metrics is None
                else evaluation_metrics.get(
                    "entropy_token_mean"
                )
            ),
            "heldout_length": (
                None
                if evaluation_metrics is None
                else evaluation_metrics.get(
                    "response_length_mean"
                )
            ),
            "heldout_truncation_rate": (
                None
                if evaluation_metrics is None
                else evaluation_metrics.get(
                    "truncation_rate"
                )
            ),
            "policy_output": (
                training_summary.get(
                    "policy_output"
                )
            ),
        }

        comparisons.append(
            comparison
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return comparisons


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Task 2 PPO clipping geometry "
            "analysis and matched short forks."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/ppo.yaml",
    )

    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
    )

    parser.add_argument(
        "--fork-updates",
        type=int,
    )

    parser.add_argument(
        "--cache-only",
        action="store_true",
    )

    parser.add_argument(
        "--skip-cache-analysis",
        action="store_true",
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
    )

    parser.add_argument(
        "--eval-max-examples",
        type=int,
    )

    parser.add_argument(
        "--skip-reward",
        action="store_true",
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    epsilons = (
        args.epsilons
        if args.epsilons is not None
        else [
            float(value)
            for value in cfg[
                "clip_values"
            ]
        ]
    )

    fork_updates = int(
        args.fork_updates
        if args.fork_updates is not None
        else cfg["fork_updates"]
    )

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_summary_path = (
        results_dir
        / "clipping_cache_summary.json"
    )

    cache_csv_path = (
        results_dir
        / "clipping_cache_geometry.csv"
    )

    comparison_path = (
        results_dir
        / "clipping_study_summary.json"
    )

    comparison_csv_path = (
        results_dir
        / "clipping_study_comparison.csv"
    )

    timer = wall_timer()

    cache_summary = None

    if not args.skip_cache_analysis:
        print(
            "Analyzing supplied cached PPO "
            "rollouts...",
            flush=True,
        )

        cache_summary = (
            analyze_cached_geometry(
                config_path=args.config,
                epsilons=epsilons,
            )
        )

        save_json(
            cache_summary_path,
            cache_summary,
        )

        save_csv(
            cache_csv_path,
            cache_summary[
                "geometry"
            ],
        )

        print(
            "Saved cached-rollout summary to "
            f"{cache_summary_path}",
            flush=True,
        )

        print(
            "Saved cached-rollout table to "
            f"{cache_csv_path}",
            flush=True,
        )

    comparisons = []

    if not args.cache_only:
        comparisons = run_matched_forks(
            config_path=args.config,
            epsilons=epsilons,
            fork_updates=fork_updates,
            eval_max_examples=(
                args.eval_max_examples
            ),
            skip_reward=(
                args.skip_reward
            ),
            skip_evaluation=(
                args.skip_evaluation
            ),
        )

    combined_summary = {
        "config": args.config,
        "epsilons": [
            float(value)
            for value in epsilons
        ],
        "fork_updates": (
            fork_updates
        ),
        "cache_analysis": (
            None
            if cache_summary is None
            else {
                "num_rollouts": (
                    cache_summary[
                        "num_rollouts"
                    ]
                ),
                "valid_tokens": (
                    cache_summary[
                        "valid_tokens"
                    ]
                ),
                "geometry": (
                    cache_summary[
                        "geometry"
                    ]
                ),
                "critic": (
                    cache_summary[
                        "critic"
                    ]
                ),
            }
        ),
        "matched_forks": comparisons,
        "wall_clock_seconds": float(
            timer()
        ),
    }

    save_json(
        comparison_path,
        combined_summary,
    )

    if comparisons:
        save_csv(
            comparison_csv_path,
            comparisons,
        )

    print(
        "\nPPO clipping study summary",
        flush=True,
    )

    if cache_summary is not None:
        print(
            "epsilon | clip fraction | "
            "affected-token fraction | "
            "selected surrogate",
            flush=True,
        )

        for row in cache_summary[
            "geometry"
        ]:
            print(
                f"{row['epsilon']} | "
                f"{row['clip_fraction']} | "
                f"{row['affected_token_fraction']} | "
                f"{row['selected_surrogate_mean']}",
                flush=True,
            )

    if comparisons:
        print(
            "\nepsilon | held-out reward | "
            "held-out KL | mean length | "
            "max gradient norm",
            flush=True,
        )

        for row in comparisons:
            print(
                f"{row['epsilon']} | "
                f"{row['heldout_reward']} | "
                f"{row['heldout_kl']} | "
                f"{row['heldout_length']} | "
                f"{row['training_max_gradient_norm']}",
                flush=True,
            )

    print(
        "Saved clipping-study summary to "
        f"{comparison_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()