from __future__ import annotations

import argparse
import gc

import pandas as pd
import torch

from common.data import (
    load_yaml,
    read_jsonl,
    repo_path,
)
from common.logging_utils import (
    load_json,
    save_json,
)
from task3_grpo.continue_train import (
    run_grpo,
)
from task3_grpo.evaluate import (
    run_evaluation,
)


CONDITIONS = (
    {
        "condition": "canonical_grpo",
        "loss_type": "grpo",
        "run_name": "normalization_grpo",
        "output": (
            "outputs/task3_grpo/"
            "normalization/grpo"
        ),
    },
    {
        "condition": "dr_grpo",
        "loss_type": "dr_grpo",
        "run_name": (
            "normalization_dr_grpo"
        ),
        "output": (
            "outputs/task3_grpo/"
            "normalization/dr_grpo"
        ),
    },
)


def _clear_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _length_bin(length):
    length = int(length)

    if length <= 128:
        return "short_1_128"

    if length <= 256:
        return "medium_129_256"

    return "long_257_plus"


def _safe_mean(values):
    if not values:
        return None

    return float(
        sum(values) / len(values)
    )


def _training_summary_path(
    results_dir,
    run_name,
):
    return (
        results_dir
        / f"{run_name}_train_summary.json"
    )


def _training_log_path(
    results_dir,
    run_name,
):
    return (
        results_dir
        / f"{run_name}_train_log.jsonl"
    )


def _training_generations_path(
    results_dir,
    run_name,
):
    return (
        results_dir
        / f"{run_name}_train_generations.jsonl"
    )


def _evaluation_metrics_path(
    results_dir,
    run_name,
):
    return (
        results_dir
        / f"{run_name}_eval_metrics.json"
    )


def _evaluation_generations_path(
    results_dir,
    run_name,
):
    return (
        results_dir
        / f"{run_name}_eval_generations.jsonl"
    )


def _verify_adapter(path):
    adapter_dir = repo_path(path)

    adapter_file = (
        adapter_dir
        / "adapter_model.safetensors"
    )

    if not adapter_file.exists():
        raise FileNotFoundError(
            "Could not find the required adapter: "
            f"{adapter_file}"
        )


def _load_existing_training_summary(
    results_dir,
    run_name,
):
    path = _training_summary_path(
        results_dir,
        run_name,
    )

    if not path.exists():
        raise FileNotFoundError(
            "Training was skipped, but the "
            "required summary does not exist: "
            f"{path}"
        )

    return load_json(path)


def _load_existing_evaluation_metrics(
    results_dir,
    run_name,
):
    path = _evaluation_metrics_path(
        results_dir,
        run_name,
    )

    if not path.exists():
        raise FileNotFoundError(
            "Evaluation was skipped, but the "
            "required metrics do not exist: "
            f"{path}"
        )

    return load_json(path)


def _run_or_load_training(
    config_path,
    cfg,
    condition,
    fork_updates,
    skip_training,
):
    results_dir = repo_path(
        cfg["results_dir"]
    )

    if skip_training:
        _verify_adapter(
            condition["output"]
        )

        return (
            _load_existing_training_summary(
                results_dir,
                condition["run_name"],
            )
        )

    print()
    print(
        "Training normalization condition: "
        f"{condition['condition']}"
    )
    print(
        f"Loss type: "
        f"{condition['loss_type']}"
    )
    print(
        f"Updates: {fork_updates}"
    )

    summary = run_grpo(
        config_path=config_path,
        output=condition["output"],
        updates=fork_updates,
        loss_type=condition[
            "loss_type"
        ],
        run_name=condition[
            "run_name"
        ],
    )

    _clear_memory()

    return summary


def _run_or_load_evaluation(
    config_path,
    cfg,
    condition,
    eval_max_examples,
    skip_reward,
    skip_evaluation,
):
    results_dir = repo_path(
        cfg["results_dir"]
    )

    evaluation_name = (
        f"{condition['run_name']}_eval"
    )

    if skip_evaluation:
        return (
            _load_existing_evaluation_metrics(
                results_dir,
                condition["run_name"],
            )
        )

    _verify_adapter(
        condition["output"]
    )

    print()
    print(
        "Evaluating normalization condition: "
        f"{condition['condition']}"
    )

    metrics = run_evaluation(
        config_path=config_path,
        adapter=condition["output"],
        name=evaluation_name,
        max_examples=eval_max_examples,
        skip_reward=skip_reward,
        batch_size=1,
    )

    _clear_memory()

    return metrics


def _normalization_denominator(
    loss_type,
    response_length,
    max_completion_length,
):
    if loss_type == "grpo":
        return float(
            max(
                int(response_length),
                1,
            )
        )

    if loss_type == "dr_grpo":
        return float(
            max_completion_length
        )

    raise ValueError(
        f"Unknown loss_type={loss_type!r}"
    )


def analyze_length_conditioned_weights(
    cfg,
    condition,
):
    results_dir = repo_path(
        cfg["results_dir"]
    )

    generations_path = (
        _training_generations_path(
            results_dir,
            condition["run_name"],
        )
    )

    if not generations_path.exists():
        raise FileNotFoundError(
            "Missing training generations: "
            f"{generations_path}"
        )

    rows = read_jsonl(
        generations_path
    )

    max_completion_length = int(
        cfg["max_completion_length"]
    )

    enriched = []

    for row in rows:
        response_length = int(
            row["response_length"]
        )

        used_for_policy_loss = bool(
            row.get(
                "used_for_policy_loss",
                True,
            )
        )

        denominator = (
            _normalization_denominator(
                loss_type=condition[
                    "loss_type"
                ],
                response_length=(
                    response_length
                ),
                max_completion_length=(
                    max_completion_length
                ),
            )
        )

        absolute_advantage = abs(
            float(row["advantage"])
        )

        valid_length = (
            response_length
            if used_for_policy_loss
            else 0
        )

        # At the beginning of a policy update, when the
        # old/current-policy ratio is approximately one,
        # this is the response advantage's scale applied
        # to each valid token.
        per_token_advantage_scale = (
            absolute_advantage
            / denominator
            if used_for_policy_loss
            else 0.0
        )

        # Total magnitude accumulated across the response
        # before model-specific token gradients are applied.
        normalized_advantage_mass = (
            absolute_advantage
            * valid_length
            / denominator
            if used_for_policy_loss
            else 0.0
        )

        enriched.append(
            {
                "condition": condition[
                    "condition"
                ],
                "loss_type": condition[
                    "loss_type"
                ],
                "run_name": condition[
                    "run_name"
                ],
                "update": int(
                    row["update"]
                ),
                "prompt_id": str(
                    row["prompt_id"]
                ),
                "source_index": int(
                    row["source_index"]
                ),
                "generation_index": int(
                    row[
                        "generation_index"
                    ]
                ),
                "length_bin": _length_bin(
                    response_length
                ),
                "response_length": (
                    response_length
                ),
                "reward": float(
                    row["reward"]
                ),
                "advantage": float(
                    row["advantage"]
                ),
                "absolute_advantage": (
                    absolute_advantage
                ),
                "normalization_denominator": (
                    denominator
                ),
                "per_token_advantage_scale": (
                    per_token_advantage_scale
                ),
                "normalized_advantage_mass": (
                    normalized_advantage_mass
                ),
                "used_for_policy_loss": (
                    used_for_policy_loss
                ),
                "truncated": bool(
                    row["truncated"]
                ),
            }
        )

    summary_rows = []

    for length_bin in (
        "short_1_128",
        "medium_129_256",
        "long_257_plus",
    ):
        selected = [
            row
            for row in enriched
            if row["length_bin"]
            == length_bin
        ]

        if not selected:
            continue

        summary_rows.append(
            {
                "condition": condition[
                    "condition"
                ],
                "loss_type": condition[
                    "loss_type"
                ],
                "length_bin": length_bin,
                "num_completions": len(
                    selected
                ),
                "num_used_for_policy_loss": sum(
                    int(
                        row[
                            "used_for_policy_loss"
                        ]
                    )
                    for row in selected
                ),
                "mean_response_length": (
                    _safe_mean(
                        [
                            float(
                                row[
                                    "response_length"
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
                "mean_reward": _safe_mean(
                    [
                        float(row["reward"])
                        for row in selected
                    ]
                ),
                "mean_absolute_advantage": (
                    _safe_mean(
                        [
                            float(
                                row[
                                    "absolute_advantage"
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
                "mean_normalization_denominator": (
                    _safe_mean(
                        [
                            float(
                                row[
                                    "normalization_denominator"
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
                "mean_per_token_advantage_scale": (
                    _safe_mean(
                        [
                            float(
                                row[
                                    "per_token_advantage_scale"
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
                "mean_normalized_advantage_mass": (
                    _safe_mean(
                        [
                            float(
                                row[
                                    "normalized_advantage_mass"
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
                "truncation_rate": (
                    _safe_mean(
                        [
                            float(
                                row[
                                    "truncated"
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
            }
        )

    return enriched, summary_rows


def analyze_update_length_conditioning(
    cfg,
    condition,
):
    results_dir = repo_path(
        cfg["results_dir"]
    )

    log_path = _training_log_path(
        results_dir,
        condition["run_name"],
    )

    if not log_path.exists():
        raise FileNotFoundError(
            f"Missing training log: {log_path}"
        )

    rows = read_jsonl(log_path)

    enriched = []

    for row in rows:
        mean_length = float(
            row["response_length"]
        )

        enriched.append(
            {
                "condition": condition[
                    "condition"
                ],
                "loss_type": condition[
                    "loss_type"
                ],
                "update": int(
                    row["update"]
                ),
                "length_bin": _length_bin(
                    mean_length
                ),
                "mean_response_length": (
                    mean_length
                ),
                "gradient_norm": float(
                    row["gradient_norm"]
                ),
                "policy_loss": float(
                    row["policy_loss"]
                ),
                "mean_reward": float(
                    row["mean_reward"]
                ),
                "sampled_kl": float(
                    row["sampled_kl"]
                ),
                "clip_fraction": float(
                    row["clip_fraction"]
                ),
                "informative_group_rate": float(
                    row[
                        "informative_group_rate"
                    ]
                ),
            }
        )

    summary_rows = []

    for length_bin in (
        "short_1_128",
        "medium_129_256",
        "long_257_plus",
    ):
        selected = [
            row
            for row in enriched
            if row["length_bin"]
            == length_bin
        ]

        if not selected:
            continue

        summary_rows.append(
            {
                "condition": condition[
                    "condition"
                ],
                "loss_type": condition[
                    "loss_type"
                ],
                "length_bin": length_bin,
                "num_updates": len(
                    selected
                ),
                "mean_response_length": (
                    _safe_mean(
                        [
                            row[
                                "mean_response_length"
                            ]
                            for row in selected
                        ]
                    )
                ),
                "mean_gradient_norm": (
                    _safe_mean(
                        [
                            row[
                                "gradient_norm"
                            ]
                            for row in selected
                        ]
                    )
                ),
                "max_gradient_norm": max(
                    row["gradient_norm"]
                    for row in selected
                ),
                "mean_absolute_policy_loss": (
                    _safe_mean(
                        [
                            abs(
                                row[
                                    "policy_loss"
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
                "mean_reward": _safe_mean(
                    [
                        row["mean_reward"]
                        for row in selected
                    ]
                ),
                "mean_sampled_kl": (
                    _safe_mean(
                        [
                            row[
                                "sampled_kl"
                            ]
                            for row in selected
                        ]
                    )
                ),
                "mean_clip_fraction": (
                    _safe_mean(
                        [
                            row[
                                "clip_fraction"
                            ]
                            for row in selected
                        ]
                    )
                ),
                "mean_informative_group_rate": (
                    _safe_mean(
                        [
                            row[
                                "informative_group_rate"
                            ]
                            for row in selected
                        ]
                    )
                ),
            }
        )

    return enriched, summary_rows


def build_comparison_row(
    condition,
    training_summary,
    evaluation_metrics,
):
    return {
        "condition": condition[
            "condition"
        ],
        "loss_type": condition[
            "loss_type"
        ],
        "updates": training_summary[
            "updates"
        ],
        "train_mean_reward": (
            training_summary[
                "mean_reward"
            ]
        ),
        "train_mean_group_reward_std": (
            training_summary[
                "mean_group_reward_std"
            ]
        ),
        "train_informative_group_rate": (
            training_summary[
                "mean_informative_group_rate"
            ]
        ),
        "train_sampled_kl": (
            training_summary[
                "mean_sampled_kl"
            ]
        ),
        "train_entropy": (
            training_summary[
                "mean_entropy"
            ]
        ),
        "train_mean_gradient_norm": (
            training_summary[
                "mean_gradient_norm"
            ]
        ),
        "train_max_gradient_norm": (
            training_summary[
                "max_gradient_norm"
            ]
        ),
        "train_mean_response_length": (
            training_summary[
                "mean_response_length"
            ]
        ),
        "train_truncation_rate": (
            training_summary[
                "mean_truncation_rate"
            ]
        ),
        "train_wall_clock_seconds": (
            training_summary[
                "wall_clock_seconds"
            ]
        ),
        "train_peak_vram_gib": (
            training_summary[
                "peak_vram_gib"
            ]
        ),
        "heldout_reward": (
            evaluation_metrics[
                "reward_model_score_mean"
            ]
        ),
        "heldout_reward_std": (
            evaluation_metrics[
                "reward_model_score_std"
            ]
        ),
        "heldout_sampled_kl": (
            evaluation_metrics[
                "sampled_kl_token_mean"
            ]
        ),
        "heldout_entropy": (
            evaluation_metrics[
                "entropy_token_mean"
            ]
        ),
        "heldout_mean_response_length": (
            evaluation_metrics[
                "response_length_mean"
            ]
        ),
        "heldout_response_length_std": (
            evaluation_metrics[
                "response_length_std"
            ]
        ),
        "heldout_response_length_iqr": (
            evaluation_metrics[
                "response_length_iqr"
            ]
        ),
        "heldout_eos_rate": (
            evaluation_metrics[
                "terminated_with_eos_rate"
            ]
        ),
        "heldout_truncation_rate": (
            evaluation_metrics[
                "truncation_rate"
            ]
        ),
    }


def build_qualitative_comparisons(
    cfg,
):
    results_dir = repo_path(
        cfg["results_dir"]
    )

    canonical_path = (
        _evaluation_generations_path(
            results_dir,
            "normalization_grpo",
        )
    )

    dr_path = (
        _evaluation_generations_path(
            results_dir,
            "normalization_dr_grpo",
        )
    )

    if (
        not canonical_path.exists()
        or not dr_path.exists()
    ):
        return {
            "paired_examples": [],
            "largest_reward_differences": [],
            "largest_length_differences": [],
            "largest_kl_differences": [],
        }

    canonical_rows = read_jsonl(
        canonical_path
    )
    dr_rows = read_jsonl(
        dr_path
    )

    canonical_by_prompt = {
        str(row["prompt_id"]): row
        for row in canonical_rows
    }

    dr_by_prompt = {
        str(row["prompt_id"]): row
        for row in dr_rows
    }

    paired = []

    common_prompt_ids = sorted(
        set(canonical_by_prompt)
        .intersection(dr_by_prompt)
    )

    for prompt_id in common_prompt_ids:
        canonical = canonical_by_prompt[
            prompt_id
        ]
        dr = dr_by_prompt[prompt_id]

        canonical_reward = canonical.get(
            "reward_model_score"
        )
        dr_reward = dr.get(
            "reward_model_score"
        )

        reward_difference = None

        if (
            canonical_reward is not None
            and dr_reward is not None
        ):
            reward_difference = float(
                dr_reward
                - canonical_reward
            )

        paired.append(
            {
                "prompt_id": prompt_id,
                "source_index": canonical[
                    "source_index"
                ],
                "prompt_messages": canonical[
                    "prompt_messages"
                ],
                "canonical_response": (
                    canonical["response"]
                ),
                "dr_grpo_response": (
                    dr["response"]
                ),
                "canonical_reward": (
                    canonical_reward
                ),
                "dr_grpo_reward": (
                    dr_reward
                ),
                "reward_difference_dr_minus_canonical": (
                    reward_difference
                ),
                "canonical_length": int(
                    canonical[
                        "response_length"
                    ]
                ),
                "dr_grpo_length": int(
                    dr["response_length"]
                ),
                "length_difference_dr_minus_canonical": (
                    int(
                        dr["response_length"]
                    )
                    - int(
                        canonical[
                            "response_length"
                        ]
                    )
                ),
                "canonical_sampled_kl": float(
                    canonical[
                        "sampled_kl"
                    ]
                ),
                "dr_grpo_sampled_kl": float(
                    dr["sampled_kl"]
                ),
                "sampled_kl_difference_dr_minus_canonical": (
                    float(
                        dr["sampled_kl"]
                    )
                    - float(
                        canonical[
                            "sampled_kl"
                        ]
                    )
                ),
            }
        )

    reward_candidates = [
        row
        for row in paired
        if row[
            "reward_difference_dr_minus_canonical"
        ] is not None
    ]

    return {
        "num_paired_examples": len(
            paired
        ),
        "largest_reward_differences": sorted(
            reward_candidates,
            key=lambda row: abs(
                row[
                    "reward_difference_dr_minus_canonical"
                ]
            ),
            reverse=True,
        )[:10],
        "largest_length_differences": sorted(
            paired,
            key=lambda row: abs(
                row[
                    "length_difference_dr_minus_canonical"
                ]
            ),
            reverse=True,
        )[:10],
        "largest_kl_differences": sorted(
            paired,
            key=lambda row: abs(
                row[
                    "sampled_kl_difference_dr_minus_canonical"
                ]
            ),
            reverse=True,
        )[:10],
    }


def run_normalization_study(
    config_path,
    fork_updates=None,
    eval_max_examples=None,
    skip_reward=False,
    skip_training=False,
    skip_evaluation=False,
):
    cfg = load_yaml(config_path)

    selected_fork_updates = int(
        cfg["fork_updates"]
        if fork_updates is None
        else fork_updates
    )

    if selected_fork_updates < 1:
        raise ValueError(
            "fork_updates must be positive."
        )

    results_dir = repo_path(
        cfg["results_dir"]
    )
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_summaries = {}
    evaluation_metrics = {}

    for condition in CONDITIONS:
        training_summaries[
            condition["condition"]
        ] = _run_or_load_training(
            config_path=config_path,
            cfg=cfg,
            condition=condition,
            fork_updates=(
                selected_fork_updates
            ),
            skip_training=skip_training,
        )

        evaluation_metrics[
            condition["condition"]
        ] = _run_or_load_evaluation(
            config_path=config_path,
            cfg=cfg,
            condition=condition,
            eval_max_examples=(
                eval_max_examples
            ),
            skip_reward=skip_reward,
            skip_evaluation=(
                skip_evaluation
            ),
        )

    comparison_rows = []
    completion_detail_rows = []
    completion_length_rows = []
    update_detail_rows = []
    update_length_rows = []

    for condition in CONDITIONS:
        condition_name = condition[
            "condition"
        ]

        comparison_rows.append(
            build_comparison_row(
                condition=condition,
                training_summary=(
                    training_summaries[
                        condition_name
                    ]
                ),
                evaluation_metrics=(
                    evaluation_metrics[
                        condition_name
                    ]
                ),
            )
        )

        (
            condition_completion_details,
            condition_completion_summary,
        ) = (
            analyze_length_conditioned_weights(
                cfg=cfg,
                condition=condition,
            )
        )

        completion_detail_rows.extend(
            condition_completion_details
        )

        completion_length_rows.extend(
            condition_completion_summary
        )

        (
            condition_update_details,
            condition_update_summary,
        ) = (
            analyze_update_length_conditioning(
                cfg=cfg,
                condition=condition,
            )
        )

        update_detail_rows.extend(
            condition_update_details
        )

        update_length_rows.extend(
            condition_update_summary
        )

    qualitative = (
        build_qualitative_comparisons(
            cfg
        )
    )

    comparison_path = (
        results_dir
        / "normalization_comparison.csv"
    )

    completion_length_path = (
        results_dir
        / (
            "normalization_length_"
            "conditioned.csv"
        )
    )

    completion_details_path = (
        results_dir
        / (
            "normalization_completion_"
            "details.csv"
        )
    )

    update_length_path = (
        results_dir
        / (
            "normalization_update_length_"
            "conditioned.csv"
        )
    )

    update_details_path = (
        results_dir
        / (
            "normalization_update_details.csv"
        )
    )

    qualitative_path = (
        results_dir
        / (
            "normalization_qualitative_"
            "candidates.json"
        )
    )

    summary_path = (
        results_dir
        / "normalization_study_summary.json"
    )

    pd.DataFrame(
        comparison_rows
    ).to_csv(
        comparison_path,
        index=False,
    )

    pd.DataFrame(
        completion_length_rows
    ).to_csv(
        completion_length_path,
        index=False,
    )

    pd.DataFrame(
        completion_detail_rows
    ).to_csv(
        completion_details_path,
        index=False,
    )

    pd.DataFrame(
        update_length_rows
    ).to_csv(
        update_length_path,
        index=False,
    )

    pd.DataFrame(
        update_detail_rows
    ).to_csv(
        update_details_path,
        index=False,
    )

    save_json(
        qualitative_path,
        qualitative,
    )

    summary = {
        "config": config_path,
        "fork_updates": (
            selected_fork_updates
        ),
        "eval_max_examples": (
            eval_max_examples
        ),
        "reward_scoring_skipped": bool(
            skip_reward
        ),
        "training_skipped": bool(
            skip_training
        ),
        "evaluation_skipped": bool(
            skip_evaluation
        ),
        "matched_controls": {
            "starting_checkpoint": str(
                cfg["paths"][
                    "grpo_midpoint_policy"
                ]
            ),
            "seed": int(cfg["seed"]),
            "prompts_per_update": int(
                cfg.get(
                    "prompts_per_update",
                    1,
                )
            ),
            "num_generations": int(
                cfg["num_generations"]
            ),
            "clip_epsilon": float(
                cfg["clip_epsilon"]
            ),
            "kl_beta": float(
                cfg["kl_beta"]
            ),
            "learning_rate": float(
                cfg["learning_rate"]
            ),
            "max_completion_length": int(
                cfg[
                    "max_completion_length"
                ]
            ),
        },
        "conditions": comparison_rows,
        "length_conditioned_completion_statistics": (
            completion_length_rows
        ),
        "length_conditioned_update_statistics": (
            update_length_rows
        ),
        "comparison_table": str(
            comparison_path.relative_to(
                repo_path(".")
            )
        ),
        "completion_length_table": str(
            completion_length_path.relative_to(
                repo_path(".")
            )
        ),
        "completion_details_table": str(
            completion_details_path.relative_to(
                repo_path(".")
            )
        ),
        "update_length_table": str(
            update_length_path.relative_to(
                repo_path(".")
            )
        ),
        "update_details_table": str(
            update_details_path.relative_to(
                repo_path(".")
            )
        ),
        "qualitative_candidates": str(
            qualitative_path.relative_to(
                repo_path(".")
            )
        ),
    }

    save_json(
        summary_path,
        summary,
    )

    print()
    print(
        "GRPO normalization study summary"
    )
    print(
        "condition | held-out reward | "
        "held-out KL | entropy | "
        "mean length | max gradient norm"
    )

    for row in comparison_rows:
        print(
            f"{row['condition']} | "
            f"{row['heldout_reward']} | "
            f"{row['heldout_sampled_kl']} | "
            f"{row['heldout_entropy']} | "
            f"{row['heldout_mean_response_length']} | "
            f"{row['train_max_gradient_norm']}"
        )

    print(
        "Saved normalization summary to "
        f"{summary_path}"
    )
    print(
        "Saved normalization comparison to "
        f"{comparison_path}"
    )
    print(
        "Saved length-conditioned completion "
        f"statistics to {completion_length_path}"
    )
    print(
        "Saved length-conditioned update "
        f"statistics to {update_length_path}"
    )
    print(
        "Saved qualitative candidates to "
        f"{qualitative_path}"
    )

    return summary


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run matched canonical-GRPO and "
            "Dr-GRPO short continuation forks."
        )
    )

    ap.add_argument(
        "--config",
        default="configs/grpo.yaml",
    )

    ap.add_argument(
        "--fork-updates",
        type=int,
    )

    ap.add_argument(
        "--eval-max-examples",
        type=int,
    )

    ap.add_argument(
        "--skip-reward",
        action="store_true",
    )

    ap.add_argument(
        "--skip-training",
        action="store_true",
    )

    ap.add_argument(
        "--skip-evaluation",
        action="store_true",
    )

    args = ap.parse_args()

    run_normalization_study(
        config_path=args.config,
        fork_updates=args.fork_updates,
        eval_max_examples=(
            args.eval_max_examples
        ),
        skip_reward=args.skip_reward,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation,
    )


if __name__ == "__main__":
    main()