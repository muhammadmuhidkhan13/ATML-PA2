from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch

from common.data import (
    load_yaml,
    read_jsonl,
    repo_path,
)
from common.logging_utils import (
    load_json,
    save_json,
    wall_timer,
)
from task2_ppo.continue_train import (
    run_ppo,
)
from task2_ppo.evaluate import (
    run_evaluation,
)


def beta_tag(beta):
    text = f"{float(beta):g}"

    return text.replace(
        ".",
        "p",
    )


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


def adapter_path_for_beta(beta):
    return (
        "outputs/task2_ppo/"
        "kl_ablation/"
        f"kl_{beta_tag(beta)}"
    )


def run_name_for_beta(beta):
    return (
        f"kl_{beta_tag(beta)}"
    )


def load_existing_training_summary(
    results_dir,
    run_name,
):
    path = (
        results_dir
        / f"{run_name}_train_summary.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Cannot skip training because the expected "
            f"training summary does not exist: {path}"
        )

    return load_json(path)


def load_condition_generations(
    evaluation_metrics,
):
    generations_path = (
        evaluation_metrics.get(
            "generations"
        )
    )

    if not generations_path:
        return []

    path = repo_path(
        generations_path
    )

    if not path.exists():
        return []

    return read_jsonl(path)


def create_cross_condition_candidates(
    condition_generations,
    baseline_beta=0.10,
):
    if not condition_generations:
        return {
            "baseline_beta": (
                baseline_beta
            ),
            "comparisons": [],
        }

    available_betas = sorted(
        condition_generations
    )

    selected_baseline = min(
        available_betas,
        key=lambda value: abs(
            float(value)
            - float(baseline_beta)
        ),
    )

    baseline_rows = (
        condition_generations[
            selected_baseline
        ]
    )

    baseline_by_prompt = {
        row.get(
            "prompt_id",
            row.get(
                "source_index"
            ),
        ): row
        for row in baseline_rows
    }

    comparisons = []

    for beta in available_betas:
        if beta == selected_baseline:
            continue

        candidate_rows = []

        for row in condition_generations[
            beta
        ]:
            prompt_key = row.get(
                "prompt_id",
                row.get(
                    "source_index"
                ),
            )

            baseline_row = (
                baseline_by_prompt.get(
                    prompt_key
                )
            )

            if baseline_row is None:
                continue

            candidate_reward = row.get(
                "reward_model_score"
            )

            baseline_reward = (
                baseline_row.get(
                    "reward_model_score"
                )
            )

            if (
                candidate_reward is None
                or baseline_reward is None
            ):
                reward_delta = None
            else:
                reward_delta = float(
                    candidate_reward
                    - baseline_reward
                )

            candidate_rows.append(
                {
                    "prompt_id": row.get(
                        "prompt_id"
                    ),
                    "source_index": row.get(
                        "source_index"
                    ),
                    "baseline_beta": float(
                        selected_baseline
                    ),
                    "comparison_beta": float(
                        beta
                    ),
                    "reward_delta": (
                        reward_delta
                    ),
                    "kl_delta": float(
                        row.get(
                            "sampled_kl",
                            0.0,
                        )
                        - baseline_row.get(
                            "sampled_kl",
                            0.0,
                        )
                    ),
                    "entropy_delta": float(
                        row.get(
                            "entropy",
                            0.0,
                        )
                        - baseline_row.get(
                            "entropy",
                            0.0,
                        )
                    ),
                    "length_delta": int(
                        row.get(
                            "response_length",
                            0,
                        )
                        - baseline_row.get(
                            "response_length",
                            0,
                        )
                    ),
                    "prompt_messages": (
                        row.get(
                            "prompt_messages"
                        )
                    ),
                    "baseline_response": (
                        baseline_row.get(
                            "response"
                        )
                    ),
                    "comparison_response": (
                        row.get(
                            "response"
                        )
                    ),
                    "baseline_reward": (
                        baseline_reward
                    ),
                    "comparison_reward": (
                        candidate_reward
                    ),
                    "baseline_kl": (
                        baseline_row.get(
                            "sampled_kl"
                        )
                    ),
                    "comparison_kl": (
                        row.get(
                            "sampled_kl"
                        )
                    ),
                    "baseline_entropy": (
                        baseline_row.get(
                            "entropy"
                        )
                    ),
                    "comparison_entropy": (
                        row.get(
                            "entropy"
                        )
                    ),
                    "baseline_length": (
                        baseline_row.get(
                            "response_length"
                        )
                    ),
                    "comparison_length": (
                        row.get(
                            "response_length"
                        )
                    ),
                }
            )

        reward_increase = [
            row
            for row in candidate_rows
            if row[
                "reward_delta"
            ] is not None
        ]

        reward_increase = sorted(
            reward_increase,
            key=lambda row: (
                row["reward_delta"]
            ),
            reverse=True,
        )

        largest_drift = sorted(
            candidate_rows,
            key=lambda row: (
                row["kl_delta"]
            ),
            reverse=True,
        )

        largest_length_change = sorted(
            candidate_rows,
            key=lambda row: abs(
                row[
                    "length_delta"
                ]
            ),
            reverse=True,
        )

        potential_overoptimization = [
            row
            for row in reward_increase
            if (
                row[
                    "reward_delta"
                ] > 0
                and row[
                    "kl_delta"
                ] > 0
            )
        ]

        comparisons.append(
            {
                "comparison_beta": float(
                    beta
                ),
                "num_matched_prompts": len(
                    candidate_rows
                ),
                "largest_reward_increases": (
                    reward_increase[:5]
                ),
                "largest_positive_kl_changes": (
                    largest_drift[:5]
                ),
                "largest_length_changes": (
                    largest_length_change[:5]
                ),
                "potential_overoptimization_candidates": (
                    potential_overoptimization[
                        :5
                    ]
                ),
            }
        )

    return {
        "baseline_beta": float(
            selected_baseline
        ),
        "note": (
            "These are candidates for manual qualitative "
            "inspection. Reward and KL changes do not by "
            "themselves establish response-quality changes."
        ),
        "comparisons": comparisons,
    }


def run_kl_ablation(
    config_path,
    betas,
    fork_updates,
    eval_max_examples=None,
    skip_reward=False,
    skip_training=False,
    skip_evaluation=False,
):
    cfg = load_yaml(config_path)

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    condition_rows = []
    condition_generations = {}

    for condition_index, beta in enumerate(
        betas
    ):
        beta = float(beta)

        run_name = (
            run_name_for_beta(beta)
        )

        output_path = (
            adapter_path_for_beta(beta)
        )

        print(
            f"\n[{condition_index + 1}/"
            f"{len(betas)}] "
            f"KL beta={beta}",
            flush=True,
        )

        if skip_training:
            training_summary = (
                load_existing_training_summary(
                    results_dir,
                    run_name,
                )
            )

            if not repo_path(
                output_path
            ).exists():
                raise FileNotFoundError(
                    "Cannot skip training because the "
                    "expected adapter does not exist: "
                    f"{repo_path(output_path)}"
                )
        else:
            training_summary = run_ppo(
                config_path=config_path,
                output=output_path,
                updates=int(
                    fork_updates
                ),
                clip_epsilon=float(
                    cfg[
                        "clip_epsilon"
                    ]
                ),
                kl_beta=beta,
                run_name=run_name,
            )

        evaluation_metrics = None
        generations = []

        if not skip_evaluation:
            evaluation_name = (
                f"{run_name}_eval"
            )

            print(
                f"Evaluating KL beta="
                f"{beta}",
                flush=True,
            )

            evaluation_metrics = (
                run_evaluation(
                    config_path=(
                        config_path
                    ),
                    adapter=output_path,
                    name=(
                        evaluation_name
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

            generations = (
                load_condition_generations(
                    evaluation_metrics
                )
            )

            condition_generations[
                beta
            ] = generations

        condition_row = {
            "kl_beta": beta,
            "run_name": run_name,
            "updates": int(
                fork_updates
            ),
            "clip_epsilon": float(
                cfg[
                    "clip_epsilon"
                ]
            ),
            "generated_tokens": (
                training_summary.get(
                    "total_generated_tokens"
                )
            ),
            "training_mean_reward": (
                training_summary.get(
                    "mean_learned_reward"
                )
            ),
            "training_mean_adjusted_reward": (
                training_summary.get(
                    "mean_adjusted_task_reward"
                )
            ),
            "training_mean_kl": (
                training_summary.get(
                    "mean_sampled_kl"
                )
            ),
            "training_mean_entropy": (
                training_summary.get(
                    "mean_entropy"
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
            "heldout_reward_std": (
                None
                if evaluation_metrics is None
                else evaluation_metrics.get(
                    "reward_model_score_std"
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
            "heldout_length_std": (
                None
                if evaluation_metrics is None
                else evaluation_metrics.get(
                    "response_length_std"
                )
            ),
            "heldout_eos_rate": (
                None
                if evaluation_metrics is None
                else evaluation_metrics.get(
                    "terminated_with_eos_rate"
                )
            ),
            "heldout_truncation_rate": (
                None
                if evaluation_metrics is None
                else evaluation_metrics.get(
                    "truncation_rate"
                )
            ),
            "num_heldout_prompts": (
                None
                if evaluation_metrics is None
                else evaluation_metrics.get(
                    "num_evaluation_prompts"
                )
            ),
            "policy_output": (
                training_summary.get(
                    "policy_output"
                )
            ),
        }

        condition_rows.append(
            condition_row
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        condition_rows,
        condition_generations,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run matched PPO KL-pressure ablations "
            "from the supplied midpoint checkpoints."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/ppo.yaml",
    )

    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
    )

    parser.add_argument(
        "--fork-updates",
        type=int,
    )

    parser.add_argument(
        "--eval-max-examples",
        type=int,
    )

    parser.add_argument(
        "--skip-reward",
        action="store_true",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    betas = (
        args.betas
        if args.betas is not None
        else [
            float(value)
            for value in cfg[
                "kl_values"
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

    summary_path = (
        results_dir
        / "kl_ablation_summary.json"
    )

    comparison_csv_path = (
        results_dir
        / "kl_ablation_comparison.csv"
    )

    qualitative_path = (
        results_dir
        / "kl_ablation_qualitative_candidates.json"
    )

    timer = wall_timer()

    (
        condition_rows,
        condition_generations,
    ) = run_kl_ablation(
        config_path=args.config,
        betas=betas,
        fork_updates=fork_updates,
        eval_max_examples=(
            args.eval_max_examples
        ),
        skip_reward=(
            args.skip_reward
        ),
        skip_training=(
            args.skip_training
        ),
        skip_evaluation=(
            args.skip_evaluation
        ),
    )

    qualitative_candidates = (
        create_cross_condition_candidates(
            condition_generations,
            baseline_beta=0.10,
        )
    )

    summary = {
        "config": args.config,
        "betas": [
            float(value)
            for value in betas
        ],
        "fork_updates": (
            fork_updates
        ),
        "fixed_clip_epsilon": float(
            cfg["clip_epsilon"]
        ),
        "conditions": (
            condition_rows
        ),
        "wall_clock_seconds": float(
            timer()
        ),
        "comparison_table": str(
            comparison_csv_path.relative_to(
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

    save_csv(
        comparison_csv_path,
        condition_rows,
    )

    save_json(
        qualitative_path,
        qualitative_candidates,
    )

    print(
        "\nPPO KL-pressure ablation summary",
        flush=True,
    )

    print(
        "beta | held-out reward | "
        "held-out KL | entropy | "
        "mean length",
        flush=True,
    )

    for row in condition_rows:
        print(
            f"{row['kl_beta']} | "
            f"{row['heldout_reward']} | "
            f"{row['heldout_kl']} | "
            f"{row['heldout_entropy']} | "
            f"{row['heldout_length']}",
            flush=True,
        )

    print(
        "Saved KL-ablation summary to "
        f"{summary_path}",
        flush=True,
    )

    print(
        "Saved KL comparison table to "
        f"{comparison_csv_path}",
        flush=True,
    )

    print(
        "Saved qualitative candidates to "
        f"{qualitative_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()