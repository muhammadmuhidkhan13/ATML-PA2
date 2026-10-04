from __future__ import annotations

import argparse
import csv
import gc
import json

import torch

from common.data import load_yaml, repo_path
from task1_dpo.evaluate import run_evaluation
from task1_dpo.train import run_training


def beta_tag(beta: float) -> str:
    """Return a filesystem-safe label for a beta value."""
    return f"{beta:g}".replace("-", "neg").replace(".", "p")


def clear_model_memory() -> None:
    """Release unused memory between training and evaluation stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_json(path):
    """Read a JSON file and return its contents."""
    return json.loads(path.read_text(encoding="utf-8"))


def run_beta_ablation(
    config_path: str,
    betas: list[float] | None = None,
    max_examples: int | None = None,
    eval_max_examples: int | None = None,
    skip_reward: bool = False,
):
    """Run matched DPO training and evaluation for several beta values."""

    cfg = load_yaml(config_path)

    beta_values = [
        float(value)
        for value in (
            cfg["betas"]
            if betas is None
            else betas
        )
    ]

    train_limit = int(
        cfg["short_ablation_examples"]
        if max_examples is None
        else max_examples
    )

    output_root = repo_path(
        "outputs/task1_dpo/beta_ablation"
    )
    results_dir = repo_path(
        cfg["results_dir"]
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    conditions = []

    for index, beta in enumerate(
        beta_values,
        start=1,
    ):
        run_name = (
            f"beta_{beta_tag(beta)}"
        )
        eval_name = (
            f"{run_name}_eval"
        )

        adapter_path = (
            output_root / run_name
        )
        train_summary_path = (
            results_dir
            / f"{run_name}_train_summary.json"
        )
        eval_metrics_path = (
            results_dir
            / f"{eval_name}_metrics.json"
        )

        print()
        print(
            f"[{index}/{len(beta_values)}] "
            f"Training matched condition "
            f"with beta={beta:g}"
        )

        try:
            run_training(
                config_path=config_path,
                run_name=run_name,
                output_path=str(
                    adapter_path
                ),
                beta=beta,
                max_examples=train_limit,
            )
        finally:
            clear_model_memory()

        if not train_summary_path.exists():
            raise FileNotFoundError(
                "Expected training summary "
                "was not created: "
                f"{train_summary_path}"
            )

        print(
            f"[{index}/{len(beta_values)}] "
            f"Evaluating beta={beta:g}"
        )

        try:
            run_evaluation(
                config_path=config_path,
                adapter=str(
                    adapter_path
                ),
                name=eval_name,
                max_examples=(
                    eval_max_examples
                ),
                skip_reward=skip_reward,
            )
        finally:
            clear_model_memory()

        if not eval_metrics_path.exists():
            raise FileNotFoundError(
                "Expected evaluation metrics "
                "were not created: "
                f"{eval_metrics_path}"
            )

        conditions.append(
            {
                "run_name": run_name,
                "beta": beta,
                "adapter": str(
                    adapter_path
                ),
                "training_summary_file": str(
                    train_summary_path
                ),
                "evaluation_metrics_file": str(
                    eval_metrics_path
                ),
                "training": read_json(
                    train_summary_path
                ),
                "evaluation": read_json(
                    eval_metrics_path
                ),
            }
        )

    summary = {
        "experiment":
            "task1_dpo_beta_ablation",

        "config":
            config_path,

        "seed":
            int(cfg["seed"]),

        "betas":
            beta_values,

        "train_examples_per_condition":
            train_limit,

        "evaluation_max_examples":
            eval_max_examples,

        "skip_reward":
            skip_reward,

        "matched_controls": {
            "fresh_initialization_per_condition":
                True,

            "training_dataset":
                cfg["paths"][
                    "dpo_standard_train"
                ],

            "evaluation_dataset":
                cfg["paths"][
                    "dpo_standard_eval"
                ],

            "seed":
                int(cfg["seed"]),

            "batch_size":
                int(cfg["batch_size"]),

            "gradient_accumulation_steps":
                int(cfg["grad_accum_steps"]),

            "epochs":
                int(cfg["epochs"]),

            "learning_rate":
                float(cfg["learning_rate"]),

            "max_sequence_length":
                int(
                    cfg[
                        "max_sequence_length"
                    ]
                ),
        },

        "conditions":
            conditions,
    }

    summary_path = (
        results_dir
        / "beta_ablation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = (
        results_dir
        / "beta_ablation_comparison.csv"
    )

    columns = [
        "beta",
        "preference_accuracy",
        "dpo_loss",
        "relative_margin_mean",
        "sampled_kl_token_mean",
        "reward_model_score_mean",
        "response_length_mean",
        "truncation_rate",
        "train_wall_clock_seconds",
        "evaluation_wall_clock_seconds",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )
        writer.writeheader()

        for condition in conditions:
            training = (
                condition["training"]
            )
            evaluation = (
                condition["evaluation"]
            )

            writer.writerow(
                {
                    "beta":
                        condition["beta"],

                    "preference_accuracy":
                        evaluation.get(
                            "preference_accuracy"
                        ),

                    "dpo_loss":
                        evaluation.get(
                            "dpo_loss"
                        ),

                    "relative_margin_mean":
                        evaluation.get(
                            "relative_margin_mean"
                        ),

                    "sampled_kl_token_mean":
                        evaluation.get(
                            "sampled_kl_token_mean"
                        ),

                    "reward_model_score_mean":
                        evaluation.get(
                            "reward_model_score_mean"
                        ),

                    "response_length_mean":
                        evaluation.get(
                            "response_length_mean"
                        ),

                    "truncation_rate":
                        evaluation.get(
                            "truncation_rate"
                        ),

                    "train_wall_clock_seconds":
                        training.get(
                            "wall_clock_seconds"
                        ),

                    "evaluation_wall_clock_seconds":
                        evaluation.get(
                            "wall_clock_seconds"
                        ),
                }
            )

    print()
    print(
        "DPO beta ablation summary"
    )
    print(
        "beta | preference accuracy | "
        "sampled KL | reward | mean length"
    )

    for condition in conditions:
        evaluation = (
            condition["evaluation"]
        )

        accuracy = evaluation.get(
            "preference_accuracy"
        )
        kl_value = evaluation.get(
            "sampled_kl_token_mean"
        )
        reward = evaluation.get(
            "reward_model_score_mean"
        )
        length = evaluation.get(
            "response_length_mean"
        )

        print(
            f"{condition['beta']:g} | "
            f"{accuracy} | "
            f"{kl_value} | "
            f"{reward} | "
            f"{length}"
        )

    print(
        "Saved beta-ablation summary to "
        f"{summary_path}"
    )
    print(
        "Saved beta comparison table to "
        f"{csv_path}"
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched Task 1 "
            "DPO beta ablations."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/dpo.yaml",
    )

    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
    )

    parser.add_argument(
        "--max-examples",
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

    args = parser.parse_args()

    run_beta_ablation(
        config_path=args.config,
        betas=args.betas,
        max_examples=args.max_examples,
        eval_max_examples=(
            args.eval_max_examples
        ),
        skip_reward=args.skip_reward,
    )


if __name__ == "__main__":
    main()