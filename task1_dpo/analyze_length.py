from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.metrics import (
    parse_word_limit,
    word_count,
    word_limit_compliance,
)
from common.models import load_policy, load_tokenizer
from task1_dpo.evaluate import evaluate_preference_pairs
from task1_dpo.train import run_training


STRATA = [
    "preferred_longer",
    "length_matched",
    "rejected_longer",
]


def clear_model_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean(values) -> float | None:
    values = [
        float(value)
        for value in values
    ]

    if not values:
        return None

    return float(
        statistics.fmean(values)
    )


def std(values) -> float | None:
    values = [
        float(value)
        for value in values
    ]

    if not values:
        return None

    return float(
        statistics.pstdev(values)
    )


def iqr(values) -> float | None:
    values = [
        float(value)
        for value in values
    ]

    if not values:
        return None

    q1, q3 = np.percentile(
        values,
        [25, 75],
    )

    return float(q3 - q1)


def write_json(
    path: Path,
    payload,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def aggregate_strata(
    details: list[dict],
) -> dict[str, dict]:
    output = {}

    for stratum in STRATA:
        selected = [
            row
            for row in details
            if row["length_stratum"]
            == stratum
        ]

        output[stratum] = {
            "num_pairs":
                len(selected),

            "preference_accuracy":
                mean(
                    float(
                        row[
                            "preference_correct"
                        ]
                    )
                    for row in selected
                ),

            "dpo_loss":
                mean(
                    row["dpo_loss"]
                    for row in selected
                ),

            "relative_margin_mean":
                mean(
                    row["relative_margin"]
                    for row in selected
                ),

            "relative_margin_std":
                std(
                    row["relative_margin"]
                    for row in selected
                ),

            "policy_margin_mean":
                mean(
                    row["policy_margin"]
                    for row in selected
                ),

            "reference_margin_mean":
                mean(
                    row["reference_margin"]
                    for row in selected
                ),

            "chosen_tokens_mean":
                mean(
                    row["chosen_tokens"]
                    for row in selected
                ),

            "rejected_tokens_mean":
                mean(
                    row["rejected_tokens"]
                    for row in selected
                ),

            "length_difference_mean":
                mean(
                    row["length_difference"]
                    for row in selected
                ),
        }

    return output


@torch.inference_mode()
def evaluate_policy(
    label: str,
    adapter_path: str,
    cfg: dict,
    tokenizer,
    stratified_rows: list[dict],
    word_rows: list[dict],
) -> dict:
    print(
        f"\nLoading {label} policy "
        f"from {adapter_path}"
    )

    set_seed(
        int(cfg["seed"])
    )

    policy = load_policy(
        cfg,
        adapter_path=adapter_path,
        trainable=False,
    )

    pair_summary, pair_details = (
        evaluate_preference_pairs(
            policy=policy,
            tokenizer=tokenizer,
            rows=stratified_rows,
            batch_size=int(
                cfg["batch_size"]
            ),
            max_sequence_length=int(
                cfg[
                    "max_sequence_length"
                ]
            ),
            beta=float(cfg["beta"]),
        )
    )

    metadata = {
        row["prompt_id"]: row
        for row in stratified_rows
    }

    for detail in pair_details:
        source = metadata[
            detail["prompt_id"]
        ]

        for key in (
            "length_stratum",
            "chosen_tokens",
            "rejected_tokens",
            "length_difference",
        ):
            detail[key] = source[key]

        detail["policy"] = label

    set_seed(
        int(cfg["seed"])
    )

    generation_cfg = (
        cfg["generation"]
    )

    generation_records = []

    batch_size = int(
        cfg["batch_size"]
    )

    for start in range(
        0,
        len(word_rows),
        batch_size,
    ):
        chunk = word_rows[
            start:
            start + batch_size
        ]

        prompts = [
            row["messages"]
            for row in chunk
        ]

        generated = batch_generate(
            model=policy,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_length=int(
                cfg[
                    "max_sequence_length"
                ]
            ),
            max_new_tokens=int(
                cfg[
                    "max_generation_tokens"
                ]
            ),
            temperature=float(
                generation_cfg[
                    "temperature"
                ]
            ),
            top_p=float(
                generation_cfg["top_p"]
            ),
            do_sample=bool(
                generation_cfg[
                    "do_sample"
                ]
            ),
        )

        for index, row in enumerate(
            chunk
        ):
            prompt_text = (
                row["messages"][-1][
                    "content"
                ]
            )

            response = (
                generated[
                    "responses"
                ][index]
            )

            generation_records.append(
                {
                    "policy":
                        label,

                    "prompt_id":
                        row["prompt_id"],

                    "prompt":
                        prompt_text,

                    "word_limit":
                        parse_word_limit(
                            prompt_text
                        ),

                    "response":
                        response,

                    "response_words":
                        word_count(
                            response
                        ),

                    "response_tokens":
                        int(
                            generated[
                                "response_lengths"
                            ][index]
                        ),

                    "word_limit_compliance":
                        word_limit_compliance(
                            prompt_text,
                            response,
                        ),

                    "terminated_with_eos":
                        bool(
                            generated[
                                "terminated_with_eos"
                            ][index]
                        ),

                    "truncated":
                        bool(
                            generated[
                                "truncated"
                            ][index]
                        ),
                }
            )

    lengths = [
        row["response_tokens"]
        for row in generation_records
    ]

    compliance_values = [
        row[
            "word_limit_compliance"
        ]
        for row in generation_records
        if row[
            "word_limit_compliance"
        ]
        is not None
    ]

    generation_summary = {
        "num_prompts":
            len(generation_records),

        "response_length_mean":
            mean(lengths),

        "response_length_std":
            std(lengths),

        "response_length_iqr":
            iqr(lengths),

        "word_limit_compliance_rate":
            mean(compliance_values),

        "terminated_with_eos_rate":
            mean(
                float(
                    row[
                        "terminated_with_eos"
                    ]
                )
                for row
                in generation_records
            ),

        "truncation_rate":
            mean(
                float(
                    row["truncated"]
                )
                for row
                in generation_records
            ),
    }

    del policy
    clear_model_memory()

    return {
        "label":
            label,

        "adapter":
            adapter_path,

        "pair_summary":
            pair_summary,

        "strata":
            aggregate_strata(
                pair_details
            ),

        "pair_details":
            pair_details,

        "word_limit_summary":
            generation_summary,

        "word_limit_generations":
            generation_records,
    }


def select_qualitative_candidates(
    standard: dict,
    balanced: dict,
    rows: list[dict],
) -> list[dict]:
    source = {
        row["prompt_id"]: row
        for row in rows
    }

    standard_map = {
        row["prompt_id"]: row
        for row
        in standard["pair_details"]
    }

    balanced_map = {
        row["prompt_id"]: row
        for row
        in balanced["pair_details"]
    }

    candidates = []

    for stratum in STRATA:
        prompt_ids = [
            row["prompt_id"]
            for row in rows
            if (
                row["length_stratum"]
                == stratum
                and row["prompt_id"]
                in standard_map
                and row["prompt_id"]
                in balanced_map
            )
        ]

        prompt_ids.sort(
            key=lambda prompt_id: abs(
                balanced_map[
                    prompt_id
                ][
                    "relative_margin"
                ]
                -
                standard_map[
                    prompt_id
                ][
                    "relative_margin"
                ]
            ),
            reverse=True,
        )

        for prompt_id in prompt_ids[:2]:
            dataset_row = source[
                prompt_id
            ]

            chosen_text = (
                dataset_row[
                    "chosen"
                ][-1]["content"]
            )

            rejected_text = (
                dataset_row[
                    "rejected"
                ][-1]["content"]
            )

            candidates.append(
                {
                    "length_stratum":
                        stratum,

                    "prompt_id":
                        prompt_id,

                    "prompt":
                        dataset_row[
                            "prompt"
                        ],

                    "chosen_excerpt":
                        chosen_text[:500],

                    "rejected_excerpt":
                        rejected_text[:500],

                    "standard_relative_margin":
                        standard_map[
                            prompt_id
                        ][
                            "relative_margin"
                        ],

                    "balanced_relative_margin":
                        balanced_map[
                            prompt_id
                        ][
                            "relative_margin"
                        ],

                    "standard_preference_correct":
                        standard_map[
                            prompt_id
                        ][
                            "preference_correct"
                        ],

                    "balanced_preference_correct":
                        balanced_map[
                            prompt_id
                        ][
                            "preference_correct"
                        ],
                }
            )

    return candidates


def write_strata_csv(
    path: Path,
    policy_results: list[dict],
) -> None:
    columns = [
        "policy",
        "stratum",
        "num_pairs",
        "preference_accuracy",
        "dpo_loss",
        "relative_margin_mean",
        "relative_margin_std",
        "policy_margin_mean",
        "reference_margin_mean",
        "chosen_tokens_mean",
        "rejected_tokens_mean",
        "length_difference_mean",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
        )

        writer.writeheader()

        for result in policy_results:
            for stratum in STRATA:
                metrics = (
                    result["strata"][
                        stratum
                    ]
                )

                writer.writerow(
                    {
                        "policy":
                            result["label"],

                        "stratum":
                            stratum,

                        **metrics,
                    }
                )


def run_length_analysis(
    config_path: str,
    standard_adapter: str | None = None,
    balanced_adapter: str | None = None,
    skip_training: bool = False,
    max_train_examples: int | None = None,
    eval_max_examples: int | None = None,
    word_limit_max_examples: int | None = None,
) -> dict:
    cfg = load_yaml(
        config_path
    )

    standard_path = repo_path(
        standard_adapter
        or cfg["standard_output"]
    )

    balanced_path = repo_path(
        balanced_adapter
        or cfg["length_output"]
    )

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not standard_path.exists():
        raise FileNotFoundError(
            "Standard DPO adapter "
            f"not found at {standard_path}. "
            "Run the standard Task 1 "
            "training command first."
        )

    if not skip_training:
        try:
            run_training(
                config_path=config_path,
                run_name=(
                    "length_balanced"
                ),
                dataset_path=(
                    cfg["paths"][
                        "dpo_length_train"
                    ]
                ),
                output_path=str(
                    balanced_path
                ),
                beta=float(
                    cfg["beta"]
                ),
                max_examples=(
                    max_train_examples
                ),
            )
        finally:
            clear_model_memory()

    if not balanced_path.exists():
        raise FileNotFoundError(
            "Length-balanced adapter "
            f"not found at {balanced_path}."
        )

    stratified_rows = read_jsonl(
        cfg["paths"][
            "dpo_length_eval"
        ]
    )

    word_rows = read_jsonl(
        cfg["paths"][
            "word_limit_prompts"
        ]
    )

    if eval_max_examples is not None:
        stratified_rows = (
            stratified_rows[
                :int(
                    eval_max_examples
                )
            ]
        )

    if (
        word_limit_max_examples
        is not None
    ):
        word_rows = (
            word_rows[
                :int(
                    word_limit_max_examples
                )
            ]
        )

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    standard = evaluate_policy(
        label="standard",
        adapter_path=str(
            standard_path
        ),
        cfg=cfg,
        tokenizer=tokenizer,
        stratified_rows=(
            stratified_rows
        ),
        word_rows=word_rows,
    )

    balanced = evaluate_policy(
        label="length_balanced",
        adapter_path=str(
            balanced_path
        ),
        cfg=cfg,
        tokenizer=tokenizer,
        stratified_rows=(
            stratified_rows
        ),
        word_rows=word_rows,
    )

    policy_results = [
        standard,
        balanced,
    ]

    summary = {
        "experiment":
            "task1_dpo_length_confounding",

        "config":
            config_path,

        "seed":
            int(cfg["seed"]),

        "num_stratified_rows":
            len(stratified_rows),

        "num_word_limit_prompts":
            len(word_rows),

        "policies": {
            result["label"]: {
                "adapter":
                    result["adapter"],

                "pair_summary":
                    result[
                        "pair_summary"
                    ],

                "strata":
                    result["strata"],

                "word_limit_summary":
                    result[
                        "word_limit_summary"
                    ],
            }
            for result
            in policy_results
        },
    }

    write_json(
        results_dir
        / "length_analysis_summary.json",
        summary,
    )

    write_strata_csv(
        results_dir
        / "length_strata_comparison.csv",
        policy_results,
    )

    write_jsonl(
        results_dir
        / "length_pair_details.jsonl",
        (
            standard[
                "pair_details"
            ]
            +
            balanced[
                "pair_details"
            ]
        ),
    )

    word_generations = (
        standard[
            "word_limit_generations"
        ]
        +
        balanced[
            "word_limit_generations"
        ]
    )

    write_jsonl(
        results_dir
        / "word_limit_generations.jsonl",
        word_generations,
    )

    write_json(
        results_dir
        / (
            "length_qualitative_"
            "candidates.json"
        ),
        select_qualitative_candidates(
            standard,
            balanced,
            stratified_rows,
        ),
    )

    print()
    print(
        "Length-confounding summary"
    )

    for result in policy_results:
        print()
        print(
            result["label"]
        )

        for stratum in STRATA:
            metrics = (
                result["strata"][
                    stratum
                ]
            )

            print(
                f"  {stratum}: "
                f"n={metrics['num_pairs']}, "
                "accuracy="
                f"{metrics['preference_accuracy']}"
            )

        word_summary = (
            result[
                "word_limit_summary"
            ]
        )

        print(
            "  word-limit prompts: "
            "mean_tokens="
            f"{word_summary['response_length_mean']}, "
            "compliance="
            f"{word_summary['word_limit_compliance_rate']}"
        )

    print()
    print(
        "Saved length analysis to "
        f"{results_dir}"
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Task 1 DPO "
            "length-confounding study."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/dpo.yaml",
    )

    parser.add_argument(
        "--standard-adapter",
    )

    parser.add_argument(
        "--balanced-adapter",
    )

    parser.add_argument(
        "--skip-training",
        action="store_true",
    )

    parser.add_argument(
        "--max-train-examples",
        type=int,
    )

    parser.add_argument(
        "--eval-max-examples",
        type=int,
    )

    parser.add_argument(
        "--word-limit-max-examples",
        type=int,
    )

    args = parser.parse_args()

    run_length_analysis(
        config_path=args.config,
        standard_adapter=(
            args.standard_adapter
        ),
        balanced_adapter=(
            args.balanced_adapter
        ),
        skip_training=(
            args.skip_training
        ),
        max_train_examples=(
            args.max_train_examples
        ),
        eval_max_examples=(
            args.eval_max_examples
        ),
        word_limit_max_examples=(
            args.word_limit_max_examples
        ),
    )


if __name__ == "__main__":
    main()