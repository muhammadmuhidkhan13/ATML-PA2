from __future__ import annotations

import argparse
from collections import defaultdict
from statistics import mean, pstdev

import numpy as np
import pandas as pd
import torch

from common.data import (
    load_yaml,
    read_jsonl,
    repo_path,
)
from common.logging_utils import save_json
from task3_grpo.grpo import (
    group_relative_advantages,
)


CACHE_GROUP_SIZE = 8


def _safe_mean(values):
    if not values:
        return None
    return float(mean(values))


def _safe_std(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def _safe_variance(values):
    if not values:
        return None

    array = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.var(
            array,
            ddof=0,
        )
    )


def load_k8_cache(path):
    rows = read_jsonl(path)

    if not rows:
        raise ValueError(
            "The supplied GRPO group cache is empty."
        )

    required = {
        "source_index",
        "prompt_id",
        "generation_index",
        "completion",
        "completion_tokens",
        "terminated_with_eos",
        "clipped_at_max",
        "reward",
    }

    missing = required.difference(
        rows[0]
    )

    if missing:
        raise ValueError(
            "Unexpected GRPO cache schema. "
            f"Missing keys: {sorted(missing)}"
        )

    by_prompt = defaultdict(list)

    for row in rows:
        prompt_key = str(
            row["source_index"]
        )

        by_prompt[prompt_key].append(
            dict(row)
        )

    short_groups = {
        prompt_key: len(group)
        for prompt_key, group
        in by_prompt.items()
        if len(group) < CACHE_GROUP_SIZE
    }

    if short_groups:
        raise ValueError(
            "Expected at least eight cached "
            "completions per prompt. "
            f"Short groups: {short_groups}"
        )

    for prompt_key, group in by_prompt.items():
        group.sort(
            key=lambda row: int(
                row.get(
                    "generation_index",
                    0,
                )
            )
        )

        # Use exactly the first eight cached
        # completions for every prompt so all K
        # conditions share an identical budget.
        by_prompt[prompt_key] = group[
            :CACHE_GROUP_SIZE
        ]

    return dict(by_prompt)


def regroup_equal_generation_budget(
    by_prompt,
    k: int,
):
    """Split each cached K=8 prompt group into disjoint K-sized groups.

    K=8:
        One group of eight per prompt.

    K=4:
        Two groups of four per prompt.

    K=2:
        Four groups of two per prompt.

    Every condition therefore uses the same eight completions per
    prompt and the same total number of cached generations.
    """
    if k < 2:
        raise ValueError(
            "Group size must be at least two."
        )

    if CACHE_GROUP_SIZE % k != 0:
        raise ValueError(
            f"K={k} does not divide the cached "
            f"group size {CACHE_GROUP_SIZE}."
        )

    regrouped = []

    for prompt_key in sorted(
        by_prompt,
        key=lambda value: int(value),
    ):
        completions = by_prompt[
            prompt_key
        ]

        if len(completions) != (
            CACHE_GROUP_SIZE
        ):
            raise ValueError(
                f"Prompt {prompt_key!r} does not "
                "contain exactly eight selected "
                "cached completions."
            )

        num_blocks = (
            CACHE_GROUP_SIZE // k
        )

        for block_index in range(
            num_blocks
        ):
            start = block_index * k
            stop = start + k

            block = completions[
                start:stop
            ]

            regrouped.append(
                {
                    "prompt_key": (
                        prompt_key
                    ),
                    "prompt_id": str(
                        block[0]["prompt_id"]
                    ),
                    "source_index": int(
                        block[0][
                            "source_index"
                        ]
                    ),
                    "block_index": (
                        block_index
                    ),
                    "k": int(k),
                    "rows": block,
                }
            )

    expected_generations = (
        len(by_prompt)
        * CACHE_GROUP_SIZE
    )

    observed_generations = sum(
        len(group["rows"])
        for group in regrouped
    )

    if (
        observed_generations
        != expected_generations
    ):
        raise RuntimeError(
            "Equal-generation regrouping failed: "
            f"expected {expected_generations}, "
            f"observed {observed_generations}."
        )

    return regrouped


def build_difficulty_bins(
    by_prompt,
):
    """Create fixed easy/hard bins from each prompt's K=8 mean reward.

    Prompts are ranked by mean reward across all eight cached
    completions. The lower half is labelled hard and the upper half
    easy. These labels are calculated once and reused for every K.
    """
    prompt_scores = []

    for prompt_key, rows in by_prompt.items():
        rewards = [
            float(row["reward"])
            for row in rows
        ]

        prompt_scores.append(
            {
                "prompt_key": (
                    prompt_key
                ),
                "source_index": int(
                    rows[0]["source_index"]
                ),
                "prompt_id": str(
                    rows[0]["prompt_id"]
                ),
                "mean_k8_reward": float(
                    np.mean(rewards)
                ),
            }
        )

    prompt_scores.sort(
        key=lambda item: (
            item["mean_k8_reward"],
            item["source_index"],
        )
    )

    split_index = (
        len(prompt_scores) // 2
    )

    difficulty_by_prompt = {}

    for rank, item in enumerate(
        prompt_scores
    ):
        difficulty = (
            "hard"
            if rank < split_index
            else "easy"
        )

        difficulty_by_prompt[
            item["prompt_key"]
        ] = difficulty

        item["difficulty"] = difficulty
        item["difficulty_rank"] = (
            rank + 1
        )

    return (
        difficulty_by_prompt,
        prompt_scores,
    )


def analyze_group(
    group,
    difficulty,
    tolerance,
):
    rows = group["rows"]

    rewards = torch.tensor(
        [
            float(row["reward"])
            for row in rows
        ],
        dtype=torch.float32,
    )

    group_ids = torch.zeros(
        len(rows),
        dtype=torch.long,
    )

    advantages = (
        group_relative_advantages(
            rewards=rewards,
            group_ids=group_ids,
            eps=tolerance,
        )
    )

    reward_std = float(
        rewards.std(
            unbiased=False
        ).item()
    )

    informative = bool(
        reward_std > tolerance
    )

    advantage_values = [
        float(value)
        for value in advantages.tolist()
    ]

    reward_values = [
        float(value)
        for value in rewards.tolist()
    ]

    completion_lengths = [
        int(row["completion_tokens"])
        for row in rows
    ]

    truncation_flags = [
        float(row["clipped_at_max"])
        for row in rows
    ]

    eos_flags = [
        float(
            row["terminated_with_eos"]
        )
        for row in rows
    ]

    return {
        "k": int(group["k"]),
        "prompt_key": str(
            group["prompt_key"]
        ),
        "prompt_id": str(
            group["prompt_id"]
        ),
        "source_index": int(
            group["source_index"]
        ),
        "block_index": int(
            group["block_index"]
        ),
        "difficulty": difficulty,
        "num_completions": len(rows),
        "generation_indices": [
            int(row["generation_index"])
            for row in rows
        ],
        "mean_reward": float(
            rewards.mean().item()
        ),
        "reward_std": reward_std,
        "reward_range": float(
            rewards.max().item()
            - rewards.min().item()
        ),
        "informative": informative,
        "uninformative": (
            not informative
        ),
        "advantage_mean": float(
            advantages.mean().item()
        ),
        "advantage_variance": float(
            advantages.var(
                unbiased=False
            ).item()
        ),
        "mean_absolute_advantage": float(
            advantages.abs().mean().item()
        ),
        "rewards": reward_values,
        "advantages": (
            advantage_values
        ),
        "mean_completion_tokens": float(
            np.mean(completion_lengths)
        ),
        "truncation_rate": float(
            np.mean(truncation_flags)
        ),
        "terminated_with_eos_rate": (
            float(np.mean(eos_flags))
        ),
    }


def summarize_groups(
    group_records,
    k,
    difficulty,
):
    selected = [
        record
        for record in group_records
        if record["k"] == k
        and (
            difficulty == "all"
            or record["difficulty"]
            == difficulty
        )
    ]

    if not selected:
        raise ValueError(
            f"No groups found for K={k}, "
            f"difficulty={difficulty!r}."
        )

    all_advantages = [
        advantage
        for record in selected
        for advantage in record[
            "advantages"
        ]
    ]

    informative_values = [
        float(record["informative"])
        for record in selected
    ]

    reward_stds = [
        float(record["reward_std"])
        for record in selected
    ]

    advantage_variances = [
        float(
            record[
                "advantage_variance"
            ]
        )
        for record in selected
    ]

    mean_absolute_advantages = [
        float(
            record[
                "mean_absolute_advantage"
            ]
        )
        for record in selected
    ]

    rewards = [
        reward
        for record in selected
        for reward in record["rewards"]
    ]

    completion_lengths = [
        float(
            record[
                "mean_completion_tokens"
            ]
        )
        for record in selected
    ]

    return {
        "k": int(k),
        "difficulty": difficulty,
        "num_groups": len(selected),
        "num_prompts": len(
            {
                record["prompt_key"]
                for record in selected
            }
        ),
        "total_generations": int(
            sum(
                record["num_completions"]
                for record in selected
            )
        ),
        "mean_reward": (
            _safe_mean(rewards)
        ),
        "mean_group_reward_std": (
            _safe_mean(reward_stds)
        ),
        "std_group_reward_std": (
            _safe_std(reward_stds)
        ),
        "informative_group_rate": (
            _safe_mean(
                informative_values
            )
        ),
        "uninformative_group_fraction": (
            1.0
            - _safe_mean(
                informative_values
            )
        ),
        "mean_within_group_advantage_variance": (
            _safe_mean(
                advantage_variances
            )
        ),
        "pooled_advantage_variance": (
            _safe_variance(
                all_advantages
            )
        ),
        "mean_absolute_advantage": (
            _safe_mean(
                mean_absolute_advantages
            )
        ),
        "mean_completion_tokens": (
            _safe_mean(
                completion_lengths
            )
        ),
        "mean_truncation_rate": (
            _safe_mean(
                [
                    float(
                        record[
                            "truncation_rate"
                        ]
                    )
                    for record in selected
                ]
            )
        ),
        "mean_terminated_with_eos_rate": (
            _safe_mean(
                [
                    float(
                        record[
                            "terminated_with_eos_rate"
                        ]
                    )
                    for record in selected
                ]
            )
        ),
    }


def run_group_size_analysis(
    config_path,
    tolerance=1.0e-6,
):
    if tolerance <= 0:
        raise ValueError(
            "tolerance must be positive."
        )

    cfg = load_yaml(config_path)

    by_prompt = load_k8_cache(
        cfg["group_cache"]
    )

    group_sizes = [
        int(value)
        for value in cfg["group_sizes"]
    ]

    if sorted(group_sizes) != [
        2,
        4,
        8,
    ]:
        raise ValueError(
            "The assignment requires "
            "group_sizes [2, 4, 8]."
        )

    (
        difficulty_by_prompt,
        prompt_difficulty_records,
    ) = build_difficulty_bins(
        by_prompt
    )

    group_records = []

    for k in group_sizes:
        regrouped = (
            regroup_equal_generation_budget(
                by_prompt,
                k,
            )
        )

        for group in regrouped:
            group_records.append(
                analyze_group(
                    group=group,
                    difficulty=(
                        difficulty_by_prompt[
                            group[
                                "prompt_key"
                            ]
                        ]
                    ),
                    tolerance=tolerance,
                )
            )

    comparison_rows = []

    for k in group_sizes:
        for difficulty in [
            "all",
            "hard",
            "easy",
        ]:
            comparison_rows.append(
                summarize_groups(
                    group_records,
                    k=k,
                    difficulty=difficulty,
                )
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
        / "group_size_summary.json"
    )

    comparison_path = (
        results_dir
        / "group_size_comparison.csv"
    )

    group_details_path = (
        results_dir
        / "group_size_group_details.csv"
    )

    difficulty_path = (
        results_dir
        / "group_size_prompt_difficulty.csv"
    )

    summary = {
        "config": config_path,
        "cache": str(
            cfg["group_cache"]
        ),
        "cache_group_size": (
            CACHE_GROUP_SIZE
        ),
        "num_cached_prompts": len(
            by_prompt
        ),
        "num_cached_generations": int(
            len(by_prompt)
            * CACHE_GROUP_SIZE
        ),
        "group_sizes": group_sizes,
        "equal_generation_rule": (
            "For every prompt, use the same eight "
            "cached completions. K=8 uses one group "
            "of eight, K=4 uses two consecutive "
            "disjoint groups of four, and K=2 uses "
            "four consecutive disjoint groups of two."
        ),
        "difficulty_rule": (
            "Rank prompts by mean reward across all "
            "eight cached completions. Label the "
            "lower half hard and the upper half easy. "
            "Reuse these fixed labels for every K."
        ),
        "informative_tolerance": float(
            tolerance
        ),
        "comparison": (
            comparison_rows
        ),
        "prompt_difficulty": (
            prompt_difficulty_records
        ),
        "comparison_table": str(
            comparison_path.relative_to(
                repo_path(".")
            )
        ),
        "group_details_table": str(
            group_details_path.relative_to(
                repo_path(".")
            )
        ),
        "prompt_difficulty_table": str(
            difficulty_path.relative_to(
                repo_path(".")
            )
        ),
    }

    save_json(
        summary_path,
        summary,
    )

    pd.DataFrame(
        comparison_rows
    ).to_csv(
        comparison_path,
        index=False,
    )

    flat_group_records = []

    for record in group_records:
        flat_record = {
            key: value
            for key, value
            in record.items()
            if key not in {
                "rewards",
                "advantages",
                "generation_indices",
            }
        }

        flat_record[
            "generation_indices"
        ] = ",".join(
            str(value)
            for value in record[
                "generation_indices"
            ]
        )

        flat_record["rewards"] = ",".join(
            str(value)
            for value in record[
                "rewards"
            ]
        )

        flat_record[
            "advantages"
        ] = ",".join(
            str(value)
            for value in record[
                "advantages"
            ]
        )

        flat_group_records.append(
            flat_record
        )

    pd.DataFrame(
        flat_group_records
    ).to_csv(
        group_details_path,
        index=False,
    )

    pd.DataFrame(
        prompt_difficulty_records
    ).to_csv(
        difficulty_path,
        index=False,
    )

    print(
        "GRPO group-size diagnostic"
    )
    print(
        "K | groups | generations | "
        "reward std | informative rate | "
        "advantage variance"
    )

    for row in comparison_rows:
        if row["difficulty"] != "all":
            continue

        print(
            f"{row['k']} | "
            f"{row['num_groups']} | "
            f"{row['total_generations']} | "
            f"{row['mean_group_reward_std']:.6f} | "
            f"{row['informative_group_rate']:.6f} | "
            f"{row['pooled_advantage_variance']:.6f}"
        )

    print()
    print(
        "Difficulty-bin comparison"
    )
    print(
        "K | difficulty | groups | "
        "reward std | informative rate"
    )

    for row in comparison_rows:
        if row["difficulty"] == "all":
            continue

        print(
            f"{row['k']} | "
            f"{row['difficulty']} | "
            f"{row['num_groups']} | "
            f"{row['mean_group_reward_std']:.6f} | "
            f"{row['informative_group_rate']:.6f}"
        )

    print(
        "Saved group-size summary to "
        f"{summary_path}"
    )
    print(
        "Saved comparison table to "
        f"{comparison_path}"
    )
    print(
        "Saved group details to "
        f"{group_details_path}"
    )
    print(
        "Saved prompt difficulty bins to "
        f"{difficulty_path}"
    )

    return summary


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Analyze cached GRPO generations at "
            "equal generation budgets for "
            "K={2,4,8}."
        )
    )

    ap.add_argument(
        "--config",
        default="configs/grpo.yaml",
    )

    ap.add_argument(
        "--tolerance",
        type=float,
        default=1.0e-6,
    )

    args = ap.parse_args()

    run_group_size_analysis(
        config_path=args.config,
        tolerance=args.tolerance,
    )


if __name__ == "__main__":
    main()