from __future__ import annotations

import argparse
import csv
import gc
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.data import (
    load_yaml,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.logging_utils import (
    append_jsonl,
    save_json,
    set_seed,
)
from task5_feedback.rlaif import (
    PairwiseAIJudge,
)
from task5_feedback.rlvr import (
    exact_reward,
    extract_designated_final,
)


VARIANT_ORDER = (
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
)

PERTURBED_VARIANTS = (
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
)


def _mean(
    values,
):
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return float(
        np.mean(values)
    )


def _std(
    values,
):
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return float(
        np.std(values)
    )


def _rate(
    numerator: int,
    denominator: int,
):
    if denominator == 0:
        return None

    return float(
        numerator
        / denominator
    )


def _save_csv(
    path: Path,
    rows: list[dict[str, Any]],
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

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def _results_dir(
    cfg: dict,
) -> Path:
    root = repo_path(
        cfg.get(
            "results_dir",
            "results",
        )
    )

    if root.name == "task5_feedback":
        output_dir = root
    else:
        output_dir = (
            root
            / "task5_feedback"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def _diagnostic_path(
    cfg: dict,
) -> str:
    paths = cfg.get(
        "paths",
        {},
    )

    return str(
        paths.get(
            "task5_controlled_reward_diagnostics",
            paths.get(
                "controlled_reward_diagnostics",
                (
                    "data/"
                    "task5_controlled_reward_"
                    "diagnostics.jsonl"
                ),
            ),
        )
    )


def load_diagnostic_groups(
    cfg: dict,
    max_problems: int | None = None,
):
    """Load and validate the fixed five-response groups."""

    rows = read_jsonl(
        _diagnostic_path(
            cfg
        )
    )

    by_problem = defaultdict(
        list
    )

    problem_order = []

    for row in rows:
        problem_id = str(
            row["problem_id"]
        )

        if problem_id not in by_problem:
            problem_order.append(
                problem_id
            )

        by_problem[
            problem_id
        ].append(
            dict(row)
        )

    if max_problems is not None:
        if int(max_problems) < 1:
            raise ValueError(
                "max_problems must be at least 1."
            )

        selected_ids = problem_order[
            : int(max_problems)
        ]

        by_problem = {
            problem_id: by_problem[
                problem_id
            ]
            for problem_id in selected_ids
        }

        problem_order = selected_ids

    expected_variants = set(
        VARIANT_ORDER
    )

    normalized = {}

    for problem_id in problem_order:
        group = by_problem[
            problem_id
        ]

        variant_map = {}

        for row in group:
            variant = str(
                row["variant_type"]
            )

            if variant in variant_map:
                raise ValueError(
                    "Duplicate diagnostic variant "
                    f"{variant!r} for problem "
                    f"{problem_id!r}."
                )

            variant_map[
                variant
            ] = row

        actual_variants = set(
            variant_map
        )

        if actual_variants != expected_variants:
            missing = sorted(
                expected_variants
                - actual_variants
            )

            extra = sorted(
                actual_variants
                - expected_variants
            )

            raise ValueError(
                "Unexpected variants for problem "
                f"{problem_id!r}; "
                f"missing={missing}, extra={extra}."
            )

        questions = {
            str(
                row["question"]
            )
            for row in group
        }

        gold_finals = {
            str(
                row["gold_final"]
            )
            for row in group
        }

        if len(questions) != 1:
            raise ValueError(
                "Variants have different questions "
                f"for problem {problem_id!r}."
            )

        if len(gold_finals) != 1:
            raise ValueError(
                "Variants have different gold answers "
                f"for problem {problem_id!r}."
            )

        normalized[
            problem_id
        ] = [
            variant_map[
                variant
            ]
            for variant in VARIANT_ORDER
        ]

    return normalized


def score_exact_verifier(
    groups,
) -> list[dict]:
    """Apply the deterministic RLVR verifier to all responses."""

    scored_rows = []

    for problem_id, group in groups.items():
        for row in group:
            response = str(
                row["response"]
            )

            gold_final = str(
                row["gold_final"]
            )

            actual_reward = float(
                exact_reward(
                    response,
                    gold_final,
                )
            )

            expected_reward = float(
                row[
                    "expected_exact_reward"
                ]
            )

            scored_rows.append(
                {
                    "problem_id": problem_id,
                    "question": str(
                        row["question"]
                    ),
                    "gold_final": gold_final,
                    "variant_type": str(
                        row[
                            "variant_type"
                        ]
                    ),
                    "response": response,
                    "predicted_final": (
                        extract_designated_final(
                            response
                        )
                    ),
                    "expected_exact_reward": (
                        expected_reward
                    ),
                    "actual_exact_reward": (
                        actual_reward
                    ),
                    "exact_matches_expected": bool(
                        actual_reward
                        == expected_reward
                    ),
                    "reasoning_quality": row.get(
                        "reasoning_quality"
                    ),
                    "style": row.get(
                        "style"
                    ),
                    "manual_validation": bool(
                        row.get(
                            "manual_validation",
                            False,
                        )
                    ),
                }
            )

    return scored_rows


def pair_id(
    problem_id: str,
    variant_a: str,
    variant_b: str,
) -> str:
    return (
        f"{problem_id}|"
        f"{variant_a}|"
        f"{variant_b}"
    )


def normalize_judgment(
    judgment,
) -> str:
    label = str(
        judgment
    ).strip().upper()

    if label not in {
        "A",
        "B",
        "TIE",
    }:
        return "TIE"

    return label


def run_ai_pairwise_scoring(
    cfg: dict,
    groups,
    pairwise_path: Path,
    judge_cache_path: Path,
    overwrite: bool,
) -> list[dict]:
    """Judge every pair among the five variants per problem."""

    if overwrite and pairwise_path.exists():
        pairwise_path.unlink()

    if pairwise_path.exists():
        existing_rows = read_jsonl(
            pairwise_path
        )
    else:
        existing_rows = []

    existing = {}

    for row in existing_rows:
        key = str(
            row["pair_id"]
        )

        if key in existing:
            raise ValueError(
                "Duplicate cached pairwise result "
                f"{key!r}."
            )

        existing[
            key
        ] = row

    pending = []

    for problem_id, group in groups.items():
        by_variant = {
            str(
                row["variant_type"]
            ): row
            for row in group
        }

        for (
            variant_a,
            variant_b,
        ) in itertools.combinations(
            VARIANT_ORDER,
            2,
        ):
            key = pair_id(
                problem_id,
                variant_a,
                variant_b,
            )

            if key in existing:
                continue

            pending.append(
                {
                    "pair_id": key,
                    "problem_id": problem_id,
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "row_a": by_variant[
                        variant_a
                    ],
                    "row_b": by_variant[
                        variant_b
                    ],
                }
            )

    print(
        "Controlled AI-judge comparisons: "
        f"existing={len(existing)}, "
        f"pending={len(pending)}",
        flush=True,
    )

    if pending:
        judge = PairwiseAIJudge(
            cfg,
            judge_cache_path,
        )

        for index, job in enumerate(
            pending,
            start=1,
        ):
            row_a = job[
                "row_a"
            ]

            row_b = job[
                "row_b"
            ]

            judgment = normalize_judgment(
                judge.compare(
                    problem=str(
                        row_a[
                            "question"
                        ]
                    ),
                    a=str(
                        row_a[
                            "response"
                        ]
                    ),
                    b=str(
                        row_b[
                            "response"
                        ]
                    ),
                )
            )

            if judgment == "A":
                score_a = 1.0
                score_b = 0.0
            elif judgment == "B":
                score_a = 0.0
                score_b = 1.0
            else:
                score_a = 0.5
                score_b = 0.5

            record = {
                "pair_id": job[
                    "pair_id"
                ],
                "problem_id": job[
                    "problem_id"
                ],
                "question": str(
                    row_a[
                        "question"
                    ]
                ),
                "variant_a": job[
                    "variant_a"
                ],
                "variant_b": job[
                    "variant_b"
                ],
                "judgment": judgment,
                "score_a": score_a,
                "score_b": score_b,
                "response_a": str(
                    row_a[
                        "response"
                    ]
                ),
                "response_b": str(
                    row_b[
                        "response"
                    ]
                ),
            }

            append_jsonl(
                pairwise_path,
                record,
            )

            existing[
                record[
                    "pair_id"
                ]
            ] = record

            if (
                index % 10 == 0
                or index == len(
                    pending
                )
            ):
                print(
                    "Controlled AI judging "
                    f"{index}/{len(pending)} "
                    "pending comparisons",
                    flush=True,
                )

        del judge

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ordered_rows = []

    for problem_id in groups:
        for (
            variant_a,
            variant_b,
        ) in itertools.combinations(
            VARIANT_ORDER,
            2,
        ):
            ordered_rows.append(
                existing[
                    pair_id(
                        problem_id,
                        variant_a,
                        variant_b,
                    )
                ]
            )

    return ordered_rows


def aggregate_group_rewards(
    groups,
    pairwise_rows: list[dict],
) -> list[dict]:
    """Convert all-pairs judgments into one AI score per variant."""

    stats = {}

    for problem_id in groups:
        for variant in VARIANT_ORDER:
            stats[
                (
                    problem_id,
                    variant,
                )
            ] = {
                "wins": 0,
                "ties": 0,
                "losses": 0,
                "pairwise_points": 0.0,
                "comparisons": 0,
            }

    for row in pairwise_rows:
        problem_id = str(
            row["problem_id"]
        )

        variant_a = str(
            row["variant_a"]
        )

        variant_b = str(
            row["variant_b"]
        )

        judgment = str(
            row["judgment"]
        )

        stat_a = stats[
            (
                problem_id,
                variant_a,
            )
        ]

        stat_b = stats[
            (
                problem_id,
                variant_b,
            )
        ]

        stat_a[
            "comparisons"
        ] += 1

        stat_b[
            "comparisons"
        ] += 1

        if judgment == "A":
            stat_a["wins"] += 1
            stat_b["losses"] += 1

            stat_a[
                "pairwise_points"
            ] += 1.0
        elif judgment == "B":
            stat_b["wins"] += 1
            stat_a["losses"] += 1

            stat_b[
                "pairwise_points"
            ] += 1.0
        else:
            stat_a["ties"] += 1
            stat_b["ties"] += 1

            stat_a[
                "pairwise_points"
            ] += 0.5

            stat_b[
                "pairwise_points"
            ] += 0.5

    rows = []

    for problem_id, group in groups.items():
        by_variant = {
            str(
                row["variant_type"]
            ): row
            for row in group
        }

        for variant in VARIANT_ORDER:
            stat = stats[
                (
                    problem_id,
                    variant,
                )
            ]

            comparisons = int(
                stat[
                    "comparisons"
                ]
            )

            ai_group_reward = (
                float(
                    stat[
                        "pairwise_points"
                    ]
                    / comparisons
                )
                if comparisons
                else None
            )

            source_row = by_variant[
                variant
            ]

            rows.append(
                {
                    "problem_id": problem_id,
                    "variant_type": variant,
                    "question": str(
                        source_row[
                            "question"
                        ]
                    ),
                    "gold_final": str(
                        source_row[
                            "gold_final"
                        ]
                    ),
                    "response": str(
                        source_row[
                            "response"
                        ]
                    ),
                    "ai_group_reward": (
                        ai_group_reward
                    ),
                    "ai_wins": int(
                        stat["wins"]
                    ),
                    "ai_ties": int(
                        stat["ties"]
                    ),
                    "ai_losses": int(
                        stat["losses"]
                    ),
                    "ai_comparisons": (
                        comparisons
                    ),
                }
            )

    return rows


def clean_comparison_rows(
    pairwise_rows: list[dict],
) -> list[dict]:
    """Extract clean-correct versus each perturbation."""

    output = []

    for row in pairwise_rows:
        variant_a = str(
            row["variant_a"]
        )

        variant_b = str(
            row["variant_b"]
        )

        if variant_a == "clean_correct":
            perturbed_variant = (
                variant_b
            )

            judgment = str(
                row["judgment"]
            )

            if judgment == "A":
                outcome = "clean_better"
            elif judgment == "B":
                outcome = "perturbation_better"
            else:
                outcome = "tie"
        elif variant_b == "clean_correct":
            perturbed_variant = (
                variant_a
            )

            judgment = str(
                row["judgment"]
            )

            if judgment == "B":
                outcome = "clean_better"
            elif judgment == "A":
                outcome = "perturbation_better"
            else:
                outcome = "tie"
        else:
            continue

        output.append(
            {
                "problem_id": str(
                    row["problem_id"]
                ),
                "perturbed_variant": (
                    perturbed_variant
                ),
                "outcome": outcome,
                "judgment": str(
                    row["judgment"]
                ),
                "response_a": row[
                    "response_a"
                ],
                "response_b": row[
                    "response_b"
                ],
            }
        )

    return output


def summarize_exact_scores(
    exact_rows: list[dict],
) -> list[dict]:
    by_variant = defaultdict(
        list
    )

    for row in exact_rows:
        by_variant[
            row[
                "variant_type"
            ]
        ].append(
            row
        )

    summary = []

    for variant in VARIANT_ORDER:
        rows = by_variant[
            variant
        ]

        summary.append(
            {
                "variant_type": variant,
                "num_responses": len(
                    rows
                ),
                "exact_reward_mean": _mean(
                    [
                        row[
                            "actual_exact_reward"
                        ]
                        for row in rows
                    ]
                ),
                "expected_exact_reward_mean": _mean(
                    [
                        row[
                            "expected_exact_reward"
                        ]
                        for row in rows
                    ]
                ),
                "expected_match_rate": _mean(
                    [
                        float(
                            row[
                                "exact_matches_expected"
                            ]
                        )
                        for row in rows
                    ]
                ),
                "designated_answer_rate": _mean(
                    [
                        float(
                            row[
                                "predicted_final"
                            ]
                            is not None
                        )
                        for row in rows
                    ]
                ),
            }
        )

    return summary


def summarize_ai_scores(
    ai_group_rows: list[dict],
) -> list[dict]:
    by_variant = defaultdict(
        list
    )

    for row in ai_group_rows:
        by_variant[
            row[
                "variant_type"
            ]
        ].append(
            row
        )

    summary = []

    for variant in VARIANT_ORDER:
        rows = by_variant[
            variant
        ]

        summary.append(
            {
                "variant_type": variant,
                "num_responses": len(
                    rows
                ),
                "ai_group_reward_mean": _mean(
                    [
                        row[
                            "ai_group_reward"
                        ]
                        for row in rows
                    ]
                ),
                "ai_group_reward_std": _std(
                    [
                        row[
                            "ai_group_reward"
                        ]
                        for row in rows
                    ]
                ),
                "mean_ai_wins": _mean(
                    [
                        row[
                            "ai_wins"
                        ]
                        for row in rows
                    ]
                ),
                "mean_ai_ties": _mean(
                    [
                        row[
                            "ai_ties"
                        ]
                        for row in rows
                    ]
                ),
                "mean_ai_losses": _mean(
                    [
                        row[
                            "ai_losses"
                        ]
                        for row in rows
                    ]
                ),
            }
        )

    return summary


def summarize_clean_comparisons(
    rows: list[dict],
) -> list[dict]:
    by_variant = defaultdict(
        list
    )

    for row in rows:
        by_variant[
            row[
                "perturbed_variant"
            ]
        ].append(
            row
        )

    summary = []

    for variant in PERTURBED_VARIANTS:
        variant_rows = by_variant[
            variant
        ]

        clean_better = sum(
            row["outcome"]
            == "clean_better"
            for row in variant_rows
        )

        ties = sum(
            row["outcome"]
            == "tie"
            for row in variant_rows
        )

        perturbation_better = sum(
            row["outcome"]
            == "perturbation_better"
            for row in variant_rows
        )

        count = len(
            variant_rows
        )

        summary.append(
            {
                "perturbed_variant": variant,
                "num_problems": count,
                "clean_better_rate": _rate(
                    clean_better,
                    count,
                ),
                "tie_rate": _rate(
                    ties,
                    count,
                ),
                "perturbation_better_rate": _rate(
                    perturbation_better,
                    count,
                ),
            }
        )

    return summary


def build_sensitivity_summary(
    exact_summary: list[dict],
    ai_summary: list[dict],
    clean_summary: list[dict],
) -> dict:
    exact_lookup = {
        row[
            "variant_type"
        ]: row
        for row in exact_summary
    }

    ai_lookup = {
        row[
            "variant_type"
        ]: row
        for row in ai_summary
    }

    clean_lookup = {
        row[
            "perturbed_variant"
        ]: row
        for row in clean_summary
    }

    clean_exact = exact_lookup[
        "clean_correct"
    ][
        "exact_reward_mean"
    ]

    clean_ai = ai_lookup[
        "clean_correct"
    ][
        "ai_group_reward_mean"
    ]

    corrupt_exact = exact_lookup[
        "corrupt_reasoning_correct_final"
    ][
        "exact_reward_mean"
    ]

    corrupt_ai = ai_lookup[
        "corrupt_reasoning_correct_final"
    ][
        "ai_group_reward_mean"
    ]

    wrong_final_exact = exact_lookup[
        "good_reasoning_wrong_final"
    ][
        "exact_reward_mean"
    ]

    wrong_final_ai = ai_lookup[
        "good_reasoning_wrong_final"
    ][
        "ai_group_reward_mean"
    ]

    filler_exact = exact_lookup[
        "persuasive_filler_correct"
    ][
        "exact_reward_mean"
    ]

    filler_ai = ai_lookup[
        "persuasive_filler_correct"
    ][
        "ai_group_reward_mean"
    ]

    distractor_exact = exact_lookup[
        "gold_distractor_wrong_final"
    ][
        "exact_reward_mean"
    ]

    distractor_ai = ai_lookup[
        "gold_distractor_wrong_final"
    ][
        "ai_group_reward_mean"
    ]

    return {
        "reasoning_sensitivity": {
            "comparison": (
                "clean_correct minus "
                "corrupt_reasoning_correct_final"
            ),
            "exact_reward_gap": (
                clean_exact
                - corrupt_exact
            ),
            "ai_group_reward_gap": (
                clean_ai
                - corrupt_ai
            ),
            "ai_clean_better_rate": clean_lookup[
                "corrupt_reasoning_correct_final"
            ][
                "clean_better_rate"
            ],
            "interpretation": (
                "The final answer is held correct while "
                "reasoning is corrupted."
            ),
        },
        "outcome_sensitivity": {
            "comparison": (
                "clean_correct minus "
                "good_reasoning_wrong_final"
            ),
            "exact_reward_gap": (
                clean_exact
                - wrong_final_exact
            ),
            "ai_group_reward_gap": (
                clean_ai
                - wrong_final_ai
            ),
            "ai_clean_better_rate": clean_lookup[
                "good_reasoning_wrong_final"
            ][
                "clean_better_rate"
            ],
            "interpretation": (
                "Reasoning is mostly preserved while the "
                "designated final answer is made wrong."
            ),
        },
        "persuasive_filler_sensitivity": {
            "comparison": (
                "clean_correct minus "
                "persuasive_filler_correct"
            ),
            "exact_reward_gap": (
                clean_exact
                - filler_exact
            ),
            "ai_group_reward_gap": (
                clean_ai
                - filler_ai
            ),
            "ai_perturbation_better_rate": clean_lookup[
                "persuasive_filler_correct"
            ][
                "perturbation_better_rate"
            ],
            "interpretation": (
                "Correctness is held fixed while persuasive "
                "but unnecessary text is added."
            ),
        },
        "gold_distractor_resistance": {
            "comparison": (
                "clean_correct minus "
                "gold_distractor_wrong_final"
            ),
            "exact_reward_gap": (
                clean_exact
                - distractor_exact
            ),
            "ai_group_reward_gap": (
                clean_ai
                - distractor_ai
            ),
            "ai_clean_better_rate": clean_lookup[
                "gold_distractor_wrong_final"
            ][
                "clean_better_rate"
            ],
            "interpretation": (
                "The correct number is mentioned as a "
                "distractor, but the designated final answer "
                "is wrong."
            ),
        },
    }


def qualitative_candidates(
    groups,
    clean_rows: list[dict],
) -> dict:
    by_problem = {
        problem_id: {
            row[
                "variant_type"
            ]: row
            for row in group
        }
        for problem_id, group in groups.items()
    }

    unexpected = []
    ties = []

    for row in clean_rows:
        problem_id = str(
            row["problem_id"]
        )

        variant = str(
            row[
                "perturbed_variant"
            ]
        )

        clean_row = by_problem[
            problem_id
        ][
            "clean_correct"
        ]

        variant_row = by_problem[
            problem_id
        ][
            variant
        ]

        candidate = {
            "problem_id": problem_id,
            "question": str(
                clean_row[
                    "question"
                ]
            ),
            "perturbed_variant": variant,
            "outcome": row[
                "outcome"
            ],
            "clean_response": str(
                clean_row[
                    "response"
                ]
            ),
            "perturbed_response": str(
                variant_row[
                    "response"
                ]
            ),
            "clean_exact_reward": float(
                exact_reward(
                    clean_row[
                        "response"
                    ],
                    clean_row[
                        "gold_final"
                    ],
                )
            ),
            "perturbed_exact_reward": float(
                exact_reward(
                    variant_row[
                        "response"
                    ],
                    variant_row[
                        "gold_final"
                    ],
                )
            ),
        }

        if (
            row["outcome"]
            == "perturbation_better"
            and len(unexpected) < 12
        ):
            unexpected.append(
                candidate
            )

        if (
            row["outcome"] == "tie"
            and len(ties) < 12
        ):
            ties.append(
                candidate
            )

    return {
        "unexpected_perturbation_wins": (
            unexpected
        ),
        "clean_perturbation_ties": ties,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Task 5 controlled RLVR-versus-"
            "RLAIF reward diagnostic."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/feedback.yaml",
    )

    parser.add_argument(
        "--max-problems",
        type=int,
        help=(
            "Optional prefix of diagnostic problems "
            "for a smoke test."
        ),
    )

    parser.add_argument(
        "--skip-ai-judge",
        action="store_true",
        help=(
            "Run only the deterministic exact-verifier "
            "analysis without loading the GPU judge."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace derived exact and pairwise result files. "
            "The deterministic judge cache is preserved."
        ),
    )

    args = parser.parse_args()

    cfg = load_yaml(
        args.config
    )

    set_seed(
        int(
            cfg["seed"]
        )
    )

    groups = load_diagnostic_groups(
        cfg,
        max_problems=args.max_problems,
    )

    output_dir = _results_dir(
        cfg
    )

    exact_path = (
        output_dir
        / "controlled_exact_scores.jsonl"
    )

    pairwise_path = (
        output_dir
        / "controlled_pairwise_judgments.jsonl"
    )

    judge_cache_path = (
        output_dir
        / "controlled_ai_judge_cache.json"
    )

    if args.overwrite and exact_path.exists():
        exact_path.unlink()

    exact_rows = score_exact_verifier(
        groups
    )

    write_jsonl(
        exact_path,
        exact_rows,
    )

    exact_summary = summarize_exact_scores(
        exact_rows
    )

    exact_mismatches = [
        row
        for row in exact_rows
        if not row[
            "exact_matches_expected"
        ]
    ]

    print(
        "Controlled diagnostic problems: "
        f"{len(groups)}",
        flush=True,
    )

    print(
        "Controlled diagnostic responses: "
        f"{len(exact_rows)}",
        flush=True,
    )

    print(
        "Exact-verifier expectation mismatches: "
        f"{len(exact_mismatches)}",
        flush=True,
    )

    _save_csv(
        output_dir
        / "controlled_exact_variant_summary.csv",
        exact_summary,
    )

    if args.skip_ai_judge:
        summary = {
            "config": args.config,
            "dataset": _diagnostic_path(
                cfg
            ),
            "num_problems": len(
                groups
            ),
            "num_responses": len(
                exact_rows
            ),
            "ai_judging_skipped": True,
            "exact_expectation_mismatches": (
                len(
                    exact_mismatches
                )
            ),
            "exact_variant_summary": (
                exact_summary
            ),
        }

        save_json(
            output_dir
            / "controlled_reward_diagnostic_summary.json",
            summary,
        )

        print(
            "Saved exact-only diagnostic results to "
            f"{output_dir}",
            flush=True,
        )

        return

    pairwise_rows = run_ai_pairwise_scoring(
        cfg=cfg,
        groups=groups,
        pairwise_path=pairwise_path,
        judge_cache_path=judge_cache_path,
        overwrite=bool(
            args.overwrite
        ),
    )

    ai_group_rows = aggregate_group_rewards(
        groups,
        pairwise_rows,
    )

    clean_rows = clean_comparison_rows(
        pairwise_rows
    )

    ai_summary = summarize_ai_scores(
        ai_group_rows
    )

    clean_summary = summarize_clean_comparisons(
        clean_rows
    )

    sensitivity = build_sensitivity_summary(
        exact_summary,
        ai_summary,
        clean_summary,
    )

    qualitative = qualitative_candidates(
        groups,
        clean_rows,
    )

    write_jsonl(
        output_dir
        / "controlled_ai_group_scores.jsonl",
        ai_group_rows,
    )

    _save_csv(
        output_dir
        / "controlled_ai_variant_summary.csv",
        ai_summary,
    )

    _save_csv(
        output_dir
        / "controlled_clean_comparisons.csv",
        clean_summary,
    )

    save_json(
        output_dir
        / "controlled_sensitivity_summary.json",
        sensitivity,
    )

    save_json(
        output_dir
        / "controlled_qualitative_candidates.json",
        qualitative,
    )

    summary = {
        "config": args.config,
        "dataset": _diagnostic_path(
            cfg
        ),
        "num_problems": len(
            groups
        ),
        "num_responses": len(
            exact_rows
        ),
        "num_pairwise_comparisons": len(
            pairwise_rows
        ),
        "comparisons_per_problem": 10,
        "comparisons_per_response": 4,
        "ai_judging_skipped": False,
        "exact_expectation_mismatches": len(
            exact_mismatches
        ),
        "exact_variant_summary": exact_summary,
        "ai_variant_summary": ai_summary,
        "clean_comparison_summary": clean_summary,
        "sensitivity_summary": sensitivity,
        "outputs": {
            "exact_scores": str(
                exact_path
            ),
            "pairwise_judgments": str(
                pairwise_path
            ),
            "ai_group_scores": str(
                output_dir
                / "controlled_ai_group_scores.jsonl"
            ),
            "qualitative_candidates": str(
                output_dir
                / "controlled_qualitative_candidates.json"
            ),
        },
    }

    save_json(
        output_dir
        / "controlled_reward_diagnostic_summary.json",
        summary,
    )

    print(
        "\nControlled reward diagnostic",
        flush=True,
    )

    print(
        "variant | exact reward | AI group reward",
        flush=True,
    )

    exact_lookup = {
        row[
            "variant_type"
        ]: row
        for row in exact_summary
    }

    ai_lookup = {
        row[
            "variant_type"
        ]: row
        for row in ai_summary
    }

    for variant in VARIANT_ORDER:
        print(
            f"{variant} | "
            f"{exact_lookup[variant]['exact_reward_mean']} | "
            f"{ai_lookup[variant]['ai_group_reward_mean']}",
            flush=True,
        )

    print(
        "\nClean-versus-perturbation comparison",
        flush=True,
    )

    print(
        "variant | clean better | tie | perturbation better",
        flush=True,
    )

    for row in clean_summary:
        print(
            f"{row['perturbed_variant']} | "
            f"{row['clean_better_rate']} | "
            f"{row['tie_rate']} | "
            f"{row['perturbation_better_rate']}",
            flush=True,
        )

    print(
        "\nSaved controlled diagnostic results to "
        f"{output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()