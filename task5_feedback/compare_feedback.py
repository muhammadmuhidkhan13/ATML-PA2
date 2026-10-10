from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json


POLICY_ORDER = (
    "sft",
    "rlvr",
    "rlaif",
)

VARIANT_ORDER = (
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
)


def task_results_dir(cfg: dict) -> Path:
    return (
        repo_path(
            cfg.get(
                "results_dir",
                "results",
            )
        )
        / "task5_feedback"
    )


def require_file(
    path: Path,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {path}"
        )


def optional_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def difference(
    first,
    second,
):
    first_value = optional_float(first)
    second_value = optional_float(second)

    if (
        first_value is None
        or second_value is None
    ):
        return None

    return first_value - second_value


def csv_value(value: Any):
    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
        ),
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    return value


def save_csv(
    path: Path,
    rows: list[dict],
) -> None:
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
                fieldnames.append(key)

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

        for row in rows:
            writer.writerow(
                {
                    key: csv_value(
                        row.get(key)
                    )
                    for key in fieldnames
                }
            )


def indexed_rows(
    rows: list[dict],
    key_fields: tuple[str, ...],
) -> dict:
    result = {}

    for row in rows:
        key = tuple(
            str(
                row.get(field)
            )
            for field in key_fields
        )

        if key in result:
            raise ValueError(
                "Duplicate summary row for "
                f"{key_fields}={key}"
            )

        result[key] = row

    return result


def validate_math_summary(
    summary: dict,
) -> None:
    policy_rows = summary.get(
        "policy_metrics"
    )

    pairwise_rows = summary.get(
        "pairwise_metrics"
    )

    if not isinstance(
        policy_rows,
        list,
    ) or not policy_rows:
        raise ValueError(
            "math_evaluation_summary.json does not "
            "contain completed policy_metrics."
        )

    if not isinstance(
        pairwise_rows,
        list,
    ) or not pairwise_rows:
        raise ValueError(
            "math_evaluation_summary.json does not "
            "contain completed pairwise_metrics."
        )

    datasets = {
        str(
            row.get(
                "dataset"
            )
        )
        for row in policy_rows
    }

    required_datasets = {
        "gsm8k",
        "transfer",
    }

    missing_datasets = (
        required_datasets
        - datasets
    )

    if missing_datasets:
        raise ValueError(
            "Math evaluation is incomplete. Missing "
            f"datasets: {sorted(missing_datasets)}"
        )

    present_pairs = {
        (
            str(
                row.get(
                    "dataset"
                )
            ),
            str(
                row.get(
                    "policy"
                )
            ),
        )
        for row in policy_rows
    }

    required_pairs = {
        (
            dataset,
            policy,
        )
        for dataset in required_datasets
        for policy in POLICY_ORDER
    }

    missing_pairs = (
        required_pairs
        - present_pairs
    )

    if missing_pairs:
        raise ValueError(
            "Math policy evaluation is incomplete. "
            f"Missing rows: {sorted(missing_pairs)}"
        )


def validate_controlled_summary(
    summary: dict,
    allow_incomplete: bool,
) -> bool:
    exact_rows = summary.get(
        "exact_variant_summary"
    )

    if not isinstance(
        exact_rows,
        list,
    ) or not exact_rows:
        raise ValueError(
            "Controlled diagnostic summary does not "
            "contain exact_variant_summary."
        )

    exact_variants = {
        str(
            row.get(
                "variant_type"
            )
        )
        for row in exact_rows
    }

    missing_variants = (
        set(VARIANT_ORDER)
        - exact_variants
    )

    if missing_variants:
        raise ValueError(
            "Controlled exact diagnostic is missing "
            f"variants: {sorted(missing_variants)}"
        )

    ai_skipped = bool(
        summary.get(
            "ai_judging_skipped",
            False,
        )
    )

    ai_rows = summary.get(
        "ai_variant_summary"
    )

    ai_complete = (
        not ai_skipped
        and isinstance(
            ai_rows,
            list,
        )
        and bool(ai_rows)
    )

    if (
        not ai_complete
        and not allow_incomplete
    ):
        raise RuntimeError(
            "The controlled diagnostic currently has "
            "only exact-verifier results. Run "
            "task5_feedback.score_perturbations without "
            "--skip-ai-judge before creating the final "
            "Task 5 comparison. Use --allow-incomplete "
            "only for a provisional CPU-only synthesis."
        )

    return ai_complete


def build_policy_comparison(
    policy_rows: list[dict],
) -> list[dict]:
    lookup = indexed_rows(
        policy_rows,
        (
            "dataset",
            "policy",
        ),
    )

    datasets = []

    for row in policy_rows:
        dataset = str(
            row.get(
                "dataset"
            )
        )

        if dataset not in datasets:
            datasets.append(
                dataset
            )

    result = []

    for dataset in datasets:
        baseline = lookup.get(
            (
                dataset,
                "sft",
            )
        )

        if baseline is None:
            raise ValueError(
                "Cannot calculate policy deltas because "
                f"the SFT baseline is missing for {dataset}."
            )

        for policy in POLICY_ORDER:
            row = lookup.get(
                (
                    dataset,
                    policy,
                )
            )

            if row is None:
                continue

            combined = dict(row)

            combined[
                "exact_accuracy_delta_vs_sft"
            ] = difference(
                row.get(
                    "exact_accuracy"
                ),
                baseline.get(
                    "exact_accuracy"
                ),
            )

            combined[
                "format_compliance_delta_vs_sft"
            ] = difference(
                row.get(
                    "format_compliance_rate"
                ),
                baseline.get(
                    "format_compliance_rate"
                ),
            )

            combined[
                "mean_length_change_vs_sft"
            ] = difference(
                row.get(
                    "response_length_mean"
                ),
                baseline.get(
                    "response_length_mean"
                ),
            )

            result.append(
                combined
            )

    return result


def build_pairwise_comparison(
    pairwise_rows: list[dict],
) -> list[dict]:
    return [
        dict(row)
        for row in pairwise_rows
    ]


def build_controlled_comparison(
    controlled_summary: dict,
) -> list[dict]:
    exact_lookup = {
        str(
            row.get(
                "variant_type"
            )
        ): row
        for row in controlled_summary.get(
            "exact_variant_summary",
            [],
        )
    }

    ai_lookup = {
        str(
            row.get(
                "variant_type"
            )
        ): row
        for row in controlled_summary.get(
            "ai_variant_summary",
            [],
        )
    }

    clean_lookup = {
        str(
            row.get(
                "variant_type"
            )
        ): row
        for row in controlled_summary.get(
            "clean_comparison_summary",
            [],
        )
    }

    result = []

    for variant in VARIANT_ORDER:
        row = {
            "variant_type": variant,
        }

        for key, value in exact_lookup.get(
            variant,
            {},
        ).items():
            if key != "variant_type":
                row[key] = value

        for key, value in ai_lookup.get(
            variant,
            {},
        ).items():
            if key != "variant_type":
                row[key] = value

        for key, value in clean_lookup.get(
            variant,
            {},
        ).items():
            if key != "variant_type":
                row[
                    f"clean_comparison_{key}"
                ] = value

        result.append(
            row
        )

    return result


def best_policy(
    rows: list[dict],
    dataset: str,
    metric: str,
):
    candidates = []

    for row in rows:
        if (
            str(
                row.get(
                    "dataset"
                )
            )
            != dataset
        ):
            continue

        value = optional_float(
            row.get(
                metric
            )
        )

        if value is None:
            continue

        candidates.append(
            (
                value,
                str(
                    row.get(
                        "policy"
                    )
                ),
            )
        )

    if not candidates:
        return None

    best_value = max(
        value
        for value, _ in candidates
    )

    winners = sorted(
        policy
        for value, policy in candidates
        if value == best_value
    )

    return {
        "metric": metric,
        "value": best_value,
        "policies": winners,
    }


def compare_rl_methods(
    policy_rows: list[dict],
    dataset: str,
) -> dict | None:
    lookup = indexed_rows(
        policy_rows,
        (
            "dataset",
            "policy",
        ),
    )

    rlvr = lookup.get(
        (
            dataset,
            "rlvr",
        )
    )

    rlaif = lookup.get(
        (
            dataset,
            "rlaif",
        )
    )

    if (
        rlvr is None
        or rlaif is None
    ):
        return None

    return {
        "dataset": dataset,
        "rlvr_exact_accuracy": rlvr.get(
            "exact_accuracy"
        ),
        "rlaif_exact_accuracy": rlaif.get(
            "exact_accuracy"
        ),
        "rlvr_minus_rlaif_exact_accuracy": difference(
            rlvr.get(
                "exact_accuracy"
            ),
            rlaif.get(
                "exact_accuracy"
            ),
        ),
        "rlvr_format_compliance": rlvr.get(
            "format_compliance_rate"
        ),
        "rlaif_format_compliance": rlaif.get(
            "format_compliance_rate"
        ),
        "rlvr_minus_rlaif_format_compliance": difference(
            rlvr.get(
                "format_compliance_rate"
            ),
            rlaif.get(
                "format_compliance_rate"
            ),
        ),
        "rlvr_mean_length": rlvr.get(
            "response_length_mean"
        ),
        "rlaif_mean_length": rlaif.get(
            "response_length_mean"
        ),
        "rlvr_minus_rlaif_mean_length": difference(
            rlvr.get(
                "response_length_mean"
            ),
            rlaif.get(
                "response_length_mean"
            ),
        ),
    }


def build_interpretation(
    math_summary: dict,
    controlled_summary: dict,
    ai_complete: bool,
) -> dict:
    policy_rows = math_summary[
        "policy_metrics"
    ]

    in_domain_best = best_policy(
        policy_rows,
        "gsm8k",
        "exact_accuracy",
    )

    transfer_best = best_policy(
        policy_rows,
        "transfer",
        "exact_accuracy",
    )

    exact_lookup = {
        str(
            row.get(
                "variant_type"
            )
        ): optional_float(
            row.get(
                "exact_reward_mean"
            )
        )
        for row in controlled_summary.get(
            "exact_variant_summary",
            [],
        )
    }

    exact_reasoning_gap = difference(
        exact_lookup.get(
            "clean_correct"
        ),
        exact_lookup.get(
            "corrupt_reasoning_correct_final"
        ),
    )

    exact_outcome_gap = difference(
        exact_lookup.get(
            "clean_correct"
        ),
        exact_lookup.get(
            "good_reasoning_wrong_final"
        ),
    )

    exact_filler_gap = difference(
        exact_lookup.get(
            "clean_correct"
        ),
        exact_lookup.get(
            "persuasive_filler_correct"
        ),
    )

    exact_distractor_gap = difference(
        exact_lookup.get(
            "clean_correct"
        ),
        exact_lookup.get(
            "gold_distractor_wrong_final"
        ),
    )

    conclusions = [
        {
            "claim": (
                "The exact verifier is driven by the "
                "designated final answer."
            ),
            "evidence": {
                "reasoning_corruption_gap": (
                    exact_reasoning_gap
                ),
                "wrong_final_gap": (
                    exact_outcome_gap
                ),
                "persuasive_filler_gap": (
                    exact_filler_gap
                ),
                "gold_distractor_wrong_final_gap": (
                    exact_distractor_gap
                ),
            },
            "interpretation": (
                "A zero reasoning-corruption gap together "
                "with a positive wrong-final gap means the "
                "verifier checks the designated numerical "
                "outcome but does not evaluate the quality "
                "of the preceding reasoning."
            ),
        }
    ]

    if ai_complete:
        sensitivity = controlled_summary.get(
            "sensitivity_summary",
            {},
        )

        conclusions.append(
            {
                "claim": (
                    "The fixed AI judge supplies the "
                    "reasoning- and style-sensitive view "
                    "that the exact verifier cannot provide."
                ),
                "evidence": sensitivity,
                "interpretation": (
                    "Compare the AI group-reward gaps with "
                    "the exact-reward gaps. A nonzero AI gap "
                    "where the exact gap is zero indicates "
                    "that the AI judge distinguishes response "
                    "quality beyond final-answer correctness."
                ),
            }
        )

    return {
        "best_exact_accuracy": {
            "gsm8k": in_domain_best,
            "transfer": transfer_best,
        },
        "rlvr_vs_rlaif": [
            comparison
            for comparison in (
                compare_rl_methods(
                    policy_rows,
                    "gsm8k",
                ),
                compare_rl_methods(
                    policy_rows,
                    "transfer",
                ),
            )
            if comparison is not None
        ],
        "controlled_diagnostic_conclusions": conclusions,
        "important_caution": (
            "Exact accuracy and normalized AI pairwise "
            "score are different metrics and must not be "
            "subtracted from one another. Compare policies "
            "within each metric separately."
        ),
    }


def run_comparison(
    config_path: str,
    allow_incomplete: bool = False,
    output_prefix: str = "feedback_comparison",
) -> dict:
    cfg = load_yaml(
        config_path
    )

    output_dir = task_results_dir(
        cfg
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    math_path = (
        output_dir
        / "math_evaluation_summary.json"
    )

    controlled_path = (
        output_dir
        / "controlled_reward_diagnostic_summary.json"
    )

    require_file(
        math_path,
        "Task 5 math evaluation summary",
    )

    require_file(
        controlled_path,
        "Task 5 controlled reward diagnostic",
    )

    math_summary = load_json(
        math_path
    )

    controlled_summary = load_json(
        controlled_path
    )

    validate_math_summary(
        math_summary
    )

    ai_complete = validate_controlled_summary(
        controlled_summary,
        allow_incomplete=allow_incomplete,
    )

    policy_comparison = (
        build_policy_comparison(
            math_summary[
                "policy_metrics"
            ]
        )
    )

    pairwise_comparison = (
        build_pairwise_comparison(
            math_summary[
                "pairwise_metrics"
            ]
        )
    )

    controlled_comparison = (
        build_controlled_comparison(
            controlled_summary
        )
    )

    transfer_analysis = (
        math_summary.get(
            "transfer_analysis",
            {},
        )
    )

    interpretation = build_interpretation(
        math_summary,
        controlled_summary,
        ai_complete=ai_complete,
    )

    final_summary = {
        "config": config_path,
        "status": (
            "complete"
            if ai_complete
            else "provisional_exact_only"
        ),
        "inputs": {
            "math_evaluation_summary": str(
                math_path
            ),
            "controlled_reward_diagnostic_summary": str(
                controlled_path
            ),
        },
        "evaluation_design": {
            "policies": list(
                POLICY_ORDER
            ),
            "in_domain_dataset": "gsm8k",
            "transfer_dataset": "transfer",
            "policy_evaluators": [
                "deterministic exact-answer verifier",
                "fixed pairwise AI judge",
            ],
            "controlled_diagnostic_variants": list(
                VARIANT_ORDER
            ),
            "controlled_ai_judging_complete": (
                ai_complete
            ),
        },
        "policy_metrics": policy_comparison,
        "pairwise_metrics": pairwise_comparison,
        "transfer_analysis": transfer_analysis,
        "controlled_variant_metrics": (
            controlled_comparison
        ),
        "controlled_sensitivity_summary": (
            controlled_summary.get(
                "sensitivity_summary"
            )
        ),
        "interpretation": interpretation,
    }

    summary_path = (
        output_dir
        / f"{output_prefix}_summary.json"
    )

    policy_path = (
        output_dir
        / f"{output_prefix}_policy_metrics.csv"
    )

    pairwise_path = (
        output_dir
        / f"{output_prefix}_pairwise_metrics.csv"
    )

    controlled_table_path = (
        output_dir
        / f"{output_prefix}_controlled_metrics.csv"
    )

    rl_comparison_path = (
        output_dir
        / f"{output_prefix}_rlvr_vs_rlaif.csv"
    )

    save_json(
        summary_path,
        final_summary,
    )

    save_csv(
        policy_path,
        policy_comparison,
    )

    save_csv(
        pairwise_path,
        pairwise_comparison,
    )

    save_csv(
        controlled_table_path,
        controlled_comparison,
    )

    save_csv(
        rl_comparison_path,
        interpretation[
            "rlvr_vs_rlaif"
        ],
    )

    print(
        "\nTask 5 final feedback comparison",
        flush=True,
    )

    print(
        "Status:",
        final_summary[
            "status"
        ],
        flush=True,
    )

    for row in interpretation[
        "rlvr_vs_rlaif"
    ]:
        print(
            f"{row['dataset']}: "
            f"RLVR exact="
            f"{row['rlvr_exact_accuracy']} | "
            f"RLAIF exact="
            f"{row['rlaif_exact_accuracy']} | "
            f"RLVR-RLAIF="
            f"{row['rlvr_minus_rlaif_exact_accuracy']}",
            flush=True,
        )

    print(
        "Saved final summary to",
        summary_path,
        flush=True,
    )

    print(
        "Saved policy comparison to",
        policy_path,
        flush=True,
    )

    print(
        "Saved pairwise comparison to",
        pairwise_path,
        flush=True,
    )

    print(
        "Saved controlled comparison to",
        controlled_table_path,
        flush=True,
    )

    return final_summary


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Combine the Task 5 in-domain, controlled, "
            "and transfer results into the final "
            "RLVR-versus-RLAIF comparison."
        )
    )

    ap.add_argument(
        "--config",
        default="configs/feedback.yaml",
    )

    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Create a provisional exact-only synthesis "
            "before controlled AI judging is complete."
        ),
    )

    ap.add_argument(
        "--output-prefix",
        default="feedback_comparison",
    )

    args = ap.parse_args()

    run_comparison(
        config_path=args.config,
        allow_incomplete=args.allow_incomplete,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()