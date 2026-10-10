from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.data import (
    load_yaml,
    read_jsonl,
    repo_path,
)
from common.logging_utils import save_json


POLICY_NAMES = (
    "sft",
    "dpo",
    "ppo",
    "grpo",
)

LABELS = (
    "SAFE_ANSWER",
    "JUSTIFIED_REFUSAL",
    "UNSAFE_COMPLIANCE",
    "OVER_REFUSAL",
    "AMBIGUOUS",
)


def results_dir(cfg: dict) -> Path:
    return (
        repo_path(cfg["results_dir"])
        / "task4_safety"
    )


def judged_path(
    cfg: dict,
    judged_prefix: str,
    policy_name: str,
) -> Path:
    return (
        results_dir(cfg)
        / f"{judged_prefix}_{policy_name}.jsonl"
    )


def load_expected_xstest(cfg: dict) -> pd.DataFrame:
    path = repo_path(cfg["paths"]["xstest"])
    df = pd.read_csv(path)

    required = {
        "xstest_id",
        "prompt",
        "benchmark_class",
        "type",
    }

    missing = sorted(
        required.difference(df.columns)
    )

    if missing:
        raise ValueError(
            "XSTest is missing columns: "
            + ", ".join(missing)
        )

    if df["xstest_id"].duplicated().any():
        raise ValueError(
            "XSTest contains duplicate IDs."
        )

    df = df.copy()

    df["benchmark_class"] = (
        df["benchmark_class"]
        .astype(str)
        .str.upper()
    )

    return df.reset_index(drop=True)


def load_judged_rows(
    cfg: dict,
    judged_prefix: str,
    policy_name: str,
) -> list[dict[str, Any]]:
    path = judged_path(
        cfg,
        judged_prefix,
        policy_name,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing judged responses for "
            f"{policy_name}: {path}"
        )

    rows = read_jsonl(path)

    if not rows:
        raise ValueError(
            f"The judged file is empty: {path}"
        )

    required = {
        "xstest_id",
        "policy",
        "prompt",
        "benchmark_class",
        "type",
        "response",
        "response_tokens",
        "judge_label",
        "judge_confidence",
    }

    missing = sorted(
        required.difference(rows[0])
    )

    if missing:
        raise ValueError(
            f"{path} is missing fields: "
            + ", ".join(missing)
        )

    ids = []

    for position, row in enumerate(rows):
        if str(row["policy"]) != policy_name:
            raise ValueError(
                f"Row {position} in {path} has "
                f"policy {row['policy']!r}; expected "
                f"{policy_name!r}."
            )

        benchmark_class = str(
            row["benchmark_class"]
        ).upper()

        if benchmark_class not in {
            "SAFE",
            "UNSAFE",
        }:
            raise ValueError(
                f"Invalid benchmark class at row "
                f"{position} in {path}."
            )

        label = str(
            row["judge_label"]
        ).upper()

        if label not in LABELS:
            raise ValueError(
                f"Invalid judge label {label!r} at "
                f"row {position} in {path}."
            )

        ids.append(int(row["xstest_id"]))

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"Duplicate XSTest IDs in {path}."
        )

    return rows


def validate_policy_alignment(
    rows_by_policy: dict[
        str,
        list[dict[str, Any]],
    ],
    expected_df: pd.DataFrame,
    allow_partial: bool,
) -> None:
    expected_by_id = {
        int(row["xstest_id"]): row
        for _, row in expected_df.iterrows()
    }

    expected_ids = set(expected_by_id)

    reference_policy = next(
        iter(rows_by_policy)
    )

    reference_rows = rows_by_policy[
        reference_policy
    ]

    reference_by_id = {
        int(row["xstest_id"]): row
        for row in reference_rows
    }

    reference_ids = set(reference_by_id)

    if not allow_partial:
        if reference_ids != expected_ids:
            missing = sorted(
                expected_ids.difference(
                    reference_ids
                )
            )

            extra = sorted(
                reference_ids.difference(
                    expected_ids
                )
            )

            raise ValueError(
                "The judged files do not contain the "
                "complete fixed XSTest set. "
                f"Missing: {missing[:10]}; "
                f"extra: {extra[:10]}. "
                "Use --allow-partial only for smoke "
                "tests."
            )

    for xstest_id, reference in (
        reference_by_id.items()
    ):
        if xstest_id not in expected_by_id:
            raise ValueError(
                f"Unexpected XSTest ID "
                f"{xstest_id}."
            )

        expected = expected_by_id[
            xstest_id
        ]

        for field in (
            "prompt",
            "benchmark_class",
            "type",
        ):
            if str(reference[field]) != str(
                expected[field]
            ):
                raise ValueError(
                    f"Saved {field} does not match "
                    f"XSTest for ID {xstest_id}."
                )

    for policy_name, rows in (
        rows_by_policy.items()
    ):
        rows_by_id = {
            int(row["xstest_id"]): row
            for row in rows
        }

        if set(rows_by_id) != reference_ids:
            missing = sorted(
                reference_ids.difference(
                    rows_by_id
                )
            )

            extra = sorted(
                set(rows_by_id).difference(
                    reference_ids
                )
            )

            raise ValueError(
                f"Policy {policy_name} does not use "
                "the common prompt set. "
                f"Missing: {missing[:10]}; "
                f"extra: {extra[:10]}."
            )

        for xstest_id, reference in (
            reference_by_id.items()
        ):
            row = rows_by_id[xstest_id]

            for field in (
                "prompt",
                "benchmark_class",
                "type",
            ):
                if str(row[field]) != str(
                    reference[field]
                ):
                    raise ValueError(
                        f"Policy {policy_name} has "
                        f"different {field} for "
                        f"XSTest ID {xstest_id}."
                    )


def rate(
    rows: list[dict[str, Any]],
    label: str,
) -> float | None:
    if not rows:
        return None

    return float(
        sum(
            str(row["judge_label"]).upper()
            == label
            for row in rows
        )
        / len(rows)
    )


def policy_metrics(
    policy_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    safe_rows = [
        row
        for row in rows
        if str(
            row["benchmark_class"]
        ).upper()
        == "SAFE"
    ]

    unsafe_rows = [
        row
        for row in rows
        if str(
            row["benchmark_class"]
        ).upper()
        == "UNSAFE"
    ]

    lengths = np.asarray(
        [
            int(row["response_tokens"])
            for row in rows
        ],
        dtype=float,
    )

    confidences = np.asarray(
        [
            float(row["judge_confidence"])
            for row in rows
        ],
        dtype=float,
    )

    return {
        "policy": policy_name,
        "num_examples": len(rows),
        "num_safe": len(safe_rows),
        "num_unsafe": len(unsafe_rows),
        "safe_answer_rate": rate(
            safe_rows,
            "SAFE_ANSWER",
        ),
        "safe_over_refusal_rate": rate(
            safe_rows,
            "OVER_REFUSAL",
        ),
        "unsafe_compliance_rate": rate(
            unsafe_rows,
            "UNSAFE_COMPLIANCE",
        ),
        "justified_refusal_rate": rate(
            unsafe_rows,
            "JUSTIFIED_REFUSAL",
        ),
        "ambiguous_judge_rate": rate(
            rows,
            "AMBIGUOUS",
        ),
        "mean_judge_confidence": float(
            confidences.mean()
        ),
        "response_length_mean": float(
            lengths.mean()
        ),
        "response_length_std": float(
            lengths.std()
        ),
        "response_length_iqr": float(
            np.percentile(lengths, 75)
            - np.percentile(lengths, 25)
        ),
        "generated_token_count": int(
            lengths.sum()
        ),
    }


def category_table(
    rows_by_policy: dict[
        str,
        list[dict[str, Any]],
    ],
) -> pd.DataFrame:
    records = []

    for policy_name, rows in (
        rows_by_policy.items()
    ):
        frame = pd.DataFrame(rows)

        frame["judge_label"] = (
            frame["judge_label"]
            .astype(str)
            .str.upper()
        )

        frame["benchmark_class"] = (
            frame["benchmark_class"]
            .astype(str)
            .str.upper()
        )

        grouped = frame.groupby(
            [
                "type",
                "benchmark_class",
            ],
            sort=True,
            dropna=False,
        )

        for (
            category,
            benchmark_class,
        ), group in grouped:
            record = {
                "policy": policy_name,
                "type": str(category),
                "benchmark_class": str(
                    benchmark_class
                ),
                "num_examples": int(
                    len(group)
                ),
                "response_length_mean": float(
                    group[
                        "response_tokens"
                    ].astype(float).mean()
                ),
            }

            for label in LABELS:
                count = int(
                    (
                        group["judge_label"]
                        == label
                    ).sum()
                )

                record[
                    f"{label.lower()}_count"
                ] = count

                record[
                    f"{label.lower()}_rate"
                ] = float(
                    count / len(group)
                )

            records.append(record)

    return pd.DataFrame(records)


def label_distribution_table(
    rows_by_policy: dict[
        str,
        list[dict[str, Any]],
    ],
) -> pd.DataFrame:
    records = []

    for policy_name, rows in (
        rows_by_policy.items()
    ):
        for benchmark_class in (
            "ALL",
            "SAFE",
            "UNSAFE",
        ):
            if benchmark_class == "ALL":
                selected = rows
            else:
                selected = [
                    row
                    for row in rows
                    if str(
                        row[
                            "benchmark_class"
                        ]
                    ).upper()
                    == benchmark_class
                ]

            for label in LABELS:
                count = sum(
                    str(
                        row["judge_label"]
                    ).upper()
                    == label
                    for row in selected
                )

                records.append(
                    {
                        "policy": policy_name,
                        "benchmark_class": (
                            benchmark_class
                        ),
                        "judge_label": label,
                        "count": int(count),
                        "rate": float(
                            count
                            / len(selected)
                        )
                        if selected
                        else None,
                    }
                )

    return pd.DataFrame(records)


def metric_from_manual_rows(
    policy_name: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    safe = frame.loc[
        frame["benchmark_class"]
        == "SAFE"
    ]

    unsafe = frame.loc[
        frame["benchmark_class"]
        == "UNSAFE"
    ]

    def manual_rate(
        selected: pd.DataFrame,
        label: str,
    ) -> float | None:
        if selected.empty:
            return None

        return float(
            (
                selected["manual_label"]
                == label
            ).mean()
        )

    return {
        "policy": policy_name,
        "num_manually_labeled": int(
            len(frame)
        ),
        "manual_safe_answer_rate": (
            manual_rate(
                safe,
                "SAFE_ANSWER",
            )
        ),
        "manual_safe_over_refusal_rate": (
            manual_rate(
                safe,
                "OVER_REFUSAL",
            )
        ),
        "manual_unsafe_compliance_rate": (
            manual_rate(
                unsafe,
                "UNSAFE_COMPLIANCE",
            )
        ),
        "manual_justified_refusal_rate": (
            manual_rate(
                unsafe,
                "JUSTIFIED_REFUSAL",
            )
        ),
        "manual_ambiguous_rate": (
            manual_rate(
                frame,
                "AMBIGUOUS",
            )
        ),
    }


def analyze_manual_audit(
    audit_path: Path,
    rows_by_policy: dict[
        str,
        list[dict[str, Any]],
    ],
    require_complete: bool,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if not audit_path.exists():
        if require_complete:
            raise FileNotFoundError(
                f"Manual audit sheet not found: "
                f"{audit_path}"
            )

        return (
            {
                "status": "not_available",
                "path": str(audit_path),
            },
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    audit = pd.read_csv(
        audit_path,
        keep_default_na=False,
    )

    required = {
        "xstest_id",
        "policy",
        "benchmark_class",
        "prompt",
        "response",
        "manual_label",
    }

    missing = sorted(
        required.difference(
            audit.columns
        )
    )

    if missing:
        raise ValueError(
            "Manual audit sheet is missing: "
            + ", ".join(missing)
        )

    if audit.duplicated(
        subset=[
            "xstest_id",
            "policy",
        ]
    ).any():
        raise ValueError(
            "Manual audit sheet contains duplicate "
            "prompt-policy rows."
        )

    audit["manual_label"] = (
        audit["manual_label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    completed = audit.loc[
        audit["manual_label"] != ""
    ].copy()

    invalid = sorted(
        set(completed["manual_label"])
        .difference(LABELS)
    )

    if invalid:
        raise ValueError(
            "Invalid manual labels: "
            + ", ".join(invalid)
        )

    missing_count = int(
        len(audit) - len(completed)
    )

    if require_complete and missing_count:
        raise ValueError(
            f"Manual audit is incomplete: "
            f"{missing_count} rows still need labels."
        )

    judged_lookup = {}

    for policy_name, rows in (
        rows_by_policy.items()
    ):
        for row in rows:
            judged_lookup[
                (
                    int(row["xstest_id"]),
                    policy_name,
                )
            ] = row

    joined_records = []

    for _, manual_row in completed.iterrows():
        key = (
            int(manual_row["xstest_id"]),
            str(manual_row["policy"]),
        )

        if key not in judged_lookup:
            raise KeyError(
                f"Manual audit row {key} is missing "
                "from judged responses."
            )

        judged = judged_lookup[key]

        if str(manual_row["prompt"]) != str(
            judged["prompt"]
        ):
            raise ValueError(
                f"Prompt mismatch for audit row "
                f"{key}."
            )

        if str(manual_row["response"]) != str(
            judged["response"]
        ):
            raise ValueError(
                f"Response mismatch for audit row "
                f"{key}."
            )

        joined_records.append(
            {
                "xstest_id": key[0],
                "policy": key[1],
                "benchmark_class": str(
                    judged[
                        "benchmark_class"
                    ]
                ).upper(),
                "type": str(
                    judged["type"]
                ),
                "prompt": str(
                    judged["prompt"]
                ),
                "response": str(
                    judged["response"]
                ),
                "judge_label": str(
                    judged["judge_label"]
                ).upper(),
                "manual_label": str(
                    manual_row[
                        "manual_label"
                    ]
                ).upper(),
                "judge_confidence": float(
                    judged[
                        "judge_confidence"
                    ]
                ),
                "manual_notes": str(
                    manual_row.get(
                        "manual_notes",
                        "",
                    )
                ),
            }
        )

    joined = pd.DataFrame(
        joined_records
    )

    if joined.empty:
        if require_complete:
            raise ValueError(
                "No completed manual labels found."
            )

        return (
            {
                "status": "incomplete",
                "path": str(audit_path),
                "total_rows": int(len(audit)),
                "completed_rows": 0,
                "missing_rows": missing_count,
            },
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    joined["agreement"] = (
        joined["judge_label"]
        == joined["manual_label"]
    )

    agreement_records = []

    for scope, group in [
        ("all", joined),
        *[
            (
                policy_name,
                joined.loc[
                    joined["policy"]
                    == policy_name
                ],
            )
            for policy_name in sorted(
                joined["policy"].unique()
            )
        ],
    ]:
        if group.empty:
            continue

        agreement_records.append(
            {
                "scope": scope,
                "num_examples": int(
                    len(group)
                ),
                "agreement_rate": float(
                    group[
                        "agreement"
                    ].mean()
                ),
                "judge_ambiguous_rate": float(
                    (
                        group["judge_label"]
                        == "AMBIGUOUS"
                    ).mean()
                ),
                "manual_ambiguous_rate": float(
                    (
                        group["manual_label"]
                        == "AMBIGUOUS"
                    ).mean()
                ),
            }
        )

    agreement_df = pd.DataFrame(
        agreement_records
    )

    confusion_records = []

    scopes = [
        ("all", joined),
        *[
            (
                policy_name,
                joined.loc[
                    joined["policy"]
                    == policy_name
                ],
            )
            for policy_name in sorted(
                joined["policy"].unique()
            )
        ],
    ]

    for scope, group in scopes:
        if group.empty:
            continue

        for manual_label in LABELS:
            manual_group = group.loc[
                group["manual_label"]
                == manual_label
            ]

            denominator = len(
                manual_group
            )

            for judge_label in LABELS:
                count = int(
                    (
                        manual_group[
                            "judge_label"
                        ]
                        == judge_label
                    ).sum()
                )

                confusion_records.append(
                    {
                        "scope": scope,
                        "manual_label": (
                            manual_label
                        ),
                        "judge_label": (
                            judge_label
                        ),
                        "count": count,
                        "row_rate": (
                            float(
                                count
                                / denominator
                            )
                            if denominator
                            else None
                        ),
                    }
                )

    confusion_df = pd.DataFrame(
        confusion_records
    )

    disagreement_df = joined.loc[
        ~joined["agreement"]
    ].copy()

    manual_metrics_records = []

    for policy_name in sorted(
        joined["policy"].unique()
    ):
        policy_frame = joined.loc[
            joined["policy"]
            == policy_name
        ]

        manual_metrics_records.append(
            metric_from_manual_rows(
                policy_name,
                policy_frame,
            )
        )

    manual_metrics_df = pd.DataFrame(
        manual_metrics_records
    )

    overall = agreement_df.loc[
        agreement_df["scope"] == "all"
    ].iloc[0]

    summary = {
        "status": (
            "complete"
            if missing_count == 0
            else "incomplete"
        ),
        "path": str(audit_path),
        "total_rows": int(len(audit)),
        "completed_rows": int(
            len(completed)
        ),
        "missing_rows": missing_count,
        "agreement_rate": float(
            overall["agreement_rate"]
        ),
        "judge_ambiguous_rate": float(
            overall[
                "judge_ambiguous_rate"
            ]
        ),
        "manual_ambiguous_rate": float(
            overall[
                "manual_ambiguous_rate"
            ]
        ),
        "num_disagreements": int(
            len(disagreement_df)
        ),
    }

    return (
        summary,
        agreement_df,
        confusion_df,
        disagreement_df,
        manual_metrics_df,
    )


def qualitative_candidates(
    rows_by_policy: dict[
        str,
        list[dict[str, Any]],
    ],
    disagreements: pd.DataFrame,
    limit_per_kind: int = 8,
) -> dict[str, Any]:
    all_rows = []

    for rows in rows_by_policy.values():
        all_rows.extend(rows)

    def select(
        benchmark_class: str,
        judge_label: str,
    ) -> list[dict[str, Any]]:
        candidates = [
            row
            for row in all_rows
            if str(
                row["benchmark_class"]
            ).upper()
            == benchmark_class
            and str(
                row["judge_label"]
            ).upper()
            == judge_label
        ]

        candidates.sort(
            key=lambda row: float(
                row["judge_confidence"]
            ),
            reverse=True,
        )

        return [
            {
                "xstest_id": int(
                    row["xstest_id"]
                ),
                "policy": str(
                    row["policy"]
                ),
                "type": str(
                    row["type"]
                ),
                "prompt": str(
                    row["prompt"]
                ),
                "response": str(
                    row["response"]
                ),
                "judge_label": str(
                    row["judge_label"]
                ),
                "judge_confidence": float(
                    row["judge_confidence"]
                ),
            }
            for row in candidates[
                :limit_per_kind
            ]
        ]

    policy_disagreements = []

    frame = pd.DataFrame(all_rows)

    for xstest_id, group in frame.groupby(
        "xstest_id",
        sort=True,
    ):
        if (
            group["judge_label"]
            .astype(str)
            .nunique()
            <= 1
        ):
            continue

        policy_disagreements.append(
            {
                "xstest_id": int(
                    xstest_id
                ),
                "prompt": str(
                    group.iloc[0]["prompt"]
                ),
                "benchmark_class": str(
                    group.iloc[0][
                        "benchmark_class"
                    ]
                ),
                "type": str(
                    group.iloc[0]["type"]
                ),
                "policy_labels": {
                    str(row["policy"]): str(
                        row["judge_label"]
                    )
                    for _, row in (
                        group.iterrows()
                    )
                },
            }
        )

    manual_disagreements = []

    if not disagreements.empty:
        for _, row in disagreements.head(
            limit_per_kind
        ).iterrows():
            manual_disagreements.append(
                {
                    "xstest_id": int(
                        row["xstest_id"]
                    ),
                    "policy": str(
                        row["policy"]
                    ),
                    "type": str(
                        row["type"]
                    ),
                    "prompt": str(
                        row["prompt"]
                    ),
                    "response": str(
                        row["response"]
                    ),
                    "judge_label": str(
                        row["judge_label"]
                    ),
                    "manual_label": str(
                        row["manual_label"]
                    ),
                    "manual_notes": str(
                        row["manual_notes"]
                    ),
                }
            )

    return {
        "harmful_compliance": select(
            "UNSAFE",
            "UNSAFE_COMPLIANCE",
        ),
        "justified_refusal": select(
            "UNSAFE",
            "JUSTIFIED_REFUSAL",
        ),
        "over_refusal": select(
            "SAFE",
            "OVER_REFUSAL",
        ),
        "policy_label_disagreements": (
            policy_disagreements[
                :limit_per_kind
            ]
        ),
        "manual_judge_disagreements": (
            manual_disagreements
        ),
    }


def run_evaluation(
    config_path: str,
    policies: list[str],
    judged_prefix: str = "judged",
    output_prefix: str = "safety",
    audit_sheet: str | None = None,
    allow_partial: bool = False,
    require_manual_audit: bool = False,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)

    if not policies:
        raise ValueError(
            "Select at least one policy."
        )

    invalid = sorted(
        set(policies).difference(
            POLICY_NAMES
        )
    )

    if invalid:
        raise ValueError(
            "Unknown policies: "
            + ", ".join(invalid)
        )

    if len(policies) != len(set(policies)):
        raise ValueError(
            "Each policy may be selected only once."
        )

    expected_df = load_expected_xstest(
        cfg
    )

    rows_by_policy = {
        policy_name: load_judged_rows(
            cfg=cfg,
            judged_prefix=judged_prefix,
            policy_name=policy_name,
        )
        for policy_name in policies
    }

    validate_policy_alignment(
        rows_by_policy=rows_by_policy,
        expected_df=expected_df,
        allow_partial=allow_partial,
    )

    output_dir = results_dir(cfg)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy_records = [
        policy_metrics(
            policy_name,
            rows_by_policy[policy_name],
        )
        for policy_name in policies
    ]

    policy_df = pd.DataFrame(
        policy_records
    )

    categories_df = category_table(
        rows_by_policy
    )

    distributions_df = (
        label_distribution_table(
            rows_by_policy
        )
    )

    if audit_sheet is None:
        audit_path = (
            output_dir
            / "manual_audit_sheet.csv"
        )
    else:
        audit_path = repo_path(
            audit_sheet
        )

    (
        manual_summary,
        agreement_df,
        confusion_df,
        disagreements_df,
        manual_metrics_df,
    ) = analyze_manual_audit(
        audit_path=audit_path,
        rows_by_policy=rows_by_policy,
        require_complete=(
            require_manual_audit
        ),
    )

    candidates = qualitative_candidates(
        rows_by_policy=rows_by_policy,
        disagreements=disagreements_df,
    )

    policy_path = (
        output_dir
        / f"{output_prefix}_policy_comparison.csv"
    )

    category_path = (
        output_dir
        / f"{output_prefix}_category_results.csv"
    )

    distribution_path = (
        output_dir
        / f"{output_prefix}_label_distribution.csv"
    )

    summary_path = (
        output_dir
        / f"{output_prefix}_summary.json"
    )

    candidate_path = (
        output_dir
        / (
            f"{output_prefix}_"
            "qualitative_candidates.json"
        )
    )

    policy_df.to_csv(
        policy_path,
        index=False,
    )

    categories_df.to_csv(
        category_path,
        index=False,
    )

    distributions_df.to_csv(
        distribution_path,
        index=False,
    )

    manual_paths = {}

    if not agreement_df.empty:
        agreement_path = (
            output_dir
            / "manual_audit_agreement.csv"
        )

        confusion_path = (
            output_dir
            / "manual_audit_confusion.csv"
        )

        disagreement_path = (
            output_dir
            / "manual_audit_disagreements.csv"
        )

        manual_metrics_path = (
            output_dir
            / "manual_audit_policy_metrics.csv"
        )

        agreement_df.to_csv(
            agreement_path,
            index=False,
        )

        confusion_df.to_csv(
            confusion_path,
            index=False,
        )

        disagreements_df.to_csv(
            disagreement_path,
            index=False,
        )

        manual_metrics_df.to_csv(
            manual_metrics_path,
            index=False,
        )

        manual_paths = {
            "agreement": str(
                agreement_path.relative_to(
                    repo_path(".")
                )
            ),
            "confusion": str(
                confusion_path.relative_to(
                    repo_path(".")
                )
            ),
            "disagreements": str(
                disagreement_path.relative_to(
                    repo_path(".")
                )
            ),
            "policy_metrics": str(
                manual_metrics_path.relative_to(
                    repo_path(".")
                )
            ),
        }

    save_json(
        candidate_path,
        candidates,
    )

    summary = {
        "config": config_path,
        "dataset": str(
            cfg["paths"]["xstest"]
        ),
        "policies": policies,
        "judged_prefix": judged_prefix,
        "allow_partial": bool(
            allow_partial
        ),
        "num_expected_xstest_prompts": int(
            len(expected_df)
        ),
        "num_evaluated_prompts_per_policy": int(
            len(
                next(
                    iter(
                        rows_by_policy.values()
                    )
                )
            )
        ),
        "metrics": policy_records,
        "manual_audit": manual_summary,
        "outputs": {
            "policy_comparison": str(
                policy_path.relative_to(
                    repo_path(".")
                )
            ),
            "category_results": str(
                category_path.relative_to(
                    repo_path(".")
                )
            ),
            "label_distribution": str(
                distribution_path.relative_to(
                    repo_path(".")
                )
            ),
            "qualitative_candidates": str(
                candidate_path.relative_to(
                    repo_path(".")
                )
            ),
            "manual_audit": manual_paths,
        },
    }

    save_json(
        summary_path,
        summary,
    )

    print(
        "\nTask 4 safety-calibration summary",
        flush=True,
    )

    print(
        "policy | safe answer | over-refusal | "
        "unsafe compliance | justified refusal | "
        "ambiguous | mean length",
        flush=True,
    )

    for record in policy_records:
        print(
            f"{record['policy']} | "
            f"{record['safe_answer_rate']:.4f} | "
            f"{record['safe_over_refusal_rate']:.4f} | "
            f"{record['unsafe_compliance_rate']:.4f} | "
            f"{record['justified_refusal_rate']:.4f} | "
            f"{record['ambiguous_judge_rate']:.4f} | "
            f"{record['response_length_mean']:.2f}",
            flush=True,
        )

    print(
        f"\nManual audit status: "
        f"{manual_summary['status']}",
        flush=True,
    )

    if (
        manual_summary["status"]
        in {"complete", "incomplete"}
        and "agreement_rate"
        in manual_summary
    ):
        print(
            "Manual-versus-AI agreement: "
            f"{manual_summary['agreement_rate']:.4f}",
            flush=True,
        )

    print(
        f"Saved Task 4 summary to "
        f"{summary_path}",
        flush=True,
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Task 4 safety calibration "
            "and manual-audit results."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/feedback.yaml",
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICY_NAMES,
        default=list(POLICY_NAMES),
    )

    parser.add_argument(
        "--judged-prefix",
        default="judged",
    )

    parser.add_argument(
        "--output-prefix",
        default="safety",
    )

    parser.add_argument(
        "--audit-sheet",
        help=(
            "Optional manual-audit CSV path."
        ),
    )

    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Allow a prompt prefix for smoke tests. "
            "Do not use for final results."
        ),
    )

    parser.add_argument(
        "--require-manual-audit",
        action="store_true",
        help=(
            "Fail unless every manual-audit row "
            "has a valid label."
        ),
    )

    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        policies=list(args.policies),
        judged_prefix=args.judged_prefix,
        output_prefix=args.output_prefix,
        audit_sheet=args.audit_sheet,
        allow_partial=args.allow_partial,
        require_manual_audit=(
            args.require_manual_audit
        ),
    )


if __name__ == "__main__":
    main()