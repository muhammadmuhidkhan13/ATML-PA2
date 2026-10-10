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

MANUAL_LABELS = (
    "SAFE_ANSWER",
    "JUSTIFIED_REFUSAL",
    "UNSAFE_COMPLIANCE",
    "OVER_REFUSAL",
    "AMBIGUOUS",
)


def fixed_audit_ids(
    base_rows: list[dict[str, Any]],
    per_class: int,
    seed: int,
) -> list[int]:
    """Select fixed safe and unsafe prompt IDs."""
    if per_class <= 0:
        raise ValueError(
            "per_class must be positive."
        )

    metadata = pd.DataFrame(base_rows)

    required = {
        "xstest_id",
        "benchmark_class",
    }

    missing = sorted(
        required.difference(
            metadata.columns
        )
    )

    if missing:
        raise ValueError(
            "Generation rows are missing: "
            + ", ".join(missing)
        )

    metadata = (
        metadata[
            [
                "xstest_id",
                "benchmark_class",
            ]
        ]
        .drop_duplicates(
            subset=["xstest_id"]
        )
        .copy()
    )

    metadata["benchmark_class"] = (
        metadata["benchmark_class"]
        .astype(str)
        .str.upper()
    )

    rng = np.random.default_rng(seed)
    selected_ids: list[int] = []

    for benchmark_class in (
        "SAFE",
        "UNSAFE",
    ):
        pool = (
            metadata.loc[
                metadata[
                    "benchmark_class"
                ]
                == benchmark_class,
                "xstest_id",
            ]
            .astype(int)
            .to_numpy()
        )

        if len(pool) < per_class:
            raise ValueError(
                f"Not enough {benchmark_class} "
                f"rows for the audit: need "
                f"{per_class}, found {len(pool)}."
            )

        chosen = rng.choice(
            pool,
            size=per_class,
            replace=False,
        )

        selected_ids.extend(
            int(value)
            for value in chosen.tolist()
        )

    return sorted(selected_ids)


def task4_results_dir(cfg: dict) -> Path:
    return (
        repo_path(cfg["results_dir"])
        / "task4_safety"
    )


def generation_path(
    cfg: dict,
    generation_prefix: str,
    policy_name: str,
) -> Path:
    return (
        task4_results_dir(cfg)
        / (
            f"{generation_prefix}_"
            f"{policy_name}.jsonl"
        )
    )


def load_policy_generations(
    cfg: dict,
    generation_prefix: str,
    policy_name: str,
) -> list[dict[str, Any]]:
    """Load and validate one policy generation file."""
    path = generation_path(
        cfg,
        generation_prefix,
        policy_name,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {policy_name} generations: "
            f"{path}"
        )

    rows = read_jsonl(path)

    if not rows:
        raise ValueError(
            f"The generation file is empty: {path}"
        )

    required = {
        "xstest_id",
        "policy",
        "prompt",
        "benchmark_class",
        "type",
        "response",
        "response_tokens",
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
                f"policy {row['policy']!r}, "
                f"expected {policy_name!r}."
            )

        benchmark_class = str(
            row["benchmark_class"]
        ).upper()

        if benchmark_class not in {
            "SAFE",
            "UNSAFE",
        }:
            raise ValueError(
                f"Invalid benchmark class in "
                f"{path} at row {position}."
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
) -> None:
    """Check that every policy uses the same prompts."""
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

    for policy_name, rows in (
        rows_by_policy.items()
    ):
        rows_by_id = {
            int(row["xstest_id"]): row
            for row in rows
        }

        if set(rows_by_id) != set(
            reference_by_id
        ):
            missing = sorted(
                set(reference_by_id)
                .difference(rows_by_id)
            )

            extra = sorted(
                set(rows_by_id)
                .difference(reference_by_id)
            )

            raise ValueError(
                f"Policy {policy_name} is not "
                "aligned with the common prompt set. "
                f"Missing IDs: {missing[:10]}; "
                f"extra IDs: {extra[:10]}."
            )

        for xstest_id, reference in (
            reference_by_id.items()
        ):
            row = rows_by_id[xstest_id]

            fields = (
                "prompt",
                "benchmark_class",
                "type",
            )

            for field in fields:
                if str(row[field]) != str(
                    reference[field]
                ):
                    raise ValueError(
                        f"Policy {policy_name} has "
                        f"different {field!r} for "
                        f"xstest_id={xstest_id}."
                    )


def load_or_create_audit_ids(
    cfg: dict,
    base_rows: list[dict[str, Any]],
    ids_path: Path,
    overwrite: bool,
) -> list[int]:
    """Reuse fixed IDs or create them exactly once."""
    per_class = int(
        cfg["manual_audit_per_class"]
    )

    expected_total = 2 * per_class

    if ids_path.exists() and not overwrite:
        ids_df = pd.read_csv(ids_path)

        if "xstest_id" not in ids_df.columns:
            raise ValueError(
                f"{ids_path} does not contain an "
                "xstest_id column."
            )

        ids = (
            ids_df["xstest_id"]
            .astype(int)
            .tolist()
        )

        if len(ids) != expected_total:
            raise ValueError(
                f"Expected {expected_total} fixed "
                f"audit IDs, found {len(ids)}."
            )

        if len(ids) != len(set(ids)):
            raise ValueError(
                "The saved audit IDs contain "
                "duplicates."
            )

        available = {
            int(row["xstest_id"]): str(
                row["benchmark_class"]
            ).upper()
            for row in base_rows
        }

        missing = sorted(
            set(ids).difference(available)
        )

        if missing:
            raise ValueError(
                "Saved audit IDs are missing from "
                f"the generation cache: {missing}."
            )

        safe_count = sum(
            available[xstest_id] == "SAFE"
            for xstest_id in ids
        )

        unsafe_count = sum(
            available[xstest_id] == "UNSAFE"
            for xstest_id in ids
        )

        if (
            safe_count != per_class
            or unsafe_count != per_class
        ):
            raise ValueError(
                "Saved audit IDs are not balanced: "
                f"SAFE={safe_count}, "
                f"UNSAFE={unsafe_count}."
            )

        return ids

    ids = fixed_audit_ids(
        base_rows=base_rows,
        per_class=per_class,
        seed=int(cfg["seed"]),
    )

    base_by_id = {
        int(row["xstest_id"]): row
        for row in base_rows
    }

    ids_records = []

    for xstest_id in ids:
        row = base_by_id[xstest_id]

        ids_records.append(
            {
                "xstest_id": xstest_id,
                "benchmark_class": str(
                    row["benchmark_class"]
                ).upper(),
                "type": str(row["type"]),
            }
        )

    pd.DataFrame(ids_records).to_csv(
        ids_path,
        index=False,
    )

    return ids


def build_audit_sheet(
    rows_by_policy: dict[
        str,
        list[dict[str, Any]],
    ],
    audit_ids: list[int],
    seed: int,
) -> pd.DataFrame:
    """Create a blinded sheet without AI labels."""
    records = []

    for policy_name, rows in (
        rows_by_policy.items()
    ):
        rows_by_id = {
            int(row["xstest_id"]): row
            for row in rows
        }

        for xstest_id in audit_ids:
            if xstest_id not in rows_by_id:
                raise KeyError(
                    f"Policy {policy_name} is missing "
                    f"audit ID {xstest_id}."
                )

            row = rows_by_id[xstest_id]

            records.append(
                {
                    "xstest_id": xstest_id,
                    "policy": policy_name,
                    "benchmark_class": str(
                        row["benchmark_class"]
                    ).upper(),
                    "type": str(row["type"]),
                    "prompt": str(
                        row["prompt"]
                    ),
                    "response": str(
                        row["response"]
                    ),
                    "response_tokens": int(
                        row["response_tokens"]
                    ),
                    "manual_label": "",
                    "manual_notes": "",
                }
            )

    audit_df = pd.DataFrame(records)

    # Shuffle deterministically so manual labels
    # are not assigned in policy-grouped blocks.
    audit_df = audit_df.sample(
        frac=1.0,
        random_state=int(seed),
    ).reset_index(drop=True)

    audit_df.insert(
        0,
        "audit_order",
        np.arange(
            1,
            len(audit_df) + 1,
        ),
    )

    return audit_df


def run_make_audit_sheet(
    config_path: str,
    generation_prefix: str = "generated",
    policies: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)

    if policies is None:
        policies = list(POLICY_NAMES)

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

    rows_by_policy = {
        policy_name: load_policy_generations(
            cfg=cfg,
            generation_prefix=(
                generation_prefix
            ),
            policy_name=policy_name,
        )
        for policy_name in policies
    }

    validate_policy_alignment(
        rows_by_policy
    )

    output_dir = task4_results_dir(cfg)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ids_path = (
        output_dir
        / "manual_audit_ids.csv"
    )

    sheet_path = (
        output_dir
        / "manual_audit_sheet.csv"
    )

    manifest_path = (
        output_dir
        / "manual_audit_manifest.json"
    )

    if sheet_path.exists() and not overwrite:
        raise FileExistsError(
            f"The manual audit sheet already exists: "
            f"{sheet_path}\n"
            "It may contain manual work. Preserve it, "
            "or use --overwrite only if you explicitly "
            "intend to replace it."
        )

    base_policy = (
        "sft"
        if "sft" in rows_by_policy
        else policies[0]
    )

    audit_ids = load_or_create_audit_ids(
        cfg=cfg,
        base_rows=rows_by_policy[
            base_policy
        ],
        ids_path=ids_path,
        overwrite=overwrite,
    )

    audit_df = build_audit_sheet(
        rows_by_policy=rows_by_policy,
        audit_ids=audit_ids,
        seed=int(cfg["seed"]),
    )

    audit_df.to_csv(
        sheet_path,
        index=False,
    )

    per_class = int(
        cfg["manual_audit_per_class"]
    )

    manifest = {
        "config": config_path,
        "generation_prefix": (
            generation_prefix
        ),
        "seed": int(cfg["seed"]),
        "policies": policies,
        "base_policy_for_id_selection": (
            base_policy
        ),
        "manual_audit_per_class": per_class,
        "num_prompt_ids": len(
            audit_ids
        ),
        "num_policy_response_rows": int(
            len(audit_df)
        ),
        "num_safe_prompt_ids": int(
            sum(
                str(
                    row[
                        "benchmark_class"
                    ]
                ).upper()
                == "SAFE"
                for row in rows_by_policy[
                    base_policy
                ]
                if int(
                    row["xstest_id"]
                )
                in set(audit_ids)
            )
        ),
        "num_unsafe_prompt_ids": int(
            sum(
                str(
                    row[
                        "benchmark_class"
                    ]
                ).upper()
                == "UNSAFE"
                for row in rows_by_policy[
                    base_policy
                ]
                if int(
                    row["xstest_id"]
                )
                in set(audit_ids)
            )
        ),
        "allowed_manual_labels": list(
            MANUAL_LABELS
        ),
        "blind_to_ai_labels": True,
        "audit_ids": str(
            ids_path.relative_to(
                repo_path(".")
            )
        ),
        "audit_sheet": str(
            sheet_path.relative_to(
                repo_path(".")
            )
        ),
        "instructions": [
            (
                "Read only the prompt and response "
                "before assigning manual_label."
            ),
            (
                "Do not open the judged response "
                "files until all manual labels are "
                "complete."
            ),
            (
                "Use exactly one allowed label for "
                "every row."
            ),
            (
                "Use manual_notes for brief audit "
                "comments or uncertainty."
            ),
        ],
    }

    save_json(
        manifest_path,
        manifest,
    )

    print(
        f"Wrote fixed audit IDs to "
        f"{ids_path}",
        flush=True,
    )

    print(
        f"Wrote blind manual-audit sheet to "
        f"{sheet_path}",
        flush=True,
    )

    print(
        f"Wrote audit manifest to "
        f"{manifest_path}",
        flush=True,
    )

    print(
        f"Audit contains {len(audit_ids)} "
        f"prompt IDs and {len(audit_df)} "
        "policy-response rows.",
        flush=True,
    )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create the fixed blind Task 4 "
            "manual-audit sheet."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/feedback.yaml",
    )

    parser.add_argument(
        "--generation-prefix",
        default="generated",
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICY_NAMES,
        default=list(POLICY_NAMES),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace the audit IDs and sheet. "
            "This can destroy completed manual labels."
        ),
    )

    args = parser.parse_args()

    run_make_audit_sheet(
        config_path=args.config,
        generation_prefix=(
            args.generation_prefix
        ),
        policies=list(args.policies),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()