from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path("results/task3_grpo")

RUN_NAMES = [
    "standard",
    "normalization_grpo",
    "normalization_dr_grpo",
]

FIELD_MAPPING = {
    "sampled_kl": "mean_sampled_kl",
    "entropy": "mean_entropy",
    "clip_fraction": "mean_clip_fraction",
    "ratio_mean": "mean_ratio",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error

    return records


def valid_token_weighted_mean(
    records: list[dict],
    field: str,
) -> float:
    weighted_sum = 0.0
    total_tokens = 0.0

    for record in records:
        value = record.get(field)
        tokens = record.get("valid_policy_tokens")

        if not is_finite_number(value):
            raise RuntimeError(
                f"Missing or invalid {field!r} "
                f"in update {record.get('update')}"
            )

        if not is_finite_number(tokens):
            raise RuntimeError(
                "Missing or invalid "
                "'valid_policy_tokens' in update "
                f"{record.get('update')}"
            )

        tokens = float(tokens)

        if tokens < 0:
            raise RuntimeError(
                "valid_policy_tokens cannot be negative"
            )

        if tokens == 0:
            continue

        weighted_sum += float(value) * tokens
        total_tokens += tokens

    if total_tokens <= 0:
        raise RuntimeError(
            f"No valid tokens available for {field}"
        )

    return weighted_sum / total_tokens


log_hashes_before = {}
report_rows = []
correction_runs = []

for run_name in RUN_NAMES:
    log_path = (
        ROOT / f"{run_name}_train_log.jsonl"
    )
    summary_path = (
        ROOT / f"{run_name}_train_summary.json"
    )

    if not log_path.exists():
        raise FileNotFoundError(log_path)

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    log_hashes_before[str(log_path)] = (
        file_sha256(log_path)
    )

    records = load_jsonl(log_path)

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    attempted_updates = len(records)

    valid_updates = [
        record
        for record in records
        if float(
            record.get(
                "valid_policy_tokens",
                0,
            )
        ) > 0
    ]

    zero_token_updates = [
        record
        for record in records
        if float(
            record.get(
                "valid_policy_tokens",
                0,
            )
        ) == 0
    ]

    total_valid_tokens = int(
        sum(
            float(
                record["valid_policy_tokens"]
            )
            for record in records
        )
    )

    total_generated_tokens = int(
        summary["total_generated_tokens"]
    )

    retained_fraction = (
        total_valid_tokens
        / total_generated_tokens
        if total_generated_tokens > 0
        else None
    )

    current_values = {
        summary_field: summary.get(summary_field)
        for summary_field
        in FIELD_MAPPING.values()
    }

    corrected_values = {}

    for log_field, summary_field in (
        FIELD_MAPPING.items()
    ):
        corrected_value = (
            valid_token_weighted_mean(
                records,
                log_field,
            )
        )

        summary[summary_field] = (
            corrected_value
        )
        corrected_values[summary_field] = (
            corrected_value
        )

    if (
        "pre_correction_token_metrics"
        not in summary
    ):
        summary[
            "pre_correction_token_metrics"
        ] = current_values

    old_values = dict(
        summary[
            "pre_correction_token_metrics"
        ]
    )

    summary["attempted_updates"] = (
        attempted_updates
    )
    summary["updates_with_valid_policy_tokens"] = (
        len(valid_updates)
    )
    summary[
        "updates_without_valid_policy_tokens"
    ] = len(zero_token_updates)
    summary[
        "zero_valid_policy_token_update_ids"
    ] = [
        record.get("update", index)
        for index, record in enumerate(
            zero_token_updates,
            start=1,
        )
    ]
    summary["total_valid_policy_tokens"] = (
        total_valid_tokens
    )
    summary[
        "retained_policy_token_fraction"
    ] = retained_fraction

    summary[
        "token_metric_aggregation"
    ] = {
        "method": (
            "global valid-response-token weighted mean"
        ),
        "weight_field": "valid_policy_tokens",
        "zero_valid_token_updates": (
            "excluded from token-level metric means"
        ),
        "manual_alignment": (
            "Entropy, sampled KL, clip fraction, "
            "and ratio are aggregated over valid "
            "response tokens."
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    correction_runs.append(
        {
            "run_name": run_name,
            "source_log": str(log_path),
            "corrected_summary": str(
                summary_path
            ),
            "attempted_updates": (
                attempted_updates
            ),
            "optimizer_updates": summary.get(
                "optimizer_updates"
            ),
            "updates_with_valid_policy_tokens": (
                len(valid_updates)
            ),
            "updates_without_valid_policy_tokens": (
                len(zero_token_updates)
            ),
            "zero_valid_policy_token_update_ids": (
                summary[
                    "zero_valid_policy_token_update_ids"
                ]
            ),
            "total_generated_tokens": (
                total_generated_tokens
            ),
            "total_valid_policy_tokens": (
                total_valid_tokens
            ),
            "retained_policy_token_fraction": (
                retained_fraction
            ),
            "old_values": old_values,
            "corrected_values": (
                corrected_values
            ),
        }
    )

    report_rows.append(
        {
            "run_name": run_name,
            "attempted_updates": attempted_updates,
            "optimizer_updates": summary.get(
                "optimizer_updates"
            ),
            "valid_token_updates": len(
                valid_updates
            ),
            "zero_token_updates": len(
                zero_token_updates
            ),
            "valid_policy_tokens": (
                total_valid_tokens
            ),
            "generated_tokens": (
                total_generated_tokens
            ),
            "retained_fraction": (
                retained_fraction
            ),
            "old_sampled_kl": old_values[
                "mean_sampled_kl"
            ],
            "corrected_sampled_kl": (
                corrected_values[
                    "mean_sampled_kl"
                ]
            ),
            "old_entropy": old_values[
                "mean_entropy"
            ],
            "corrected_entropy": (
                corrected_values[
                    "mean_entropy"
                ]
            ),
            "old_clip_fraction": old_values[
                "mean_clip_fraction"
            ],
            "corrected_clip_fraction": (
                corrected_values[
                    "mean_clip_fraction"
                ]
            ),
            "old_ratio": old_values[
                "mean_ratio"
            ],
            "corrected_ratio": (
                corrected_values[
                    "mean_ratio"
                ]
            ),
        }
    )

log_hashes_after = {
    path: file_sha256(Path(path))
    for path in log_hashes_before
}

if log_hashes_before != log_hashes_after:
    raise RuntimeError(
        "Raw training logs changed unexpectedly."
    )

report_json = (
    ROOT
    / "training_token_metric_correction.json"
)

report_json.write_text(
    json.dumps(
        {
            "correction_scope": (
                "Run-level token diagnostics only"
            ),
            "raw_logs_modified": False,
            "checkpoints_modified": False,
            "retraining_performed": False,
            "raw_log_hashes": log_hashes_after,
            "runs": correction_runs,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

report_csv = (
    ROOT
    / "training_token_metric_correction.csv"
)

with report_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(
            report_rows[0].keys()
        ),
    )
    writer.writeheader()
    writer.writerows(report_rows)

print("\nTASK 3 TOKEN-METRIC CORRECTION")
print("=" * 92)

for run in correction_runs:
    print(f"\nRun: {run['run_name']}")
    print(
        "Attempted / optimizer / valid-token updates:",
        f"{run['attempted_updates']} / "
        f"{run['optimizer_updates']} / "
        f"{run['updates_with_valid_policy_tokens']}",
    )
    print(
        "Zero-token update IDs:",
        run[
            "zero_valid_policy_token_update_ids"
        ],
    )
    print(
        "Valid/generated tokens:",
        f"{run['total_valid_policy_tokens']:,} / "
        f"{run['total_generated_tokens']:,}",
    )
    print(
        "Retained token fraction:",
        f"{run['retained_policy_token_fraction']:.4%}",
    )

    for field in FIELD_MAPPING.values():
        print(
            f"{field}:",
            f"{run['old_values'][field]} -> "
            f"{run['corrected_values'][field]}",
        )

print("\nRaw training logs unchanged: PASS")
print("JSON report:", report_json.resolve())
print("CSV report: ", report_csv.resolve())
