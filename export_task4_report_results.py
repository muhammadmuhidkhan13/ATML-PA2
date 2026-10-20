import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path("results/task4_safety")
ASSETS = ROOT / "report_assets_corrected"
OUTPUT = ASSETS / "task4_complete_report_results.txt"

REQUIRED = [
    ROOT / "corrected_final_evaluation_summary.json",
    ROOT / "corrected_final_evaluation_policy_comparison.csv",
    ROOT / "corrected_final_evaluation_category_results.csv",
    ROOT / "corrected_final_evaluation_label_distribution.csv",
    ROOT / "manual_audit_sheet_corrected_grpo.csv",
    ROOT / "manual_audit_policy_metrics.csv",
    ROOT / "manual_audit_agreement.csv",
    ROOT / "manual_audit_confusion.csv",
    ROOT / "manual_audit_disagreements.csv",
    ASSETS / "task4_main_comparison_table.csv",
    ASSETS / "task4_category_label_distribution.csv",
    ASSETS / "task4_judge_confusion_counts.csv",
    ASSETS / "task4_judge_confusion_row_rates.csv",
    ASSETS / "task4_qualitative_examples.csv",
    ASSETS / "task4_report_summary.json",
    ASSETS / "task4_report_summary.md",
    Path("configs/feedback_task4_corrected.yaml"),
]

for path in REQUIRED:
    if not path.exists():
        raise FileNotFoundError(path)


def read_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def read_jsonl(path):
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: {error}"
                ) from error

    return records


def file_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def dataframe_text(frame):
    if frame.empty:
        return "(empty table)"

    return frame.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )


sections = []


def add_section(title, content):
    sections.append(
        "\n".join([
            "",
            "=" * 110,
            title,
            "=" * 110,
            str(content).rstrip(),
        ])
    )


summary = read_json(
    ROOT
    / "corrected_final_evaluation_summary.json"
)

policy_comparison = pd.read_csv(
    ROOT
    / "corrected_final_evaluation_policy_comparison.csv"
)

main_comparison = pd.read_csv(
    ASSETS
    / "task4_main_comparison_table.csv"
)

manual_metrics = pd.read_csv(
    ROOT
    / "manual_audit_policy_metrics.csv"
)

agreement = pd.read_csv(
    ROOT
    / "manual_audit_agreement.csv"
)

confusion = pd.read_csv(
    ROOT
    / "manual_audit_confusion.csv"
)

judge_confusion_counts = pd.read_csv(
    ASSETS
    / "task4_judge_confusion_counts.csv",
    index_col=0,
)

judge_confusion_rates = pd.read_csv(
    ASSETS
    / "task4_judge_confusion_row_rates.csv",
    index_col=0,
)

category_results = pd.read_csv(
    ROOT
    / "corrected_final_evaluation_category_results.csv"
)

category_distribution = pd.read_csv(
    ASSETS
    / "task4_category_label_distribution.csv"
)

qualitative = pd.read_csv(
    ASSETS
    / "task4_qualitative_examples.csv"
)

audit = pd.read_csv(
    ROOT
    / "manual_audit_sheet_corrected_grpo.csv"
)

disagreements = pd.read_csv(
    ROOT
    / "manual_audit_disagreements.csv"
)


add_section(
    "TASK 4 EXPERIMENTAL PROTOCOL",
    "\n".join([
        f"Config: {summary['config']}",
        f"Dataset: {summary['dataset']}",
        (
            "Policies: "
            + ", ".join(summary["policies"])
        ),
        (
            "Expected prompts per policy: "
            f"{summary['num_expected_xstest_prompts']}"
        ),
        (
            "Evaluated prompts per policy: "
            f"{summary['num_evaluated_prompts_per_policy']}"
        ),
        "XSTest composition: 250 SAFE and 200 UNSAFE prompts.",
        "Generation: deterministic Task 4 decoding.",
        "Judge labels: SAFE_ANSWER, OVER_REFUSAL, "
        "UNSAFE_COMPLIANCE, JUSTIFIED_REFUSAL, AMBIGUOUS.",
        "Manual audit: 60 matched prompts × 4 policies = 240 responses.",
        "Manual sample composition per policy: 30 SAFE and 30 UNSAFE.",
        "Corrected GRPO adapter: "
        "outputs/task3_grpo/standard_corrected_20261018",
    ]),
)


add_section(
    "MAIN REPORT COMPARISON: AI JUDGE AND MANUAL AUDIT",
    dataframe_text(main_comparison),
)


add_section(
    "FULL XSTEST POLICY METRICS",
    dataframe_text(policy_comparison),
)


add_section(
    "MANUAL-AUDIT POLICY METRICS",
    dataframe_text(manual_metrics),
)


manual_label_counts = (
    audit["manual_label"]
    .fillna("")
    .astype(str)
    .str.strip()
    .value_counts()
    .rename_axis("manual_label")
    .reset_index(name="count")
)

manual_policy_counts = (
    audit["policy"]
    .astype(str)
    .str.lower()
    .value_counts()
    .rename_axis("policy")
    .reset_index(name="count")
)

add_section(
    "MANUAL-AUDIT SAMPLE AND LABEL COUNTS",
    "\n\n".join([
        "Rows by policy:\n"
        + dataframe_text(manual_policy_counts),
        "Manual labels:\n"
        + dataframe_text(manual_label_counts),
        (
            "Unique ID-policy pairs: "
            f"{audit[['xstest_id', 'policy']].drop_duplicates().shape[0]}"
        ),
        (
            "Blank manual labels: "
            f"{audit['manual_label'].isna().sum() + (audit['manual_label'].astype(str).str.strip() == '').sum()}"
        ),
    ]),
)


add_section(
    "AI-JUDGE/MANUAL AGREEMENT SUMMARY",
    dataframe_text(agreement),
)


add_section(
    "AI-JUDGE CONFUSION COUNTS",
    judge_confusion_counts.to_string(),
)


add_section(
    "AI-JUDGE CONFUSION ROW RATES",
    judge_confusion_rates.to_string(
        float_format=lambda value: f"{value:.6f}"
    ),
)


add_section(
    "FULL MANUAL/AI CONFUSION DATA",
    dataframe_text(confusion),
)


add_section(
    "CATEGORY-LEVEL SAFETY RESULTS",
    dataframe_text(category_results),
)


add_section(
    "CATEGORY LABEL DISTRIBUTION",
    dataframe_text(category_distribution),
)


add_section(
    "REPORT QUALITATIVE EXAMPLES",
    dataframe_text(qualitative),
)


disagreement_summary = (
    disagreements
    .groupby(
        ["policy", "manual_label", "judge_label"],
        dropna=False,
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        ["policy", "count"],
        ascending=[True, False],
    )
)

add_section(
    "MANUAL/AI DISAGREEMENT BREAKDOWN",
    dataframe_text(disagreement_summary),
)


judgment_rows = []

for policy in [
    "sft",
    "dpo",
    "ppo",
    "grpo",
]:
    path = (
        ROOT
        / f"corrected_final_judged_{policy}.jsonl"
    )

    records = read_jsonl(path)
    labels = Counter(
        record.get("judge_label")
        for record in records
    )

    judgment_rows.append({
        "policy": policy,
        "records": len(records),
        "unique_ids": len({
            record["xstest_id"]
            for record in records
        }),
        "parse_success": sum(
            bool(
                record.get(
                    "judge_parse_success",
                    False,
                )
            )
            for record in records
        ),
        "SAFE_ANSWER": labels.get(
            "SAFE_ANSWER",
            0,
        ),
        "OVER_REFUSAL": labels.get(
            "OVER_REFUSAL",
            0,
        ),
        "UNSAFE_COMPLIANCE": labels.get(
            "UNSAFE_COMPLIANCE",
            0,
        ),
        "JUSTIFIED_REFUSAL": labels.get(
            "JUSTIFIED_REFUSAL",
            0,
        ),
        "AMBIGUOUS": labels.get(
            "AMBIGUOUS",
            0,
        ),
    })

judgment_integrity = pd.DataFrame(
    judgment_rows
)

add_section(
    "FULL-JUDGMENT INTEGRITY AND LABEL COUNTS",
    dataframe_text(judgment_integrity),
)


old_comparison_path = (
    ROOT
    / "final_evaluation_policy_comparison.csv"
)

if old_comparison_path.exists():
    old_comparison = pd.read_csv(
        old_comparison_path
    )

    old_grpo = (
        old_comparison[
            old_comparison["policy"]
            .astype(str)
            .str.lower()
            .eq("grpo")
        ]
        .iloc[0]
    )

    corrected_grpo = (
        policy_comparison[
            policy_comparison["policy"]
            .astype(str)
            .str.lower()
            .eq("grpo")
        ]
        .iloc[0]
    )

    comparison_fields = [
        "safe_answer_rate",
        "safe_over_refusal_rate",
        "unsafe_compliance_rate",
        "justified_refusal_rate",
        "ambiguous_judge_rate",
        "response_length_mean",
        "response_length_std",
        "response_length_iqr",
        "generated_token_count",
    ]

    old_new_rows = []

    for field in comparison_fields:
        old_value = old_grpo[field]
        corrected_value = corrected_grpo[field]

        old_new_rows.append({
            "metric": field,
            "original_grpo": old_value,
            "corrected_grpo": corrected_value,
            "difference": (
                corrected_value
                - old_value
            ),
        })

    add_section(
        "ORIGINAL VERSUS CORRECTED GRPO TASK 4 RESULTS",
        dataframe_text(
            pd.DataFrame(old_new_rows)
        ),
    )


manual_summary = summary.get(
    "manual_audit",
    {}
)

add_section(
    "MANUAL-AUDIT CALIBRATION SUMMARY",
    "\n".join([
        (
            "Status: "
            f"{manual_summary.get('status')}"
        ),
        (
            "Completed rows: "
            f"{manual_summary.get('completed_rows')}"
        ),
        (
            "Missing rows: "
            f"{manual_summary.get('missing_rows')}"
        ),
        (
            "Manual/AI agreement: "
            f"{manual_summary.get('agreement_rate')}"
        ),
        (
            "Disagreements: "
            f"{manual_summary.get('num_disagreements')}"
        ),
        (
            "AI ambiguous rate: "
            f"{manual_summary.get('judge_ambiguous_rate')}"
        ),
        (
            "Manual ambiguous rate: "
            f"{manual_summary.get('manual_ambiguous_rate')}"
        ),
    ]),
)


figure_paths = sorted(
    path
    for path in ASSETS.iterdir()
    if path.suffix.lower()
    in {".png", ".pdf"}
)

figure_inventory = "\n".join(
    f"{path.name} | "
    f"{path.stat().st_size:,} bytes"
    for path in figure_paths
)

add_section(
    "REPORT FIGURE INVENTORY",
    figure_inventory,
)


source_inventory = []

for path in REQUIRED:
    source_inventory.append({
        "file": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    })

add_section(
    "EXACT REPORT SOURCE FILES",
    dataframe_text(
        pd.DataFrame(source_inventory)
    ),
)


config_text = Path(
    "configs/feedback_task4_corrected.yaml"
).read_text(
    encoding="utf-8",
)

add_section(
    "CORRECTED TASK 4 CONFIGURATION",
    config_text,
)


existing_summary = (
    ASSETS
    / "task4_report_summary.md"
).read_text(
    encoding="utf-8",
)

add_section(
    "EXISTING REPORT SUMMARY",
    existing_summary,
)


complete_text = (
    "TASK 4 COMPLETE REPORT RESULTS\n"
    + "\n".join(sections)
    + "\n"
)

OUTPUT.write_text(
    complete_text,
    encoding="utf-8",
)

print(complete_text)

print("=" * 110)
print("TASK 4 REPORT RESULT EXPORT COMPLETED")
print("=" * 110)
print("Output:", OUTPUT.resolve())
print("Bytes:", OUTPUT.stat().st_size)
print("SHA-256:", file_sha256(OUTPUT))
