import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


REPO = Path.cwd()
RESULTS = REPO / "results"
OUTPUT = RESULTS / "task4_task5_complete_report_data.txt"

MAX_TEXT_LENGTH = 1500
MAX_FULL_JSON_LIST = 250
MAX_FULL_JSONL_ROWS = 120

PREFERRED_TASK4_FILES = [
    # Primary corrected full-policy results
    "corrected_final_evaluation_policy_comparison.csv",
    "corrected_final_evaluation_category_results.csv",
    "corrected_final_evaluation_label_distribution.csv",
    "corrected_final_evaluation_summary.json",
    "corrected_final_evaluation_qualitative_candidates.json",

    # Manual-audit calibration
    "manual_audit_policy_metrics.csv",
    "manual_audit_agreement.csv",
    "manual_audit_confusion.csv",
    "manual_audit_disagreements.csv",

    # Corrected report assets
    "task4_main_comparison_table.csv",
    "task4_category_label_distribution.csv",
    "task4_judge_confusion_counts.csv",
    "task4_judge_confusion_row_rates.csv",
    "task4_qualitative_examples.csv",
    "task4_report_summary.json",
    "task4_report_summary.md",

    # RQ1 cross-task evidence
    "task4_rq1_common_safety_results.csv",
    "task4_rq1_upstream_alignment_metrics.csv",
    "task4_rq1_evidence_summary.md",
]

TASK5_REPORT_KEYWORDS = (
    "summary",
    "comparison",
    "metrics",
    "diagnostic",
    "sensitivity",
    "agreement",
    "qualitative",
    "in_domain",
    "out_of_domain",
    "transfer",
    "gsm8k",
    "svamp",
    "perturbation",
    "verifier",
    "judge",
    "format",
    "accuracy",
    "result",
)

SKIP_KEYWORDS = (
    "manifest",
    "console",
    "generation_index",
    "preprocessing",
    "checkpoint",
)


def heading(title, character="="):
    return (
        "\n"
        + title
        + "\n"
        + character * max(88, len(title))
        + "\n"
    )


def clean_value(value):
    if value is None:
        return ""

    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)

        if value != 0 and abs(value) < 0.0001:
            return f"{value:.8g}"

        return f"{value:.6f}".rstrip("0").rstrip(".")

    if isinstance(value, (dict, list)):
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
    else:
        text = str(value)

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if len(text) > MAX_TEXT_LENGTH:
        return (
            text[:MAX_TEXT_LENGTH]
            + f"\n...[truncated; original characters={len(text)}]"
        )

    return text


def relative(path):
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def read_csv_rows(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def table_text(rows):
    if not rows:
        return "[No rows]\n"

    columns = []

    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    prepared = []

    for row in rows:
        prepared.append({
            column: clean_value(row.get(column, ""))
            for column in columns
        })

    widths = {}

    for column in columns:
        widths[column] = min(
            42,
            max(
                len(column),
                *[
                    max(
                        len(line)
                        for line in value.splitlines()
                    )
                    if value
                    else 0
                    for value in (
                        row[column]
                        for row in prepared
                    )
                ],
            ),
        )

    def render_cell(value, width):
        first_line = value.splitlines()[0] if value else ""
        if len(first_line) > width:
            return first_line[: width - 3] + "..."
        return first_line

    header = " | ".join(
        column.ljust(widths[column])
        for column in columns
    )

    separator = "-+-".join(
        "-" * widths[column]
        for column in columns
    )

    lines = [header, separator]

    for row in prepared:
        lines.append(
            " | ".join(
                render_cell(
                    row[column],
                    widths[column],
                ).ljust(widths[column])
                for column in columns
            )
        )

    return "\n".join(lines) + "\n"


def flatten_json(value, prefix=""):
    rows = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            if isinstance(child, dict):
                rows.extend(
                    flatten_json(
                        child,
                        child_prefix,
                    )
                )
            elif isinstance(child, list):
                if all(
                    not isinstance(item, (dict, list))
                    for item in child
                ):
                    rows.append({
                        "metric": child_prefix,
                        "value": clean_value(child),
                    })
                else:
                    rows.append({
                        "metric": child_prefix,
                        "value": (
                            f"[structured list: "
                            f"{len(child)} items]"
                        ),
                    })
            else:
                rows.append({
                    "metric": child_prefix,
                    "value": clean_value(child),
                })

    else:
        rows.append({
            "metric": prefix or "value",
            "value": clean_value(value),
        })

    return rows


def format_json(path):
    data = json.loads(
        path.read_text(encoding="utf-8-sig")
    )

    output = []

    if isinstance(data, dict):
        scalar_rows = flatten_json(data)

        if scalar_rows:
            output.append("Scalar and summary fields:\n")
            output.append(table_text(scalar_rows))

        for key, value in data.items():
            if not isinstance(value, list):
                continue

            output.append(
                f"\nStructured list: {key} "
                f"({len(value)} items)\n"
            )

            if (
                value
                and all(isinstance(item, dict) for item in value)
                and len(value) <= MAX_FULL_JSON_LIST
            ):
                output.append(table_text(value))
            else:
                output.append(
                    clean_value(value) + "\n"
                )

    elif isinstance(data, list):
        output.append(f"List items: {len(data)}\n")

        if (
            data
            and all(isinstance(item, dict) for item in data)
            and len(data) <= MAX_FULL_JSON_LIST
        ):
            output.append(table_text(data))
        else:
            output.append(clean_value(data) + "\n")

    else:
        output.append(clean_value(data) + "\n")

    return "".join(output)


def format_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    output = [
        f"Records: {len(rows)}\n",
    ]

    if not rows:
        return "".join(output)

    fields = sorted({
        key
        for row in rows
        if isinstance(row, dict)
        for key in row
    })

    output.append(
        "Fields: " + ", ".join(fields) + "\n"
    )

    categorical_fields = (
        "policy",
        "model",
        "condition",
        "dataset",
        "split",
        "category",
        "perturbation_category",
        "feedback_type",
        "preference",
        "judge_preference",
        "verifier_preference",
        "verifier_correct",
        "judge_correct",
        "correct",
        "format_compliant",
        "terminated_with_eos",
        "truncated",
    )

    for field in categorical_fields:
        values = [
            row.get(field)
            for row in rows
            if (
                isinstance(row, dict)
                and field in row
            )
        ]

        if not values:
            continue

        counts = Counter(
            clean_value(value)
            for value in values
        )

        if len(counts) <= 30:
            output.append(
                f"{field} counts: {dict(counts)}\n"
            )

    lower_name = path.name.lower()

    if (
        len(rows) <= MAX_FULL_JSONL_ROWS
        and (
            "diagnostic" in lower_name
            or "qualitative" in lower_name
            or "comparison" in lower_name
            or "candidate" in lower_name
        )
    ):
        output.append("\nDetailed records:\n")
        output.append(table_text(rows))
    else:
        output.append(
            "\nDetailed responses omitted from the consolidated "
            "file to keep it readable.\n"
        )

    return "".join(output)


def format_file(path):
    suffix = path.suffix.lower()

    if suffix == ".csv":
        rows = read_csv_rows(path)
        return (
            f"Rows: {len(rows)}\n\n"
            + table_text(rows)
        )

    if suffix == ".json":
        return format_json(path)

    if suffix == ".jsonl":
        return format_jsonl(path)

    if suffix in {".txt", ".md"}:
        return path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

    return "[Unsupported file type]\n"


def find_preferred_task4_files(task4_root):
    found = []
    missing = []

    search_roots = [
        task4_root / "report_assets_corrected",
        task4_root / "report_assets",
        task4_root,
    ]

    used_paths = set()

    for name in PREFERRED_TASK4_FILES:
        matches = []

        for root in search_roots:
            if not root.exists():
                continue

            matches.extend(
                root.rglob(name)
            )

        matches = sorted(
            {
                match.resolve()
                for match in matches
                if match.is_file()
            }
        )

        if not matches:
            missing.append(name)
            continue

        chosen = matches[0]

        if chosen not in used_paths:
            found.append(chosen)
            used_paths.add(chosen)

    return found, missing


def find_task5_roots():
    roots = []

    if not RESULTS.exists():
        return roots

    for path in RESULTS.iterdir():
        if path.is_dir() and "task5" in path.name.lower():
            roots.append(path)

    return sorted(roots)


def find_task5_files(task5_roots):
    candidates = []

    for root in task5_roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in {
                ".csv",
                ".json",
                ".jsonl",
                ".txt",
                ".md",
            }:
                continue

            lower_name = path.name.lower()

            if any(
                keyword in lower_name
                for keyword in SKIP_KEYWORDS
            ):
                continue

            if any(
                keyword in lower_name
                for keyword in TASK5_REPORT_KEYWORDS
            ):
                candidates.append(path)

    return sorted(set(candidates))


def write_file_section(handle, path):
    handle.write(
        heading(relative(path), "-")
    )

    handle.write(
        f"Bytes: {path.stat().st_size:,}\n"
    )

    handle.write(
        f"Last modified: "
        f"{datetime.fromtimestamp(path.stat().st_mtime)}\n\n"
    )

    try:
        handle.write(format_file(path))
    except Exception as error:
        handle.write(
            f"[FAILED TO READ: "
            f"{type(error).__name__}: {error}]\n"
        )


task4_root = RESULTS / "task4_safety"
task4_files, missing_task4 = (
    find_preferred_task4_files(task4_root)
)

task5_roots = find_task5_roots()
task5_files = find_task5_files(task5_roots)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as report:
    report.write(
        "ATML PA2 — TASK 4 AND TASK 5 "
        "COMPLETE REPORT DATA\n"
    )
    report.write("=" * 100 + "\n")
    report.write(f"Repository: {REPO}\n")
    report.write(
        f"Generated: {datetime.now().isoformat()}\n"
    )

    report.write(
        heading("REQUIRED-EVIDENCE CHECKLIST")
    )

    report.write(
        """
TASK 4 REQUIRED EVIDENCE
1. For SFT, DPO, PPO and GRPO:
   - safe-answer rate
   - safe over-refusal rate
   - unsafe-compliance rate
   - justified-refusal rate
   - ambiguous-judge rate
   - mean response length
2. Manual-audit calibration and AI-judge agreement.
3. Category-level safety behavior.
4. Qualitative examples:
   - harmful compliance
   - justified refusal
   - exaggerated refusal/over-refusal
   - whether each disagreement is a policy difference,
     judge error, or both.

TASK 5 REQUIRED EVIDENCE
1. In-domain GSM8K comparison for SFT, RLVR and RLAIF:
   - exact accuracy
   - AI pairwise result
   - format compliance
   - mean response length
   - verifier-judge agreement
2. Five-category controlled diagnostics for both feedback mechanisms:
   - better-response rate
   - tie rate
   - wrong-preference rate
   - reasoning sensitivity (Sreason)
   - outcome sensitivity (Soutcome)
3. Out-of-domain SVAMP comparison:
   - exact accuracy
   - AI pairwise result
   - format compliance
   - mean response length
   - verifier-judge agreement
4. Qualitative examples:
   - correct-reasoning/wrong-final
   - wrong-reasoning/correct-final
   - filler/style behavior
   - wrong-final/gold-distractor behavior
5. Feedback-source comparison:
   - coverage
   - noise
   - exploitability
   - inference cost
""".strip()
        + "\n"
    )

    report.write(
        heading("TASK 4 — CORRECTED SAFETY RESULTS")
    )

    if not task4_root.exists():
        report.write(
            "ERROR: results/task4_safety does not exist.\n"
        )
    elif not task4_files:
        report.write(
            "ERROR: No preferred Task 4 result files "
            "were found.\n"
        )
    else:
        report.write(
            f"Task 4 report files selected: "
            f"{len(task4_files)}\n"
        )

        for path in task4_files:
            write_file_section(report, path)

    report.write(
        heading("TASK 4 — EXPECTED FILES NOT FOUND")
    )

    if missing_task4:
        for name in missing_task4:
            report.write(f"- {name}\n")
    else:
        report.write(
            "All preferred Task 4 report files were found.\n"
        )

    report.write(
        heading("TASK 5 — RLVR VERSUS RLAIF RESULTS")
    )

    if not task5_roots:
        report.write(
            "ERROR: No results directory whose name contains "
            "'task5' was found under results/.\n"
        )
        report.write(
            "Task 5 may not yet have been run or may use an "
            "unexpected directory name.\n"
        )
    else:
        report.write("Detected Task 5 directories:\n")

        for root in task5_roots:
            report.write(f"- {relative(root)}\n")

        report.write(
            f"\nTask 5 report-focused files selected: "
            f"{len(task5_files)}\n"
        )

        if not task5_files:
            report.write(
                "WARNING: Task 5 directories exist, but no "
                "report-focused CSV/JSON/TXT files matched.\n"
            )

        for path in task5_files:
            write_file_section(report, path)

    report.write(
        heading("TASK 5 — COMPLETE FILE INVENTORY")
    )

    if task5_roots:
        inventory = []

        for root in task5_roots:
            for path in root.rglob("*"):
                if path.is_file():
                    inventory.append({
                        "file": relative(path),
                        "bytes": path.stat().st_size,
                    })

        report.write(table_text(inventory))
    else:
        report.write("[No Task 5 result files found]\n")

    report.write(
        heading("FINAL COVERAGE WARNINGS")
    )

    if missing_task4:
        report.write(
            "- Some preferred Task 4 files were not found. "
            "Review the missing-file list above.\n"
        )
    else:
        report.write(
            "- Task 4 preferred report outputs were found.\n"
        )

    if not task5_roots:
        report.write(
            "- Task 5 results were not located. Do not write "
            "numerical Task 5 claims until the correct result "
            "directory is identified or generated.\n"
        )
    elif not task5_files:
        report.write(
            "- Task 5 files exist, but the extractor could not "
            "identify report-focused outputs automatically.\n"
        )
    else:
        report.write(
            "- Task 5 report-focused outputs were found. Check "
            "the required-evidence checklist against the printed "
            "tables before finalizing the report.\n"
        )

print()
print("=" * 88)
print("TASK 4 + TASK 5 REPORT DATA EXTRACTED")
print("=" * 88)
print("Output:", OUTPUT.resolve())
print("Bytes:", f"{OUTPUT.stat().st_size:,}")
print("Task 4 selected files:", len(task4_files))
print("Task 5 directories:", len(task5_roots))
print("Task 5 selected files:", len(task5_files))
