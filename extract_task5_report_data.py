from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path.cwd()
OUTPUT_DIR = (
    ROOT
    / "results"
    / "task5_report_assets"
)

RELEVANT_TERMS = (
    "task5",
    "rlvr",
    "rlaif",
    "gsm8k",
    "math_transfer",
)

TEXT_FIELDS = (
    "prompt",
    "question",
    "input",
    "response",
    "completion",
    "generated_response",
    "generated_text",
    "prediction",
    "reference_answer",
    "ground_truth",
    "answer",
    "solution",
)

IDENTITY_FIELDS = (
    "policy",
    "method",
    "condition",
    "run_name",
    "name",
    "dataset",
    "split",
    "prompt_id",
    "evaluation_index",
    "source_id",
)

QUALITY_FIELDS = (
    "correct",
    "is_correct",
    "exact_match",
    "accuracy",
    "reward",
    "learned_reward",
    "reward_model_score",
    "sampled_kl",
    "kl",
    "entropy",
    "response_length",
    "completion_length",
    "terminated_with_eos",
    "truncated",
    "format_correct",
    "parse_success",
)

FIGURE_EXTENSIONS = {
    ".png",
    ".pdf",
    ".svg",
    ".jpg",
    ".jpeg",
}

STRUCTURED_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
}


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_output(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "Git unavailable"

    output = completed.stdout.strip()

    if not output:
        output = completed.stderr.strip()

    return output


def is_relevant(path: Path) -> bool:
    text = relative(path).lower().replace("\\", "/")

    return any(
        term in text
        for term in RELEVANT_TERMS
    )


def discover_files() -> list[Path]:
    discovered: set[Path] = set()

    search_roots = [
        ROOT / "results",
        ROOT / "configs",
        ROOT / "task5",
        ROOT / "task5_reasoning",
        ROOT / "task5_rlvr",
        ROOT / "task5_rlaif",
        ROOT / "scripts",
        ROOT / "diagnostics",
        ROOT / "outputs",
    ]

    for search_root in search_roots:
        if not search_root.exists():
            continue

        if search_root.is_file():
            if is_relevant(search_root):
                discovered.add(
                    search_root.resolve()
                )
            continue

        for path in search_root.rglob("*"):
            if (
                path.is_file()
                and is_relevant(path)
            ):
                discovered.add(path.resolve())

    for path in ROOT.glob("task5*.py"):
        if path.is_file():
            discovered.add(path.resolve())

    return sorted(
        discovered,
        key=lambda path: relative(path).lower(),
    )


def flatten_json(
    value: Any,
    prefix: str = "",
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )
            flattened.update(
                flatten_json(
                    child,
                    child_prefix,
                )
            )

    elif isinstance(value, list):
        flattened[
            f"{prefix}.__length__"
        ] = len(value)

        for index, child in enumerate(value):
            if index >= 20:
                flattened[
                    f"{prefix}.__remaining__"
                ] = len(value) - index
                break

            child_prefix = (
                f"{prefix}[{index}]"
            )

            flattened.update(
                flatten_json(
                    child,
                    child_prefix,
                )
            )

    else:
        flattened[prefix] = value

    return flattened


def scalar_type(value: Any) -> str:
    if value is None:
        return "null"

    if isinstance(value, bool):
        return "boolean"

    if isinstance(value, int):
        return "integer"

    if isinstance(value, float):
        return "float"

    return "string"


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def read_jsonl(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{relative(path)} line "
                    f"{line_number}: invalid JSON"
                ) from error

            yield line_number, value


def first_present(
    record: dict[str, Any],
    fields: tuple[str, ...],
) -> Any:
    for field in fields:
        if field in record:
            return record[field]

    return None


def text_value(
    record: dict[str, Any],
    fields: tuple[str, ...],
    maximum: int = 4000,
) -> str:
    value = first_present(
        record,
        fields,
    )

    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        rendered = json.dumps(
            value,
            ensure_ascii=False,
        )
    else:
        rendered = str(value)

    if len(rendered) > maximum:
        rendered = (
            rendered[:maximum]
            + "\n[TRUNCATED IN INDEX]"
        )

    return rendered


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    preferred_fields: list[str] | None = None,
):
    if not rows:
        path.write_text(
            "No matching records found.\n",
            encoding="utf-8",
        )
        return

    all_fields: list[str] = []

    if preferred_fields:
        all_fields.extend(preferred_fields)

    for row in rows:
        for key in row:
            if key not in all_fields:
                all_fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=all_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_csv(
    path: Path,
) -> dict[str, Any]:
    delimiter = (
        "\t"
        if path.suffix.lower() == ".tsv"
        else ","
    )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter=delimiter,
        )
        rows = list(reader)

    return {
        "source_file": relative(path),
        "rows": len(rows),
        "columns": len(reader.fieldnames or []),
        "column_names": " | ".join(
            reader.fieldnames or []
        ),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = discover_files()

    if not files:
        raise RuntimeError(
            "No Task 5/RLVR/RLAIF/GSM8K/"
            "math-transfer files were discovered."
        )

    inventory_rows = []
    json_metric_rows = []
    jsonl_summary_rows = []
    jsonl_numeric_rows = []
    csv_summary_rows = []
    qualitative_rows = []
    figure_rows = []
    configuration_sections = []
    warnings = []

    for path in files:
        suffix = path.suffix.lower()

        inventory_rows.append(
            {
                "path": relative(path),
                "extension": suffix,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

        if suffix in FIGURE_EXTENSIONS:
            figure_rows.append(
                {
                    "figure": path.name,
                    "path": relative(path),
                    "extension": suffix,
                    "bytes": path.stat().st_size,
                }
            )

        if suffix in {".yaml", ".yml"}:
            try:
                contents = path.read_text(
                    encoding="utf-8-sig"
                )
                configuration_sections.append(
                    "\n".join(
                        [
                            "=" * 100,
                            relative(path),
                            "=" * 100,
                            contents.rstrip(),
                        ]
                    )
                )
            except (OSError, UnicodeError) as error:
                warnings.append(
                    f"Could not read config "
                    f"{relative(path)}: {error}"
                )

        if suffix == ".json":
            try:
                data = load_json(path)
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
            ) as error:
                warnings.append(
                    f"Could not parse JSON "
                    f"{relative(path)}: {error}"
                )
                continue

            flattened = flatten_json(data)

            for key, value in flattened.items():
                if isinstance(
                    value,
                    (dict, list),
                ):
                    continue

                json_metric_rows.append(
                    {
                        "source_file": relative(path),
                        "metric_path": key,
                        "value": value,
                        "value_type": scalar_type(value),
                    }
                )

        elif suffix == ".jsonl":
            schemas: Counter[str] = Counter()
            numeric_values: dict[
                str,
                list[float],
            ] = defaultdict(list)
            boolean_counts: dict[
                str,
                Counter[str],
            ] = defaultdict(Counter)

            record_count = 0
            parse_failures = 0

            try:
                iterator = read_jsonl(path)

                for line_number, record in iterator:
                    record_count += 1

                    if not isinstance(record, dict):
                        schemas[
                            type(record).__name__
                        ] += 1
                        continue

                    schemas[
                        " | ".join(
                            sorted(record.keys())
                        )
                    ] += 1

                    for key, value in record.items():
                        if isinstance(value, bool):
                            boolean_counts[key][
                                str(value)
                            ] += 1
                        elif (
                            isinstance(
                                value,
                                (int, float),
                            )
                            and not isinstance(
                                value,
                                bool,
                            )
                            and math.isfinite(
                                float(value)
                            )
                        ):
                            numeric_values[key].append(
                                float(value)
                            )

                    prompt = text_value(
                        record,
                        (
                            "prompt",
                            "question",
                            "input",
                            "prompt_messages",
                            "messages",
                        ),
                    )

                    response = text_value(
                        record,
                        (
                            "response",
                            "completion",
                            "generated_response",
                            "generated_text",
                            "prediction",
                        ),
                    )

                    if prompt or response:
                        row = {
                            "source_file": relative(path),
                            "record_number": record_count,
                            "line_number": line_number,
                        }

                        for field in IDENTITY_FIELDS:
                            row[field] = record.get(
                                field,
                                "",
                            )

                        row["prompt"] = prompt
                        row["response"] = response

                        row["reference_or_answer"] = (
                            text_value(
                                record,
                                (
                                    "reference_answer",
                                    "ground_truth",
                                    "answer",
                                    "solution",
                                    "target",
                                ),
                            )
                        )

                        for field in QUALITY_FIELDS:
                            row[field] = record.get(
                                field,
                                "",
                            )

                        qualitative_rows.append(row)

            except RuntimeError as error:
                parse_failures += 1
                warnings.append(str(error))

            jsonl_summary_rows.append(
                {
                    "source_file": relative(path),
                    "records": record_count,
                    "distinct_schemas": len(schemas),
                    "most_common_schema": (
                        schemas.most_common(1)[0][0]
                        if schemas
                        else ""
                    ),
                    "parse_failures": parse_failures,
                }
            )

            for field, values in sorted(
                numeric_values.items()
            ):
                jsonl_numeric_rows.append(
                    {
                        "source_file": relative(path),
                        "field": field,
                        "count": len(values),
                        "mean": statistics.fmean(values),
                        "std_population": (
                            statistics.pstdev(values)
                            if len(values) > 1
                            else 0.0
                        ),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                )

            for field, counts in sorted(
                boolean_counts.items()
            ):
                total = sum(counts.values())

                jsonl_numeric_rows.append(
                    {
                        "source_file": relative(path),
                        "field": field,
                        "count": total,
                        "mean": (
                            counts.get("True", 0)
                            / total
                            if total
                            else None
                        ),
                        "std_population": "",
                        "minimum": 0,
                        "maximum": 1,
                    }
                )

        elif suffix in {".csv", ".tsv"}:
            try:
                csv_summary_rows.append(
                    summarize_csv(path)
                )
            except (
                OSError,
                UnicodeError,
                csv.Error,
            ) as error:
                warnings.append(
                    f"Could not summarize table "
                    f"{relative(path)}: {error}"
                )

    inventory_path = (
        OUTPUT_DIR
        / "task5_artifact_inventory.csv"
    )
    metrics_path = (
        OUTPUT_DIR
        / "task5_all_json_metrics.csv"
    )
    jsonl_summary_path = (
        OUTPUT_DIR
        / "task5_jsonl_file_summary.csv"
    )
    numeric_path = (
        OUTPUT_DIR
        / "task5_jsonl_numeric_fields.csv"
    )
    csv_summary_path = (
        OUTPUT_DIR
        / "task5_csv_file_summary.csv"
    )
    qualitative_path = (
        OUTPUT_DIR
        / "task5_qualitative_record_index.csv"
    )
    figure_path = (
        OUTPUT_DIR
        / "task5_figure_inventory.csv"
    )
    config_path = (
        OUTPUT_DIR
        / "task5_configuration_sources.txt"
    )
    report_path = (
        OUTPUT_DIR
        / "task5_complete_report_results.txt"
    )
    markdown_path = (
        OUTPUT_DIR
        / "task5_report_data_summary.md"
    )

    write_csv(
        inventory_path,
        inventory_rows,
    )
    write_csv(
        metrics_path,
        json_metric_rows,
    )
    write_csv(
        jsonl_summary_path,
        jsonl_summary_rows,
    )
    write_csv(
        numeric_path,
        jsonl_numeric_rows,
    )
    write_csv(
        csv_summary_path,
        csv_summary_rows,
    )
    write_csv(
        qualitative_path,
        qualitative_rows,
    )
    write_csv(
        figure_path,
        figure_rows,
    )

    config_path.write_text(
        "\n\n".join(configuration_sections)
        if configuration_sections
        else "No relevant YAML configurations found.\n",
        encoding="utf-8",
    )

    git_commit = git_output(
        "rev-parse",
        "HEAD",
    )
    git_status = git_output(
        "status",
        "--short",
    )

    report_lines = [
        "TASK 5 COMPLETE REPORT DATA",
        "=" * 100,
        "",
        "REPRODUCIBILITY",
        "-" * 100,
        f"Repository: {ROOT}",
        f"Git commit: {git_commit}",
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        "",
        "Git status:",
        git_status or "Clean",
        "",
        "DISCOVERY SUMMARY",
        "-" * 100,
        f"Relevant files: {len(files)}",
        f"JSON scalar metrics: {len(json_metric_rows)}",
        f"JSONL files: {len(jsonl_summary_rows)}",
        f"CSV/TSV files: {len(csv_summary_rows)}",
        f"Qualitative indexed records: {len(qualitative_rows)}",
        f"Figures: {len(figure_rows)}",
        "",
        "JSON METRICS",
        "-" * 100,
    ]

    for row in json_metric_rows:
        report_lines.append(
            f"{row['source_file']} | "
            f"{row['metric_path']} = "
            f"{row['value']}"
        )

    report_lines.extend(
        [
            "",
            "JSONL FILE SUMMARIES",
            "-" * 100,
        ]
    )

    for row in jsonl_summary_rows:
        report_lines.append(
            f"{row['source_file']} | "
            f"records={row['records']} | "
            f"schemas={row['distinct_schemas']} | "
            f"parse_failures={row['parse_failures']}"
        )

    report_lines.extend(
        [
            "",
            "JSONL NUMERIC FIELD SUMMARIES",
            "-" * 100,
        ]
    )

    for row in jsonl_numeric_rows:
        report_lines.append(
            f"{row['source_file']} | "
            f"{row['field']} | "
            f"n={row['count']} | "
            f"mean={row['mean']} | "
            f"sd={row['std_population']} | "
            f"min={row['minimum']} | "
            f"max={row['maximum']}"
        )

    report_lines.extend(
        [
            "",
            "AVAILABLE FIGURES",
            "-" * 100,
        ]
    )

    for row in figure_rows:
        report_lines.append(
            f"{row['figure']} | "
            f"{row['path']} | "
            f"{row['bytes']} bytes"
        )

    report_lines.extend(
        [
            "",
            "WARNINGS",
            "-" * 100,
        ]
    )

    if warnings:
        report_lines.extend(warnings)
    else:
        report_lines.append("None")

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    markdown_lines = [
        "# Task 5 report-data package",
        "",
        "## Reproducibility",
        "",
        f"- Git commit: `{git_commit}`",
        f"- Relevant artifacts found: {len(files)}",
        f"- JSONL files summarized: {len(jsonl_summary_rows)}",
        f"- Qualitative records indexed: {len(qualitative_rows)}",
        f"- Figures found: {len(figure_rows)}",
        "",
        "## Generated files",
        "",
        f"- `{relative(report_path)}`",
        f"- `{relative(metrics_path)}`",
        f"- `{relative(numeric_path)}`",
        f"- `{relative(qualitative_path)}`",
        f"- `{relative(figure_path)}`",
        f"- `{relative(config_path)}`",
        f"- `{relative(inventory_path)}`",
        "",
        "## Usage",
        "",
        "Use JSON summary metrics and CSV comparison tables "
        "for quantitative reporting. Use the qualitative "
        "record index to inspect actual prompts, responses, "
        "correctness indicators, rewards, and lengths. "
        "Do not compare metrics across different datasets "
        "without explicitly identifying the dataset and split.",
        "",
    ]

    markdown_path.write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )

    print("\nTASK 5 REPORT DATA EXTRACTION")
    print("=" * 100)
    print("Repository:", ROOT)
    print("Git commit:", git_commit)
    print("Relevant files:", len(files))
    print("JSON metrics:", len(json_metric_rows))
    print("JSONL files:", len(jsonl_summary_rows))
    print("CSV/TSV files:", len(csv_summary_rows))
    print(
        "Qualitative indexed records:",
        len(qualitative_rows),
    )
    print("Figures:", len(figure_rows))

    print("\nGENERATED REPORT FILES")
    print("=" * 100)

    generated = [
        report_path,
        markdown_path,
        metrics_path,
        numeric_path,
        qualitative_path,
        jsonl_summary_path,
        csv_summary_path,
        figure_path,
        config_path,
        inventory_path,
    ]

    for path in generated:
        print(
            f"{relative(path):70s} "
            f"{path.stat().st_size:>12,} bytes"
        )

    if warnings:
        print("\nWARNINGS")
        print("=" * 100)

        for warning in warnings:
            print("-", warning)

    print(
        "\nTASK 5 REPORT DATA EXTRACTION COMPLETED"
    )


if __name__ == "__main__":
    main()
