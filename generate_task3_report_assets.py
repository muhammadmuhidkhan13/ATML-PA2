from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("results/task3_grpo")
OUTPUT = RESULTS / "report_assets_corrected"

STANDARD_PREFIX = "standard_corrected_20261018"

REQUIRED_FILES = [
    RESULTS / f"{STANDARD_PREFIX}_train_log.jsonl",
    RESULTS / f"{STANDARD_PREFIX}_train_summary.json",
    RESULTS / f"{STANDARD_PREFIX}_eval_metrics.json",
    RESULTS / f"{STANDARD_PREFIX}_eval_qualitative_candidates.json",
    RESULTS / "normalization_grpo_train_log.jsonl",
    RESULTS / "normalization_grpo_train_summary.json",
    RESULTS / "normalization_grpo_eval_metrics.json",
    RESULTS / "normalization_dr_grpo_train_log.jsonl",
    RESULTS / "normalization_dr_grpo_train_summary.json",
    RESULTS / "normalization_dr_grpo_eval_metrics.json",
    RESULTS / "normalization_comparison.csv",
    RESULTS / "normalization_completion_details.csv",
    RESULTS / "normalization_study_summary.json",
    RESULTS / "normalization_length_conditioned.csv",
    RESULTS / "normalization_update_details.csv",
    RESULTS / "normalization_update_length_conditioned.csv",
    RESULTS / "group_size_comparison.csv",
    RESULTS / "group_size_group_details.csv",
    RESULTS / "group_size_prompt_difficulty.csv",
    RESULTS / "group_size_summary.json",
]


FIELD_ALIASES = {
    "attempt": [
        "attempt",
        "rollout_attempt",
        "update",
        "update_index",
    ],
    "optimizer_updates": [
        "optimizer_updates",
        "optimizer_update",
    ],
    "optimizer_step_performed": [
        "optimizer_step_performed",
    ],
    "learned_reward": [
        "mean_reward",
        "reward",
        "learned_reward",
        "mean_learned_reward",
    ],
    "group_reward_std": [
        "group_reward_std",
        "mean_group_reward_std",
    ],
    "sampled_kl": [
        "sampled_kl",
        "mean_sampled_kl",
    ],
    "entropy": [
        "entropy",
        "mean_entropy",
    ],
    "response_length": [
        "response_length",
        "mean_response_length",
    ],
    "clip_fraction": [
        "clip_fraction",
        "mean_clip_fraction",
    ],
    "ratio_mean": [
        "ratio_mean",
        "mean_ratio",
    ],
    "gradient_norm": [
        "gradient_norm",
        "mean_gradient_norm",
    ],
    "valid_policy_tokens": [
        "valid_policy_tokens",
    ],
    "generated_tokens": [
        "generated_tokens",
        "total_generated_tokens",
    ],
    "truncation_rate": [
        "truncation_rate",
        "mean_truncation_rate",
    ],
    "informative_group_rate": [
        "informative_group_rate",
        "mean_informative_group_rate",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSONL in {path}, "
                    f"line {line_number}"
                ) from error

    return rows


def first_available(row: dict, names: list[str]):
    for name in names:
        if name in row:
            return row[name]

    return None


def finite_number(value):
    if value is None or isinstance(value, bool):
        return value

    try:
        number = float(value)
    except (TypeError, ValueError):
        return value

    if math.isfinite(number):
        return number

    return None


def normalize_log(
    path: Path,
    condition: str,
) -> pd.DataFrame:
    raw_rows = load_jsonl(path)
    normalized = []

    for row_number, row in enumerate(
        raw_rows,
        start=1,
    ):
        output_row = {
            "condition": condition,
            "log_row": row_number,
        }

        for target, aliases in FIELD_ALIASES.items():
            output_row[target] = finite_number(
                first_available(
                    row,
                    aliases,
                )
            )

        if output_row["attempt"] is None:
            output_row["attempt"] = row_number

        normalized.append(output_row)

    return pd.DataFrame(normalized)


def save_figure(
    figure,
    stem: str,
):
    png_path = OUTPUT / f"{stem}.png"
    pdf_path = OUTPUT / f"{stem}.pdf"

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)


def numeric_series(frame, column):
    if column not in frame.columns:
        return None

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    )

    if values.notna().sum() == 0:
        return None

    return values


def plot_training_trajectories(
    frame: pd.DataFrame,
    title: str,
    output_stem: str,
):
    panels = [
        ("learned_reward", "Learned reward"),
        ("sampled_kl", "Sampled KL/token"),
        ("entropy", "Entropy/token"),
        ("response_length", "Mean response length"),
        ("group_reward_std", "Group reward SD"),
        ("valid_policy_tokens", "Valid policy tokens"),
    ]

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.8, 7.0),
    )

    colors = {
        "corrected_standard": "#1f77b4",
        "canonical_grpo": "#2ca02c",
        "dr_grpo": "#d62728",
    }

    for axis, (field, label) in zip(
        axes.flat,
        panels,
    ):
        for condition, group in frame.groupby(
            "condition",
            sort=False,
        ):
            x = numeric_series(
                group,
                "attempt",
            )
            y = numeric_series(
                group,
                field,
            )

            if x is None or y is None:
                continue

            valid = x.notna() & y.notna()

            axis.plot(
                x[valid],
                y[valid],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=condition,
                color=colors.get(condition),
            )

            if (
                "optimizer_step_performed"
                in group.columns
            ):
                performed = group[
                    "optimizer_step_performed"
                ]

                skipped = performed == False  # noqa: E712
                skipped = skipped & valid

                if skipped.any():
                    axis.scatter(
                        x[skipped],
                        y[skipped],
                        marker="x",
                        color="black",
                        s=55,
                        linewidths=1.5,
                        zorder=5,
                    )

        axis.set_xlabel("Rollout attempt")
        axis.set_ylabel(label)
        axis.grid(
            alpha=0.25,
            linewidth=0.7,
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()

    if not handles:
        for axis in axes.flat:
            handles, labels = axis.get_legend_handles_labels()

            if handles:
                break

    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=max(1, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )

    figure.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=(0, 0.05, 1, 0.95),
    )

    save_figure(
        figure,
        output_stem,
    )


def normalized_column_name(column: str) -> str:
    return "".join(
        character.lower()
        for character in str(column)
        if character.isalnum()
    )


def locate_column(
    frame: pd.DataFrame,
    candidate_terms: list[str],
):
    normalized = {
        normalized_column_name(column): column
        for column in frame.columns
    }

    for term in candidate_terms:
        normalized_term = normalized_column_name(term)

        if normalized_term in normalized:
            return normalized[normalized_term]

    for normalized_name, original in normalized.items():
        for term in candidate_terms:
            normalized_term = normalized_column_name(term)

            if normalized_term in normalized_name:
                return original

    return None


def plot_comparison_table(
    frame: pd.DataFrame,
    title: str,
    output_stem: str,
    x_candidates: list[str],
    metric_candidates: list[
        tuple[str, list[str]]
    ],
):
    x_column = locate_column(
        frame,
        x_candidates,
    )

    if x_column is None:
        return []

    selected = []

    for label, aliases in metric_candidates:
        column = locate_column(
            frame,
            aliases,
        )

        if column is None or column == x_column:
            continue

        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if values.notna().sum() == 0:
            continue

        selected.append(
            (
                label,
                column,
                values,
            )
        )

    if not selected:
        return []

    selected = selected[:6]

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(12.8, 7.0),
    )

    axes = axes.flat
    x_labels = frame[x_column].astype(str).tolist()
    positions = np.arange(len(frame))

    for axis, (
        label,
        source_column,
        values,
    ) in zip(axes, selected):
        axis.bar(
            positions,
            values,
            color="#4472C4",
            edgecolor="black",
            linewidth=0.4,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            x_labels,
            rotation=20,
            ha="right",
        )
        axis.set_ylabel(label)
        axis.grid(
            axis="y",
            alpha=0.25,
        )
        axis.set_title(label)

    for axis in list(axes)[len(selected):]:
        axis.axis("off")

    figure.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(
        rect=(0, 0, 1, 0.95),
    )

    save_figure(
        figure,
        output_stem,
    )

    return [
        {
            "panel": label,
            "source_column": source_column,
            "x_column": x_column,
        }
        for label, source_column, _ in selected
    ]


for required in REQUIRED_FILES:
    if not required.exists():
        raise FileNotFoundError(
            f"Required Task 3 file missing: {required}"
        )


source_hashes_before = {
    str(path): sha256(path)
    for path in REQUIRED_FILES
}

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

OUTPUT.mkdir(
    parents=True,
    exist_ok=False,
)


# ---------------------------------------------------------------------
# Corrected standard run
# ---------------------------------------------------------------------

standard_train_summary = load_json(
    RESULTS
    / f"{STANDARD_PREFIX}_train_summary.json"
)

standard_eval_metrics = load_json(
    RESULTS
    / f"{STANDARD_PREFIX}_eval_metrics.json"
)

standard_summary = {
    "run": STANDARD_PREFIX,
    "rollout_attempts": standard_train_summary.get(
        "rollout_attempts"
    ),
    "optimizer_updates": standard_train_summary.get(
        "optimizer_updates"
    ),
    "zero_valid_token_attempts": standard_train_summary.get(
        "zero_valid_token_attempts"
    ),
    "total_generated_tokens": standard_train_summary.get(
        "total_generated_tokens"
    ),
    "total_valid_policy_tokens": standard_train_summary.get(
        "total_valid_policy_tokens"
    ),
    "mean_training_reward": standard_train_summary.get(
        "mean_reward"
    ),
    "mean_training_sampled_kl": standard_train_summary.get(
        "mean_sampled_kl"
    ),
    "mean_training_entropy": standard_train_summary.get(
        "mean_entropy"
    ),
    "mean_training_response_length": (
        standard_train_summary.get(
            "mean_response_length"
        )
    ),
    "max_gradient_norm": standard_train_summary.get(
        "max_gradient_norm"
    ),
    "held_out_prompts": standard_eval_metrics.get(
        "num_evaluation_prompts"
    ),
    "held_out_reward": standard_eval_metrics.get(
        "reward_model_score_mean"
    ),
    "held_out_reward_sd": standard_eval_metrics.get(
        "reward_model_score_std"
    ),
    "held_out_sampled_kl": standard_eval_metrics.get(
        "sampled_kl_token_mean"
    ),
    "held_out_entropy": standard_eval_metrics.get(
        "entropy_token_mean"
    ),
    "held_out_mean_length": standard_eval_metrics.get(
        "response_length_mean"
    ),
    "held_out_length_sd": standard_eval_metrics.get(
        "response_length_std"
    ),
    "held_out_eos_rate": standard_eval_metrics.get(
        "terminated_with_eos_rate"
    ),
    "held_out_truncation_rate": standard_eval_metrics.get(
        "truncation_rate"
    ),
}

standard_summary_frame = pd.DataFrame(
    [standard_summary]
)

standard_summary_frame.to_csv(
    OUTPUT / "task3_corrected_standard_summary.csv",
    index=False,
)

standard_trajectory = normalize_log(
    RESULTS
    / f"{STANDARD_PREFIX}_train_log.jsonl",
    "corrected_standard",
)

standard_trajectory.to_csv(
    OUTPUT
    / "task3_corrected_standard_training_trajectory.csv",
    index=False,
)

plot_training_trajectories(
    standard_trajectory,
    "Corrected standard GRPO training dynamics",
    "task3_corrected_standard_training_dynamics",
)


# ---------------------------------------------------------------------
# Canonical GRPO versus Dr-GRPO
# ---------------------------------------------------------------------

canonical_log = normalize_log(
    RESULTS / "normalization_grpo_train_log.jsonl",
    "canonical_grpo",
)

dr_log = normalize_log(
    RESULTS / "normalization_dr_grpo_train_log.jsonl",
    "dr_grpo",
)

normalization_trajectories = pd.concat(
    [
        canonical_log,
        dr_log,
    ],
    ignore_index=True,
)

normalization_trajectories.to_csv(
    OUTPUT
    / "task3_normalization_training_trajectories.csv",
    index=False,
)

plot_training_trajectories(
    normalization_trajectories,
    "Canonical GRPO versus Dr-GRPO training dynamics",
    "task3_normalization_training_dynamics",
)

normalization_comparison = pd.read_csv(
    RESULTS / "normalization_comparison.csv"
)

normalization_comparison.to_csv(
    OUTPUT / "task3_normalization_comparison.csv",
    index=False,
)

normalization_source_map = plot_comparison_table(
    normalization_comparison,
    "Canonical GRPO versus Dr-GRPO held-out comparison",
    "task3_normalization_heldout_comparison",
    [
        "condition",
        "loss_type",
        "normalization",
        "name",
    ],
    [
        (
            "Held-out reward",
            [
                "held_out_reward",
                "reward_model_score_mean",
                "reward",
            ],
        ),
        (
            "Sampled KL/token",
            [
                "held_out_kl",
                "sampled_kl_token_mean",
                "sampled_kl",
            ],
        ),
        (
            "Entropy/token",
            [
                "held_out_entropy",
                "entropy_token_mean",
                "entropy",
            ],
        ),
        (
            "Mean response length",
            [
                "held_out_mean_length",
                "response_length_mean",
                "mean_length",
            ],
        ),
        (
            "Maximum gradient norm",
            [
                "max_gradient_norm",
                "gradient_norm_max",
            ],
        ),
        (
            "Truncation rate",
            [
                "truncation_rate",
            ],
        ),
    ],
)


# Preserve detailed normalization tables.

NORMALIZATION_TABLES = [
    "normalization_completion_details.csv",
    "normalization_length_conditioned.csv",
    "normalization_update_details.csv",
    "normalization_update_length_conditioned.csv",
]

for filename in NORMALIZATION_TABLES:
    source = RESULTS / filename
    destination = OUTPUT / f"task3_{filename}"

    shutil.copy2(
        source,
        destination,
    )


# ---------------------------------------------------------------------
# Group-size experiment
# ---------------------------------------------------------------------

group_size_comparison = pd.read_csv(
    RESULTS / "group_size_comparison.csv"
)

group_size_comparison.to_csv(
    OUTPUT / "task3_group_size_comparison.csv",
    index=False,
)

group_size_source_map = plot_comparison_table(
    group_size_comparison,
    "GRPO group-size comparison",
    "task3_group_size_comparison",
    [
        "group_size",
        "num_generations",
        "k",
    ],
    [
        (
            "Informative-group rate",
            [
                "informative_group_rate",
                "mean_informative_group_rate",
            ],
        ),
        (
            "Uninformative-group fraction",
            [
                "uninformative_group_fraction",
                "mean_uninformative_group_fraction",
            ],
        ),
        (
            "Group reward SD",
            [
                "group_reward_std",
                "mean_group_reward_std",
            ],
        ),
        (
            "Mean reward",
            [
                "mean_reward",
                "reward",
            ],
        ),
        (
            "At least one correct",
            [
                "any_correct_rate",
                "at_least_one_correct_rate",
                "positive_group_rate",
            ],
        ),
        (
            "All-correct rate",
            [
                "all_correct_rate",
            ],
        ),
    ],
)

for filename in [
    "group_size_group_details.csv",
    "group_size_prompt_difficulty.csv",
]:
    shutil.copy2(
        RESULTS / filename,
        OUTPUT / f"task3_{filename}",
    )


# ---------------------------------------------------------------------
# Qualitative and provenance files
# ---------------------------------------------------------------------

shutil.copy2(
    RESULTS
    / f"{STANDARD_PREFIX}_eval_qualitative_candidates.json",
    OUTPUT
    / "task3_corrected_standard_qualitative_candidates.json",
)

shutil.copy2(
    RESULTS / "normalization_qualitative_candidates.json",
    OUTPUT
    / "task3_normalization_qualitative_candidates.json",
)

shutil.copy2(
    RESULTS / "group_size_summary.json",
    OUTPUT / "task3_group_size_summary.json",
)

shutil.copy2(
    RESULTS / "normalization_study_summary.json",
    OUTPUT / "task3_normalization_study_summary.json",
)


# ---------------------------------------------------------------------
# Source map and consolidated numerical summary
# ---------------------------------------------------------------------

source_map_rows = [
    {
        "report_component": (
            "Corrected standard training dynamics"
        ),
        "source_file": (
            f"{STANDARD_PREFIX}_train_log.jsonl"
        ),
        "fields": (
            "attempt, optimizer_step_performed, "
            "mean_reward/reward, sampled_kl, entropy, "
            "response_length, group_reward_std, "
            "valid_policy_tokens"
        ),
    },
    {
        "report_component": (
            "Corrected standard held-out performance"
        ),
        "source_file": (
            f"{STANDARD_PREFIX}_eval_metrics.json"
        ),
        "fields": (
            "reward_model_score_mean, "
            "sampled_kl_token_mean, entropy_token_mean, "
            "response_length_mean, "
            "terminated_with_eos_rate, truncation_rate"
        ),
    },
    {
        "report_component": (
            "Normalization training trajectories"
        ),
        "source_file": (
            "normalization_grpo_train_log.jsonl; "
            "normalization_dr_grpo_train_log.jsonl"
        ),
        "fields": (
            "reward, sampled_kl, entropy, "
            "response_length, group_reward_std, "
            "valid_policy_tokens"
        ),
    },
    {
        "report_component": (
            "Normalization held-out comparison"
        ),
        "source_file": "normalization_comparison.csv",
        "fields": json.dumps(
            normalization_source_map,
            sort_keys=True,
        ),
    },
    {
        "report_component": "Group-size comparison",
        "source_file": "group_size_comparison.csv",
        "fields": json.dumps(
            group_size_source_map,
            sort_keys=True,
        ),
    },
]

source_map = pd.DataFrame(source_map_rows)

source_map.to_csv(
    OUTPUT / "task3_report_source_map.csv",
    index=False,
)

report_summary = {
    "corrected_standard": standard_summary,
    "normalization_comparison": (
        normalization_comparison.to_dict(
            orient="records"
        )
    ),
    "group_size_comparison": (
        group_size_comparison.to_dict(
            orient="records"
        )
    ),
    "normalization_plot_source_fields": (
        normalization_source_map
    ),
    "group_size_plot_source_fields": (
        group_size_source_map
    ),
}

with (
    OUTPUT / "task3_report_numerical_summary.json"
).open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        report_summary,
        handle,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# Human-readable text summary.

summary_lines = [
    "TASK 3 CORRECTED REPORT DATA",
    "=" * 80,
    "",
    "CORRECTED STANDARD GRPO",
    "-" * 80,
]

for key, value in standard_summary.items():
    summary_lines.append(
        f"{key}: {value}"
    )

summary_lines.extend(
    [
        "",
        "NORMALIZATION COMPARISON",
        "-" * 80,
        normalization_comparison.to_string(
            index=False
        ),
        "",
        "GROUP-SIZE COMPARISON",
        "-" * 80,
        group_size_comparison.to_string(
            index=False
        ),
        "",
    ]
)

(
    OUTPUT / "task3_report_numerical_summary.txt"
).write_text(
    "\n".join(summary_lines),
    encoding="utf-8",
)


# ---------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------

source_hashes_after = {
    str(path): sha256(path)
    for path in REQUIRED_FILES
}

if source_hashes_before != source_hashes_after:
    raise RuntimeError(
        "At least one source result changed during "
        "report generation."
    )

integrity_rows = [
    {
        "source_file": path,
        "sha256_before": source_hashes_before[path],
        "sha256_after": source_hashes_after[path],
        "unchanged": (
            source_hashes_before[path]
            == source_hashes_after[path]
        ),
    }
    for path in sorted(source_hashes_before)
]

pd.DataFrame(integrity_rows).to_csv(
    OUTPUT / "task3_report_source_integrity.csv",
    index=False,
)


print()
print("TASK 3 CORRECTED REPORT PACKAGE")
print("=" * 80)
print(
    standard_summary_frame.to_string(
        index=False
    )
)

print()
print("NORMALIZATION COMPARISON")
print("=" * 80)
print(
    normalization_comparison.to_string(
        index=False
    )
)

print()
print("GROUP-SIZE COMPARISON")
print("=" * 80)
print(
    group_size_comparison.to_string(
        index=False
    )
)

print()
print("GENERATED REPORT FILES")
print("=" * 80)

for path in sorted(OUTPUT.iterdir()):
    if path.is_file():
        print(
            f"{path.name:<60} "
            f"{path.stat().st_size:>12,} bytes"
        )

print()
print("Raw source files unchanged: PASS")
print(f"Report directory: {OUTPUT.resolve()}")
print()
print("TASK 3 CORRECTED REPORT ASSETS COMPLETED")
