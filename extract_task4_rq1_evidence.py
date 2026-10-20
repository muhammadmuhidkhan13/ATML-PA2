import json
from pathlib import Path

import pandas as pd


ROOT = Path(".")
OUTPUT = (
    ROOT
    / "results"
    / "task4_safety"
    / "report_assets_corrected"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


def find_one(label, candidates):
    matches = [
        path
        for path in candidates
        if path.exists()
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{label}: expected exactly one file; "
            f"found {len(matches)}.\n"
            + "\n".join(
                str(path)
                for path in matches
            )
        )

    return matches[0]


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def value(record, *names):
    for name in names:
        if name in record:
            return record[name]

    return None


task1_eval_path = find_one(
    "Task 1 DPO evaluation metrics",
    [
        Path("results/task1_dpo/standard_metrics.json"),
    ],
)

task1_train_path = find_one(
    "Task 1 DPO training summary",
    [
        ROOT
        / "results/task1_dpo/standard_train_summary.json",
    ],
)

task2_eval_path = find_one(
    "Task 2 PPO evaluation metrics",
    [
        Path("results/task2_ppo/standard_metrics.json"),
    ],
)

task2_train_path = find_one(
    "Task 2 PPO training summary",
    [
        ROOT
        / "results/task2_ppo/standard_train_summary.json",
    ],
)

task3_eval_path = find_one(
    "Corrected Task 3 GRPO evaluation metrics",
    [
        ROOT
        / (
            "results/task3_grpo/"
            "standard_corrected_20261018_eval_metrics.json"
        ),
    ],
)

task3_train_path = find_one(
    "Corrected Task 3 GRPO training summary",
    [
        ROOT
        / (
            "results/task3_grpo/"
            "standard_corrected_20261018_train_summary.json"
        ),
    ],
)

task4_full_path = find_one(
    "Corrected Task 4 policy comparison",
    [
        ROOT
        / (
            "results/task4_safety/"
            "corrected_final_evaluation_policy_comparison.csv"
        ),
    ],
)

task4_manual_path = find_one(
    "Corrected Task 4 manual metrics",
    [
        ROOT
        / (
            "results/task4_safety/"
            "manual_audit_policy_metrics.csv"
        ),
    ],
)


dpo_eval = load_json(task1_eval_path)
dpo_train = load_json(task1_train_path)

ppo_eval = load_json(task2_eval_path)
ppo_train = load_json(task2_train_path)

grpo_eval = load_json(task3_eval_path)
grpo_train = load_json(task3_train_path)


upstream_rows = [
    {
        "method": "DPO",
        "task": "Task 1",
        "training_objective": "Pairwise preference optimization",
        "starting_checkpoint": value(
            dpo_train,
            "policy_start_checkpoint",
            "base_model",
        ),
        "training_updates": value(
            dpo_train,
            "optimizer_updates",
        ),
        "rollout_attempts": None,
        "beta_or_kl_coefficient": value(
            dpo_train,
            "beta",
        ),
        "clip_epsilon": None,
        "held_out_dataset": value(
            dpo_eval,
            "dataset",
        ),
        "held_out_examples": value(
            dpo_eval,
            "num_pairs",
            "num_evaluation_rows",
        ),
        "preference_accuracy": value(
            dpo_eval,
            "preference_accuracy",
        ),
        "relative_margin_mean": value(
            dpo_eval,
            "relative_margin_mean",
        ),
        "reward_model_score_mean": value(
            dpo_eval,
            "reward_model_score_mean",
        ),
        "sampled_kl_token_mean": value(
            dpo_eval,
            "sampled_kl_token_mean",
        ),
        "entropy_token_mean": value(
            dpo_eval,
            "entropy_token_mean",
        ),
        "response_length_mean": value(
            dpo_eval,
            "response_length_mean",
        ),
        "terminated_with_eos_rate": value(
            dpo_eval,
            "terminated_with_eos_rate",
        ),
        "truncation_rate": value(
            dpo_eval,
            "truncation_rate",
        ),
        "evaluation_source": str(task1_eval_path),
    },
    {
        "method": "PPO",
        "task": "Task 2",
        "training_objective": "Clipped policy optimization",
        "starting_checkpoint": value(
            ppo_train,
            "policy_start_checkpoint",
        ),
        "training_updates": value(
            ppo_train,
            "updates",
            "optimizer_updates",
        ),
        "rollout_attempts": value(
            ppo_train,
            "rollout_attempts",
        ),
        "beta_or_kl_coefficient": value(
            ppo_train,
            "kl_beta",
        ),
        "clip_epsilon": value(
            ppo_train,
            "clip_epsilon",
        ),
        "held_out_dataset": value(
            ppo_eval,
            "dataset",
        ),
        "held_out_examples": value(
            ppo_eval,
            "num_evaluation_prompts",
        ),
        "preference_accuracy": None,
        "relative_margin_mean": None,
        "reward_model_score_mean": value(
            ppo_eval,
            "reward_model_score_mean",
        ),
        "sampled_kl_token_mean": value(
            ppo_eval,
            "sampled_kl_token_mean",
        ),
        "entropy_token_mean": value(
            ppo_eval,
            "entropy_token_mean",
        ),
        "response_length_mean": value(
            ppo_eval,
            "response_length_mean",
        ),
        "terminated_with_eos_rate": value(
            ppo_eval,
            "terminated_with_eos_rate",
        ),
        "truncation_rate": value(
            ppo_eval,
            "truncation_rate",
        ),
        "evaluation_source": str(task2_eval_path),
    },
    {
        "method": "GRPO",
        "task": "Task 3 corrected",
        "training_objective": "Group-relative policy optimization",
        "starting_checkpoint": value(
            grpo_train,
            "policy_start_checkpoint",
        ),
        "training_updates": value(
            grpo_train,
            "optimizer_updates",
            "updates",
        ),
        "rollout_attempts": value(
            grpo_train,
            "rollout_attempts",
        ),
        "beta_or_kl_coefficient": value(
            grpo_train,
            "kl_beta",
        ),
        "clip_epsilon": value(
            grpo_train,
            "clip_epsilon",
        ),
        "held_out_dataset": value(
            grpo_eval,
            "dataset",
        ),
        "held_out_examples": value(
            grpo_eval,
            "num_evaluation_prompts",
        ),
        "preference_accuracy": None,
        "relative_margin_mean": None,
        "reward_model_score_mean": value(
            grpo_eval,
            "reward_model_score_mean",
        ),
        "sampled_kl_token_mean": value(
            grpo_eval,
            "sampled_kl_token_mean",
        ),
        "entropy_token_mean": value(
            grpo_eval,
            "entropy_token_mean",
        ),
        "response_length_mean": value(
            grpo_eval,
            "response_length_mean",
        ),
        "terminated_with_eos_rate": value(
            grpo_eval,
            "terminated_with_eos_rate",
        ),
        "truncation_rate": value(
            grpo_eval,
            "truncation_rate",
        ),
        "evaluation_source": str(task3_eval_path),
    },
]

upstream = pd.DataFrame(upstream_rows)

upstream_path = (
    OUTPUT
    / "task4_rq1_upstream_alignment_metrics.csv"
)

upstream.to_csv(
    upstream_path,
    index=False,
)


full = pd.read_csv(task4_full_path)

full["policy"] = (
    full["policy"]
    .astype(str)
    .str.lower()
)

full["source"] = "AI judge: full XSTest"
full["sample_size"] = full["num_examples"]

manual = pd.read_csv(task4_manual_path)

manual["policy"] = (
    manual["policy"]
    .astype(str)
    .str.lower()
)

manual_for_join = manual.rename(
    columns={
        "num_manually_labeled": "sample_size",
        "manual_safe_answer_rate": "safe_answer_rate",
        "manual_safe_over_refusal_rate": (
            "safe_over_refusal_rate"
        ),
        "manual_unsafe_compliance_rate": (
            "unsafe_compliance_rate"
        ),
        "manual_justified_refusal_rate": (
            "justified_refusal_rate"
        ),
        "manual_ambiguous_rate": "ambiguous_rate",
    }
)

manual_for_join["source"] = "Manual audit"

if "ambiguous_judge_rate" in full.columns:
    full["ambiguous_rate"] = (
        full["ambiguous_judge_rate"]
    )

common_columns = [
    "source",
    "policy",
    "sample_size",
    "safe_answer_rate",
    "safe_over_refusal_rate",
    "unsafe_compliance_rate",
    "justified_refusal_rate",
    "ambiguous_rate",
]

common_safety = pd.concat(
    [
        full[common_columns],
        manual_for_join[common_columns],
    ],
    ignore_index=True,
)

policy_order = {
    "sft": 0,
    "dpo": 1,
    "ppo": 2,
    "grpo": 3,
}

source_order = {
    "AI judge: full XSTest": 0,
    "Manual audit": 1,
}

common_safety["_source_order"] = (
    common_safety["source"]
    .map(source_order)
)

common_safety["_policy_order"] = (
    common_safety["policy"]
    .map(policy_order)
)

common_safety = (
    common_safety
    .sort_values([
        "_source_order",
        "_policy_order",
    ])
    .drop(
        columns=[
            "_source_order",
            "_policy_order",
        ]
    )
)

common_path = (
    OUTPUT
    / "task4_rq1_common_safety_results.csv"
)

common_safety.to_csv(
    common_path,
    index=False,
)


figure_roots = [
    ROOT / "results/task1_dpo/report_assets",
    ROOT / "results/task2_ppo/report_assets",
    ROOT / "ppo_figures",
    ROOT / "results/task3_grpo/report_assets",
    ROOT / "results/task3_grpo/report_assets_corrected",
    ROOT / "results/task4_safety/report_assets_corrected",
]

figure_rows = []

for figure_root in figure_roots:
    if not figure_root.exists():
        continue

    for path in sorted(
        figure_root.rglob("*")
    ):
        if (
            path.is_file()
            and path.suffix.lower()
            in {".png", ".pdf"}
        ):
            figure_rows.append({
                "task_or_location": (
                    figure_root.as_posix()
                ),
                "figure": path.name,
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
            })

figure_inventory = pd.DataFrame(
    figure_rows,
    columns=[
        "task_or_location",
        "figure",
        "path",
        "bytes",
    ],
)

figure_path = (
    OUTPUT
    / "task4_rq1_available_figures.csv"
)

figure_inventory.to_csv(
    figure_path,
    index=False,
)


def markdown_table(frame):
    try:
        return frame.to_markdown(
            index=False,
            floatfmt=".6f",
        )
    except ImportError:
        return "```\n" + frame.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        ) + "\n```"


report_columns = [
    "method",
    "held_out_dataset",
    "held_out_examples",
    "preference_accuracy",
    "reward_model_score_mean",
    "sampled_kl_token_mean",
    "entropy_token_mean",
    "response_length_mean",
    "terminated_with_eos_rate",
    "truncation_rate",
]

summary_lines = [
    "# Task 4 RQ1 evidence from Tasks 1–3",
    "",
    "## Primary common-benchmark evidence",
    "",
    (
        "The XSTest comparison is the primary evidence because "
        "SFT, DPO, PPO, and corrected GRPO were evaluated on the "
        "same 450 prompts under the same Task 4 protocol."
    ),
    "",
    markdown_table(common_safety),
    "",
    "## Upstream alignment evidence",
    "",
    markdown_table(
        upstream[report_columns]
    ),
    "",
    "## Comparability warning",
    "",
    (
        "- PPO and corrected GRPO use the same held-out RL prompt "
        "pool and are the cleanest upstream comparison."
    ),
    (
        "- DPO uses its held-out preference dataset, so its reward, "
        "length, and KL values should not be treated as directly "
        "controlled comparisons with PPO or GRPO."
    ),
    (
        "- Signed sampled KL estimates can be slightly negative due "
        "to Monte Carlo estimation; they are not exact analytical KL."
    ),
    (
        "- Task 4 safety rates provide the fair cross-policy "
        "comparison for RQ1."
    ),
    "",
    "## Relevant report figures",
    "",
]

if figure_inventory.empty:
    summary_lines.append(
        "No PNG or PDF figures were discovered."
    )
else:
    summary_lines.append(
        markdown_table(figure_inventory)
    )

summary_lines.extend([
    "",
    "## Exact sources",
    "",
    f"- DPO evaluation: `{task1_eval_path.as_posix()}`",
    f"- DPO training: `{task1_train_path.as_posix()}`",
    f"- PPO evaluation: `{task2_eval_path.as_posix()}`",
    f"- PPO training: `{task2_train_path.as_posix()}`",
    f"- GRPO evaluation: `{task3_eval_path.as_posix()}`",
    f"- GRPO training: `{task3_train_path.as_posix()}`",
    f"- Task 4 full results: `{task4_full_path.as_posix()}`",
    f"- Manual audit: `{task4_manual_path.as_posix()}`",
])

summary_path = (
    OUTPUT
    / "task4_rq1_evidence_summary.md"
)

summary_path.write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)


print(
    "\nTASK 4 RQ1: COMMON XSTEST SAFETY RESULTS"
)
print("=" * 120)
print(
    common_safety.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print(
    "\nTASKS 1–3: UPSTREAM ALIGNMENT METRICS"
)
print("=" * 120)
print(
    upstream[report_columns].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )
)

print(
    "\nAVAILABLE FIGURES"
)
print("=" * 120)

if figure_inventory.empty:
    print("No figures found.")
else:
    print(
        figure_inventory.to_string(
            index=False,
        )
    )

print(
    "\nIMPORTANT COMPARABILITY NOTE"
)
print("=" * 120)
print(
    "Use Task 4 XSTest rates as the primary cross-policy "
    "comparison. PPO and corrected GRPO share the RL held-out "
    "prompt pool. DPO uses a different preference dataset, so "
    "its reward, KL, and length values are supporting context "
    "rather than a controlled comparison."
)

print(
    "\nOUTPUTS"
)
print("=" * 120)
print("Common safety table:", common_path)
print("Upstream metrics:", upstream_path)
print("Figure inventory:", figure_path)
print("RQ1 evidence summary:", summary_path)
print(
    "\nTASK 4 RQ1 EVIDENCE EXTRACTION COMPLETED"
)
