# ATML PA2 report evidence bundle

This bundle collects the report evidence required by the assignment manual. Existing authoritative figures were copied without modification. Newly generated figures are derived only from saved CSV/JSON results. The corrected Task 3 and Task 4 artifacts replace their older counterparts; no superseded Task 3 or Task 4 result is included.

The assignment permits required quantities to be organized as figures or tables. Each figure is provided as both PNG and vector PDF. Source tables for newly generated figures are in each task's `sources/` directory.

## Evidence map

| Task | Required evidence | Primary file | Note |
| --- | --- | --- | --- |
| Task 1 | DPO summary: standard plus three beta forks | `task1_dpo/figures/task1_dpo_beta_tradeoff.pdf` | Generated from standard_metrics.json and beta_ablation_comparison.csv; budgets labeled separately. |
| Task 1 | Length-confounding strata, generated length, word-limit compliance | `task1_dpo/figures/task1_length_confounding.pdf` | Generated from length_analysis_summary.json. |
| Task 1 | Qualitative reward/preference-quality and length/instruction examples | `task1_dpo/qualitative/task1_qualitative_evidence_final.md` | Copied from the reviewed Task 1 evidence. |
| Task 2 | Standard PPO trajectories and compute | `task2_ppo/figures/standard_ppo_diagnostics.pdf` | Existing figure copied unchanged; compute values are in standard_train_summary.json. |
| Task 2 | Cached clipping geometry for epsilon 0.05, 0.20, 0.50 | `task2_ppo/figures/task2_cached_clipping_geometry.pdf` | Generated from clipping_cache_geometry.csv. |
| Task 2 | Matched clipping and KL-pressure fork behavior | `task2_ppo/figures/task2_ppo_ablation_trajectories.pdf` | Existing figure copied unchanged; held-out tables are packaged alongside it. |
| Task 2 | Reward-quality agreement and disagreement | `task2_ppo/qualitative/rq3_targeted_short_analysis.md` | Validated blind targeted diagnostic; not a population sample. |
| Task 3 | Corrected standard GRPO trajectories and compute | `task3_grpo_corrected/figures/task3_corrected_standard_training_dynamics.pdf` | Corrected 20-effective-update run only. |
| Task 3 | Equal-generation group-size study, including difficulty bins | `task3_grpo_corrected/figures/task3_group_size_comparison.pdf` | Corrected report asset copied unchanged. |
| Task 3 | Canonical versus Dr-GRPO held-out comparison | `task3_grpo_corrected/figures/task3_normalization_heldout_comparison.pdf` | Corrected report asset copied unchanged. |
| Task 3 | Length-conditioned normalization dynamics/statistics | `task3_grpo_corrected/figures/task3_normalization_training_dynamics.pdf` | Length-conditioned tables are packaged in tables/. |
| Task 3 | Qualitative normalization/group-informativeness evidence | `task3_grpo_corrected/qualitative/task3_normalization_qualitative_candidates.json` | Corrected candidates only. |
| Task 4 | Corrected safety-calibration comparison | `task4_safety_corrected/figures/task4_calibration_rates.pdf` | Corrected final evaluation only. |
| Task 4 | Category-level behavior | `task4_safety_corrected/figures/task4_category_heatmap.pdf` | Corrected final evaluation only. |
| Task 4 | Manual-audit agreement/confusion | `task4_safety_corrected/figures/task4_judge_confusion_matrix.pdf` | Corrected manual-audit evidence. |
| Task 4 | Harmful compliance, justified refusal, and over-refusal examples | `task4_safety_corrected/qualitative/task4_qualitative_examples.csv` | Corrected qualitative examples. |
| Task 5 | In-domain and out-of-domain policy comparison | `task5_feedback/figures/task5_policy_domain_comparison.pdf` | Generated from policy and pairwise comparison tables. |
| Task 5 | Controlled diagnostic, better/tie/wrong rates, S_reason and S_outcome | `task5_feedback/figures/task5_controlled_reward_diagnostic.pdf` | Generated from all five variant metrics and four clean-versus-perturbation comparisons. |
| Task 5 | Reasoning corruption, persuasive filler, and wrong-final/gold-distractor examples | `task5_feedback/qualitative/controlled_qualitative_candidates.json` | Existing qualitative candidates copied unchanged. |
| Task 5 | Feedback coverage/noise/exploitability/cost source data | `task5_feedback/tables/feedback_comparison_summary.json` | Machine-readable summary for the required discussion. |
| Task 6 | Cross-task synthesis | `EVIDENCE_INDEX.md` | No new experiment is required; synthesize Tasks 1-5 using this index. |

## Replacement policy

- Task 3 uses `standard_corrected_20261018` and `report_assets_corrected` only.
- Task 4 uses `corrected_final_evaluation_*` and `report_assets_corrected` only.
- Older Task 3 and Task 4 artifacts are intentionally excluded.

## Report-use caution

The bundle is evidence, not report prose. Preserve the manual's qualifications: short and standard training budgets differ; targeted qualitative subsets are not representative; cross-prompt reward differences do not isolate policy quality; and reward-quality disagreement is not by itself proof of reward overoptimization.
