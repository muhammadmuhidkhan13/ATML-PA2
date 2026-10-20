# Task 4 RQ1 evidence from Tasks 1–3

## Primary common-benchmark evidence

The XSTest comparison is the primary evidence because SFT, DPO, PPO, and corrected GRPO were evaluated on the same 450 prompts under the same Task 4 protocol.

```
               source policy  sample_size  safe_answer_rate  safe_over_refusal_rate  unsafe_compliance_rate  justified_refusal_rate  ambiguous_rate
AI judge: full XSTest    sft          450          0.568000                0.000000                0.000000                0.880000        0.000000
AI judge: full XSTest    dpo          450          0.556000                0.000000                0.000000                0.875000        0.000000
AI judge: full XSTest    ppo          450          0.584000                0.000000                0.000000                0.890000        0.000000
AI judge: full XSTest   grpo          450          0.576000                0.000000                0.000000                0.880000        0.000000
         Manual audit    sft           60          0.633333                0.366667                0.033333                0.966667        0.000000
         Manual audit    dpo           60          0.600000                0.400000                0.033333                0.966667        0.000000
         Manual audit    ppo           60          0.633333                0.366667                0.033333                0.966667        0.000000
         Manual audit   grpo           60          0.633333                0.366667                0.033333                0.966667        0.000000
```

## Upstream alignment evidence

```
method               held_out_dataset  held_out_examples  preference_accuracy  reward_model_score_mean  sampled_kl_token_mean  entropy_token_mean  response_length_mean  terminated_with_eos_rate  truncation_rate
   DPO   data/dpo_standard_eval.jsonl                290             0.631034                 1.246320               0.000315                 NaN            153.903448                  0.551724         0.448276
   PPO data/rl_prompt_pool_eval.jsonl                200                  NaN                 1.569832              -0.000026            0.839519            300.415000                  0.945000         0.055000
  GRPO data/rl_prompt_pool_eval.jsonl                200                  NaN                 1.396812               0.000113            0.825010            304.140000                  0.920000         0.080000
```

## Comparability warning

- PPO and corrected GRPO use the same held-out RL prompt pool and are the cleanest upstream comparison.
- DPO uses its held-out preference dataset, so its reward, length, and KL values should not be treated as directly controlled comparisons with PPO or GRPO.
- Signed sampled KL estimates can be slightly negative due to Monte Carlo estimation; they are not exact analytical KL.
- Task 4 safety rates provide the fair cross-policy comparison for RQ1.

## Relevant report figures

```
                            task_or_location                                         figure                                                                                      path  bytes
             results/task2_ppo/report_assets            task2_ppo_ablation_trajectories.pdf                       results/task2_ppo/report_assets/task2_ppo_ablation_trajectories.pdf  29085
             results/task2_ppo/report_assets            task2_ppo_ablation_trajectories.png                       results/task2_ppo/report_assets/task2_ppo_ablation_trajectories.png 709815
                                 ppo_figures                   standard_ppo_diagnostics.pdf                                                  ppo_figures/standard_ppo_diagnostics.pdf  32604
                                 ppo_figures                   standard_ppo_diagnostics.png                                                  ppo_figures/standard_ppo_diagnostics.png 365619
  results/task3_grpo/report_assets_corrected task3_corrected_standard_training_dynamics.pdf results/task3_grpo/report_assets_corrected/task3_corrected_standard_training_dynamics.pdf  33223
  results/task3_grpo/report_assets_corrected task3_corrected_standard_training_dynamics.png results/task3_grpo/report_assets_corrected/task3_corrected_standard_training_dynamics.png 498276
  results/task3_grpo/report_assets_corrected                task3_group_size_comparison.pdf                results/task3_grpo/report_assets_corrected/task3_group_size_comparison.pdf  23770
  results/task3_grpo/report_assets_corrected                task3_group_size_comparison.png                results/task3_grpo/report_assets_corrected/task3_group_size_comparison.png 205205
  results/task3_grpo/report_assets_corrected     task3_normalization_heldout_comparison.pdf     results/task3_grpo/report_assets_corrected/task3_normalization_heldout_comparison.pdf  26156
  results/task3_grpo/report_assets_corrected     task3_normalization_heldout_comparison.png     results/task3_grpo/report_assets_corrected/task3_normalization_heldout_comparison.png 236554
  results/task3_grpo/report_assets_corrected      task3_normalization_training_dynamics.pdf      results/task3_grpo/report_assets_corrected/task3_normalization_training_dynamics.pdf  33682
  results/task3_grpo/report_assets_corrected      task3_normalization_training_dynamics.png      results/task3_grpo/report_assets_corrected/task3_normalization_training_dynamics.png 554534
results/task4_safety/report_assets_corrected                    task4_calibration_rates.pdf                  results/task4_safety/report_assets_corrected/task4_calibration_rates.pdf  19140
results/task4_safety/report_assets_corrected                    task4_calibration_rates.png                  results/task4_safety/report_assets_corrected/task4_calibration_rates.png 182093
results/task4_safety/report_assets_corrected                     task4_category_heatmap.pdf                   results/task4_safety/report_assets_corrected/task4_category_heatmap.pdf  20368
results/task4_safety/report_assets_corrected                     task4_category_heatmap.png                   results/task4_safety/report_assets_corrected/task4_category_heatmap.png 234997
results/task4_safety/report_assets_corrected               task4_judge_confusion_matrix.pdf             results/task4_safety/report_assets_corrected/task4_judge_confusion_matrix.pdf  20833
results/task4_safety/report_assets_corrected               task4_judge_confusion_matrix.png             results/task4_safety/report_assets_corrected/task4_judge_confusion_matrix.png 244771
```

## Exact sources

- DPO evaluation: `results/task1_dpo/standard_metrics.json`
- DPO training: `results/task1_dpo/standard_train_summary.json`
- PPO evaluation: `results/task2_ppo/standard_metrics.json`
- PPO training: `results/task2_ppo/standard_train_summary.json`
- GRPO evaluation: `results/task3_grpo/standard_corrected_20261018_eval_metrics.json`
- GRPO training: `results/task3_grpo/standard_corrected_20261018_train_summary.json`
- Task 4 full results: `results/task4_safety/corrected_final_evaluation_policy_comparison.csv`
- Manual audit: `results/task4_safety/manual_audit_policy_metrics.csv`
