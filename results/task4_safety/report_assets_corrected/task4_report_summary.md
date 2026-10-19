# Task 4 corrected report summary

- Full AI judgments: 1,800
- Manual-audit responses: 240
- Manual/AI exact agreement: 0.7000
- Manual/AI disagreements: 72

## Corrected full-XSTest policy metrics

| policy   |   num_examples |   num_safe |   num_unsafe |   safe_answer_rate |   safe_over_refusal_rate |   unsafe_compliance_rate |   justified_refusal_rate |   ambiguous_judge_rate |   mean_judge_confidence |   response_length_mean |   response_length_std |   response_length_iqr |   generated_token_count |
|:---------|---------------:|-----------:|-------------:|-------------------:|-------------------------:|-------------------------:|-------------------------:|-----------------------:|------------------------:|-----------------------:|----------------------:|----------------------:|------------------------:|
| sft      |            450 |        250 |          200 |              0.568 |                        0 |                        0 |                    0.88  |                      0 |                0.999889 |                108.087 |               80.9158 |                124.25 |                   48639 |
| dpo      |            450 |        250 |          200 |              0.556 |                        0 |                        0 |                    0.875 |                      0 |                1        |                110.233 |               82.0403 |                131.25 |                   49605 |
| ppo      |            450 |        250 |          200 |              0.584 |                        0 |                        0 |                    0.89  |                      0 |                0.999889 |                107.469 |               79.9945 |                115    |                   48361 |
| grpo     |            450 |        250 |          200 |              0.576 |                        0 |                        0 |                    0.88  |                      0 |                0.999889 |                108.311 |               80.4316 |                124.25 |                   48740 |

## Corrected manual-audit metrics

| source       | policy   |   num_records |   num_safe |   num_unsafe |   safe_answer_rate |   safe_over_refusal_rate |   unsafe_compliance_rate |   justified_refusal_rate |   ambiguous_rate |   mean_response_tokens |
|:-------------|:---------|--------------:|-----------:|-------------:|-------------------:|-------------------------:|-------------------------:|-------------------------:|-----------------:|-----------------------:|
| Manual audit | sft      |            60 |         30 |           30 |           0.633333 |                 0.366667 |                0.0333333 |                 0.966667 |                0 |                109.367 |
| Manual audit | dpo      |            60 |         30 |           30 |           0.6      |                 0.4      |                0.0333333 |                 0.966667 |                0 |                107.5   |
| Manual audit | ppo      |            60 |         30 |           30 |           0.633333 |                 0.366667 |                0.0333333 |                 0.966667 |                0 |                108.9   |
| Manual audit | grpo     |            60 |         30 |           30 |           0.633333 |                 0.366667 |                0.0333333 |                 0.966667 |                0 |                110.333 |
