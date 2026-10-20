# Task 2 PPO clipping investigation

This is a read-only investigation. No tracked repository file, checkpoint, or
existing result was modified. The isolated replay script is
`diagnostics/investigate_ppo_clipping.py`.

## Manual requirement

The assignment requires two parts for the clipping study:

1. Use the supplied cached rollout batch to measure PPO clipped-surrogate and
   affected-token fractions for epsilon in `{0.05, 0.20, 0.50}`.
2. Run matched short continuation forks from the same PPO checkpoint and
   compare held-out behavior under identical reward/KL settings.

The exact additional wording is: “Compare short forks at equal
generated-token and update budgets.” The manual does not define “equal
generated-token budget” as identical realized totals after EOS termination, nor
does it specify an exact sample-count formula. In context, “budget” is best
read as the controlled rollout/training allocation: the same number of
updates, prompts/completions per update, and maximum-generation setting unless
the release specifies otherwise. Identical sampled response text is not
required. Reusing identical rollouts is a stricter optional control for
isolating clipping geometry. The cached rollout is explicitly required for the
geometry part.

Therefore, the existing forks should not be labeled noncompliant solely
because their realized response-token totals differ after sampled EOS
termination. They use the same eight prompt IDs, one sampled completion per
update, eight updates, and the same 512-token maximum. The manual's wording is
ambiguous about whether a realized-token cap or exact realized total was
intended; the evidence does not establish a manual violation.

## Confirmed implementation findings

- `task2_ppo/continue_train.py:547-577` computes old log-probabilities from
  the rollout policy inside `torch.no_grad()`, then explicitly detaches them.
- `task2_ppo/continue_train.py:708-761` recomputes new log-probabilities each
  PPO epoch, but reuses the same detached `old_logp`; it calls the policy
  optimizer between epochs.
- `task2_ppo/ppo.py:186-219` uses `exp(new_logp - old_logp)`, applies the
  clipped minimum surrogate, and computes clip fraction using the supplied
  response mask and strict outside-boundary checks.
- `task2_ppo/continue_train.py:522-524` passes the response mask into reward
  construction; `:647-673` masks critic values and normalizes advantages over
  valid response tokens; `:796-812` uses the same response mask for value loss.
- `task2_ppo/continue_train.py:752-761` records the value returned by
  `torch.nn.utils.clip_grad_norm_`. That return value is the norm before
  gradient clipping, not the post-clipping norm. The logged field is averaged
  across both PPO epochs at `:938-955`.
- `task2_ppo/analyze_clipping.py:781-806` independently confirms the ratio
  formula and the same clip-fraction calculation for cached geometry.

No confirmed old-logprob, ratio, masking, optimizer-step, or clip-fraction
implementation defect was found.

## Fork isolation and reproducibility

`task2_ppo/analyze_clipping.py:1030-1065` calls `run_ppo` once per epsilon.
Each `run_ppo` call invokes `prepare_ppo_continuation` at
`task2_ppo/continue_train.py:255-257`; that function resets the seed at
`:69-70`, reloads the midpoint policy and value checkpoint, and constructs
fresh AdamW optimizers at `:109-137`. No scheduler is configured. Thus stale
policy/value/optimizer state is not the likely explanation.

The seed helper resets Python, NumPy, PyTorch CPU, and CUDA-all-device RNGs at
`common/logging_utils.py:15-20`. However, generation uses sampled decoding
(`configs/base.yaml:generation`, temperature 0.7, top-p 0.9), and the policy is
run on CUDA when available. A shared seed does not guarantee identical CUDA
execution or identical sampled trajectories after even tiny numerical/model
state differences.

## Existing-results audit

All three logs use the same eight prompt IDs in the same order. Their update-1
responses are byte-identical. Response text first diverges at update 2.
Therefore equal response lengths alone are not being used as evidence of text
identity; the saved `*_generations.jsonl` files were compared directly.

The logged training trajectories are:

| epsilon | total generated tokens | mean grad norm | max grad norm | mean clip fraction |
|---:|---:|---:|---:|---:|
| 0.05 | 2,180 | 0.5254 | 1.5750 | 0.0 |
| 0.20 | 2,017 | 0.5286 | 1.6545 | 0.0 |
| 0.50 | 2,592 | 0.3847 | 0.9788 | 0.0 |

Update-1 logged policy gradient norms are 0.230205, 0.230920, and 0.231174;
the range is 0.000969, about 0.42% of their mean. Later differences become
large once the sampled trajectories diverge. Update-2 response lengths are
324, 278, and 279 tokens; update-3 lengths are 308, 58, and 437 tokens.

The three existing forks have equal update counts, prompt IDs, one sampled
completion per update, and the same maximum-generation setting. Their realized
totals differ because sampled completions terminate at different lengths. This
is a possible source of unequal training exposure, but it is not enough to
declare the forks noncompliant with the manual's underspecified “budget”
wording.

## Cached fixed-rollout geometry

`results/task2_ppo/clipping_cache_geometry.csv` reports 8,814 valid tokens with
the same midpoint ratio distribution for every epsilon:

- ratio minimum: 0.02571
- ratio maximum: 32.68220
- clip fraction: 0.10846 at epsilon 0.05, 0.008282 at 0.20, and 0.005559 at 0.50
- affected-token fraction: 0.05310, 0.004538, and 0.003290 respectively

This confirms that the production objective and logging path can detect
clipping. It is not direct evidence about each short fork's per-update ratios:
the cached geometry uses a supplied fixed rollout and is a different regime
from each fork's freshly generated rollout.

## Fixed-rollout gradient replay

The isolated replay ran in the pinned `atml-pa2` environment with PyTorch
2.14.1+cu130 on an NVIDIA GeForce RTX 3060 Laptop GPU. It used one supplied
cached rollout row, restored one identical midpoint policy state for each
epsilon, performed forward/backward only, and called neither
`optimizer.step()` nor gradient clipping. The output is
`diagnostics/ppo_clipping_investigation/fixed_rollout_replay.json`.

The ratio quantiles are identical across epsilon because the policy and fixed
inputs are identical:

| epsilon | min | q01 | q05 | q50 | q95 | q99 | max | configured clip fraction | policy loss | gradient norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.909041 | 0.926699 | 0.954940 | 1.000000 | 1.048697 | 1.083173 | 1.222170 | 0.095847 | 0.0001949 | 0.207080 |
| 0.20 | 0.909041 | 0.926699 | 0.954940 | 1.000000 | 1.048697 | 1.083173 | 1.222170 | 0.003195 | -0.0011296 | 0.224026 |
| 0.50 | 0.909041 | 0.926699 | 0.954940 | 1.000000 | 1.048697 | 1.083173 | 1.222170 | 0.000000 | -0.0012022 | 0.229425 |

Pairwise gradient comparisons:

| epsilons | max absolute difference | relative L2 difference | cosine similarity |
|---|---:|---:|---:|
| 0.05 vs 0.20 | 0.0035309 | 0.41469 | 0.92361 |
| 0.05 vs 0.50 | 0.0035706 | 0.48477 | 0.89920 |
| 0.20 vs 0.50 | 0.0023488 | 0.22739 | 0.97504 |

These differences are expected for this cached row because the ratio crosses
the 0.05 and 0.20 boundaries. They are not evidence of dropout or a faulty
epsilon implementation.

### Inactive-clipping control

The same diagnostic was run again with the same checkpoint, response tokens,
mask, advantages, and one fixed midpoint-policy log-probability tensor reused
for all three epsilons. Every ratio was exactly 1.0, so no ratio crossed the
narrowest 0.05 boundary.

| epsilon | policy loss | gradient norm | relative L2 differences | cosine similarities |
|---:|---:|---:|---|---|
| 0.05 | 1.066409e-08 | 0.228530 | 0.05-vs-0.20: 0.0; 0.05-vs-0.50: 0.0 | 1.0; 1.0 |
| 0.20 | 1.066409e-08 | 0.228530 | 0.20-vs-0.50: 0.0 | 1.0 |
| 0.50 | 1.066409e-08 | 0.228530 | - | - |

All pairwise maximum absolute gradient differences were exactly 0.0. This
control verifies that epsilon does not change the loss or gradient when
clipping is inactive. The output is
`diagnostics/ppo_clipping_investigation/inactive_clip_control.json`.

## Dropout-mode check

Policy/LoRA dropout is not active in either production log-probability path.
`task2_ppo/continue_train.py:455-458` sets `policy.eval()` before rollout
collection and old-logprob computation. The new-logprob path explicitly calls
`policy.eval()` again at `:714`. `batch_generate` also temporarily uses eval
mode at `common/generation.py:77-86`. Therefore old and new log-probabilities
are both collected with dropout disabled. There is no distinct “production
mode” versus “identical evaluation mode” condition to compare: they are the
same mode. Dropout cannot explain the small update-1 gradient differences.

## Verdict

- **CONFIRMED STOCHASTIC-TRAJECTORY EFFECT:** the saved sampled response text
  first diverges at update 2, and later rewards, lengths, and gradient norms
  diverge substantially.
- **UNRESOLVED:** whether the small update-1 gradient differences are ordinary
  CUDA/numerical nondeterminism or another lower-level source. The fixed replay
  shows epsilon-dependent gradients when clipping is active, but it does not
  recreate update 1 of the short forks, whose logged clip fractions are zero.
- **No confirmed implementation or logging defect.** Zero logged PPO clipping
  is consistent with very small per-update policy changes under the current
  learning rate and rollout regime; it is not contradicted by the nonzero
  clipping seen in the separate cached geometry study.

## Report-safe wording

“The matched PPO clipping forks recorded zero PPO ratio clipping during their
short continuations. The implementation uses detached rollout log-probabilities,
the standard `exp(new-old)` ratio, and the correct response-token mask. Their
configured generation budgets were matched; realized token totals differed
because of stochastic EOS termination. The forks followed different sampled
trajectories after the first update, so the results are valid evidence that
clipping was inactive in those particular sampled updates but are only
partially informative for attributing downstream differences to epsilon. The
supplied fixed-rollout geometry separately demonstrates nonzero clipping at all
three epsilons, while the inactive-clipping control verifies that epsilon
produces identical gradients when no ratio crosses the narrowest boundary.”
