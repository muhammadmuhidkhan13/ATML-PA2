# Task 3 GRPO zero-valid-token update audit

## Executive verdict

The all-masked updates are not complete no-ops in the existing production
training path. The truncation mask correctly removes every token from a
completion that reaches the 512-token generation cap, and the resulting GRPO
loss is a differentiable exact zero. However, the loop still calls backward,
gradient clipping, and `AdamW.step()` for the update. Because the optimizer has
already accumulated momentum by update 5, the all-masked updates can change
LoRA parameters through AdamW's existing first moment while also advancing the
optimizer step counter and decaying its moment estimates.

The exact historical parameter deltas cannot be reconstructed: no optimizer
state checkpoints were saved for these runs. The adapters remain loadable and
usable, but their training trajectories include momentum-only updates at the
fully masked iterations. The issue is material enough to qualify Task 3
training conclusions. A corrected rerun is recommended but is not technically
required to use the existing adapters or to report the observed runs.

No training was run. No checkpoints, adapters, existing results, production
code, configs, commits, or remotes were modified.

## Exact update lifecycle

1. `common/generation.py:7-30` creates an EOS-aware response mask, and
   `:102-135` marks a completion as truncated when its valid length reaches
   `max_new_tokens` without EOS.
2. `task3_grpo/continue_train.py:267-284` applies
   `mask_truncated_sequences` when `mask_truncated_completions` is true.
   `task3_grpo/grpo.py:296-327` multiplies every token mask in a truncated
   completion by zero.
3. The training loop collects four completions per prompt at
   `task3_grpo/continue_train.py:517-579`, computes group advantages at
   `:616-639`, and records whether each completion has any effective policy
   tokens at `:691-696`.
4. For every policy epoch, the loop calls
   `optimizer.zero_grad(set_to_none=True)` at
   `task3_grpo/continue_train.py:711-716`, computes the loss for every
   completion at `:718-764`, divides by the number of completions at
   `:766-771`, and always calls `scaled_loss.backward()` at `:773`.
5. After all completions, it always calls
   `torch.nn.utils.clip_grad_norm_` at `task3_grpo/continue_train.py:847-852`,
   then `optimizer.step()` and increments `optimizer_updates` at `:854-855`.
6. The update is logged at `task3_grpo/continue_train.py:947-1028`, including
   `valid_policy_tokens` at `:1010-1012`. The summary counts attempted and
   optimizer updates separately at `:1166-1190`, but both counts include the
   fully masked iterations because the optimizer step was performed.

There is no learning-rate scheduler in `task3_grpo/continue_train.py`,
`task3_grpo/grpo.py`, `configs/grpo.yaml`, or the shared utilities inspected.

## Loss behavior for an all-zero mask

`task3_grpo/grpo.py:192-223` computes the masked sequence objective. With an
all-zero mask, its numerator is zero and the denominator is clamped to at
least one, so both canonical and Dr-GRPO policy terms are zero. The KL penalty
at `:239-257` uses `common.metrics.masked_mean`, whose denominator is also
clamped, so it is zero as well. The total loss is therefore exactly zero while
remaining connected to `new_logp` through multiplication by the zero mask.

The real production masking utility and loss were tested with a minimal
differentiable tensor. For a normal mask, the loss required gradients and
produced a nonzero gradient. For an all-zero mask produced by
`mask_truncated_sequences(..., [True])`, the loss was exactly `0.0`, still had
`requires_grad=True`, and backward produced an explicit zero gradient tensor,
not `grad=None`.

## Raw-log evidence

The three logs exactly match the known zero-token evidence. Every fully masked
row had four 512-token completions, 2,048 generated tokens, truncation rate
1.0, zero valid policy tokens, and zero token-level diagnostics.

| Run | Update | Generated tokens | Valid policy tokens | Truncation rate | Policy loss | Total loss | Gradient norm | Ratio mean | Sampled KL | Clip fraction | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard | 5 | 2048 | 0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| standard | 13 | 2048 | 0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| standard | 15 | 2048 | 0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| standard | 17 | 2048 | 0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| canonical normalization | 5 | 2048 | 0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Dr-GRPO normalization | 5 | 2048 | 0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

The standard summary reports 20 attempted updates and 20 optimizer updates,
with four updates without valid policy tokens. Each normalization summary
reports 8 attempted and 8 optimizer updates, with one update without valid
policy tokens. This confirms that the rows were reported as optimizer updates,
not skipped iterations.

The corresponding generation records confirm that all four completions in
standard updates 5, 13, 15, and 17 were `truncated=true` and
`used_for_policy_loss=false`; the same holds for update 5 in both normalization
runs.

## Optimizer semantics

The production optimizer is `torch.optim.AdamW`, constructed at
`task3_grpo/continue_train.py:104-113` over all trainable policy parameters
returned at `:97-102`. The verified settings are:

| Setting | Value | Source |
|---|---:|---|
| Learning rate | `5e-6` | `configs/grpo.yaml:6` |
| Betas | `(0.9, 0.999)` | AdamW default; not overridden at `:104-113` |
| Optimizer epsilon | `1e-8` | `task3_grpo/continue_train.py:110-112` default path and AdamW default |
| Weight decay | `0.0` | `task3_grpo/continue_train.py:107-109` default path |
| Gradient max norm | `1.0` | `configs/grpo.yaml:14`, used at `:847-852` |
| Zeroing behavior | `set_to_none=True` | `task3_grpo/continue_train.py:714-715` |
| Scheduler | None | no scheduler creation or step call found |
| Policy epochs | `1` | `configs/grpo.yaml:5` |
| Completions per prompt | `4` | `configs/grpo.yaml:9` |
| Truncated-completion masking | enabled | `configs/grpo.yaml:13` |

Because `backward()` is called on a differentiable zero loss, the production
path supplies explicit zero gradients to AdamW. With zero weight decay, a
fresh AdamW optimizer leaves parameters unchanged but creates state and
increments its step counter. After momentum has been established, AdamW with
an explicit zero gradient changes parameters using the existing first moment,
decays the first and second moments, and increments the step counter.

## Deterministic CPU optimizer control

`run_control.py` used CPU-only deterministic tensors and the production AdamW
hyperparameters. Results are saved in `control_results.json`.

| Case | Parameter change | Step counter | Moment/state behavior |
|---|---:|---|---|
| Fresh optimizer, explicit zero gradient | 0 | `None -> 1` | State created with zero moments |
| Fresh optimizer, grad=None | 0 | remains `None` | No state created or changed |
| Established momentum, explicit zero gradient | L2 `4.720447e-06`; max abs `3.337861e-06` | `1 -> 2` | First-moment change `0.0360555`; second-moment change `9.84915e-06`; momentum contributed |
| Established momentum, grad=None | 0 | remains `1` | First and second moments unchanged; momentum did not contribute |

Thus the exact classification for the observed production all-masked updates
is **parameter and optimizer-state change**, because those updates occur after
earlier effective updates have established AdamW state. A hypothetical first
update with a fresh optimizer and explicit zero gradient would instead be an
optimizer-state-only change.

## Checkpoint impact

No optimizer-state checkpoint was found for these historical runs. The saved
policy adapters and evaluation outputs are present, but the optimizer state
needed to reconstruct the exact historical deltas is not. Therefore the
historical parameter change can be established semantically but not measured
exactly from saved artifacts.

The adapters remain usable as policy checkpoints. They should be described as
policies trained under a continuation loop that included momentum-only updates
at fully masked iterations, rather than as outputs of a strictly skip-on-empty
optimizer regime.

## Scientific impact and recommendation

- Four of twenty standard updates, or 20%, were fully masked. This is
  material: those iterations supplied no policy-token learning signal but
  still advanced AdamW and could move parameters through momentum.
- One of eight updates, or 12.5%, was fully masked in each normalization fork.
- Canonical GRPO and Dr-GRPO remain mechanically matched at the same empty
  update ID, so the paired comparison is not invalidated by an asymmetric
  empty update. However, both conditions include the same unintended
  momentum-only event, so conclusions about normalization should qualify that
  shared behavior.
- A corrected rerun is **recommended but optional**, not technically required
  to use the adapters or report the observed trajectories. It becomes
  technically necessary only if the report claims results for a corrected
  algorithm that skips ineffective updates.

## Conventional fix, not implemented

The smallest conventional correction is to detect an update with
`valid_token_total == 0`, log it as attempted but ineffective, and skip
backward, gradient clipping, optimizer.step, and any scheduler.step if one is
later introduced. The generation/update budget should remain transparent: the
iteration can still count as an attempted rollout update while the effective
optimizer-update count is recorded separately.

## Limitations

The CPU control establishes AdamW semantics and the real GRPO loss/mask
gradient behavior, but it does not reconstruct the historical LoRA parameter
delta because optimizer state was not saved. It also does not load the full
1.5B model, which is unnecessary for determining the zero-mask computation and
optimizer semantics.
