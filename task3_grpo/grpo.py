from __future__ import annotations

import torch

from common.metrics import (
    masked_mean,
    sampled_kl as sampled_kl_metric,
    sample_entropy,
)


def group_relative_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    eps: float = 1e-6,
):
    """Calculate reward advantages separately within each prompt group.

    Each completion is standardized using only the rewards of the other
    completions generated for the same prompt:

        advantage = (reward - group_mean) / (group_std + eps)

    Groups with no reward variation receive zero advantages.
    """
    if rewards.ndim != 1:
        raise ValueError(
            "rewards must be a one-dimensional tensor"
        )

    if group_ids.ndim != 1:
        raise ValueError(
            "group_ids must be a one-dimensional tensor"
        )

    if rewards.shape[0] != group_ids.shape[0]:
        raise ValueError(
            "rewards and group_ids must contain the same "
            "number of completions"
        )

    if rewards.numel() == 0:
        return torch.empty_like(rewards)

    if eps <= 0:
        raise ValueError("eps must be positive")

    group_ids = group_ids.to(device=rewards.device)
    advantages = torch.zeros_like(rewards)

    for group_id in torch.unique(group_ids):
        group_mask = group_ids == group_id
        group_rewards = rewards[group_mask]

        group_mean = group_rewards.mean()
        group_std = group_rewards.std(
            unbiased=False
        )

        if not torch.isfinite(group_mean):
            raise ValueError(
                f"Non-finite reward mean in group "
                f"{group_id.item()!r}"
            )

        if not torch.isfinite(group_std):
            raise ValueError(
                f"Non-finite reward standard deviation in "
                f"group {group_id.item()!r}"
            )

        if group_std <= eps:
            advantages[group_mask] = 0.0
        else:
            advantages[group_mask] = (
                group_rewards - group_mean
            ) / (group_std + eps)

    return advantages


def grpo_policy_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    seq_adv: torch.Tensor,
    token_mask: torch.Tensor,
    ref_logp: torch.Tensor,
    eps: float,
    beta: float,
    loss_type: str = "grpo",
    max_completion_length: int | None = None,
):
    """Calculate the clipped GRPO policy loss and diagnostics.

    Args:
        new_logp:
            Current-policy log-probabilities for sampled response
            tokens, shaped [batch, response_steps].
        old_logp:
            Rollout-policy log-probabilities for the same tokens.
        seq_adv:
            One group-relative scalar advantage per completion,
            shaped [batch].
        token_mask:
            Valid response-token mask. A deliberately excluded
            completion may have an all-zero mask.
        ref_logp:
            Frozen-reference log-probabilities for the sampled tokens.
        eps:
            PPO clipping epsilon.
        beta:
            Coefficient for the reference-policy KL penalty.
        loss_type:
            "grpo" for realized-length normalization or "dr_grpo"
            for fixed maximum-length normalization.
        max_completion_length:
            Required when loss_type is "dr_grpo".
    """
    if new_logp.shape != old_logp.shape:
        raise ValueError(
            "new_logp and old_logp must have identical shapes"
        )

    if new_logp.shape != ref_logp.shape:
        raise ValueError(
            "new_logp and ref_logp must have identical shapes"
        )

    if new_logp.shape != token_mask.shape:
        raise ValueError(
            "token_mask must have the same shape as token log-probabilities"
        )

    if new_logp.ndim != 2:
        raise ValueError(
            "token log-probabilities must have shape "
            "[batch, response_steps]"
        )

    if seq_adv.ndim != 1:
        raise ValueError(
            "seq_adv must have shape [batch]"
        )

    if seq_adv.shape[0] != new_logp.shape[0]:
        raise ValueError(
            "seq_adv must contain one advantage per completion"
        )

    if eps <= 0:
        raise ValueError(
            "Clipping epsilon must be positive"
        )

    if beta < 0:
        raise ValueError(
            "KL coefficient beta must be non-negative"
        )

    token_mask = token_mask.to(
        device=new_logp.device,
        dtype=new_logp.dtype,
    )
    seq_adv = seq_adv.to(
        device=new_logp.device,
        dtype=new_logp.dtype,
    )

    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)

    token_advantage = seq_adv.unsqueeze(-1)

    unclipped_objective = (
        ratio * token_advantage
    )

    clipped_ratio = ratio.clamp(
        1.0 - float(eps),
        1.0 + float(eps),
    )

    clipped_objective = (
        clipped_ratio * token_advantage
    )

    selected_objective = torch.minimum(
        unclipped_objective,
        clipped_objective,
    )

    sequence_objective_sum = (
        selected_objective * token_mask
    ).sum(dim=-1)

    valid_token_count = token_mask.sum(dim=-1)

    if loss_type == "grpo":
        # Canonical GRPO gives each completion equal sequence-level
        # weight after averaging over its realized valid tokens.
        sequence_objective = (
            sequence_objective_sum
            / valid_token_count.clamp_min(1.0)
        )

    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError(
                "dr_grpo requires max_completion_length"
            )

        if max_completion_length <= 0:
            raise ValueError(
                "max_completion_length must be positive"
            )

        # Dr-GRPO uses one fixed denominator for every completion.
        # This avoids upweighting short completions solely because
        # they contain fewer generated tokens.
        sequence_objective = (
            sequence_objective_sum
            / float(max_completion_length)
        )

    else:
        raise ValueError(
            f"Unknown loss_type={loss_type!r}"
        )

    policy_term = -sequence_objective.mean()

    # Non-negative per-token KL estimator commonly used in GRPO:
    #
    #   exp(log pi_ref - log pi_theta)
    #   - (log pi_ref - log pi_theta)
    #   - 1
    #
    # It is used as the differentiable reference-policy penalty.
    ref_over_policy_log_ratio = (
        ref_logp - new_logp
    )

    per_token_kl_penalty = (
        torch.exp(ref_over_policy_log_ratio)
        - ref_over_policy_log_ratio
        - 1.0
    )

    kl_penalty = masked_mean(
        per_token_kl_penalty,
        token_mask,
    )

    loss = (
        policy_term
        + float(beta) * kl_penalty
    )

    outside_clip_range = (
        (ratio < (1.0 - float(eps)))
        | (ratio > (1.0 + float(eps)))
    ).to(dtype=new_logp.dtype)

    # This is the course-defined sampled-response KL diagnostic.
    # It may be slightly negative on a finite sampled batch.
    reported_sampled_kl = sampled_kl_metric(
        new_logp.detach(),
        ref_logp.detach(),
        token_mask,
    )

    diagnostics = {
        "policy_term": policy_term.detach(),
        "kl_penalty": kl_penalty.detach(),
        "sampled_kl": reported_sampled_kl.detach(),
        "clip_fraction": masked_mean(
            outside_clip_range,
            token_mask,
        ).detach(),
        "ratio_mean": masked_mean(
            ratio.detach(),
            token_mask,
        ),
        "sample_entropy": sample_entropy(
            new_logp.detach(),
            token_mask,
        ),
        "valid_token_count": (
            valid_token_count.detach()
        ),
    }

    return loss, diagnostics


def mask_truncated_sequences(
    token_mask: torch.Tensor,
    truncated: list[bool] | torch.Tensor,
):
    """Mask every token of a completion that hit the generation cap."""
    if token_mask.ndim != 2:
        raise ValueError(
            "token_mask must have shape "
            "[batch, response_steps]"
        )

    truncated = torch.as_tensor(
        truncated,
        device=token_mask.device,
        dtype=torch.bool,
    )

    if truncated.ndim != 1:
        raise ValueError(
            "truncated must contain one Boolean per completion"
        )

    if truncated.shape[0] != token_mask.shape[0]:
        raise ValueError(
            "truncated must contain one Boolean per completion"
        )

    keep_sequence = (
        ~truncated
    ).to(dtype=token_mask.dtype).unsqueeze(-1)

    return token_mask * keep_sequence