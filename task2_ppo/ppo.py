from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(
    rewards,
    values,
    mask,
    gamma=1.0,
    lam=0.95,
):
    """Compute token-level Generalized Advantage Estimation.

    Args:
        rewards:
            Reward assigned to every response position.
            Shape: [batch, response_steps].

        values:
            Critic's estimated value at every response position.
            Shape: [batch, response_steps].

        mask:
            One for valid response tokens and zero for padding.
            Shape: [batch, response_steps].

        gamma:
            Reward discount factor.

        lam:
            GAE smoothing parameter.

    The final valid response position bootstraps with zero because the
    generated response is treated as a completed trajectory.
    """

    batch_size, response_steps = rewards.shape

    advantages = torch.zeros_like(
        rewards
    )

    last_advantage = torch.zeros(
        batch_size,
        device=rewards.device,
        dtype=rewards.dtype,
    )

    for timestep in reversed(
        range(response_steps)
    ):
        current_valid = mask[
            :,
            timestep,
        ]

        if timestep + 1 < response_steps:
            next_valid = mask[
                :,
                timestep + 1,
            ]

            next_value = (
                values[:, timestep + 1]
                * next_valid
            )
        else:
            next_valid = torch.zeros_like(
                current_valid
            )

            next_value = torch.zeros_like(
                last_advantage
            )

        temporal_difference = (
            rewards[:, timestep]
            + gamma * next_value
            - values[:, timestep]
        )

        last_advantage = (
            temporal_difference
            + gamma
            * lam
            * next_valid
            * last_advantage
        )

        # Padding positions must not contribute to the trajectory.
        last_advantage = (
            last_advantage
            * current_valid
        )

        advantages[
            :,
            timestep,
        ] = last_advantage

    returns = advantages + values

    return advantages, returns


def shaped_rewards(
    task_reward,
    policy_logp,
    ref_logp,
    response_mask,
    beta_kl,
):
    """Create token-level rewards using KL shaping and terminal reward.

    Every valid response token receives a sampled KL penalty:

        -beta_kl * (policy_logp - reference_logp)

    The learned task reward is added only to the final valid response
    token.
    """

    sampled_log_ratio = (
        policy_logp - ref_logp
    )

    rewards = (
        -float(beta_kl)
        * sampled_log_ratio
        * response_mask
    )

    for batch_index in range(
        rewards.shape[0]
    ):
        valid_tokens = int(
            response_mask[
                batch_index
            ].sum().item()
        )

        if valid_tokens > 0:
            final_token_index = (
                valid_tokens - 1
            )

            rewards[
                batch_index,
                final_token_index,
            ] += task_reward[
                batch_index
            ]

    return rewards


def ppo_policy_loss(
    new_logp,
    old_logp,
    advantage,
    mask,
    eps=0.2,
):
    """Compute PPO's clipped policy loss and diagnostics.

    The probability ratio is:

        ratio = exp(new_logp - old_logp)

    PPO compares the ordinary surrogate objective with a clipped
    surrogate and keeps the more conservative value:

        min(
            ratio * advantage,
            clip(ratio, 1-eps, 1+eps) * advantage,
        )

    The function returns a loss because PyTorch optimizers minimize.
    Therefore, the selected PPO objective is negated.
    """

    ratio = torch.exp(
        new_logp - old_logp
    )

    unclipped_surrogate = (
        ratio * advantage
    )

    clipped_ratio = ratio.clamp(
        1.0 - eps,
        1.0 + eps,
    )

    clipped_surrogate = (
        clipped_ratio * advantage
    )

    # PPO deliberately selects the less optimistic objective.
    objective = torch.minimum(
        unclipped_surrogate,
        clipped_surrogate,
    )

    loss = -masked_mean(
        objective,
        mask,
    )

    outside_clip_range = (
        (ratio < (1.0 - eps))
        | (ratio > (1.0 + eps))
    ).float()

    clip_fraction = masked_mean(
        outside_clip_range,
        mask,
    )

    return (
        loss,
        ratio.detach(),
        clip_fraction.detach(),
    )


def value_mse_loss(
    predicted_values,
    returns,
    mask,
):
    """Compute masked mean-squared error for the critic."""

    squared_error = (
        predicted_values - returns
    ) ** 2

    return masked_mean(
        squared_error,
        mask,
    )


def normalize_advantages(
    advantages,
    mask,
    eps=1e-6,
):
    """Normalize advantages over valid response positions only."""

    valid_advantages = advantages[
        mask.bool()
    ]

    if valid_advantages.numel() <= 1:
        return advantages

    mean = valid_advantages.mean()

    standard_deviation = (
        valid_advantages
        .std(unbiased=False)
        .clamp_min(eps)
    )

    normalized = (
        advantages - mean
    ) / standard_deviation

    return normalized * mask