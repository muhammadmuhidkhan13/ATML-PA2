"""Read-only fixed-rollout PPO clipping replay.

This diagnostic intentionally does not call the training loop or an optimizer.
It loads one or more supplied cached rollouts, restores the same midpoint
policy state for each epsilon, and compares the differentiable PPO loss and
policy gradients. Results are written only beneath diagnostics/.

Run from the repository root in the assignment environment, for example:

    python diagnostics/investigate_ppo_clipping.py --rows 1

The script is deliberately separate from production training code so that it
cannot overwrite checkpoints or results/task2_ppo files.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer
from task2_ppo.analyze_clipping import (
    find_prompt_row,
    load_cached_rollouts,
    reconstruct_cached_sequence,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "ppo_clipping_investigation"


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    q = torch.tensor([0.0, 0.01, 0.05, 0.50, 0.95, 0.99, 1.0], device=values.device)
    result = torch.quantile(values.float(), q).detach().cpu().tolist()
    names = ("q00", "q01", "q05", "q50", "q95", "q99", "q100")
    return {name: float(value) for name, value in zip(names, result)}


def _grad_vector(policy) -> torch.Tensor:
    chunks = []
    for parameter in policy.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                chunks.append(torch.zeros_like(parameter).float().flatten())
            else:
                chunks.append(parameter.grad.detach().float().flatten())
    return torch.cat(chunks)


def _parameter_snapshot(policy) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }


def _restore_parameters(policy, snapshot: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(snapshot[name])


def _fixed_inputs(cfg: dict, rows: list[dict], tokenizer, policy):
    prompt_rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    by_source = {row.get("source_index"): row for row in prompt_rows}
    by_prompt = {row.get("prompt_id"): row for row in prompt_rows}

    device = next(policy.parameters()).device
    old_parts = []
    ref_parts = []
    value_parts = []
    response_parts = []
    sequence_parts = []
    prompt_widths = []
    details = []

    for cached in rows:
        source = find_prompt_row(cached, by_source, by_prompt)
        rebuilt = reconstruct_cached_sequence(
            tokenizer=tokenizer,
            source_row=source,
            cached_row=cached,
            max_prompt_length=int(cfg["max_prompt_length"]),
        )
        full = torch.tensor([rebuilt["full_ids"]], dtype=torch.long, device=device)
        response = torch.tensor([rebuilt["response_ids"]], dtype=torch.long, device=device)
        attention = torch.ones_like(full)
        with torch.no_grad():
            old_logp = torch.as_tensor(cached["old_logprobs"], dtype=torch.float32, device=device)
            ref_logp = torch.as_tensor(cached["ref_logprobs"], dtype=torch.float32, device=device)
            values = torch.as_tensor(cached["values"], dtype=torch.float32, device=device)
            mask = torch.ones((1, response.shape[1]), dtype=torch.float32, device=device)
            terminal = torch.tensor([float(cached["effective_terminal_reward"])], device=device)
            rewards = shaped_rewards(
                task_reward=terminal,
                policy_logp=old_logp.unsqueeze(0),
                ref_logp=ref_logp.unsqueeze(0),
                response_mask=mask,
                beta_kl=float(cfg["kl_beta"]),
            )
            advantages, _ = compute_gae(
                rewards=rewards,
                values=values.unsqueeze(0),
                mask=mask,
                gamma=float(cfg["gamma"]),
                lam=float(cfg["gae_lambda"]),
            )
            advantages = normalize_advantages(advantages, mask).detach()

        old_parts.append(old_logp)
        ref_parts.append(ref_logp)
        value_parts.append(values)
        response_parts.append(response)
        sequence_parts.append((full, attention))
        prompt_widths.append(len(rebuilt["prompt_ids"]))
        details.append({"prompt_id": cached.get("prompt_id"), "source_index": cached.get("source_index")})

    # The default diagnostic uses one row because variable-length rows require
    # padding and a block-diagonal sequence batch. Keeping one row makes every
    # tensor and mask explicit and avoids introducing another batching effect.
    if len(rows) != 1:
        raise ValueError("Use exactly one cached row per replay invocation.")

    return {
        "sequence": sequence_parts[0][0],
        "attention": sequence_parts[0][1],
        "response": response_parts[0],
        "prompt_width": prompt_widths[0],
        "old_logp": old_parts[0].unsqueeze(0),
        "ref_logp": ref_parts[0].unsqueeze(0),
        "advantages": advantages,
        "mask": torch.ones_like(advantages),
        "details": details[0],
    }


def _run_epsilon(policy, inputs, epsilon: float) -> tuple[dict, torch.Tensor]:
    policy.zero_grad(set_to_none=True)
    new_logp, _ = response_token_logprobs(
        policy,
        inputs["sequence"],
        inputs["attention"],
        inputs["prompt_width"],
        inputs["response"],
    )
    loss, ratio, clip_fraction = ppo_policy_loss(
        new_logp=new_logp,
        old_logp=inputs["old_logp"],
        advantage=inputs["advantages"],
        mask=inputs["mask"],
        eps=epsilon,
    )
    clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    unclipped = ratio * inputs["advantages"]
    clipped = clipped_ratio * inputs["advantages"]
    loss.backward()
    gradient = _grad_vector(policy)
    pre_clip_norm = float(torch.linalg.vector_norm(gradient).item())
    return {
        "epsilon": epsilon,
        "ratio_min": float(ratio.min().detach().item()),
        "ratio_max": float(ratio.max().detach().item()),
        "ratio_quantiles": _quantiles(ratio.reshape(-1)),
        "outside_005": float((((ratio < 0.95) | (ratio > 1.05)).float().mean()).item()),
        "outside_020": float((((ratio < 0.80) | (ratio > 1.20)).float().mean()).item()),
        "outside_050": float((((ratio < 0.50) | (ratio > 1.50)).float().mean()).item()),
        "configured_clip_fraction": float(clip_fraction.item()),
        "unclipped_policy_loss": float((-unclipped.mean()).detach().item()),
        "clipped_policy_loss": float((-clipped.mean()).detach().item()),
        "final_policy_loss": float(loss.detach().item()),
        "gradient_norm_before_clip": pre_clip_norm,
    }, gradient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo.yaml")
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.05, 0.20, 0.50])
    parser.add_argument(
        "--inactive-control",
        action="store_true",
        help="Use one fixed midpoint-policy log-probability tensor as old_logp.",
    )
    args = parser.parse_args()
    if args.rows != 1:
        raise ValueError("This isolated replay intentionally accepts --rows 1 only.")

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    cached = load_cached_rollouts(cfg["cached_rollouts"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    policy.eval()
    inputs = _fixed_inputs(cfg, cached[:1], tokenizer, policy)
    output_name = "fixed_rollout_replay.json"

    if args.inactive_control:
        # Construct a control in which old_logp is the fixed checkpoint's own
        # evaluation log-probability on the same response. This makes the
        # theoretical ratio exactly one; the post-forward check below guards
        # against any unexpected numerical deviation.
        with torch.no_grad():
            midpoint_logp, _ = response_token_logprobs(
                policy,
                inputs["sequence"],
                inputs["attention"],
                inputs["prompt_width"],
                inputs["response"],
            )
        inputs["old_logp"] = midpoint_logp.detach()
        output_name = "inactive_clip_control.json"

    snapshot = _parameter_snapshot(policy)
    rows = []
    gradients = []
    for epsilon in args.epsilons:
        _restore_parameters(policy, snapshot)
        row, gradient = _run_epsilon(policy, inputs, float(epsilon))
        rows.append(row)
        gradients.append(gradient)

    if args.inactive_control and any(
        row["outside_005"] != 0.0 for row in rows
    ):
        raise RuntimeError(
            "Inactive control violated the narrowest epsilon boundary."
        )

    pairwise = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = gradients[i], gradients[j]
            pairwise.append({
                "epsilon_a": rows[i]["epsilon"],
                "epsilon_b": rows[j]["epsilon"],
                "max_abs_gradient_difference": float((a - b).abs().max().item()),
                "relative_l2_difference": float((torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a).clamp_min(1e-12)).item()),
                "cosine_similarity": float(torch.nn.functional.cosine_similarity(a, b, dim=0).item()),
            })

    result = {
        "config": args.config,
        "cache": cfg["cached_rollouts"],
        "row": inputs["details"],
        "control": bool(args.inactive_control),
        "note": "No optimizer.step() and no gradient clipping were called.",
        "epsilon_results": rows,
        "pairwise_gradient_comparison": pairwise,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / output_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
