"""CPU-only optimizer and GRPO all-masked controls.

This file is diagnostic-only. It does not load a model, call the training
loop, or write into results/, checkpoints/, outputs/, or source files.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from task3_grpo.grpo import (
    grpo_policy_loss,
    mask_truncated_sequences,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "diagnostics" / "grpo_zero_token_update_audit" / "control_results.json"

LR = 5.0e-6
BETAS = (0.9, 0.999)
EPS = 1.0e-8
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0


def seed_everything(seed: int = 6304) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(parameter: torch.nn.Parameter):
    return AdamW(
        [parameter],
        lr=LR,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )


def clone_state(optimizer, parameter):
    state = optimizer.state.get(parameter, {})
    def tensor_value(value):
        return None if value is None else value.detach().cpu().clone().tolist()
    return {
        "step": tensor_value(state.get("step")),
        "exp_avg": tensor_value(state.get("exp_avg")),
        "exp_avg_sq": tensor_value(state.get("exp_avg_sq")),
    }


def state_change(before, after):
    def flat(value):
        return None if value is None else torch.tensor(value, dtype=torch.float64)
    changes = {}
    for key in ("step", "exp_avg", "exp_avg_sq"):
        a, b = flat(before[key]), flat(after[key])
        if a is None or b is None:
            changes[key] = None
        else:
            changes[key] = float(torch.linalg.vector_norm(b - a).item())
    return changes


def run_step(parameter, optimizer):
    before_parameter = parameter.detach().cpu().clone()
    before_state = clone_state(optimizer, parameter)
    optimizer.step()
    after_parameter = parameter.detach().cpu().clone()
    after_state = clone_state(optimizer, parameter)
    delta = after_parameter - before_parameter
    return {
        "parameter_before": before_parameter.tolist(),
        "parameter_after": after_parameter.tolist(),
        "parameter_l2_change": float(torch.linalg.vector_norm(delta).item()),
        "max_abs_parameter_change": float(delta.abs().max().item()),
        "optimizer_step_before": before_state["step"],
        "optimizer_step_after": after_state["step"],
        "first_moment_change": state_change(before_state, after_state)["exp_avg"],
        "second_moment_change": state_change(before_state, after_state)["exp_avg_sq"],
        "optimizer_state_before": before_state,
        "optimizer_state_after": after_state,
        "weight_decay_contributed": bool(WEIGHT_DECAY != 0.0),
    }


def fresh_explicit_zero():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = make_optimizer(parameter)
    optimizer.zero_grad(set_to_none=True)
    (parameter * 0.0).sum().backward()
    return run_step(parameter, optimizer) | {
        "case": "fresh_optimizer_explicit_zero_gradient",
        "gradient_kind": "explicit_zero_tensor",
        "momentum_contributed": False,
    }


def fresh_none():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = make_optimizer(parameter)
    optimizer.zero_grad(set_to_none=True)
    return run_step(parameter, optimizer) | {
        "case": "fresh_optimizer_grad_none",
        "gradient_kind": "none",
        "momentum_contributed": False,
    }


def established(explicit_zero: bool):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = make_optimizer(parameter)
    optimizer.zero_grad(set_to_none=True)
    (parameter * torch.tensor([2.0, -3.0])).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if explicit_zero:
        (parameter * 0.0).sum().backward()
    result = run_step(parameter, optimizer)
    result.update({
        "case": "established_momentum_explicit_zero_gradient" if explicit_zero else "established_momentum_grad_none",
        "gradient_kind": "explicit_zero_tensor" if explicit_zero else "none",
        "momentum_contributed": bool(explicit_zero),
        "weight_decay_contributed": False,
    })
    return result


def production_mask_test():
    results = []
    normal_mask = torch.tensor([[1.0, 1.0, 1.0]])
    zero_mask = mask_truncated_sequences(
        normal_mask,
        [True],
    )
    for name, mask in (
        ("nonzero_mask", normal_mask),
        ("all_zero_mask_from_truncation_utility", zero_mask),
    ):
        parameter = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        new_logp = parameter.expand(1, 3)
        old_logp = torch.zeros_like(new_logp)
        ref_logp = torch.zeros_like(new_logp)
        advantage = torch.tensor([1.0])
        loss, diagnostics = grpo_policy_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            seq_adv=advantage,
            token_mask=mask,
            ref_logp=ref_logp,
            eps=0.20,
            beta=0.10,
            loss_type="grpo",
            max_completion_length=3,
        )
        loss.backward()
        results.append({
            "case": name,
            "mask": mask.tolist(),
            "loss": float(loss.detach().item()),
            "loss_requires_grad": bool(loss.requires_grad),
            "gradient_is_none": parameter.grad is None,
            "gradient": None if parameter.grad is None else parameter.grad.detach().tolist(),
            "valid_token_count": diagnostics["valid_token_count"].detach().cpu().tolist(),
            "sampled_kl": float(diagnostics["sampled_kl"].item()),
            "clip_fraction": float(diagnostics["clip_fraction"].item()),
        })
    return results


def main():
    seed_everything()
    result = {
        "device": "cpu",
        "optimizer": {
            "class": "torch.optim.AdamW",
            "learning_rate": LR,
            "betas": list(BETAS),
            "epsilon": EPS,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "zero_grad_set_to_none": True,
            "scheduler": None,
        },
        "optimizer_controls": [
            fresh_explicit_zero(),
            fresh_none(),
            established(True),
            established(False),
        ],
        "production_mask_test": production_mask_test(),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
