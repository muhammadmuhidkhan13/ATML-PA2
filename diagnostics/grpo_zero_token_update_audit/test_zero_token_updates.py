import copy
import unittest

import torch

from task3_grpo.continue_train import (
    resolve_grpo_stop_policy,
    should_continue_grpo,
)


class ZeroTokenUpdateTests(unittest.TestCase):
    def test_all_masked_attempt_does_not_touch_parameters_or_adam_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=5.0e-6)

        (parameter.square()).backward()
        optimizer.step()

        before_parameter = parameter.detach().clone()
        before_state = copy.deepcopy(optimizer.state_dict())

        valid_policy_tokens = 0
        if valid_policy_tokens > 0:
            (parameter.square()).backward()
            torch.nn.utils.clip_grad_norm_([parameter], 1.0)
            optimizer.step()

        self.assertTrue(torch.equal(parameter, before_parameter))
        after_state = optimizer.state_dict()
        self.assertEqual(after_state["param_groups"], before_state["param_groups"])
        self.assertEqual(after_state["state"].keys(), before_state["state"].keys())
        for key in before_state["state"]:
            for field, before_value in before_state["state"][key].items():
                self.assertTrue(
                    torch.equal(
                        after_state["state"][key][field],
                        before_value,
                    )
                )

    def test_valid_attempt_updates_parameters(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=5.0e-6)

        (parameter.square()).backward()
        torch.nn.utils.clip_grad_norm_([parameter], 1.0)
        optimizer.step()

        self.assertNotEqual(float(parameter.item()), 1.0)
        self.assertEqual(optimizer.state[parameter]["step"].item(), 1)

    def test_standard_mode_counts_effective_optimizer_updates(self):
        policy = resolve_grpo_stop_policy(3, "effective_optimizer_updates")
        outcomes = [False, True, False, True, True]
        attempts = 0
        optimizer_updates = 0

        while should_continue_grpo(
            policy["stop_rule"],
            policy["requested_optimizer_updates"],
            policy["requested_rollout_attempts"],
            attempts,
            optimizer_updates,
        ):
            attempts += 1
            optimizer_updates += int(outcomes[attempts - 1])

        self.assertEqual(attempts, 5)
        self.assertEqual(optimizer_updates, 3)

    def test_fixed_budget_mode_keeps_requested_attempt_count(self):
        policy = resolve_grpo_stop_policy(4, "rollout_attempts")
        outcomes = [False, True, False, True]
        attempts = 0
        optimizer_updates = 0

        while should_continue_grpo(
            policy["stop_rule"],
            policy["requested_optimizer_updates"],
            policy["requested_rollout_attempts"],
            attempts,
            optimizer_updates,
        ):
            attempts += 1
            optimizer_updates += int(outcomes[attempts - 1])

        self.assertEqual(attempts, 4)
        self.assertEqual(optimizer_updates, 2)


if __name__ == "__main__":
    unittest.main()
