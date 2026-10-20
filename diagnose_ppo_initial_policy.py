from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.models import (
    load_policy,
    load_tokenizer,
    reference_mode,
)


CONDITIONS = {
    "standard_20_updates": {
        "adapter": "outputs/task2_ppo/standard",
        "generations": (
            "results/task2_ppo/"
            "standard_generations.jsonl"
        ),
    },
    "kl_0_8_updates": {
        "adapter": (
            "outputs/task2_ppo/"
            "kl_ablation/kl_0"
        ),
        "generations": (
            "results/task2_ppo/"
            "kl_0_eval_generations.jsonl"
        ),
    },
    "kl_0p1_8_updates": {
        "adapter": (
            "outputs/task2_ppo/"
            "kl_ablation/kl_0p1"
        ),
        "generations": (
            "results/task2_ppo/"
            "kl_0p1_eval_generations.jsonl"
        ),
    },
    "kl_0p2_8_updates": {
        "adapter": (
            "outputs/task2_ppo/"
            "kl_ablation/kl_0p2"
        ),
        "generations": (
            "results/task2_ppo/"
            "kl_0p2_eval_generations.jsonl"
        ),
    },
}


def mean(values):
    if not values:
        return None
    return float(np.mean(values))


def std(values):
    if not values:
        return None
    return float(np.std(values))


def write_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_generation(
    tokenizer,
    record,
    max_prompt_length,
):
    """Reconstruct the exact prompt/response token boundary.

    The response text saved by the evaluator excludes
    special tokens, so EOS is restored only when the
    saved record says generation ended with EOS.
    """

    prompt = record["prompt_messages"]

    rendered = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_encoding = tokenizer(
        rendered,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=True,
    )

    prompt_ids = list(
        prompt_encoding["input_ids"]
    )

    response_ids = list(
        tokenizer(
            record["response"],
            add_special_tokens=False,
        )["input_ids"]
    )

    if record.get("terminated_with_eos"):
        eos_id = tokenizer.eos_token_id

        if eos_id is None:
            raise RuntimeError(
                "Record says EOS termination occurred, "
                "but tokenizer has no EOS token."
            )

        response_ids.append(eos_id)

    expected_length = int(
        record["response_length"]
    )

    if len(response_ids) != expected_length:
        raise RuntimeError(
            "Saved response could not be reconstructed "
            "exactly.\n"
            f"evaluation_index="
            f"{record.get('evaluation_index')}\n"
            f"saved response_length={expected_length}\n"
            f"reconstructed tokens={len(response_ids)}\n"
            "The diagnostic has stopped rather than "
            "silently using mismatched tokens."
        )

    sequence = torch.tensor(
        [prompt_ids + response_ids],
        dtype=torch.long,
    )

    attention_mask = torch.ones_like(
        sequence,
        dtype=torch.long,
    )

    response_tensor = torch.tensor(
        [response_ids],
        dtype=torch.long,
    )

    return {
        "sequence": sequence,
        "attention_mask": attention_mask,
        "response_ids": response_tensor,
        "prompt_width": len(prompt_ids),
    }


def score_response(
    model,
    tokenized,
):
    device = next(model.parameters()).device

    sequence = tokenized[
        "sequence"
    ].to(device)

    attention_mask = tokenized[
        "attention_mask"
    ].to(device)

    response_ids = tokenized[
        "response_ids"
    ].to(device)

    with torch.no_grad():
        token_logp, _ = response_token_logprobs(
            model=model,
            sequences=sequence,
            attention_mask=attention_mask,
            prompt_width=tokenized[
                "prompt_width"
            ],
            response_ids=response_ids,
        )

    return (
        token_logp[0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )


def summarize_deltas(
    condition,
    records,
    token_deltas_initial,
    token_deltas_base,
    sequence_deltas_initial,
    sequence_deltas_base,
):
    initial_values = np.concatenate(
        token_deltas_initial
    )

    base_values = np.concatenate(
        token_deltas_base
    )

    rewards = [
        float(record["reward_model_score"])
        for record in records
        if record.get(
            "reward_model_score"
        ) is not None
    ]

    lengths = [
        int(record["response_length"])
        for record in records
    ]

    return {
        "condition": condition,
        "num_prompts": len(records),
        "num_response_tokens": int(
            initial_values.size
        ),

        # Candidate-generated sampled estimate of
        # KL(candidate || midpoint).
        "signed_logratio_vs_initial_token_mean":
            float(initial_values.mean()),

        # These reveal changes hidden by cancellation.
        "absolute_logratio_vs_initial_token_mean":
            float(
                np.abs(initial_values).mean()
            ),
        "rms_logratio_vs_initial":
            float(
                np.sqrt(
                    np.mean(
                        initial_values ** 2
                    )
                )
            ),
        "sequence_logratio_vs_initial_mean":
            mean(sequence_deltas_initial),
        "sequence_logratio_vs_initial_std":
            std(sequence_deltas_initial),

        "token_fraction_abs_shift_gt_0p01":
            float(
                np.mean(
                    np.abs(initial_values)
                    > 0.01
                )
            ),
        "token_fraction_abs_shift_gt_0p05":
            float(
                np.mean(
                    np.abs(initial_values)
                    > 0.05
                )
            ),
        "token_fraction_abs_shift_gt_0p10":
            float(
                np.mean(
                    np.abs(initial_values)
                    > 0.10
                )
            ),

        # Recreates the comparison concept used by
        # the existing evaluator: candidate vs base.
        "signed_logratio_vs_base_token_mean":
            float(base_values.mean()),
        "absolute_logratio_vs_base_token_mean":
            float(
                np.abs(base_values).mean()
            ),
        "rms_logratio_vs_base":
            float(
                np.sqrt(
                    np.mean(
                        base_values ** 2
                    )
                )
            ),
        "sequence_logratio_vs_base_mean":
            mean(sequence_deltas_base),

        "saved_reward_mean":
            mean(rewards),
        "saved_reward_std":
            std(rewards),
        "response_length_mean":
            mean(lengths),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/ppo.yaml",
    )

    parser.add_argument(
        "--number",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "diagnostics/"
            "ppo_initial_policy_comparison"
        ),
    )

    args = parser.parse_args()

    if args.number < 1:
        raise ValueError(
            "--number must be at least one."
        )

    cfg = load_yaml(args.config)

    initial_adapter = cfg["paths"][
        "ppo_midpoint_policy"
    ]

    if not repo_path(
        initial_adapter
    ).exists():
        raise FileNotFoundError(
            repo_path(initial_adapter)
        )

    print(
        "PPO INITIAL-POLICY DIAGNOSTIC"
    )
    print("=" * 88)
    print(
        "Initial policy:",
        initial_adapter,
    )
    print(
        "Prompts requested:",
        args.number,
    )

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    # Load the base model only once with the midpoint
    # adapter as the default PEFT adapter.
    model = load_policy(
        cfg,
        adapter_path=initial_adapter,
        trainable=False,
    )

    adapter_names = {
        "initial": "default",
    }

    for index, (
        condition,
        specification,
    ) in enumerate(
        CONDITIONS.items(),
        start=1,
    ):
        adapter_path = repo_path(
            specification["adapter"]
        )

        if not adapter_path.exists():
            raise FileNotFoundError(
                adapter_path
            )

        adapter_name = f"candidate_{index}"

        model.load_adapter(
            str(adapter_path),
            adapter_name=adapter_name,
            is_trainable=False,
        )

        adapter_names[
            condition
        ] = adapter_name

    selected_records = {}

    for condition, specification in (
        CONDITIONS.items()
    ):
        path = repo_path(
            specification["generations"]
        )

        if not path.exists():
            raise FileNotFoundError(path)

        rows = read_jsonl(path)

        rows = sorted(
            rows,
            key=lambda row: int(
                row["evaluation_index"]
            ),
        )[: args.number]

        if len(rows) != args.number:
            raise RuntimeError(
                f"{condition}: expected "
                f"{args.number} records, "
                f"found {len(rows)}."
            )

        selected_records[
            condition
        ] = rows

    # Verify that every comparison uses the same
    # held-out contexts.
    first_condition = next(
        iter(selected_records)
    )

    reference_keys = [
        (
            int(row["evaluation_index"]),
            str(row.get("prompt_id")),
            json.dumps(
                row["prompt_messages"],
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        for row in selected_records[
            first_condition
        ]
    ]

    for condition, rows in (
        selected_records.items()
    ):
        current_keys = [
            (
                int(row["evaluation_index"]),
                str(row.get("prompt_id")),
                json.dumps(
                    row["prompt_messages"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            for row in rows
        ]

        if current_keys != reference_keys:
            raise RuntimeError(
                f"{condition}: held-out prompt "
                "sequence does not match."
            )

    print(
        "Matched evaluation contexts: PASS"
    )

    output_dir = repo_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detailed_rows = []
    summary_rows = []

    for condition, records in (
        selected_records.items()
    ):
        print()
        print(
            f"Scoring {condition}: "
            f"{len(records)} responses"
        )

        candidate_name = adapter_names[
            condition
        ]

        token_deltas_initial = []
        token_deltas_base = []
        sequence_deltas_initial = []
        sequence_deltas_base = []

        for row_number, record in enumerate(
            records,
            start=1,
        ):
            tokenized = reconstruct_generation(
                tokenizer=tokenizer,
                record=record,
                max_prompt_length=int(
                    cfg["max_prompt_length"]
                ),
            )

            model.set_adapter(
                candidate_name
            )
            candidate_logp = score_response(
                model,
                tokenized,
            )

            model.set_adapter(
                adapter_names["initial"]
            )
            initial_logp = score_response(
                model,
                tokenized,
            )

            model.set_adapter(
                candidate_name
            )

            with reference_mode(model):
                base_logp = score_response(
                    model,
                    tokenized,
                )

            delta_initial = (
                candidate_logp
                - initial_logp
            )

            delta_base = (
                candidate_logp
                - base_logp
            )

            token_deltas_initial.append(
                delta_initial
            )
            token_deltas_base.append(
                delta_base
            )

            sequence_initial = float(
                delta_initial.sum()
            )

            sequence_base = float(
                delta_base.sum()
            )

            sequence_deltas_initial.append(
                sequence_initial
            )
            sequence_deltas_base.append(
                sequence_base
            )

            detailed_rows.append(
                {
                    "condition": condition,
                    "evaluation_index": int(
                        record[
                            "evaluation_index"
                        ]
                    ),
                    "prompt_id": record.get(
                        "prompt_id"
                    ),
                    "response_tokens": int(
                        record[
                            "response_length"
                        ]
                    ),
                    "reward_model_score":
                        record.get(
                            "reward_model_score"
                        ),
                    "saved_sampled_kl":
                        record.get(
                            "sampled_kl"
                        ),
                    "signed_logratio_vs_initial_token_mean":
                        float(
                            delta_initial.mean()
                        ),
                    "absolute_logratio_vs_initial_token_mean":
                        float(
                            np.abs(
                                delta_initial
                            ).mean()
                        ),
                    "rms_logratio_vs_initial":
                        float(
                            np.sqrt(
                                np.mean(
                                    delta_initial
                                    ** 2
                                )
                            )
                        ),
                    "sequence_logratio_vs_initial":
                        sequence_initial,
                    "signed_logratio_vs_base_token_mean":
                        float(
                            delta_base.mean()
                        ),
                    "absolute_logratio_vs_base_token_mean":
                        float(
                            np.abs(
                                delta_base
                            ).mean()
                        ),
                    "sequence_logratio_vs_base":
                        sequence_base,
                }
            )

            if (
                row_number % 10 == 0
                or row_number == len(records)
            ):
                print(
                    f"  {row_number}/"
                    f"{len(records)}",
                    flush=True,
                )

        summary_rows.append(
            summarize_deltas(
                condition=condition,
                records=records,
                token_deltas_initial=(
                    token_deltas_initial
                ),
                token_deltas_base=(
                    token_deltas_base
                ),
                sequence_deltas_initial=(
                    sequence_deltas_initial
                ),
                sequence_deltas_base=(
                    sequence_deltas_base
                ),
            )
        )

    csv_path = (
        output_dir
        / "ppo_initial_policy_summary.csv"
    )

    details_path = (
        output_dir
        / "ppo_initial_policy_details.csv"
    )

    json_path = (
        output_dir
        / "ppo_initial_policy_summary.json"
    )

    write_csv(
        csv_path,
        summary_rows,
    )

    write_csv(
        details_path,
        detailed_rows,
    )

    json_path.write_text(
        json.dumps(
            {
                "config": args.config,
                "initial_policy":
                    initial_adapter,
                "num_shared_prompts":
                    args.number,
                "sampling_basis": (
                    "Each condition's saved "
                    "generated responses."
                ),
                "reference_note": (
                    "The existing evaluator's "
                    "reference_mode disables the "
                    "LoRA adapter and therefore "
                    "uses the bare base model. "
                    "This diagnostic separately "
                    "compares every trained adapter "
                    "with the PPO midpoint adapter."
                ),
                "conditions": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "SUMMARY"
    )
    print("=" * 88)

    for row in summary_rows:
        print(
            f"{row['condition']:24s} | "
            f"KL→initial="
            f"{row['signed_logratio_vs_initial_token_mean']:.8f} | "
            f"|shift|="
            f"{row['absolute_logratio_vs_initial_token_mean']:.8f} | "
            f"RMS="
            f"{row['rms_logratio_vs_initial']:.8f} | "
            f"reward="
            f"{row['saved_reward_mean']:.4f}"
        )

    print()
    print(
        "Saved:",
        csv_path,
    )
    print(
        "Saved:",
        details_path,
    )
    print(
        "Saved:",
        json_path,
    )
    print()
    print(
        "PPO INITIAL-POLICY DIAGNOSTIC PASSED"
    )


if __name__ == "__main__":
    main()
