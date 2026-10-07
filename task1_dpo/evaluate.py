from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common.data import (
    filter_overlength_preference_rows,
    load_yaml,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import save_json, set_seed, wall_timer
from common.metrics import sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)
from task1_dpo.train import make_collate


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def _finite_or_none(value: float):
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values: list[float]):
    return float(np.mean(values)) if values else None


def _std(values: list[float]):
    return float(np.std(values)) if values else None


def _iqr(values: list[float]):
    if not values:
        return None

    q25, q75 = np.percentile(values, [25, 75])
    return float(q75 - q25)


def load_evaluation_bundle(
    config_path: str,
    adapter: str,
    dataset_path: str | None = None,
    max_examples: int | None = None,
):
    cfg = load_yaml(config_path)

    selected_dataset = (
        dataset_path
        or cfg["paths"]["dpo_standard_eval"]
    )

    raw_rows = read_jsonl(selected_dataset)

    if not raw_rows:
        raise ValueError(
            "The selected DPO evaluation dataset is empty."
        )

    if (
        max_examples is not None
        and int(max_examples) < 1
    ):
        raise ValueError(
            "max_examples must be at least 1 when provided."
        )

    tokenizer = load_tokenizer(cfg["base_model"])

    max_sequence_length = int(
        cfg["max_sequence_length"]
    )

    eligible_rows, excluded_rows = (
        filter_overlength_preference_rows(
            raw_rows,
            tokenizer,
            max_sequence_length,
        )
    )

    if not eligible_rows:
        raise ValueError(
            "No DPO evaluation examples remain after filtering "
            "prompts that do not fit inside "
            f"max_sequence_length={max_sequence_length}."
        )

    rows = eligible_rows

    if max_examples is not None:
        rows = rows[: int(max_examples)]

    print(
        "DPO evaluation preprocessing: "
        f"raw={len(raw_rows)}, "
        f"eligible={len(eligible_rows)}, "
        f"excluded={len(excluded_rows)}, "
        f"selected={len(rows)}, "
        f"max_sequence_length={max_sequence_length}"
    )

    policy = load_policy(
        cfg,
        adapter_path=adapter,
        trainable=False,
    )

    return {
        "cfg": cfg,
        "rows": rows,
        "raw_num_rows": len(raw_rows),
        "eligible_num_rows": len(eligible_rows),
        "excluded_rows": excluded_rows,
        "requested_max_examples": (
            None
            if max_examples is None
            else int(max_examples)
        ),
        "dataset_path": selected_dataset,
        "tokenizer": tokenizer,
        "policy": policy,
    }


@torch.inference_mode()
def evaluate_preference_pairs(
    policy,
    tokenizer,
    rows: list[dict],
    batch_size: int,
    max_sequence_length: int,
    beta: float,
):
    loader = DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(
            tokenizer,
            max_sequence_length,
        ),
    )

    device = next(policy.parameters()).device

    details: list[dict] = []
    offset = 0

    for chosen_batch, rejected_batch in tqdm(
        loader,
        desc="Held-out preference pairs",
    ):
        chosen_batch = _move_batch(
            chosen_batch,
            device,
        )

        rejected_batch = _move_batch(
            rejected_batch,
            device,
        )

        policy_chosen, _, _ = (
            response_sequence_logprobs(
                policy,
                chosen_batch,
            )
        )

        policy_rejected, _, _ = (
            response_sequence_logprobs(
                policy,
                rejected_batch,
            )
        )

        with reference_mode(policy):
            ref_chosen, _, _ = (
                response_sequence_logprobs(
                    policy,
                    chosen_batch,
                )
            )

            ref_rejected, _, _ = (
                response_sequence_logprobs(
                    policy,
                    rejected_batch,
                )
            )

        policy_margin = (
            policy_chosen
            - policy_rejected
        )

        ref_margin = (
            ref_chosen
            - ref_rejected
        )

        relative_margin = (
            policy_margin
            - ref_margin
        )

        logits = beta * relative_margin
        losses = -F.logsigmoid(logits)

        batch_n = int(policy_chosen.shape[0])

        for j in range(batch_n):
            row = rows[offset + j]

            details.append(
                {
                    "prompt_id":
                        row.get("prompt_id"),

                    "source_index":
                        row.get("source_index"),

                    "policy_chosen_logp":
                        float(
                            policy_chosen[j].item()
                        ),

                    "policy_rejected_logp":
                        float(
                            policy_rejected[j].item()
                        ),

                    "reference_chosen_logp":
                        float(
                            ref_chosen[j].item()
                        ),

                    "reference_rejected_logp":
                        float(
                            ref_rejected[j].item()
                        ),

                    "policy_margin":
                        float(
                            policy_margin[j].item()
                        ),

                    "reference_margin":
                        float(
                            ref_margin[j].item()
                        ),

                    "relative_margin":
                        float(
                            relative_margin[j].item()
                        ),

                    "dpo_logit":
                        float(
                            logits[j].item()
                        ),

                    "dpo_loss":
                        float(
                            losses[j].item()
                        ),

                    "preference_correct":
                        bool(
                            relative_margin[j].item()
                            > 0
                        ),
                }
            )

        offset += batch_n

    losses = [
        row["dpo_loss"]
        for row in details
    ]

    relative_margins = [
        row["relative_margin"]
        for row in details
    ]

    summary = {
        "num_pairs":
            len(details),

        "dpo_loss":
            _mean(losses),

        "preference_accuracy":
            _mean(
                [
                    float(
                        row[
                            "preference_correct"
                        ]
                    )
                    for row in details
                ]
            ),

        "relative_margin_mean":
            _mean(relative_margins),

        "relative_margin_std":
            _std(relative_margins),
    }

    return summary, details


@torch.inference_mode()
def generate_and_measure_kl(
    policy,
    tokenizer,
    rows: list[dict],
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    generation_cfg: dict,
):
    records: list[dict] = []

    kl_weighted_sum = 0.0
    kl_token_count = 0.0

    for start in tqdm(
        range(0, len(rows), batch_size),
        desc="Generate and measure KL",
    ):
        batch_rows = rows[
            start:
            start + batch_size
        ]

        prompts = [
            prompt_messages_from_preference(
                row
            )
            for row in batch_rows
        ]

        generated = batch_generate(
            policy,
            tokenizer,
            prompts,
            max_prompt_length=(
                max_prompt_length
            ),
            max_new_tokens=max_new_tokens,
            temperature=float(
                generation_cfg.get(
                    "temperature",
                    0.7,
                )
            ),
            top_p=float(
                generation_cfg.get(
                    "top_p",
                    0.9,
                )
            ),
            do_sample=bool(
                generation_cfg.get(
                    "do_sample",
                    True,
                )
            ),
        )

        policy_logp, _ = (
            response_token_logprobs(
                policy,
                generated["sequences"],
                generated["attention_mask"],
                generated["prompt_width"],
                generated["response_ids"],
            )
        )

        with reference_mode(policy):
            reference_logp, _ = (
                response_token_logprobs(
                    policy,
                    generated["sequences"],
                    generated[
                        "attention_mask"
                    ],
                    generated[
                        "prompt_width"
                    ],
                    generated[
                        "response_ids"
                    ],
                )
            )

        mask = generated["response_mask"]

        valid_tokens = float(
            mask.sum().item()
        )

        batch_kl = float(
            sampled_kl(
                policy_logp,
                reference_logp,
                mask,
            ).item()
        )

        kl_weighted_sum += (
            batch_kl
            * valid_tokens
        )

        kl_token_count += valid_tokens

        token_differences = (
            policy_logp
            - reference_logp
        ) * mask

        per_response_kl = (
            token_differences.sum(-1)
            / mask.sum(-1).clamp_min(1.0)
        )

        for j, row in enumerate(batch_rows):
            chosen, rejected = (
                preference_responses(row)
            )

            records.append(
                {
                    "prompt_id":
                        row.get("prompt_id"),

                    "source_index":
                        row.get("source_index"),

                    "prompt_messages":
                        prompts[j],

                    "prompt":
                        row.get("prompt", ""),

                    "chosen":
                        chosen,

                    "rejected":
                        rejected,

                    "response":
                        generated[
                            "responses"
                        ][j],

                    "response_length_tokens":
                        int(
                            generated[
                                "response_lengths"
                            ][j]
                        ),

                    "terminated_with_eos":
                        bool(
                            generated[
                                "terminated_with_eos"
                            ][j]
                        ),

                    "truncated":
                        bool(
                            generated[
                                "truncated"
                            ][j]
                        ),

                    "sampled_kl_token_mean":
                        float(
                            per_response_kl[
                                j
                            ].item()
                        ),
                }
            )

        del (
            generated,
            policy_logp,
            reference_logp,
            mask,
            token_differences,
        )

    lengths = [
        float(
            row[
                "response_length_tokens"
            ]
        )
        for row in records
    ]

    summary = {
        "num_generations":
            len(records),

        "sampled_kl_token_mean":
            (
                kl_weighted_sum
                / kl_token_count
                if kl_token_count
                else None
            ),

        "generated_token_count":
            int(kl_token_count),

        "response_length_mean":
            _mean(lengths),

        "response_length_std":
            _std(lengths),

        "response_length_iqr":
            _iqr(lengths),

        "terminated_with_eos_rate":
            _mean(
                [
                    float(
                        row[
                            "terminated_with_eos"
                        ]
                    )
                    for row in records
                ]
            ),

        "truncation_rate":
            _mean(
                [
                    float(
                        row["truncated"]
                    )
                    for row in records
                ]
            ),
    }

    return summary, records


@torch.inference_mode()
def add_reward_scores(
    cfg: dict,
    generation_records: list[dict],
    batch_size: int,
):
    reward_model, reward_tokenizer = (
        load_reward_model(cfg)
    )

    scores: list[float] = []

    for start in tqdm(
        range(
            0,
            len(generation_records),
            batch_size,
        ),
        desc="Reward-model scoring",
    ):
        batch = generation_records[
            start:
            start + batch_size
        ]

        batch_scores = score_reward_pairs(
            reward_model,
            reward_tokenizer,
            [
                row["prompt_messages"]
                for row in batch
            ],
            [
                row["response"]
                for row in batch
            ],
            max_length=1024,
        )

        values = [
            float(value)
            for value
            in batch_scores
            .detach()
            .cpu()
            .tolist()
        ]

        scores.extend(values)

        for row, score in zip(
            batch,
            values,
        ):
            row[
                "reward_model_score"
            ] = score

    del reward_model
    del reward_tokenizer
    clear_gpu()

    return {
        "reward_model_score_mean":
            _mean(scores),

        "reward_model_score_std":
            _std(scores),
    }


def select_qualitative_candidates(
    records: list[dict],
    count: int = 3,
):
    if not records:
        return []

    candidates: list[dict] = []
    seen: set[tuple] = set()

    def add(
        label: str,
        ordered: list[dict],
    ):
        added = 0

        for row in ordered:
            key = (
                row.get("prompt_id"),
                row.get("source_index"),
            )

            if key in seen:
                continue

            seen.add(key)

            candidates.append(
                {
                    "selection_reason":
                        label,
                    **row,
                }
            )

            added += 1

            if added >= count:
                break

    reward_rows = [
        row
        for row in records
        if "reward_model_score" in row
    ]

    if reward_rows:
        add(
            "highest_reward_model_score",
            sorted(
                reward_rows,
                key=lambda row:
                    row[
                        "reward_model_score"
                    ],
                reverse=True,
            ),
        )

        add(
            "lowest_reward_model_score",
            sorted(
                reward_rows,
                key=lambda row:
                    row[
                        "reward_model_score"
                    ],
            ),
        )

    preference_rows = [
        row
        for row in records
        if "heldout_relative_margin" in row
    ]

    if preference_rows:
        add(
            "strongest_reference_adjusted_preference",
            sorted(
                preference_rows,
                key=lambda row:
                    row[
                        "heldout_relative_margin"
                    ],
                reverse=True,
            ),
        )

    add(
        "longest_generated_response",
        sorted(
            records,
            key=lambda row:
                row[
                    "response_length_tokens"
                ],
            reverse=True,
        ),
    )

    add(
        "largest_absolute_sampled_kl",
        sorted(
            records,
            key=lambda row:
                abs(
                    row[
                        "sampled_kl_token_mean"
                    ]
                ),
            reverse=True,
        ),
    )

    return candidates


def run_evaluation(
    config_path: str,
    adapter: str,
    name: str,
    dataset_path: str | None = None,
    max_examples: int | None = None,
    skip_reward: bool = False,
):
    timer = wall_timer()

    bundle = load_evaluation_bundle(
        config_path,
        adapter,
        dataset_path=dataset_path,
        max_examples=max_examples,
    )

    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    raw_num_rows = bundle["raw_num_rows"]
    eligible_num_rows = bundle["eligible_num_rows"]
    excluded_rows = bundle["excluded_rows"]
    requested_max_examples = bundle[
        "requested_max_examples"
    ]

    set_seed(int(cfg["seed"]))

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preference_path = (
        results_dir
        / f"{name}_preference_details.jsonl"
    )

    generations_path = (
        results_dir
        / f"{name}_generations.jsonl"
    )

    qualitative_path = (
        results_dir
        / f"{name}_qualitative_candidates.json"
    )

    metrics_path = (
        results_dir
        / f"{name}_metrics.json"
    )

    preprocessing_manifest_path = (
        results_dir
        / f"{name}_preprocessing_manifest.json"
    )

    preprocessing_manifest = {
        "name": name,
        "dataset": str(bundle["dataset_path"]),
        "rule": (
            "Preserve the complete prompt. Exclude an example when the "
            "prompt plus generation prefix alone has at least "
            "max_sequence_length tokens. For retained preference pairs, "
            "truncate chosen and rejected responses from the right and "
            "preserve EOS."
        ),
        "filter_order": "filter_overlength_before_max_examples",
        "max_sequence_length": int(
            cfg["max_sequence_length"]
        ),
        "raw_num_rows": raw_num_rows,
        "eligible_num_rows": eligible_num_rows,
        "excluded_num_rows": len(excluded_rows),
        "requested_max_examples": requested_max_examples,
        "selected_num_rows": len(rows),
        "selected_examples": [
            {
                "selected_index": selected_index,
                "prompt_id": row.get("prompt_id"),
                "source_index": row.get("source_index"),
            }
            for selected_index, row in enumerate(rows)
        ],
        "excluded_examples": excluded_rows,
    }

    save_json(
        preprocessing_manifest_path,
        preprocessing_manifest,
    )

    (
        preference_summary,
        preference_details,
    ) = evaluate_preference_pairs(
        policy,
        tokenizer,
        rows,
        batch_size=int(
            cfg["batch_size"]
        ),
        max_sequence_length=int(
            cfg["max_sequence_length"]
        ),
        beta=float(cfg["beta"]),
    )

    write_jsonl(
        preference_path,
        preference_details,
    )

    (
        generation_summary,
        generation_records,
    ) = generate_and_measure_kl(
        policy,
        tokenizer,
        rows,
        batch_size=int(
            cfg["batch_size"]
        ),
        max_prompt_length=int(
            cfg["max_sequence_length"]
        ),
        max_new_tokens=int(
            cfg["max_generation_tokens"]
        ),
        generation_cfg=cfg.get(
            "generation",
            {},
        ),
    )

    preference_by_id = {
        (
            row.get("prompt_id"),
            row.get("source_index"),
        ): row
        for row in preference_details
    }

    for row in generation_records:
        preference = preference_by_id.get(
            (
                row.get("prompt_id"),
                row.get("source_index"),
            )
        )

        if preference is not None:
            row[
                "heldout_relative_margin"
            ] = preference[
                "relative_margin"
            ]

            row[
                "heldout_preference_correct"
            ] = preference[
                "preference_correct"
            ]

            row[
                "heldout_dpo_loss"
            ] = preference[
                "dpo_loss"
            ]

    del policy
    clear_gpu()

    reward_summary = {
        "reward_model_score_mean":
            None,

        "reward_model_score_std":
            None,
    }

    if not skip_reward:
        reward_summary = (
            add_reward_scores(
                cfg,
                generation_records,
                batch_size=int(
                    cfg["batch_size"]
                ),
            )
        )

    write_jsonl(
        generations_path,
        generation_records,
    )

    qualitative = (
        select_qualitative_candidates(
            generation_records
        )
    )

    save_json(
        qualitative_path,
        qualitative,
    )

    summary = {
        "name":
            name,

        "config":
            str(config_path),

        "adapter":
            str(adapter),

        "dataset":
            str(bundle["dataset_path"]),

        "seed":
            int(cfg["seed"]),

        "beta":
            float(cfg["beta"]),

        "num_evaluation_rows":
            len(rows),

        "raw_num_evaluation_rows":
            raw_num_rows,

        "eligible_num_evaluation_rows":
            eligible_num_rows,

        "excluded_overlength_rows":
            len(excluded_rows),

        "requested_max_examples":
            requested_max_examples,

        "preprocessing_rule":
            "preserve_prompt_filter_prompt_overflow",

        "preprocessing_manifest":
            str(
                preprocessing_manifest_path.relative_to(
                    repo_path(".")
                )
            ),

        "max_sequence_length":
            int(
                cfg[
                    "max_sequence_length"
                ]
            ),

        "max_generation_tokens":
            int(
                cfg[
                    "max_generation_tokens"
                ]
            ),

        "generation":
            dict(
                cfg.get(
                    "generation",
                    {},
                )
            ),

        **preference_summary,
        **generation_summary,
        **reward_summary,

        "wall_clock_seconds":
            float(timer()),

        "preference_details":
            str(
                preference_path.relative_to(
                    repo_path(".")
                )
            ),

        "generations":
            str(
                generations_path.relative_to(
                    repo_path(".")
                )
            ),

        "qualitative_candidates":
            str(
                qualitative_path.relative_to(
                    repo_path(".")
                )
            ),
    }

    summary = {
        key: (
            _finite_or_none(value)
            if isinstance(value, float)
            else value
        )
        for key, value
        in summary.items()
    }

    save_json(
        metrics_path,
        summary,
    )

    print(
        "Held-out DPO loss: "
        f"{summary['dpo_loss']:.4f}"
    )

    print(
        "Held-out preference accuracy: "
        f"{summary['preference_accuracy']:.4f}"
    )

    print(
        "Sampled KL per generated token: "
        f"{summary['sampled_kl_token_mean']:.4f}"
    )

    if (
        summary[
            "reward_model_score_mean"
        ]
        is not None
    ):
        print(
            "Mean reward-model score: "
            f"{summary['reward_model_score_mean']:.4f}"
        )

    print(
        "Generated response length: "
        f"mean="
        f"{summary['response_length_mean']:.2f}, "
        f"std="
        f"{summary['response_length_std']:.2f}, "
        f"IQR="
        f"{summary['response_length_iqr']:.2f}"
    )

    print(
        "Saved evaluation metrics to "
        f"{metrics_path}"
    )

    print(
        "Saved preprocessing manifest to "
        f"{preprocessing_manifest_path}"
    )

    return summary


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default="configs/dpo.yaml",
    )

    ap.add_argument(
        "--adapter",
        required=True,
    )

    ap.add_argument(
        "--name",
        default="standard",
    )

    ap.add_argument(
        "--dataset",
    )

    ap.add_argument(
        "--max-examples",
        type=int,
    )

    ap.add_argument(
        "--skip-reward",
        action="store_true",
    )

    args = ap.parse_args()

    run_evaluation(
        args.config,
        args.adapter,
        args.name,
        dataset_path=args.dataset,
        max_examples=args.max_examples,
        skip_reward=args.skip_reward,
    )


if __name__ == "__main__":
    main()
