from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import (
    load_yaml,
    prompt_messages,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import (
    save_json,
    set_seed,
    wall_timer,
)
from common.metrics import (
    masked_mean,
    sample_entropy,
    sampled_kl,
)
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)


def load_evaluation_bundle(
    config_path: str,
    adapter: str,
    load_reward: bool = True,
):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    bundle = {
        "cfg": cfg,
        "rows": read_jsonl(
            cfg["paths"][
                "rl_prompt_eval"
            ]
        ),
        "tokenizer": load_tokenizer(
            cfg["base_model"]
        ),
        "policy": load_policy(
            cfg,
            adapter_path=adapter,
            trainable=False,
        ),
        "reward_model": None,
        "reward_tokenizer": None,
    }

    if load_reward:
        (
            reward_model,
            reward_tokenizer,
        ) = load_reward_model(cfg)

        bundle[
            "reward_model"
        ] = reward_model

        bundle[
            "reward_tokenizer"
        ] = reward_tokenizer

    return bundle


def _mean(values):
    if not values:
        return None

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=float,
            )
        )
    )


def _std(values):
    if not values:
        return None

    return float(
        np.std(
            np.asarray(
                values,
                dtype=float,
            )
        )
    )


def _iqr(values):
    if not values:
        return None

    array = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.percentile(array, 75)
        - np.percentile(array, 25)
    )


def run_evaluation(
    config_path: str,
    adapter: str,
    name: str,
    max_examples: int | None = None,
    skip_reward: bool = False,
    batch_size: int | None = None,
):
    """Evaluate a frozen PPO policy on the held-out prompt set."""

    timer = wall_timer()

    bundle = load_evaluation_bundle(
        config_path=config_path,
        adapter=adapter,
        load_reward=not skip_reward,
    )

    cfg = bundle["cfg"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    policy = bundle["policy"]
    reward_model = bundle[
        "reward_model"
    ]
    reward_tokenizer = bundle[
        "reward_tokenizer"
    ]

    if max_examples is not None:
        if int(max_examples) < 1:
            raise ValueError(
                "max_examples must be at least 1."
            )

        rows = rows[
            : int(max_examples)
        ]

    if not rows:
        raise ValueError(
            "The PPO evaluation dataset is empty."
        )

    selected_batch_size = int(
        batch_size
        if batch_size is not None
        else cfg.get(
            "evaluation_batch_size",
            1,
        )
    )

    if selected_batch_size < 1:
        raise ValueError(
            "Evaluation batch size must be at least 1."
        )

    results_dir = repo_path(
        cfg["results_dir"]
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        results_dir
        / f"{name}_metrics.json"
    )

    generations_path = (
        results_dir
        / f"{name}_generations.jsonl"
    )

    qualitative_path = (
        results_dir
        / f"{name}_qualitative_candidates.json"
    )

    policy.eval()

    policy_device = next(
        policy.parameters()
    ).device

    generation_cfg = cfg.get(
        "generation",
        {},
    )

    records = []

    total_kl_sum = 0.0
    total_entropy_sum = 0.0
    total_response_tokens = 0

    for start in range(
        0,
        len(rows),
        selected_batch_size,
    ):
        batch_rows = rows[
            start:
            start + selected_batch_size
        ]

        prompts = [
            prompt_messages(row)
            for row in batch_rows
        ]

        generated = batch_generate(
            model=policy,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_length=int(
                cfg[
                    "max_prompt_length"
                ]
            ),
            max_new_tokens=int(
                cfg[
                    "eval_max_response_length"
                ]
            ),
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

        sequences = generated[
            "sequences"
        ].to(policy_device)

        attention_mask = generated[
            "attention_mask"
        ].to(policy_device)

        response_ids = generated[
            "response_ids"
        ].to(policy_device)

        response_mask = generated[
            "response_mask"
        ].to(policy_device)

        prompt_width = int(
            generated[
                "prompt_width"
            ]
        )

        with torch.no_grad():
            policy_logp, _ = (
                response_token_logprobs(
                    policy,
                    sequences,
                    attention_mask,
                    prompt_width,
                    response_ids,
                )
            )

            with reference_mode(
                policy
            ):
                reference_logp, _ = (
                    response_token_logprobs(
                        policy,
                        sequences,
                        attention_mask,
                        prompt_width,
                        response_ids,
                    )
                )

        batch_kl = sampled_kl(
            policy_logp,
            reference_logp,
            response_mask,
        )

        batch_entropy = sample_entropy(
            policy_logp,
            response_mask,
        )

        valid_token_count = int(
            response_mask.sum().item()
        )

        total_kl_sum += (
            float(
                batch_kl.item()
            )
            * valid_token_count
        )

        total_entropy_sum += (
            float(
                batch_entropy.item()
            )
            * valid_token_count
        )

        total_response_tokens += (
            valid_token_count
        )

        if skip_reward:
            reward_scores = [
                None
                for _ in batch_rows
            ]
        else:
            scored = score_reward_pairs(
                rm_model=reward_model,
                rm_tokenizer=(
                    reward_tokenizer
                ),
                prompts=prompts,
                responses=generated[
                    "responses"
                ],
                max_length=int(
                    cfg[
                        "reward_max_length"
                    ]
                ),
            )

            reward_scores = [
                float(value)
                for value in (
                    scored.detach()
                    .float()
                    .cpu()
                    .tolist()
                )
            ]

        for local_index, row in enumerate(
            batch_rows
        ):
            token_mask = response_mask[
                local_index
            ]

            token_count = float(
                token_mask.sum().item()
            )

            per_example_kl = float(
                (
                    (
                        policy_logp[
                            local_index
                        ]
                        - reference_logp[
                            local_index
                        ]
                    )
                    * token_mask
                ).sum().item()
                / max(1.0, token_count)
            )

            per_example_entropy = float(
                (
                    -policy_logp[
                        local_index
                    ]
                    * token_mask
                ).sum().item()
                / max(1.0, token_count)
            )

            record = {
                "name": name,
                "evaluation_index": (
                    start + local_index
                ),
                "prompt_id": row.get(
                    "prompt_id"
                ),
                "source_index": row.get(
                    "source_index"
                ),
                "prompt_messages": (
                    prompts[
                        local_index
                    ]
                ),
                "response": generated[
                    "responses"
                ][local_index],
                "reward_model_score": (
                    reward_scores[
                        local_index
                    ]
                ),
                "sampled_kl": (
                    per_example_kl
                ),
                "entropy": (
                    per_example_entropy
                ),
                "response_length": int(
                    generated[
                        "response_lengths"
                    ][local_index]
                ),
                "terminated_with_eos": bool(
                    generated[
                        "terminated_with_eos"
                    ][local_index]
                ),
                "truncated": bool(
                    generated[
                        "truncated"
                    ][local_index]
                ),
            }

            records.append(record)

        completed = min(
            start
            + selected_batch_size,
            len(rows),
        )

        print(
            f"[{name}] evaluated "
            f"{completed}/{len(rows)} prompts",
            flush=True,
        )

        del sequences
        del attention_mask
        del response_ids
        del response_mask
        del policy_logp
        del reference_logp

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(
        generations_path,
        records,
    )

    reward_values = [
        row["reward_model_score"]
        for row in records
        if row[
            "reward_model_score"
        ] is not None
    ]

    response_lengths = [
        row["response_length"]
        for row in records
    ]

    terminated_values = [
        float(
            row[
                "terminated_with_eos"
            ]
        )
        for row in records
    ]

    truncated_values = [
        float(
            row["truncated"]
        )
        for row in records
    ]

    if total_response_tokens > 0:
        sampled_kl_mean = float(
            total_kl_sum
            / total_response_tokens
        )

        entropy_mean = float(
            total_entropy_sum
            / total_response_tokens
        )
    else:
        sampled_kl_mean = None
        entropy_mean = None

    elapsed_seconds = float(
        timer()
    )

    metrics = {
        "name": name,
        "config": config_path,
        "adapter": adapter,
        "dataset": cfg[
            "paths"
        ]["rl_prompt_eval"],
        "seed": int(cfg["seed"]),
        "num_evaluation_prompts": len(
            rows
        ),
        "evaluation_batch_size": (
            selected_batch_size
        ),
        "max_prompt_length": int(
            cfg["max_prompt_length"]
        ),
        "max_response_length": int(
            cfg[
                "eval_max_response_length"
            ]
        ),
        "generation": {
            "temperature": float(
                generation_cfg.get(
                    "temperature",
                    0.7,
                )
            ),
            "top_p": float(
                generation_cfg.get(
                    "top_p",
                    0.9,
                )
            ),
            "do_sample": bool(
                generation_cfg.get(
                    "do_sample",
                    True,
                )
            ),
        },
        "sampled_kl_token_mean": (
            sampled_kl_mean
        ),
        "entropy_token_mean": (
            entropy_mean
        ),
        "generated_token_count": (
            total_response_tokens
        ),
        "reward_model_score_mean": (
            _mean(reward_values)
        ),
        "reward_model_score_std": (
            _std(reward_values)
        ),
        "response_length_mean": (
            _mean(response_lengths)
        ),
        "response_length_std": (
            _std(response_lengths)
        ),
        "response_length_iqr": (
            _iqr(response_lengths)
        ),
        "terminated_with_eos_rate": (
            _mean(
                terminated_values
            )
        ),
        "truncation_rate": (
            _mean(
                truncated_values
            )
        ),
        "reward_scoring_skipped": bool(
            skip_reward
        ),
        "wall_clock_seconds": (
            elapsed_seconds
        ),
        "generations": str(
            generations_path.relative_to(
                repo_path(".")
            )
        ),
        "qualitative_candidates": str(
            qualitative_path.relative_to(
                repo_path(".")
            )
        ),
    }

    save_json(
        metrics_path,
        metrics,
    )

    if reward_values:
        ranked = sorted(
            records,
            key=lambda row: (
                row[
                    "reward_model_score"
                ]
            ),
        )

        qualitative = {
            "lowest_reward": (
                ranked[:3]
            ),
            "highest_reward": (
                ranked[-3:][::-1]
            ),
            "largest_positive_kl": sorted(
                records,
                key=lambda row: (
                    row[
                        "sampled_kl"
                    ]
                ),
                reverse=True,
            )[:3],
            "longest_responses": sorted(
                records,
                key=lambda row: (
                    row[
                        "response_length"
                    ]
                ),
                reverse=True,
            )[:3],
        }
    else:
        qualitative = {
            "largest_positive_kl": sorted(
                records,
                key=lambda row: (
                    row[
                        "sampled_kl"
                    ]
                ),
                reverse=True,
            )[:3],
            "longest_responses": sorted(
                records,
                key=lambda row: (
                    row[
                        "response_length"
                    ]
                ),
                reverse=True,
            )[:3],
        }

    save_json(
        qualitative_path,
        qualitative,
    )

    print(
        f"Evaluated prompts: "
        f"{len(rows)}",
        flush=True,
    )

    print(
        "Sampled KL per generated token: "
        f"{sampled_kl_mean}",
        flush=True,
    )

    print(
        "Sampled entropy per generated token: "
        f"{entropy_mean}",
        flush=True,
    )

    print(
        "Mean reward-model score: "
        f"{_mean(reward_values)}",
        flush=True,
    )

    print(
        "Mean response length: "
        f"{_mean(response_lengths)}",
        flush=True,
    )

    print(
        f"Saved evaluation metrics to "
        f"{metrics_path}",
        flush=True,
    )

    print(
        f"Saved generations to "
        f"{generations_path}",
        flush=True,
    )

    print(
        f"Saved qualitative candidates to "
        f"{qualitative_path}",
        flush=True,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen PPO policy on the "
            "fixed held-out prompt set."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/ppo.yaml",
    )

    parser.add_argument(
        "--adapter",
        required=True,
    )

    parser.add_argument(
        "--name",
        default="standard",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
    )

    parser.add_argument(
        "--skip-reward",
        action="store_true",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
    )

    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        adapter=args.adapter,
        name=args.name,
        max_examples=(
            args.max_examples
        ),
        skip_reward=(
            args.skip_reward
        ),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()