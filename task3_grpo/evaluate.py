from __future__ import annotations

import argparse
from statistics import mean, pstdev

import numpy as np
import torch

from common.data import (
    load_yaml,
    prompt_messages,
    read_jsonl,
    repo_path,
)
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import (
    append_jsonl,
    save_json,
    set_seed,
    wall_timer,
)
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)


def _safe_mean(values):
    if not values:
        return None
    return float(mean(values))


def _safe_std(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def _safe_iqr(values):
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


def _prompt_id(row, fallback_index):
    value = row.get("prompt_id")

    if value is None:
        return f"row_{fallback_index}"

    return str(value)


def _source_index(row, fallback_index):
    value = row.get("source_index")

    if value is None:
        return int(fallback_index)

    return int(value)


def _evaluation_generation_cap(cfg):
    return int(
        cfg.get(
            "evaluation_max_response_length",
            cfg.get(
                "cache_generation_cap",
                cfg["max_completion_length"],
            ),
        )
    )


def load_evaluation_bundle(
    config_path: str,
    adapter: str,
    load_reward: bool = True,
):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    policy = load_policy(
        cfg,
        adapter_path=adapter,
        trainable=False,
    )

    reward_model = None
    reward_tokenizer = None

    if load_reward:
        (
            reward_model,
            reward_tokenizer,
        ) = load_reward_model(cfg)

    rows = read_jsonl(
        cfg["paths"]["rl_prompt_eval"]
    )

    if not rows:
        raise ValueError(
            "The GRPO evaluation prompt set is empty."
        )

    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
    }


def _evaluate_one_prompt(
    policy,
    tokenizer,
    reward_model,
    reward_tokenizer,
    row,
    row_index,
    cfg,
    skip_reward,
):
    messages = prompt_messages(row)

    max_response_length = (
        _evaluation_generation_cap(cfg)
    )

    generated = batch_generate(
        model=policy,
        tokenizer=tokenizer,
        prompts=[messages],
        max_prompt_length=int(
            cfg["max_prompt_length"]
        ),
        max_new_tokens=max_response_length,
        temperature=float(
            cfg.get(
                "temperature",
                cfg.get(
                    "generation_temperature",
                    0.7,
                ),
            )
        ),
        top_p=float(
            cfg.get(
                "top_p",
                cfg.get(
                    "generation_top_p",
                    0.9,
                ),
            )
        ),
        do_sample=bool(
            cfg.get("do_sample", True)
        ),
    )

    sequences = generated["sequences"]
    attention_mask = generated[
        "attention_mask"
    ]
    prompt_width = int(
        generated["prompt_width"]
    )
    response_ids = generated[
        "response_ids"
    ]
    response_mask = generated[
        "response_mask"
    ]

    with torch.no_grad():
        policy_logp, _ = (
            response_token_logprobs(
                model=policy,
                sequences=sequences,
                attention_mask=attention_mask,
                prompt_width=prompt_width,
                response_ids=response_ids,
            )
        )

        with reference_mode(policy):
            reference_logp, _ = (
                response_token_logprobs(
                    model=policy,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    prompt_width=prompt_width,
                    response_ids=response_ids,
                )
            )

    valid_tokens = float(
        response_mask.sum().item()
    )

    if valid_tokens > 0:
        sampled_kl = float(
            (
                (
                    policy_logp
                    - reference_logp
                )
                * response_mask
            ).sum().item()
            / valid_tokens
        )

        sampled_entropy = float(
            (
                -policy_logp
                * response_mask
            ).sum().item()
            / valid_tokens
        )
    else:
        sampled_kl = 0.0
        sampled_entropy = 0.0

    response = str(
        generated["responses"][0]
    )

    reward_score = None

    if not skip_reward:
        reward = score_reward_pairs(
            rm_model=reward_model,
            rm_tokenizer=reward_tokenizer,
            prompts=[messages],
            responses=[response],
            max_length=int(
                cfg.get(
                    "reward_max_length",
                    1024,
                )
            ),
        )

        reward_score = float(
            reward[0].item()
        )

    return {
        "prompt_id": _prompt_id(
            row,
            row_index,
        ),
        "source_index": _source_index(
            row,
            row_index,
        ),
        "prompt_messages": messages,
        "response": response,
        "reward_model_score": (
            reward_score
        ),
        "sampled_kl": sampled_kl,
        "sampled_entropy": (
            sampled_entropy
        ),
        "response_length": int(
            generated[
                "response_lengths"
            ][0]
        ),
        "terminated_with_eos": bool(
            generated[
                "terminated_with_eos"
            ][0]
        ),
        "truncated": bool(
            generated["truncated"][0]
        ),
        "generated_token_count": int(
            valid_tokens
        ),
    }


def _select_qualitative_candidates(
    records,
    count=3,
):
    candidates = {}

    reward_records = [
        record
        for record in records
        if record[
            "reward_model_score"
        ] is not None
    ]

    if reward_records:
        candidates["highest_reward"] = sorted(
            reward_records,
            key=lambda item: item[
                "reward_model_score"
            ],
            reverse=True,
        )[:count]

        candidates["lowest_reward"] = sorted(
            reward_records,
            key=lambda item: item[
                "reward_model_score"
            ],
        )[:count]

    candidates[
        "largest_absolute_sampled_kl"
    ] = sorted(
        records,
        key=lambda item: abs(
            item["sampled_kl"]
        ),
        reverse=True,
    )[:count]

    candidates["longest_responses"] = sorted(
        records,
        key=lambda item: item[
            "response_length"
        ],
        reverse=True,
    )[:count]

    candidates["shortest_responses"] = sorted(
        records,
        key=lambda item: item[
            "response_length"
        ],
    )[:count]

    candidates["truncated_responses"] = [
        record
        for record in records
        if record["truncated"]
    ][:count]

    return candidates


def run_evaluation(
    config_path: str,
    adapter: str,
    name: str = "standard",
    max_examples: int | None = None,
    skip_reward: bool = False,
    batch_size: int | None = None,
):
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

    raw_num_rows = len(rows)

    if max_examples is not None:
        if max_examples < 1:
            raise ValueError(
                "max_examples must be positive"
            )

        rows = rows[: int(max_examples)]

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
            "batch_size must be positive"
        )

    # Evaluation is intentionally processed one prompt
    # at a time for predictable memory usage on small GPUs.
    # The value is still recorded for protocol transparency.
    effective_batch_size = 1

    results_dir = repo_path(
        cfg["results_dir"]
    )
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generations_path = (
        results_dir
        / f"{name}_generations.jsonl"
    )

    metrics_path = (
        results_dir
        / f"{name}_metrics.json"
    )

    qualitative_path = (
        results_dir
        / f"{name}_qualitative_candidates.json"
    )

    generations_path.write_text(
        "",
        encoding="utf-8",
    )

    device = next(
        policy.parameters()
    ).device

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(
            device
        )

    timer = wall_timer()
    records = []

    policy.eval()

    for row_index, row in enumerate(rows):
        record = _evaluate_one_prompt(
            policy=policy,
            tokenizer=tokenizer,
            reward_model=reward_model,
            reward_tokenizer=(
                reward_tokenizer
            ),
            row=row,
            row_index=row_index,
            cfg=cfg,
            skip_reward=skip_reward,
        )

        record = {
            "name": name,
            **record,
        }

        records.append(record)

        append_jsonl(
            generations_path,
            record,
        )

        completed = row_index + 1

        if (
            completed == len(rows)
            or completed % 10 == 0
        ):
            print(
                f"[{name}] evaluated "
                f"{completed}/{len(rows)} prompts",
                flush=True,
            )

    response_lengths = [
        float(record["response_length"])
        for record in records
    ]

    rewards = [
        float(
            record["reward_model_score"]
        )
        for record in records
        if record[
            "reward_model_score"
        ] is not None
    ]

    total_valid_tokens = sum(
        int(
            record[
                "generated_token_count"
            ]
        )
        for record in records
    )

    if total_valid_tokens > 0:
        sampled_kl_token_mean = float(
            sum(
                record["sampled_kl"]
                * record[
                    "generated_token_count"
                ]
                for record in records
            )
            / total_valid_tokens
        )

        entropy_token_mean = float(
            sum(
                record[
                    "sampled_entropy"
                ]
                * record[
                    "generated_token_count"
                ]
                for record in records
            )
            / total_valid_tokens
        )
    else:
        sampled_kl_token_mean = 0.0
        entropy_token_mean = 0.0

    eos_rate = _safe_mean(
        [
            float(
                record[
                    "terminated_with_eos"
                ]
            )
            for record in records
        ]
    )

    truncation_rate = _safe_mean(
        [
            float(record["truncated"])
            for record in records
        ]
    )

    elapsed_seconds = float(timer())

    peak_vram_gib = None

    if torch.cuda.is_available():
        peak_vram_gib = float(
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024 ** 3)
        )

    qualitative_candidates = (
        _select_qualitative_candidates(
            records
        )
    )

    save_json(
        qualitative_path,
        qualitative_candidates,
    )

    metrics = {
        "name": name,
        "config": config_path,
        "adapter": adapter,
        "dataset": str(
            cfg["paths"]["rl_prompt_eval"]
        ),
        "seed": int(cfg["seed"]),
        "raw_num_evaluation_prompts": (
            raw_num_rows
        ),
        "num_evaluation_prompts": len(
            records
        ),
        "requested_max_examples": (
            max_examples
        ),
        "requested_batch_size": (
            selected_batch_size
        ),
        "effective_batch_size": (
            effective_batch_size
        ),
        "max_prompt_length": int(
            cfg["max_prompt_length"]
        ),
        "max_response_length": (
            _evaluation_generation_cap(cfg)
        ),
        "generation": {
            "temperature": float(
                cfg.get(
                    "temperature",
                    cfg.get(
                        "generation_temperature",
                        0.7,
                    ),
                )
            ),
            "top_p": float(
                cfg.get(
                    "top_p",
                    cfg.get(
                        "generation_top_p",
                        0.9,
                    ),
                )
            ),
            "do_sample": bool(
                cfg.get(
                    "do_sample",
                    True,
                )
            ),
        },
        "sampled_kl_token_mean": (
            sampled_kl_token_mean
        ),
        "entropy_token_mean": (
            entropy_token_mean
        ),
        "generated_token_count": int(
            total_valid_tokens
        ),
        "reward_model_score_mean": (
            _safe_mean(rewards)
        ),
        "reward_model_score_std": (
            _safe_std(rewards)
        ),
        "response_length_mean": (
            _safe_mean(response_lengths)
        ),
        "response_length_std": (
            _safe_std(response_lengths)
        ),
        "response_length_iqr": (
            _safe_iqr(response_lengths)
        ),
        "terminated_with_eos_rate": (
            eos_rate
        ),
        "truncation_rate": (
            truncation_rate
        ),
        "reward_scoring_skipped": bool(
            skip_reward
        ),
        "wall_clock_seconds": (
            elapsed_seconds
        ),
        "peak_vram_gib": peak_vram_gib,
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

    print(
        f"Evaluated prompts: "
        f"{len(records)}"
    )
    print(
        "Sampled KL per generated token: "
        f"{sampled_kl_token_mean}"
    )
    print(
        "Sampled entropy per generated token: "
        f"{entropy_token_mean}"
    )
    print(
        "Mean reward-model score: "
        f"{metrics['reward_model_score_mean']}"
    )
    print(
        "Mean response length: "
        f"{metrics['response_length_mean']}"
    )
    print(
        "Saved evaluation metrics to "
        f"{metrics_path}"
    )
    print(
        "Saved generations to "
        f"{generations_path}"
    )
    print(
        "Saved qualitative candidates to "
        f"{qualitative_path}"
    )

    return metrics


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen GRPO policy on the "
            "fixed held-out prompt set."
        )
    )

    ap.add_argument(
        "--config",
        default="configs/grpo.yaml",
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
        "--max-examples",
        type=int,
    )

    ap.add_argument(
        "--skip-reward",
        action="store_true",
    )

    ap.add_argument(
        "--batch-size",
        type=int,
    )

    args = ap.parse_args()

    run_evaluation(
        config_path=args.config,
        adapter=args.adapter,
        name=args.name,
        max_examples=args.max_examples,
        skip_reward=args.skip_reward,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()