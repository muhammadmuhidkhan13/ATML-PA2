from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path

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
    append_jsonl,
    save_json,
    set_seed,
    wall_timer,
)
from common.metrics import (
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
    """Compatibility helper for loading a PPO evaluation bundle."""

    cfg = load_yaml(
        config_path
    )

    set_seed(
        int(
            cfg["seed"]
        )
    )

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
        ) = load_reward_model(
            cfg
        )

        bundle[
            "reward_model"
        ] = reward_model

        bundle[
            "reward_tokenizer"
        ] = reward_tokenizer

    return bundle


def _mean(
    values,
):
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


def _std(
    values,
):
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


def _iqr(
    values,
):
    if not values:
        return None

    array = np.asarray(
        values,
        dtype=float,
    )

    return float(
        np.percentile(
            array,
            75,
        )
        - np.percentile(
            array,
            25,
        )
    )


def _release_cuda():
    """Release unused Python and CUDA allocations."""

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_rng_state(
    path: Path,
    next_evaluation_index: int,
):
    """Save generation RNG state for exact resumption."""

    state = {
        "next_evaluation_index": int(
            next_evaluation_index
        ),
        "python_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_state": torch.get_rng_state(),
        "cuda_states": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    torch.save(
        state,
        path,
    )


def _restore_rng_state(
    path: Path,
    expected_index: int,
):
    """Restore generation RNG state after model loading."""

    if not path.exists():
        raise RuntimeError(
            "A partial generation file exists, but its RNG "
            f"resume state is missing: {path}. Remove the "
            "partial generations file and rerun the evaluation."
        )

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    saved_index = int(
        state[
            "next_evaluation_index"
        ]
    )

    if saved_index != int(
        expected_index
    ):
        raise RuntimeError(
            "Generation cache and RNG state disagree: "
            f"cached rows={expected_index}, "
            f"RNG next index={saved_index}."
        )

    random.setstate(
        state[
            "python_state"
        ]
    )

    np.random.set_state(
        state[
            "numpy_state"
        ]
    )

    torch.set_rng_state(
        state[
            "torch_state"
        ]
    )

    cuda_states = state.get(
        "cuda_states"
    )

    if (
        cuda_states is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            cuda_states
        )


def _validate_cached_records(
    records: list[dict],
    expected_rows: list[dict],
    name: str,
):
    """Require cached rows to be a contiguous dataset prefix."""

    if len(records) > len(
        expected_rows
    ):
        raise ValueError(
            f"{name} generation cache has more rows than "
            "the selected evaluation dataset."
        )

    for index, record in enumerate(
        records
    ):
        cached_index = int(
            record.get(
                "evaluation_index",
                -1,
            )
        )

        if cached_index != index:
            raise ValueError(
                "Generation cache is not a contiguous prefix: "
                f"row {index} has evaluation_index="
                f"{cached_index}."
            )

        expected_row = expected_rows[
            index
        ]

        expected_prompt_id = (
            expected_row.get(
                "prompt_id"
            )
        )

        expected_source_index = (
            expected_row.get(
                "source_index"
            )
        )

        if (
            record.get(
                "prompt_id"
            )
            != expected_prompt_id
            or record.get(
                "source_index"
            )
            != expected_source_index
        ):
            raise ValueError(
                "Generation cache does not match the selected "
                f"evaluation dataset at index {index}."
            )


def _generate_policy_records(
    cfg: dict,
    rows: list[dict],
    adapter: str,
    name: str,
    generations_path: Path,
    rng_state_path: Path,
    selected_batch_size: int,
) -> list[dict]:
    """Generate and incrementally save policy responses."""

    if generations_path.exists():
        records = read_jsonl(
            generations_path
        )
    else:
        records = []

    _validate_cached_records(
        records,
        rows,
        name,
    )

    completed = len(
        records
    )

    if completed >= len(
        rows
    ):
        if rng_state_path.exists():
            rng_state_path.unlink()

        print(
            f"[{name}] generation cache already contains "
            f"{completed}/{len(rows)} prompts",
            flush=True,
        )

        return records

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    policy = load_policy(
        cfg,
        adapter_path=adapter,
        trainable=False,
    )

    policy.eval()

    policy_device = next(
        parameter.device
        for parameter in policy.parameters()
        if parameter.device.type
        != "meta"
    )

    if completed > 0:
        _restore_rng_state(
            rng_state_path,
            completed,
        )

        print(
            f"[{name}] resuming generation from "
            f"{completed}/{len(rows)} prompts",
            flush=True,
        )
    else:
        set_seed(
            int(
                cfg["seed"]
            )
        )

        if rng_state_path.exists():
            rng_state_path.unlink()

    generation_cfg = cfg.get(
        "generation",
        {},
    )

    for start in range(
        completed,
        len(rows),
        selected_batch_size,
    ):
        batch_rows = rows[
            start:
            start
            + selected_batch_size
        ]

        prompts = [
            prompt_messages(
                row
            )
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
        ].to(
            policy_device
        )

        attention_mask = generated[
            "attention_mask"
        ].to(
            policy_device
        )

        response_ids = generated[
            "response_ids"
        ].to(
            policy_device
        )

        response_mask = generated[
            "response_mask"
        ].to(
            policy_device
        )

        prompt_width = int(
            generated[
                "prompt_width"
            ]
        )

        with torch.no_grad():
            (
                policy_logp,
                policy_logits,
            ) = response_token_logprobs(
                policy,
                sequences,
                attention_mask,
                prompt_width,
                response_ids,
            )

        # Logits are very large and are not needed after
        # extracting the sampled-token log probabilities.
        del policy_logits

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            with reference_mode(
                policy
            ):
                (
                    reference_logp,
                    reference_logits,
                ) = response_token_logprobs(
                    policy,
                    sequences,
                    attention_mask,
                    prompt_width,
                    response_ids,
                )

        del reference_logits

        # Calculate these for validation and consistency with
        # the original implementation. Per-example values are
        # stored below and aggregated after resumption.
        batch_kl = sampled_kl(
            policy_logp,
            reference_logp,
            response_mask,
        )

        batch_entropy = sample_entropy(
            policy_logp,
            response_mask,
        )

        new_records = []

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
                )
                .sum()
                .item()
                / max(
                    1.0,
                    token_count,
                )
            )

            per_example_entropy = float(
                (
                    -policy_logp[
                        local_index
                    ]
                    * token_mask
                )
                .sum()
                .item()
                / max(
                    1.0,
                    token_count,
                )
            )

            record = {
                "name": name,
                "evaluation_index": int(
                    start
                    + local_index
                ),
                "prompt_id": row.get(
                    "prompt_id"
                ),
                "source_index": row.get(
                    "source_index"
                ),
                "prompt_messages": prompts[
                    local_index
                ],
                "response": generated[
                    "responses"
                ][
                    local_index
                ],
                "reward_model_score": None,
                "sampled_kl": (
                    per_example_kl
                ),
                "entropy": (
                    per_example_entropy
                ),
                "response_length": int(
                    generated[
                        "response_lengths"
                    ][
                        local_index
                    ]
                ),
                "terminated_with_eos": bool(
                    generated[
                        "terminated_with_eos"
                    ][
                        local_index
                    ]
                ),
                "truncated": bool(
                    generated[
                        "truncated"
                    ][
                        local_index
                    ]
                ),
            }

            append_jsonl(
                generations_path,
                record,
            )

            records.append(
                record
            )

            new_records.append(
                record
            )

        next_index = (
            start
            + len(
                batch_rows
            )
        )

        _save_rng_state(
            rng_state_path,
            next_index,
        )

        print(
            f"[{name}] generated "
            f"{next_index}/{len(rows)} prompts",
            flush=True,
        )

        del new_records
        del sequences
        del attention_mask
        del response_ids
        del response_mask
        del policy_logp
        del reference_logp
        del batch_kl
        del batch_entropy
        del generated
        del prompts

        _release_cuda()

    del policy
    del tokenizer

    _release_cuda()

    if rng_state_path.exists():
        rng_state_path.unlink()

    return records


def _score_rewards_separately(
    cfg: dict,
    records: list[dict],
    generations_path: Path,
    name: str,
    selected_batch_size: int,
) -> list[dict]:
    """Score cached responses after completely unloading policy."""

    missing_indices = [
        index
        for index, record in enumerate(
            records
        )
        if record.get(
            "reward_model_score"
        )
        is None
    ]

    if not missing_indices:
        print(
            f"[{name}] all responses already have "
            "reward-model scores",
            flush=True,
        )

        return records

    (
        reward_model,
        reward_tokenizer,
    ) = load_reward_model(
        cfg
    )

    total_missing = len(
        missing_indices
    )

    for offset in range(
        0,
        total_missing,
        selected_batch_size,
    ):
        selected_indices = (
            missing_indices[
                offset:
                offset
                + selected_batch_size
            ]
        )

        prompts = [
            records[
                index
            ][
                "prompt_messages"
            ]
            for index in selected_indices
        ]

        responses = [
            records[
                index
            ][
                "response"
            ]
            for index in selected_indices
        ]

        scored = score_reward_pairs(
            rm_model=reward_model,
            rm_tokenizer=(
                reward_tokenizer
            ),
            prompts=prompts,
            responses=responses,
            max_length=int(
                cfg[
                    "reward_max_length"
                ]
            ),
        )

        scores = (
            scored.detach()
            .float()
            .cpu()
            .tolist()
        )

        for index, score in zip(
            selected_indices,
            scores,
        ):
            records[
                index
            ][
                "reward_model_score"
            ] = float(
                score
            )

        # Rewrite the small 200-row cache after each reward
        # batch so reward scoring is also resumable.
        write_jsonl(
            generations_path,
            records,
        )

        finished = min(
            offset
            + len(
                selected_indices
            ),
            total_missing,
        )

        print(
            f"[{name}] reward-scored "
            f"{finished}/{total_missing} pending responses",
            flush=True,
        )

        del scored
        del scores
        del prompts
        del responses

        _release_cuda()

    del reward_model
    del reward_tokenizer

    _release_cuda()

    return records


def _build_qualitative_candidates(
    records: list[dict],
) -> dict:
    reward_values = [
        record[
            "reward_model_score"
        ]
        for record in records
        if record.get(
            "reward_model_score"
        )
        is not None
    ]

    qualitative = {
        "largest_positive_kl": sorted(
            records,
            key=lambda row: row[
                "sampled_kl"
            ],
            reverse=True,
        )[:3],
        "longest_responses": sorted(
            records,
            key=lambda row: row[
                "response_length"
            ],
            reverse=True,
        )[:3],
    }

    if reward_values:
        ranked = sorted(
            records,
            key=lambda row: row[
                "reward_model_score"
            ],
        )

        qualitative[
            "lowest_reward"
        ] = ranked[
            :3
        ]

        qualitative[
            "highest_reward"
        ] = ranked[
            -3:
        ][
            ::-1
        ]

    return qualitative


def run_evaluation(
    config_path: str,
    adapter: str,
    name: str,
    max_examples: int | None = None,
    skip_reward: bool = False,
    batch_size: int | None = None,
):
    """Evaluate a frozen PPO policy using resumable phases."""

    timer = wall_timer()

    cfg = load_yaml(
        config_path
    )

    set_seed(
        int(
            cfg["seed"]
        )
    )

    rows = read_jsonl(
        cfg["paths"][
            "rl_prompt_eval"
        ]
    )

    if max_examples is not None:
        if int(
            max_examples
        ) < 1:
            raise ValueError(
                "max_examples must be at least 1."
            )

        rows = rows[
            : int(
                max_examples
            )
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
        cfg[
            "results_dir"
        ]
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
        / (
            f"{name}_"
            "qualitative_candidates.json"
        )
    )

    rng_state_path = (
        results_dir
        / (
            f"{name}_"
            "generation_rng_state.pt"
        )
    )

    records = _generate_policy_records(
        cfg=cfg,
        rows=rows,
        adapter=adapter,
        name=name,
        generations_path=generations_path,
        rng_state_path=rng_state_path,
        selected_batch_size=(
            selected_batch_size
        ),
    )

    if not skip_reward:
        records = (
            _score_rewards_separately(
                cfg=cfg,
                records=records,
                generations_path=(
                    generations_path
                ),
                name=name,
                selected_batch_size=(
                    selected_batch_size
                ),
            )
        )

    reward_values = [
        record[
            "reward_model_score"
        ]
        for record in records
        if record.get(
            "reward_model_score"
        )
        is not None
    ]

    response_lengths = [
        int(
            record[
                "response_length"
            ]
        )
        for record in records
    ]

    terminated_values = [
        float(
            record[
                "terminated_with_eos"
            ]
        )
        for record in records
    ]

    truncated_values = [
        float(
            record[
                "truncated"
            ]
        )
        for record in records
    ]

    total_response_tokens = int(
        sum(
            response_lengths
        )
    )

    if total_response_tokens > 0:
        sampled_kl_mean = float(
            sum(
                float(
                    record[
                        "sampled_kl"
                    ]
                )
                * int(
                    record[
                        "response_length"
                    ]
                )
                for record in records
            )
            / total_response_tokens
        )

        entropy_mean = float(
            sum(
                float(
                    record[
                        "entropy"
                    ]
                )
                * int(
                    record[
                        "response_length"
                    ]
                )
                for record in records
            )
            / total_response_tokens
        )
    else:
        sampled_kl_mean = None
        entropy_mean = None

    generation_cfg = cfg.get(
        "generation",
        {},
    )

    metrics = {
        "name": name,
        "config": config_path,
        "adapter": adapter,
        "dataset": cfg[
            "paths"
        ][
            "rl_prompt_eval"
        ],
        "seed": int(
            cfg["seed"]
        ),
        "num_evaluation_prompts": len(
            rows
        ),
        "evaluation_batch_size": (
            selected_batch_size
        ),
        "max_prompt_length": int(
            cfg[
                "max_prompt_length"
            ]
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
            _mean(
                reward_values
            )
        ),
        "reward_model_score_std": (
            _std(
                reward_values
            )
        ),
        "response_length_mean": (
            _mean(
                response_lengths
            )
        ),
        "response_length_std": (
            _std(
                response_lengths
            )
        ),
        "response_length_iqr": (
            _iqr(
                response_lengths
            )
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
        "wall_clock_seconds": float(
            timer()
        ),
        "resumable_evaluation": True,
        "separate_reward_scoring_phase": (
            not skip_reward
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

    qualitative = (
        _build_qualitative_candidates(
            records
        )
    )

    save_json(
        qualitative_path,
        qualitative,
    )

    print(
        f"Evaluated prompts: {len(rows)}",
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
        "Saved evaluation metrics to "
        f"{metrics_path}",
        flush=True,
    )

    print(
        "Saved generations to "
        f"{generations_path}",
        flush=True,
    )

    print(
        "Saved qualitative candidates to "
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
        max_examples=args.max_examples,
        skip_reward=args.skip_reward,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()