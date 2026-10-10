from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.data import (
    load_yaml,
    read_jsonl,
    repo_path,
)
from common.generation import batch_generate
from common.logging_utils import (
    append_jsonl,
    save_json,
    set_seed,
)
from common.models import (
    clear_gpu,
    load_policy,
    load_tokenizer,
)
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import (
    exact_reward,
    extract_designated_final,
)


POLICY_NAMES = (
    "sft",
    "rlvr",
    "rlaif",
)

DATASET_NAMES = (
    "gsm8k",
    "transfer",
)

PAIRWISE_TARGETS = (
    "rlvr",
    "rlaif",
)

_FINAL_AT_END_RE = re.compile(
    r"####\s*[-+]?\d[\d,]*(?:\.\d+)?\s*$"
)


def policy_specs(
    cfg: dict,
) -> dict[str, str | None]:
    """Return the three fixed Task 5 policies."""

    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_specs(
    cfg: dict,
) -> dict[str, dict[str, str]]:
    """Return names and paths for the two Task 5 evaluations."""

    paths = cfg.get(
        "paths",
        {},
    )

    gsm8k_path = paths.get(
        "gsm8k_eval",
        paths.get(
            "gsm_eval",
            "data/gsm8k_eval.jsonl",
        ),
    )

    transfer_path = paths.get(
        "math_transfer_eval",
        "data/math_transfer_eval.jsonl",
    )

    return {
        "gsm8k": {
            "display_name": "GSM8K",
            "path": str(gsm8k_path),
        },
        "transfer": {
            "display_name": "SVAMP",
            "path": str(transfer_path),
        },
    }


def task_results_dir(
    cfg: dict,
) -> Path:
    """Return results/task5_feedback."""

    root = repo_path(
        cfg.get(
            "results_dir",
            "results",
        )
    )

    if root.name == "task5_feedback":
        output_dir = root
    else:
        output_dir = (
            root
            / "task5_feedback"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def source_key(
    row: dict,
) -> str:
    """Return a stable identifier shared across policy outputs."""

    if row.get(
        "prompt_id"
    ) is not None:
        return str(
            row["prompt_id"]
        )

    if row.get(
        "source_index"
    ) is not None:
        return str(
            row["source_index"]
        )

    raise KeyError(
        "Evaluation row has neither prompt_id nor source_index"
    )


def prompt_messages(
    row: dict,
) -> list[dict]:
    messages = row.get(
        "messages"
    )

    if (
        not isinstance(
            messages,
            list,
        )
        or not messages
    ):
        raise ValueError(
            "Expected a non-empty messages list"
        )

    return messages


def read_existing_jsonl(
    path: Path,
) -> list[dict]:
    if not path.exists():
        return []

    return read_jsonl(
        path
    )


def jsonl_index(
    rows: list[dict],
    key_name: str,
) -> dict[str, dict]:
    indexed = {}

    for row in rows:
        key = str(
            row[key_name]
        )

        if key in indexed:
            raise ValueError(
                f"Duplicate {key_name}={key!r} in cached JSONL"
            )

        indexed[key] = row

    return indexed


def is_format_compliant(
    response: str,
) -> bool:
    """Require the designated answer to occur at the end."""

    return bool(
        _FINAL_AT_END_RE.search(
            str(response).strip()
        )
    )


def safe_mean(
    values,
) -> float | None:
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return float(
        np.mean(values)
    )


def safe_std(
    values,
) -> float | None:
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return float(
        np.std(values)
    )


def safe_iqr(
    values,
) -> float | None:
    values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return None

    q25, q75 = np.percentile(
        values,
        [
            25,
            75,
        ],
    )

    return float(
        q75 - q25
    )


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(
                    key
                )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def generation_path(
    output_dir: Path,
    dataset_name: str,
    policy_name: str,
    output_prefix: str,
) -> Path:
    return (
        output_dir
        / (
            f"{output_prefix}_"
            f"{dataset_name}_"
            f"{policy_name}.jsonl"
        )
    )


def pairwise_path(
    output_dir: Path,
    dataset_name: str,
    target_policy: str,
    judge_prefix: str,
) -> Path:
    return (
        output_dir
        / (
            f"{judge_prefix}_"
            f"{dataset_name}_"
            f"{target_policy}_vs_sft.jsonl"
        )
    )


def load_dataset_rows(
    cfg: dict,
    dataset_name: str,
    max_examples: int | None,
) -> tuple[str, str, list[dict]]:
    specs = dataset_specs(
        cfg
    )

    if dataset_name not in specs:
        raise KeyError(
            dataset_name
        )

    spec = specs[
        dataset_name
    ]

    rows = read_jsonl(
        spec["path"]
    )

    if max_examples is not None:
        rows = rows[
            :int(max_examples)
        ]

    return (
        spec["display_name"],
        spec["path"],
        rows,
    )


def generate_policy_responses(
    cfg: dict,
    dataset_name: str,
    dataset_path: str,
    dataset_rows: list[dict],
    policy_name: str,
    adapter_path: str | None,
    output_path: Path,
    batch_size: int,
    overwrite: bool,
) -> dict:
    """Generate or resume deterministic responses for one policy."""

    if overwrite and output_path.exists():
        output_path.unlink()

    existing_rows = read_existing_jsonl(
        output_path
    )

    completed = jsonl_index(
        existing_rows,
        "example_id",
    )

    expected_keys = {
        source_key(row)
        for row in dataset_rows
    }

    unexpected = (
        set(completed)
        - expected_keys
    )

    if unexpected:
        preview = sorted(
            unexpected
        )[:5]

        raise ValueError(
            "Existing generation cache contains rows outside "
            f"the selected dataset prefix: {preview}. "
            "Use --overwrite or a different --output-prefix."
        )

    pending_rows = [
        row
        for row in dataset_rows
        if source_key(row)
        not in completed
    ]

    print(
        f"\n[{dataset_name}/{policy_name}] "
        f"existing={len(completed)}, "
        f"pending={len(pending_rows)}, "
        f"total={len(dataset_rows)}",
        flush=True,
    )

    start_time = time.perf_counter()

    if pending_rows:
        tokenizer = load_tokenizer(
            cfg["base_model"]
        )

        model = load_policy(
            cfg,
            adapter_path=adapter_path,
            trainable=False,
        )

        max_prompt_length = int(
            cfg.get(
                "math_max_prompt_length",
                256,
            )
        )

        max_new_tokens = int(
            cfg.get(
                "math_max_new_tokens",
                512,
            )
        )

        for start in range(
            0,
            len(pending_rows),
            batch_size,
        ):
            chunk = pending_rows[
                start:
                start + batch_size
            ]

            prompts = [
                prompt_messages(row)
                for row in chunk
            ]

            generated = batch_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_prompt_length=max_prompt_length,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )

            for (
                row,
                response,
                response_tokens,
                terminated,
                truncated,
            ) in zip(
                chunk,
                generated["responses"],
                generated["response_lengths"],
                generated["terminated_with_eos"],
                generated["truncated"],
            ):
                gold_final = str(
                    row["gold_final"]
                )

                predicted_final = (
                    extract_designated_final(
                        response
                    )
                )

                record = {
                    "dataset": dataset_name,
                    "dataset_path": dataset_path,
                    "example_id": source_key(
                        row
                    ),
                    "source_index": row.get(
                        "source_index"
                    ),
                    "prompt_id": row.get(
                        "prompt_id"
                    ),
                    "policy": policy_name,
                    "question": str(
                        row["question"]
                    ),
                    "messages": prompt_messages(
                        row
                    ),
                    "gold_final": gold_final,
                    "response": response,
                    "predicted_final": predicted_final,
                    "exact_reward": float(
                        exact_reward(
                            response,
                            gold_final,
                        )
                    ),
                    "format_compliant": bool(
                        is_format_compliant(
                            response
                        )
                    ),
                    "response_tokens": int(
                        response_tokens
                    ),
                    "terminated_with_eos": bool(
                        terminated
                    ),
                    "truncated": bool(
                        truncated
                    ),
                }

                append_jsonl(
                    output_path,
                    record,
                )

                completed[
                    record["example_id"]
                ] = record

            finished = min(
                start + len(chunk),
                len(pending_rows),
            )

            print(
                f"[{dataset_name}/{policy_name}] "
                f"generated {finished}/"
                f"{len(pending_rows)} pending rows",
                flush=True,
            )

        clear_gpu(
            model
        )

        del model
        del tokenizer

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_records = [
        completed[
            source_key(row)
        ]
        for row in dataset_rows
    ]

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"[{dataset_name}/{policy_name}] "
        f"saved {len(selected_records)} rows to {output_path}",
        flush=True,
    )

    return {
        "dataset": dataset_name,
        "policy": policy_name,
        "num_examples": len(
            selected_records
        ),
        "new_examples": len(
            pending_rows
        ),
        "elapsed_seconds": float(
            elapsed
        ),
        "output": str(
            output_path
        ),
    }


def run_generation(
    cfg: dict,
    datasets: list[str],
    policies: list[str],
    max_examples: int | None,
    batch_size: int,
    output_prefix: str,
    overwrite: bool,
) -> list[dict]:
    specs = policy_specs(
        cfg
    )

    output_dir = task_results_dir(
        cfg
    )

    summaries = []

    for dataset_name in datasets:
        (
            _,
            dataset_path,
            rows,
        ) = load_dataset_rows(
            cfg,
            dataset_name,
            max_examples,
        )

        for policy_name in policies:
            summaries.append(
                generate_policy_responses(
                    cfg=cfg,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path,
                    dataset_rows=rows,
                    policy_name=policy_name,
                    adapter_path=specs[
                        policy_name
                    ],
                    output_path=generation_path(
                        output_dir,
                        dataset_name,
                        policy_name,
                        output_prefix,
                    ),
                    batch_size=batch_size,
                    overwrite=overwrite,
                )
            )

    save_json(
        output_dir
        / f"{output_prefix}_generation_summary.json",
        {
            "config_seed": int(
                cfg["seed"]
            ),
            "deterministic_generation": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_examples": max_examples,
            "runs": summaries,
        },
    )

    return summaries


def verifier_preference(
    sft_row: dict,
    target_row: dict,
) -> str:
    sft_score = float(
        sft_row["exact_reward"]
    )

    target_score = float(
        target_row["exact_reward"]
    )

    if target_score > sft_score:
        return "B"

    if target_score < sft_score:
        return "A"

    return "TIE"


def normalize_judge_label(
    label,
) -> str:
    normalized = str(
        label
    ).strip().upper()

    if normalized not in {
        "A",
        "B",
        "TIE",
    }:
        return "TIE"

    return normalized


def pairwise_score_for_target(
    judge_label: str,
) -> float:
    if judge_label == "B":
        return 1.0

    if judge_label == "TIE":
        return 0.5

    return 0.0


def load_aligned_policy_rows(
    output_dir: Path,
    dataset_name: str,
    output_prefix: str,
) -> dict[str, dict[str, dict]]:
    aligned = {}

    for policy_name in POLICY_NAMES:
        path = generation_path(
            output_dir,
            dataset_name,
            policy_name,
            output_prefix,
        )

        if not path.exists():
            raise FileNotFoundError(
                "Missing generated responses: "
                f"{path}. Run the generation phase first."
            )

        rows = read_jsonl(
            path
        )

        aligned[
            policy_name
        ] = jsonl_index(
            rows,
            "example_id",
        )

    key_sets = [
        set(rows)
        for rows in aligned.values()
    ]

    common = set.intersection(
        *key_sets
    )

    union = set.union(
        *key_sets
    )

    if common != union:
        counts = {
            name: len(rows)
            for name, rows in aligned.items()
        }

        raise ValueError(
            "Policy generation files are not aligned. "
            f"Counts: {counts}"
        )

    return aligned


def run_pairwise_judging(
    cfg: dict,
    datasets: list[str],
    output_prefix: str,
    judge_prefix: str,
    overwrite: bool,
) -> list[dict]:
    """Judge RLVR and RLAIF against the matched SFT response."""

    output_dir = task_results_dir(
        cfg
    )

    jobs = []

    for dataset_name in datasets:
        aligned = load_aligned_policy_rows(
            output_dir,
            dataset_name,
            output_prefix,
        )

        ordered_keys = list(
            aligned["sft"].keys()
        )

        for target_policy in PAIRWISE_TARGETS:
            output_path = pairwise_path(
                output_dir,
                dataset_name,
                target_policy,
                judge_prefix,
            )

            if overwrite and output_path.exists():
                output_path.unlink()

            existing_rows = read_existing_jsonl(
                output_path
            )

            existing = jsonl_index(
                existing_rows,
                "example_id",
            )

            for key in ordered_keys:
                if key in existing:
                    continue

                jobs.append(
                    {
                        "dataset_name": dataset_name,
                        "target_policy": target_policy,
                        "output_path": output_path,
                        "example_id": key,
                        "sft_row": aligned[
                            "sft"
                        ][key],
                        "target_row": aligned[
                            target_policy
                        ][key],
                    }
                )

    print(
        f"\nPairwise judging jobs remaining: {len(jobs)}",
        flush=True,
    )

    start_time = time.perf_counter()

    if jobs:
        judge_cache = (
            output_dir
            / "pairwise_ai_judge_cache.json"
        )

        judge = PairwiseAIJudge(
            cfg,
            judge_cache,
        )

        for index, job in enumerate(
            jobs,
            start=1,
        ):
            sft_row = job[
                "sft_row"
            ]

            target_row = job[
                "target_row"
            ]

            label = normalize_judge_label(
                judge.compare(
                    problem=str(
                        sft_row["question"]
                    ),
                    a=str(
                        sft_row["response"]
                    ),
                    b=str(
                        target_row["response"]
                    ),
                )
            )

            verifier_label = (
                verifier_preference(
                    sft_row,
                    target_row,
                )
            )

            record = {
                "dataset": job[
                    "dataset_name"
                ],
                "example_id": job[
                    "example_id"
                ],
                "source_index": target_row.get(
                    "source_index"
                ),
                "prompt_id": target_row.get(
                    "prompt_id"
                ),
                "question": str(
                    sft_row["question"]
                ),
                "policy_a": "sft",
                "policy_b": job[
                    "target_policy"
                ],
                "judge_preference": label,
                "target_pairwise_score": (
                    pairwise_score_for_target(
                        label
                    )
                ),
                "verifier_preference": verifier_label,
                "verifier_judge_agreement": bool(
                    label
                    == verifier_label
                ),
                "verifier_decisive": bool(
                    verifier_label
                    != "TIE"
                ),
                "sft_exact_reward": float(
                    sft_row[
                        "exact_reward"
                    ]
                ),
                "target_exact_reward": float(
                    target_row[
                        "exact_reward"
                    ]
                ),
                "sft_predicted_final": sft_row.get(
                    "predicted_final"
                ),
                "target_predicted_final": target_row.get(
                    "predicted_final"
                ),
                "sft_response_tokens": int(
                    sft_row[
                        "response_tokens"
                    ]
                ),
                "target_response_tokens": int(
                    target_row[
                        "response_tokens"
                    ]
                ),
            }

            append_jsonl(
                job["output_path"],
                record,
            )

            if (
                index % 10 == 0
                or index == len(jobs)
            ):
                print(
                    "Pairwise judged "
                    f"{index}/{len(jobs)}",
                    flush=True,
                )

        del judge

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = (
        time.perf_counter()
        - start_time
    )

    summaries = []

    for dataset_name in datasets:
        for target_policy in PAIRWISE_TARGETS:
            path = pairwise_path(
                output_dir,
                dataset_name,
                target_policy,
                judge_prefix,
            )

            rows = read_existing_jsonl(
                path
            )

            summaries.append(
                {
                    "dataset": dataset_name,
                    "target_policy": target_policy,
                    "num_pairs": len(
                        rows
                    ),
                    "output": str(
                        path
                    ),
                }
            )

    save_json(
        output_dir
        / f"{judge_prefix}_judging_summary.json",
        {
            "judge_model": cfg[
                "ai_judge_model"
            ],
            "new_judgments": len(
                jobs
            ),
            "elapsed_seconds": float(
                elapsed
            ),
            "outputs": summaries,
        },
    )

    return summaries


def policy_metrics(
    policy_name: str,
    dataset_name: str,
    rows: list[dict],
) -> dict:
    exact_scores = [
        float(
            row["exact_reward"]
        )
        for row in rows
    ]

    format_scores = [
        float(
            bool(
                row["format_compliant"]
            )
        )
        for row in rows
    ]

    lengths = [
        int(
            row["response_tokens"]
        )
        for row in rows
    ]

    eos_scores = [
        float(
            bool(
                row[
                    "terminated_with_eos"
                ]
            )
        )
        for row in rows
    ]

    truncation_scores = [
        float(
            bool(
                row["truncated"]
            )
        )
        for row in rows
    ]

    return {
        "dataset": dataset_name,
        "policy": policy_name,
        "num_examples": len(
            rows
        ),
        "exact_accuracy": safe_mean(
            exact_scores
        ),
        "format_compliance_rate": safe_mean(
            format_scores
        ),
        "response_length_mean": safe_mean(
            lengths
        ),
        "response_length_std": safe_std(
            lengths
        ),
        "response_length_iqr": safe_iqr(
            lengths
        ),
        "generated_token_count": int(
            sum(lengths)
        ),
        "terminated_with_eos_rate": safe_mean(
            eos_scores
        ),
        "truncation_rate": safe_mean(
            truncation_scores
        ),
    }


def pairwise_metrics(
    dataset_name: str,
    target_policy: str,
    rows: list[dict],
) -> dict:
    labels = [
        row[
            "judge_preference"
        ]
        for row in rows
    ]

    target_wins = sum(
        label == "B"
        for label in labels
    )

    sft_wins = sum(
        label == "A"
        for label in labels
    )

    ties = sum(
        label == "TIE"
        for label in labels
    )

    normalized_scores = [
        float(
            row[
                "target_pairwise_score"
            ]
        )
        for row in rows
    ]

    all_agreement = [
        float(
            bool(
                row[
                    "verifier_judge_agreement"
                ]
            )
        )
        for row in rows
    ]

    decisive_rows = [
        row
        for row in rows
        if bool(
            row[
                "verifier_decisive"
            ]
        )
    ]

    decisive_agreement = [
        float(
            bool(
                row[
                    "verifier_judge_agreement"
                ]
            )
        )
        for row in decisive_rows
    ]

    return {
        "dataset": dataset_name,
        "comparison": (
            f"{target_policy}_vs_sft"
        ),
        "target_policy": target_policy,
        "num_pairs": len(
            rows
        ),
        "target_ai_win_rate": (
            target_wins
            / len(rows)
            if rows
            else None
        ),
        "sft_ai_win_rate": (
            sft_wins
            / len(rows)
            if rows
            else None
        ),
        "ai_tie_rate": (
            ties
            / len(rows)
            if rows
            else None
        ),
        "target_normalized_pairwise_score": safe_mean(
            normalized_scores
        ),
        "verifier_judge_agreement_all_rate": safe_mean(
            all_agreement
        ),
        "num_verifier_decisive_pairs": len(
            decisive_rows
        ),
        "verifier_judge_agreement_decisive_rate": safe_mean(
            decisive_agreement
        ),
    }


def qualitative_candidates(
    dataset_name: str,
    aligned: dict[str, dict[str, dict]],
    pairwise_rows: dict[str, list[dict]],
    max_per_category: int = 5,
) -> dict:
    result = {
        "dataset": dataset_name,
        "verifier_judge_disagreements": [],
        "rlvr_ai_wins": [],
        "rlaif_ai_wins": [],
        "policy_correctness_disagreements": [],
    }

    for target_policy in PAIRWISE_TARGETS:
        for pair in pairwise_rows[
            target_policy
        ]:
            key = str(
                pair["example_id"]
            )

            sft_row = aligned[
                "sft"
            ][key]

            target_row = aligned[
                target_policy
            ][key]

            candidate = {
                "example_id": key,
                "target_policy": target_policy,
                "question": sft_row[
                    "question"
                ],
                "gold_final": sft_row[
                    "gold_final"
                ],
                "judge_preference": pair[
                    "judge_preference"
                ],
                "verifier_preference": pair[
                    "verifier_preference"
                ],
                "sft_exact_reward": sft_row[
                    "exact_reward"
                ],
                "target_exact_reward": target_row[
                    "exact_reward"
                ],
                "sft_response": sft_row[
                    "response"
                ],
                "target_response": target_row[
                    "response"
                ],
            }

            if (
                not pair[
                    "verifier_judge_agreement"
                ]
                and len(
                    result[
                        "verifier_judge_disagreements"
                    ]
                )
                < max_per_category
            ):
                result[
                    "verifier_judge_disagreements"
                ].append(
                    candidate
                )

            win_key = (
                f"{target_policy}_ai_wins"
            )

            if (
                pair[
                    "judge_preference"
                ]
                == "B"
                and len(
                    result[
                        win_key
                    ]
                )
                < max_per_category
            ):
                result[
                    win_key
                ].append(
                    candidate
                )

            if (
                float(
                    sft_row[
                        "exact_reward"
                    ]
                )
                != float(
                    target_row[
                        "exact_reward"
                    ]
                )
                and len(
                    result[
                        "policy_correctness_disagreements"
                    ]
                )
                < max_per_category
            ):
                result[
                    "policy_correctness_disagreements"
                ].append(
                    candidate
                )

    return result


def compute_transfer_drops(
    policy_rows: list[dict],
    pairwise_rows: list[dict],
) -> dict:
    policy_lookup = {
        (
            row["dataset"],
            row["policy"],
        ): row
        for row in policy_rows
    }

    pairwise_lookup = {
        (
            row["dataset"],
            row["target_policy"],
        ): row
        for row in pairwise_rows
    }

    policy_drops = []

    for policy_name in POLICY_NAMES:
        in_domain = policy_lookup.get(
            (
                "gsm8k",
                policy_name,
            )
        )

        transfer = policy_lookup.get(
            (
                "transfer",
                policy_name,
            )
        )

        if (
            in_domain is None
            or transfer is None
        ):
            continue

        policy_drops.append(
            {
                "policy": policy_name,
                "gsm8k_exact_accuracy": in_domain[
                    "exact_accuracy"
                ],
                "transfer_exact_accuracy": transfer[
                    "exact_accuracy"
                ],
                "exact_accuracy_drop": (
                    in_domain[
                        "exact_accuracy"
                    ]
                    - transfer[
                        "exact_accuracy"
                    ]
                ),
                "gsm8k_format_compliance": in_domain[
                    "format_compliance_rate"
                ],
                "transfer_format_compliance": transfer[
                    "format_compliance_rate"
                ],
                "format_compliance_drop": (
                    in_domain[
                        "format_compliance_rate"
                    ]
                    - transfer[
                        "format_compliance_rate"
                    ]
                ),
                "gsm8k_mean_length": in_domain[
                    "response_length_mean"
                ],
                "transfer_mean_length": transfer[
                    "response_length_mean"
                ],
                "mean_length_change": (
                    transfer[
                        "response_length_mean"
                    ]
                    - in_domain[
                        "response_length_mean"
                    ]
                ),
            }
        )

    pairwise_drops = []

    for target_policy in PAIRWISE_TARGETS:
        in_domain = pairwise_lookup.get(
            (
                "gsm8k",
                target_policy,
            )
        )

        transfer = pairwise_lookup.get(
            (
                "transfer",
                target_policy,
            )
        )

        if (
            in_domain is None
            or transfer is None
        ):
            continue

        pairwise_drops.append(
            {
                "target_policy": target_policy,
                "gsm8k_pairwise_score": in_domain[
                    "target_normalized_pairwise_score"
                ],
                "transfer_pairwise_score": transfer[
                    "target_normalized_pairwise_score"
                ],
                "pairwise_score_drop": (
                    in_domain[
                        "target_normalized_pairwise_score"
                    ]
                    - transfer[
                        "target_normalized_pairwise_score"
                    ]
                ),
            }
        )

    return {
        "policy_transfer_drops": policy_drops,
        "pairwise_transfer_drops": pairwise_drops,
    }


def summarize_results(
    cfg: dict,
    datasets: list[str],
    output_prefix: str,
    judge_prefix: str,
) -> dict:
    output_dir = task_results_dir(
        cfg
    )

    all_policy_metrics = []
    all_pairwise_metrics = []
    all_qualitative = {}

    for dataset_name in datasets:
        aligned = load_aligned_policy_rows(
            output_dir,
            dataset_name,
            output_prefix,
        )

        for policy_name in POLICY_NAMES:
            rows = list(
                aligned[
                    policy_name
                ].values()
            )

            all_policy_metrics.append(
                policy_metrics(
                    policy_name,
                    dataset_name,
                    rows,
                )
            )

        pairwise_rows = {}

        for target_policy in PAIRWISE_TARGETS:
            path = pairwise_path(
                output_dir,
                dataset_name,
                target_policy,
                judge_prefix,
            )

            if not path.exists():
                raise FileNotFoundError(
                    "Missing pairwise judgments: "
                    f"{path}. Run the judging phase first."
                )

            rows = read_jsonl(
                path
            )

            pairwise_rows[
                target_policy
            ] = rows

            all_pairwise_metrics.append(
                pairwise_metrics(
                    dataset_name,
                    target_policy,
                    rows,
                )
            )

        all_qualitative[
            dataset_name
        ] = qualitative_candidates(
            dataset_name,
            aligned,
            pairwise_rows,
        )

    transfer_analysis = compute_transfer_drops(
        all_policy_metrics,
        all_pairwise_metrics,
    )

    summary = {
        "config": "configs/feedback.yaml",
        "seed": int(
            cfg["seed"]
        ),
        "evaluation_protocol": {
            "policies": list(
                POLICY_NAMES
            ),
            "datasets": datasets,
            "generation": {
                "deterministic": True,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": int(
                    cfg.get(
                        "math_max_new_tokens",
                        512,
                    )
                ),
            },
            "exact_verifier": (
                "last designated #### <number> "
                "compared numerically with gold_final"
            ),
            "format_compliance": (
                "response ends with a designated #### <number>"
            ),
            "ai_comparison": (
                "fixed pairwise judge; SFT is response A and "
                "the RLVR/RLAIF policy is response B"
            ),
            "normalized_pairwise_score": (
                "target wins + 0.5 * ties, divided by number of pairs"
            ),
        },
        "policy_metrics": all_policy_metrics,
        "pairwise_metrics": all_pairwise_metrics,
        "transfer_analysis": transfer_analysis,
    }

    save_json(
        output_dir
        / "math_evaluation_summary.json",
        summary,
    )

    save_csv(
        output_dir
        / "math_policy_metrics.csv",
        all_policy_metrics,
    )

    save_csv(
        output_dir
        / "math_pairwise_metrics.csv",
        all_pairwise_metrics,
    )

    save_csv(
        output_dir
        / "math_transfer_policy_drops.csv",
        transfer_analysis[
            "policy_transfer_drops"
        ],
    )

    save_csv(
        output_dir
        / "math_transfer_pairwise_drops.csv",
        transfer_analysis[
            "pairwise_transfer_drops"
        ],
    )

    save_json(
        output_dir
        / "math_qualitative_candidates.json",
        all_qualitative,
    )

    print(
        "\nTask 5 mathematical-feedback summary",
        flush=True,
    )

    print(
        "\nDataset | policy | exact accuracy | "
        "format compliance | mean tokens",
        flush=True,
    )

    for row in all_policy_metrics:
        print(
            f"{row['dataset']} | "
            f"{row['policy']} | "
            f"{row['exact_accuracy']} | "
            f"{row['format_compliance_rate']} | "
            f"{row['response_length_mean']}",
            flush=True,
        )

    print(
        "\nDataset | comparison | target score | "
        "target wins | ties | decisive verifier/judge agreement",
        flush=True,
    )

    for row in all_pairwise_metrics:
        print(
            f"{row['dataset']} | "
            f"{row['comparison']} | "
            f"{row['target_normalized_pairwise_score']} | "
            f"{row['target_ai_win_rate']} | "
            f"{row['ai_tie_rate']} | "
            f"{row['verifier_judge_agreement_decisive_rate']}",
            flush=True,
        )

    print(
        "\nSaved Task 5 evaluation summary to "
        f"{output_dir}",
        flush=True,
    )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate and evaluate matched SFT, RLVR, and "
            "RLAIF responses on GSM8K and SVAMP."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/feedback.yaml",
    )

    parser.add_argument(
        "--phase",
        choices=[
            "all",
            "generate",
            "judge",
            "summarize",
        ],
        default="all",
        help=(
            "Run the complete pipeline or one resumable phase."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(
            DATASET_NAMES
        ),
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICY_NAMES,
        default=list(
            POLICY_NAMES
        ),
        help=(
            "Policies used during generation. Complete judging "
            "requires all three policies."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Generation batch size. Use 1 on a 6 GiB GPU."
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        help=(
            "Optional prefix size per dataset for smoke testing."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        default="generated",
        help=(
            "Prefix for policy-generation JSONL files."
        ),
    )

    parser.add_argument(
        "--judge-prefix",
        default="pairwise",
        help=(
            "Prefix for pairwise-judgment JSONL files."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete selected generation and judgment files "
            "instead of resuming them."
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error(
            "--batch-size must be at least 1"
        )

    if (
        args.max_examples is not None
        and args.max_examples < 1
    ):
        parser.error(
            "--max-examples must be at least 1"
        )

    cfg = load_yaml(
        args.config
    )

    set_seed(
        int(
            cfg["seed"]
        )
    )

    if args.phase in {
        "all",
        "generate",
    }:
        run_generation(
            cfg=cfg,
            datasets=list(
                args.datasets
            ),
            policies=list(
                args.policies
            ),
            max_examples=args.max_examples,
            batch_size=int(
                args.batch_size
            ),
            output_prefix=args.output_prefix,
            overwrite=bool(
                args.overwrite
            ),
        )

    if args.phase in {
        "all",
        "judge",
    }:
        run_pairwise_judging(
            cfg=cfg,
            datasets=list(
                args.datasets
            ),
            output_prefix=args.output_prefix,
            judge_prefix=args.judge_prefix,
            overwrite=bool(
                args.overwrite
            ),
        )

    if args.phase in {
        "all",
        "summarize",
    }:
        summarize_results(
            cfg=cfg,
            datasets=list(
                args.datasets
            ),
            output_prefix=args.output_prefix,
            judge_prefix=args.judge_prefix,
        )


if __name__ == "__main__":
    main()