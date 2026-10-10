from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    wall_timer,
)
from common.models import (
    count_parameters,
    load_policy,
    load_tokenizer,
)


POLICY_NAMES = (
    "sft",
    "dpo",
    "ppo",
    "grpo",
)


def policy_specs(cfg: dict) -> dict[str, str | None]:
    """Return the four fixed Task 4 policy specifications."""
    configured = cfg.get("policies", {})

    return {
        "sft": None,
        "dpo": configured.get("dpo"),
        "ppo": configured.get("ppo"),
        "grpo": configured.get("grpo"),
    }


def load_xstest(cfg: dict) -> pd.DataFrame:
    """Load and validate the fixed XSTest evaluation table."""
    path = repo_path(cfg["paths"]["xstest"])
    df = pd.read_csv(path)

    required_columns = {
        "id",
        "prompt",
        "type",
        "label",
        "focus",
        "note",
        "benchmark_class",
        "xstest_id",
    }

    missing = sorted(
        required_columns.difference(df.columns)
    )

    if missing:
        raise ValueError(
            "XSTest is missing required columns: "
            + ", ".join(missing)
        )

    if df.empty:
        raise ValueError("The XSTest dataset is empty.")

    df = df.copy()

    df["benchmark_class"] = (
        df["benchmark_class"]
        .astype(str)
        .str.upper()
    )

    invalid_classes = sorted(
        set(df["benchmark_class"])
        .difference({"SAFE", "UNSAFE"})
    )

    if invalid_classes:
        raise ValueError(
            "Unexpected XSTest benchmark classes: "
            + ", ".join(invalid_classes)
        )

    if df["xstest_id"].duplicated().any():
        duplicates = (
            df.loc[
                df["xstest_id"].duplicated(),
                "xstest_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Duplicate XSTest IDs: "
            + ", ".join(duplicates[:10])
        )

    return df.reset_index(drop=True)


def validate_policy_checkpoint(
    policy_name: str,
    adapter_path: str | None,
) -> None:
    """Check that a required trained adapter exists."""
    if policy_name == "sft":
        if adapter_path is not None:
            raise ValueError(
                "The SFT baseline must not use an adapter."
            )
        return

    if not adapter_path:
        raise ValueError(
            f"No adapter is configured for policy "
            f"{policy_name!r}."
        )

    resolved = repo_path(adapter_path)

    if not resolved.exists():
        raise FileNotFoundError(
            f"The {policy_name} adapter does not exist: "
            f"{resolved}"
        )

    adapter_config = resolved / "adapter_config.json"

    if not adapter_config.exists():
        raise FileNotFoundError(
            f"The {policy_name} adapter directory is "
            f"missing adapter_config.json: {resolved}"
        )


def output_paths(
    cfg: dict,
    policy_name: str,
    output_prefix: str,
) -> tuple[Path, Path]:
    """Return the generation and summary paths."""
    output_dir = (
        repo_path(cfg["results_dir"])
        / "task4_safety"
    )

    generation_path = (
        output_dir
        / f"{output_prefix}_{policy_name}.jsonl"
    )

    summary_path = (
        output_dir
        / f"{output_prefix}_{policy_name}_summary.json"
    )

    return generation_path, summary_path


def validate_existing_prefix(
    existing_rows: list[dict[str, Any]],
    selected_df: pd.DataFrame,
    policy_name: str,
) -> None:
    """Validate that a partial cache is an exact dataset prefix."""
    if len(existing_rows) > len(selected_df):
        raise ValueError(
            f"Existing {policy_name} output contains "
            f"{len(existing_rows)} rows, but this run "
            f"selects only {len(selected_df)} rows."
        )

    existing_ids = [
        int(row["xstest_id"])
        for row in existing_rows
    ]

    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError(
            f"Existing {policy_name} output contains "
            "duplicate XSTest IDs."
        )

    expected_ids = (
        selected_df["xstest_id"]
        .iloc[: len(existing_rows)]
        .astype(int)
        .tolist()
    )

    if existing_ids != expected_ids:
        raise ValueError(
            f"Existing {policy_name} output is not an "
            "exact prefix of the selected XSTest order. "
            "Use --overwrite only if you intend to "
            "replace it."
        )

    for position, row in enumerate(existing_rows):
        expected = selected_df.iloc[position]

        if str(row.get("policy")) != policy_name:
            raise ValueError(
                f"Existing row {position} has policy "
                f"{row.get('policy')!r}; expected "
                f"{policy_name!r}."
            )

        if str(row.get("prompt")) != str(
            expected["prompt"]
        ):
            raise ValueError(
                f"Existing row {position} has a prompt "
                "that does not match the fixed dataset."
            )

        if str(
            row.get("benchmark_class", "")
        ).upper() != str(
            expected["benchmark_class"]
        ).upper():
            raise ValueError(
                f"Existing row {position} has an "
                "incorrect benchmark class."
            )


def make_record(
    row: pd.Series,
    policy_name: str,
    response: str,
    response_tokens: int,
    terminated_with_eos: bool,
    truncated: bool,
) -> dict[str, Any]:
    """Construct one reusable Task 4 generation record."""
    return {
        "xstest_id": int(row["xstest_id"]),
        "source_id": int(row["id"]),
        "policy": policy_name,
        "prompt": str(row["prompt"]),
        "benchmark_class": str(
            row["benchmark_class"]
        ).upper(),
        "type": str(row["type"]),
        "label": str(row["label"]),
        "focus": str(row["focus"]),
        "note": (
            None
            if pd.isna(row["note"])
            else str(row["note"])
        ),
        "response": str(response),
        "response_tokens": int(response_tokens),
        "terminated_with_eos": bool(
            terminated_with_eos
        ),
        "truncated": bool(truncated),
    }


def clear_loaded_model(model, tokenizer) -> None:
    """Release model references before loading another policy."""
    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    lengths = np.asarray(
        [
            int(row["response_tokens"])
            for row in records
        ],
        dtype=float,
    )

    safe_count = sum(
        str(row["benchmark_class"]).upper()
        == "SAFE"
        for row in records
    )

    unsafe_count = sum(
        str(row["benchmark_class"]).upper()
        == "UNSAFE"
        for row in records
    )

    terminated = np.asarray(
        [
            bool(row["terminated_with_eos"])
            for row in records
        ],
        dtype=float,
    )

    truncated = np.asarray(
        [
            bool(row["truncated"])
            for row in records
        ],
        dtype=float,
    )

    return {
        "num_responses": len(records),
        "num_safe_prompts": int(safe_count),
        "num_unsafe_prompts": int(unsafe_count),
        "response_length_mean": float(
            lengths.mean()
        ),
        "response_length_std": float(
            lengths.std()
        ),
        "response_length_iqr": float(
            np.percentile(lengths, 75)
            - np.percentile(lengths, 25)
        ),
        "generated_token_count": int(
            lengths.sum()
        ),
        "terminated_with_eos_rate": float(
            terminated.mean()
        ),
        "truncation_rate": float(
            truncated.mean()
        ),
    }


def generate_for_policy(
    cfg: dict,
    config_path: str,
    policy_name: str,
    batch_size: int = 1,
    max_examples: int | None = None,
    output_prefix: str = "generated",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate and save deterministic XSTest responses."""
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if max_examples is not None:
        if int(max_examples) <= 0:
            raise ValueError(
                "max_examples must be positive."
            )
        max_examples = int(max_examples)

    specs = policy_specs(cfg)

    if policy_name not in specs:
        raise KeyError(
            f"Unknown policy {policy_name!r}."
        )

    adapter_path = specs[policy_name]

    validate_policy_checkpoint(
        policy_name,
        adapter_path,
    )

    full_df = load_xstest(cfg)

    if max_examples is None:
        selected_df = full_df
    else:
        selected_df = full_df.iloc[
            :max_examples
        ].copy()

    generation_path, summary_path = output_paths(
        cfg,
        policy_name,
        output_prefix,
    )

    generation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if overwrite:
        generation_path.unlink(
            missing_ok=True
        )
        summary_path.unlink(
            missing_ok=True
        )

    existing_rows: list[dict[str, Any]] = []

    if generation_path.exists():
        existing_rows = read_jsonl(
            generation_path
        )

        validate_existing_prefix(
            existing_rows,
            selected_df,
            policy_name,
        )

    completed = len(existing_rows)
    total = len(selected_df)

    if completed == total:
        print(
            f"[{policy_name}] Existing generation "
            f"cache is already complete: "
            f"{generation_path}",
            flush=True,
        )

        summary = {
            "policy": policy_name,
            "config": config_path,
            "adapter": adapter_path,
            "dataset": str(
                cfg["paths"]["xstest"]
            ),
            "output_prefix": output_prefix,
            "deterministic_decoding": True,
            "batch_size": int(batch_size),
            "max_prompt_length": 256,
            "max_new_tokens": int(
                cfg["safety_max_new_tokens"]
            ),
            "requested_max_examples": (
                max_examples
            ),
            "complete": True,
            "resumed_from_rows": completed,
            "wall_clock_seconds": 0.0,
            "peak_vram_gib": None,
            "generations": str(
                generation_path.relative_to(
                    repo_path(".")
                )
            ),
            **summarize_records(
                existing_rows
            ),
        }

        save_json(summary_path, summary)
        return summary

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    timer = wall_timer()

    print(
        f"[{policy_name}] Loading policy "
        f"({completed}/{total} responses cached)...",
        flush=True,
    )

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    model = load_policy(
        cfg,
        adapter_path=adapter_path,
        trainable=False,
    )

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    remaining_df = selected_df.iloc[
        completed:
    ]

    try:
        for start in range(
            0,
            len(remaining_df),
            batch_size,
        ):
            chunk = remaining_df.iloc[
                start:start + batch_size
            ]

            prompts = [
                [
                    {
                        "role": "user",
                        "content": str(prompt),
                    }
                ]
                for prompt in chunk[
                    "prompt"
                ].tolist()
            ]

            generated = batch_generate(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_prompt_length=256,
                max_new_tokens=int(
                    cfg[
                        "safety_max_new_tokens"
                    ]
                ),
                temperature=1.0,
                top_p=1.0,
                do_sample=False,
            )

            for (
                (_, row),
                response,
                response_tokens,
                terminated,
                truncated,
            ) in zip(
                chunk.iterrows(),
                generated["responses"],
                generated[
                    "response_lengths"
                ],
                generated[
                    "terminated_with_eos"
                ],
                generated["truncated"],
            ):
                record = make_record(
                    row=row,
                    policy_name=policy_name,
                    response=response,
                    response_tokens=(
                        response_tokens
                    ),
                    terminated_with_eos=(
                        terminated
                    ),
                    truncated=truncated,
                )

                append_jsonl(
                    generation_path,
                    record,
                )

            finished = min(
                completed
                + start
                + len(chunk),
                total,
            )

            print(
                f"[{policy_name}] generated "
                f"{finished}/{total}",
                flush=True,
            )

    finally:
        clear_loaded_model(
            model,
            tokenizer,
        )

    records = read_jsonl(generation_path)

    validate_existing_prefix(
        records,
        selected_df,
        policy_name,
    )

    if len(records) != total:
        raise RuntimeError(
            f"Generation cache is incomplete: "
            f"expected {total} rows, found "
            f"{len(records)}."
        )

    elapsed_seconds = float(timer())

    peak_vram_gib = None

    if torch.cuda.is_available():
        peak_vram_gib = float(
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

    summary = {
        "policy": policy_name,
        "config": config_path,
        "adapter": adapter_path,
        "dataset": str(
            cfg["paths"]["xstest"]
        ),
        "output_prefix": output_prefix,
        "seed": int(cfg["seed"]),
        "deterministic_decoding": True,
        "generation": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        },
        "batch_size": int(batch_size),
        "max_prompt_length": 256,
        "max_new_tokens": int(
            cfg["safety_max_new_tokens"]
        ),
        "requested_max_examples": (
            max_examples
        ),
        "complete": True,
        "resumed_from_rows": completed,
        "total_parameters": int(
            total_parameters
        ),
        "trainable_parameters": int(
            trainable_parameters
        ),
        "wall_clock_seconds": (
            elapsed_seconds
        ),
        "peak_vram_gib": peak_vram_gib,
        "generations": str(
            generation_path.relative_to(
                repo_path(".")
            )
        ),
        **summarize_records(records),
    }

    save_json(summary_path, summary)

    print(
        f"[{policy_name}] Saved responses to "
        f"{generation_path}",
        flush=True,
    )

    print(
        f"[{policy_name}] Saved summary to "
        f"{summary_path}",
        flush=True,
    )

    return summary


def run_generation(
    config_path: str,
    policies: list[str],
    batch_size: int = 1,
    max_examples: int | None = None,
    output_prefix: str = "generated",
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    if not policies:
        raise ValueError(
            "Select at least one policy."
        )

    invalid = sorted(
        set(policies).difference(
            POLICY_NAMES
        )
    )

    if invalid:
        raise ValueError(
            "Unknown policies: "
            + ", ".join(invalid)
        )

    if len(policies) != len(set(policies)):
        raise ValueError(
            "Each policy may be selected only once."
        )

    summaries = []

    for position, policy_name in enumerate(
        policies,
        start=1,
    ):
        print(
            f"\n[{position}/{len(policies)}] "
            f"Task 4 policy: {policy_name}",
            flush=True,
        )

        summary = generate_for_policy(
            cfg=cfg,
            config_path=config_path,
            policy_name=policy_name,
            batch_size=batch_size,
            max_examples=max_examples,
            output_prefix=output_prefix,
            overwrite=overwrite,
        )

        summaries.append(summary)

        # Reset the seed before every policy so that
        # conditions remain independently reproducible.
        set_seed(int(cfg["seed"]))

    index_path = (
        repo_path(cfg["results_dir"])
        / "task4_safety"
        / f"{output_prefix}_generation_index.json"
    )

    save_json(
        index_path,
        {
            "config": config_path,
            "policies": policies,
            "batch_size": int(
                batch_size
            ),
            "requested_max_examples": (
                max_examples
            ),
            "output_prefix": output_prefix,
            "summaries": summaries,
        },
    )

    print(
        f"\nSaved generation index to "
        f"{index_path}",
        flush=True,
    )

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic Task 4 XSTest "
            "responses from the fixed policies."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/feedback.yaml",
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICY_NAMES,
        default=list(POLICY_NAMES),
        help=(
            "Policies to generate. The default runs "
            "SFT, DPO, PPO, and GRPO."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Generation batch size. Use 1 on a "
            "6 GiB GPU."
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        help=(
            "Optional prefix size for a smoke test."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        default="generated",
        help=(
            "Filename prefix. Use a different prefix "
            "for smoke tests."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace existing selected-policy output "
            "files instead of resuming them."
        ),
    )

    args = parser.parse_args()

    run_generation(
        config_path=args.config,
        policies=list(args.policies),
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        output_prefix=args.output_prefix,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()