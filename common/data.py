from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(
    path: str | Path,
) -> Path:
    p = Path(path)

    if p.is_absolute():
        return p

    return REPO_ROOT / p


def _deep_merge(
    a: dict,
    b: dict,
) -> dict:
    out = dict(a)

    for key, value in b.items():
        if (
            isinstance(value, dict)
            and isinstance(
                out.get(key),
                dict,
            )
        ):
            out[key] = _deep_merge(
                out[key],
                value,
            )
        else:
            out[key] = value

    return out


def load_yaml(
    path: str | Path,
) -> dict:
    path = repo_path(path)

    cfg = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if cfg.get("base_config"):
        base_path = repo_path(
            cfg["base_config"]
        )

        base = yaml.safe_load(
            base_path.read_text(
                encoding="utf-8"
            )
        )

        cfg = _deep_merge(
            base,
            {
                key: value
                for key, value
                in cfg.items()
                if key != "base_config"
            },
        )

    return cfg


def read_jsonl(
    path: str | Path,
) -> list[dict]:
    rows = []

    with repo_path(path).open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def write_jsonl(
    path: str | Path,
    rows: Iterable[dict],
) -> None:
    output = repo_path(path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_csv(
    path: str | Path,
) -> pd.DataFrame:
    return pd.read_csv(
        repo_path(path)
    )


def last_assistant_text(
    messages,
) -> str:
    if isinstance(messages, str):
        return messages

    if isinstance(messages, list):
        for message in reversed(
            messages
        ):
            if (
                isinstance(
                    message,
                    dict,
                )
                and message.get("role")
                == "assistant"
            ):
                return str(
                    message.get(
                        "content",
                        "",
                    )
                )

    return str(messages)


def prompt_messages_from_preference(
    row: dict,
) -> list[dict]:
    chosen = row.get("chosen")

    if (
        isinstance(chosen, list)
        and chosen
    ):
        output = list(chosen)

        if (
            isinstance(
                output[-1],
                dict,
            )
            and output[-1].get("role")
            == "assistant"
        ):
            output = output[:-1]

        return output

    prompt = str(
        row.get(
            "prompt",
            "",
        )
    )

    return [
        {
            "role": "user",
            "content": prompt,
        }
    ]


def filter_overlength_preference_rows(
    rows: list[dict],
    tokenizer,
    max_length: int,
) -> tuple[list[dict], list[dict]]:
    """Filter rows whose prompt alone cannot fit.

    This follows the same boundary as encode_prompt_response.
    Prompts are preserved, responses may be truncated, and an
    example is excluded only when its prompt occupies all available
    sequence positions and therefore leaves no room for a response.
    """

    kept = []
    excluded = []

    for dataset_index, row in enumerate(
        rows
    ):
        messages = (
            prompt_messages_from_preference(
                row
            )
        )

        prompt_ids = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )

        prompt_tokens = len(
            prompt_ids
        )

        if prompt_tokens >= max_length:
            excluded.append(
                {
                    "dataset_index":
                        dataset_index,

                    "prompt_id":
                        row.get(
                            "prompt_id"
                        ),

                    "source_index":
                        row.get(
                            "source_index"
                        ),

                    "prompt_tokens":
                        prompt_tokens,

                    "max_length":
                        max_length,

                    "reason":
                        (
                            "prompt_does_"
                            "not_fit"
                        ),
                }
            )
        else:
            kept.append(row)

    return kept, excluded


def preference_responses(
    row: dict,
) -> tuple[str, str]:
    return (
        last_assistant_text(
            row["chosen"]
        ),
        last_assistant_text(
            row["rejected"]
        ),
    )


def prompt_messages(
    row: dict,
) -> list[dict]:
    if isinstance(
        row.get("messages"),
        list,
    ):
        return row["messages"]

    if isinstance(
        row.get("prompt"),
        list,
    ):
        return row["prompt"]

    return [
        {
            "role": "user",
            "content": str(
                row.get(
                    "prompt",
                    row.get(
                        "question",
                        "",
                    ),
                )
            ),
        }
    ]


def render_prompt(
    tokenizer,
    messages: list[dict],
) -> str:
    return (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def encode_prompt_response(
    tokenizer,
    messages: list[dict],
    response: str,
    max_length: int,
):
    """Encode a prompt-response pair for DPO.

    The complete prompt is retained for both the chosen and rejected
    responses. If the combined sequence is too long, the response is
    truncated from the right. Rows whose prompt alone cannot fit are
    expected to be removed beforehand by
    filter_overlength_preference_rows().
    """

    prompt_ids = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )

    if len(prompt_ids) >= max_length:
        raise ValueError(
            "Prompt alone has "
            f"{len(prompt_ids)} tokens, "
            "which does not fit inside "
            f"max_length={max_length}. "
            "Apply "
            "filter_overlength_preference_rows "
            "before batching."
        )

    response_ids = tokenizer(
        response,
        add_special_tokens=False,
    )["input_ids"]

    response_budget = (
        max_length
        - len(prompt_ids)
    )

    eos_id = tokenizer.eos_token_id

    if eos_id is not None:
        content_budget = max(
            0,
            response_budget - 1,
        )

        response_ids = (
            response_ids[
                :content_budget
            ]
            + [eos_id]
        )
    else:
        response_ids = response_ids[
            :response_budget
        ]

    ids = (
        prompt_ids
        + response_ids
    )

    response_mask = (
        [0] * len(prompt_ids)
        + [1] * len(response_ids)
    )

    assert len(ids) <= max_length
    assert (
        len(ids)
        == len(response_mask)
    )

    return ids, response_mask


def pad_batch(
    tokenizer,
    examples: list[
        tuple[
            list[int],
            list[int],
        ]
    ],
):
    import torch

    max_len = max(
        len(ids)
        for ids, _ in examples
    )

    pad_id = tokenizer.pad_token_id

    input_ids = []
    attention_mask = []
    response_mask = []

    for ids, current_mask in examples:
        padding = (
            max_len
            - len(ids)
        )

        input_ids.append(
            [pad_id] * padding
            + ids
        )

        attention_mask.append(
            [0] * padding
            + [1] * len(ids)
        )

        response_mask.append(
            [0] * padding
            + current_mask
        )

    return {
        "input_ids":
            torch.tensor(
                input_ids,
                dtype=torch.long,
            ),

        "attention_mask":
            torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),

        "response_mask":
            torch.tensor(
                response_mask,
                dtype=torch.float32,
            ),
    }