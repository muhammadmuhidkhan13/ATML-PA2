from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    filter_overlength_preference_rows,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
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
    reference_mode,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    """
    Create the function used by the DataLoader to turn preference
    examples into padded PyTorch batches.

    Each dataset row contains:

        prompt
        chosen response
        rejected response

    The chosen and rejected sequences retain the same prompt. Only
    response tokens contribute to the DPO sequence log-probabilities.
    """

    def collate(rows):
        chosen_examples = []
        rejected_examples = []

        for row in rows:
            prompt = prompt_messages_from_preference(row)

            (
                chosen_response,
                rejected_response,
            ) = preference_responses(row)

            chosen_examples.append(
                encode_prompt_response(
                    tokenizer=tokenizer,
                    messages=prompt,
                    response=chosen_response,
                    max_length=max_length,
                )
            )

            rejected_examples.append(
                encode_prompt_response(
                    tokenizer=tokenizer,
                    messages=prompt,
                    response=rejected_response,
                    max_length=max_length,
                )
            )

        chosen_batch = pad_batch(
            tokenizer,
            chosen_examples,
        )

        rejected_batch = pad_batch(
            tokenizer,
            rejected_examples,
        )

        return chosen_batch, rejected_batch

    return collate


def move_batch(batch, device):
    """Move every tensor in a batch onto the model's device."""

    return {
        name: tensor.to(device)
        for name, tensor in batch.items()
    }


def response_sequence_logps(model, batch):
    """
    Calculate one response log-probability for every sequence.

    A causal language model predicts the token at position t+1 using
    the tokens through position t. Therefore, logits and labels must
    be shifted by one position.

    The response mask is zero over prompt and padding tokens and one
    over response tokens. Summing masked token log-probabilities gives:

        log pi(response | prompt)

    for each example in the batch.
    """

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )

    # The logits at position t predict the token at position t + 1.
    shifted_logits = outputs.logits[:, :-1, :]
    shifted_labels = batch["input_ids"][:, 1:]
    shifted_response_mask = batch["response_mask"][:, 1:]

    # Select the raw logit assigned to the token that actually occurs.
    selected_logits = shifted_logits.gather(
        dim=-1,
        index=shifted_labels.unsqueeze(-1),
    ).squeeze(-1)

    # logsumexp is the denominator of log-softmax.
    log_normalizers = torch.logsumexp(
        shifted_logits,
        dim=-1,
    )

    token_logps = (
        selected_logits - log_normalizers
    ).float()

    # Ignore prompt and padding positions.
    sequence_logps = (
        token_logps
        * shifted_response_mask.float()
    ).sum(dim=-1)

    return sequence_logps


def _selected_example_record(
    row: dict,
    selected_index: int,
):
    """Create lightweight provenance information for a retained row."""

    return {
        "selected_index": selected_index,
        "prompt_id": row.get("prompt_id"),
        "source_index": row.get("source_index"),
        "source_split": row.get("source_split"),
        "length_stratum": row.get("length_stratum"),
    }


def prepare_dpo_run(
    config_path: str,
    dataset_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
):
    """
    Load configuration, preprocess data, load the policy, and create
    the optimizer and DataLoader.

    The preprocessing policy is:

    1. Preserve the complete prompt.
    2. Exclude rows whose prompt alone cannot fit.
    3. For retained rows, truncate response tokens from the right.
    4. Preserve EOS when the tokenizer provides one.
    5. Apply max_examples only after filtering.

    Filtering before max_examples ensures that short ablations still
    receive the requested number of valid examples whenever enough
    valid rows are available.
    """

    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    selected_dataset = (
        dataset_path
        or cfg["paths"]["dpo_standard_train"]
    )

    raw_rows = read_jsonl(selected_dataset)

    if not raw_rows:
        raise ValueError(
            "The selected DPO training dataset is empty."
        )

    if max_examples is not None and int(max_examples) < 1:
        raise ValueError(
            "max_examples must be positive when provided."
        )

    selected_beta = float(
        cfg["beta"]
        if beta is None
        else beta
    )

    if selected_beta <= 0:
        raise ValueError(
            "DPO beta must be positive."
        )

    max_sequence_length = int(
        cfg["max_sequence_length"]
    )

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )

    eligible_rows, excluded_rows = (
        filter_overlength_preference_rows(
            rows=raw_rows,
            tokenizer=tokenizer,
            max_length=max_sequence_length,
        )
    )

    if max_examples is None:
        rows = eligible_rows
    else:
        rows = eligible_rows[: int(max_examples)]

    if not rows:
        raise ValueError(
            "No DPO training examples remain after applying "
            "the prompt-length filter."
        )

    print(
        "DPO preprocessing: "
        f"raw={len(raw_rows)}, "
        f"eligible={len(eligible_rows)}, "
        f"excluded={len(excluded_rows)}, "
        f"selected={len(rows)}, "
        f"max_sequence_length={max_sequence_length}"
    )

    model = load_policy(
        cfg,
        trainable=True,
        fresh_lora=True,
    )

    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(
            tokenizer,
            max_sequence_length,
        ),
    )

    parameters = trainable_parameters(model)

    if not parameters:
        raise RuntimeError(
            "The DPO policy has no trainable parameters."
        )

    optimizer = AdamW(
        parameters,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(
            cfg.get("weight_decay", 0.0)
        ),
    )

    return {
        "cfg": cfg,
        "raw_rows": raw_rows,
        "eligible_rows": eligible_rows,
        "excluded_rows": excluded_rows,
        "rows": rows,
        "dataset_path": selected_dataset,
        "requested_max_examples": max_examples,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "parameters": parameters,
        "optimizer": optimizer,
        "beta": selected_beta,
    }


def run_training(
    config_path: str,
    run_name: str,
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
):
    """Run DPO training and save the adapter, logs, and metadata."""

    bundle = prepare_dpo_run(
        config_path=config_path,
        dataset_path=dataset_path,
        beta=beta,
        max_examples=max_examples,
    )

    cfg = bundle["cfg"]
    raw_rows = bundle["raw_rows"]
    eligible_rows = bundle["eligible_rows"]
    excluded_rows = bundle["excluded_rows"]
    rows = bundle["rows"]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    loader = bundle["loader"]
    parameters = bundle["parameters"]
    optimizer = bundle["optimizer"]
    selected_beta = bundle["beta"]

    output_spec = (
        output_path
        or cfg["standard_output"]
    )

    output = repo_path(output_spec)

    results_dir = repo_path(
        cfg["results_dir"]
    )
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        results_dir
        / f"{run_name}_train_log.jsonl"
    )

    summary_path = (
        results_dir
        / f"{run_name}_train_summary.json"
    )

    # Training and evaluation must use different filenames.
    # Otherwise, evaluating a run named "standard" would overwrite
    # the corresponding training preprocessing manifest.
    filter_manifest_path = (
        results_dir
        / f"{run_name}_train_preprocessing_manifest.json"
    )

    # Start this run with a clean log.
    log_path.write_text(
        "",
        encoding="utf-8",
    )

    device = next(
        model.parameters()
    ).device

    epochs = int(cfg["epochs"])

    accumulation_steps = int(
        cfg["grad_accum_steps"]
    )

    max_grad_norm = float(
        cfg["max_grad_norm"]
    )

    if epochs < 1:
        raise ValueError(
            "epochs must be at least 1."
        )

    if accumulation_steps < 1:
        raise ValueError(
            "grad_accum_steps must be at least 1."
        )

    total_parameters, trainable_count = (
        count_parameters(model)
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(
            device
        )

    timer = wall_timer()

    optimizer.zero_grad(
        set_to_none=True
    )

    metric_names = (
        "loss",
        "logit_mean",
        "policy_margin_mean",
        "preference_accuracy",
    )

    run_sums = {
        name: 0.0
        for name in metric_names
    }

    run_examples = 0
    examples_seen = 0
    update_step = 0

    model.train()

    for epoch_index in range(epochs):
        window_sums = {
            name: 0.0
            for name in metric_names
        }

        window_examples = 0
        window_microbatches = 0

        for batch_index, (
            chosen_batch,
            rejected_batch,
        ) in enumerate(loader):
            chosen_batch = move_batch(
                chosen_batch,
                device,
            )

            rejected_batch = move_batch(
                rejected_batch,
                device,
            )

            batch_size = int(
                chosen_batch[
                    "input_ids"
                ].shape[0]
            )

            # Calculate reference log-probabilities with the LoRA
            # adapter disabled. The reference values are constants,
            # so gradients are unnecessary.
            with torch.no_grad():
                with reference_mode(model):
                    ref_chosen_logp = (
                        response_sequence_logps(
                            model,
                            chosen_batch,
                        )
                    )

                    ref_rejected_logp = (
                        response_sequence_logps(
                            model,
                            rejected_batch,
                        )
                    )

            # Calculate log-probabilities from the trainable policy.
            policy_chosen_logp = (
                response_sequence_logps(
                    model,
                    chosen_batch,
                )
            )

            policy_rejected_logp = (
                response_sequence_logps(
                    model,
                    rejected_batch,
                )
            )

            loss, diagnostics = dpo_loss(
                policy_chosen_logp=policy_chosen_logp,
                policy_rejected_logp=policy_rejected_logp,
                ref_chosen_logp=ref_chosen_logp,
                ref_rejected_logp=ref_rejected_logp,
                beta=selected_beta,
            )

            # Normally an optimizer update uses accumulation_steps
            # microbatches. The final window can be smaller, so its
            # divisor must use its actual size.
            window_start = (
                batch_index
                // accumulation_steps
            ) * accumulation_steps

            window_size = min(
                accumulation_steps,
                len(loader) - window_start,
            )

            (
                loss / float(window_size)
            ).backward()

            batch_metrics = {
                "loss": float(
                    loss.detach().item()
                ),
                "logit_mean": float(
                    diagnostics[
                        "logit_mean"
                    ].item()
                ),
                "policy_margin_mean": float(
                    diagnostics[
                        "policy_margin_mean"
                    ].item()
                ),
                "preference_accuracy": float(
                    diagnostics[
                        "preference_accuracy"
                    ].item()
                ),
            }

            # Weight metrics by the actual number of examples. This
            # matters for the final batch when it is smaller.
            for name in metric_names:
                weighted_value = (
                    batch_metrics[name]
                    * batch_size
                )

                window_sums[name] += (
                    weighted_value
                )

                run_sums[name] += (
                    weighted_value
                )

            window_examples += batch_size
            window_microbatches += 1
            run_examples += batch_size
            examples_seen += batch_size

            is_accumulation_boundary = (
                window_microbatches
                == window_size
            )

            if not is_accumulation_boundary:
                continue

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    max_grad_norm,
                )
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            update_step += 1

            record = {
                "run_name": run_name,
                "epoch": epoch_index + 1,
                "update": update_step,
                "microbatches": (
                    window_microbatches
                ),
                "examples_seen": examples_seen,
                "beta": selected_beta,
                "loss": (
                    window_sums["loss"]
                    / window_examples
                ),
                "logit_mean": (
                    window_sums[
                        "logit_mean"
                    ]
                    / window_examples
                ),
                "policy_margin_mean": (
                    window_sums[
                        "policy_margin_mean"
                    ]
                    / window_examples
                ),
                "preference_accuracy": (
                    window_sums[
                        "preference_accuracy"
                    ]
                    / window_examples
                ),
                "grad_norm": float(
                    grad_norm.detach().item()
                ),
                "elapsed_seconds": float(
                    timer()
                ),
            }

            append_jsonl(
                log_path,
                record,
            )

            print(
                f"[{run_name}] "
                f"epoch={record['epoch']} "
                f"update={record['update']} "
                f"loss={record['loss']:.4f} "
                f"preference_accuracy="
                f"{record['preference_accuracy']:.4f} "
                f"grad_norm="
                f"{record['grad_norm']:.4f}",
                flush=True,
            )

            window_sums = {
                name: 0.0
                for name in metric_names
            }

            window_examples = 0
            window_microbatches = 0

    if run_examples == 0:
        raise RuntimeError(
            "DPO training processed no examples."
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(output)
    tokenizer.save_pretrained(output)

    elapsed_seconds = float(
        timer()
    )

    peak_vram_gib = None

    if torch.cuda.is_available():
        peak_vram_gib = float(
            torch.cuda.max_memory_allocated(
                device
            )
            / (1024 ** 3)
        )

    preprocessing_rule = (
        "Preserve the complete prompt. Exclude an example when "
        "the prompt plus generation prefix alone has at least "
        "max_sequence_length tokens. For retained preference "
        "pairs, truncate chosen and rejected responses from the "
        "right and preserve EOS."
    )

    preprocessing_manifest = {
        "name": run_name,
        "stage": "training",
        "dataset": str(
            bundle["dataset_path"]
        ),
        "rule": preprocessing_rule,
        "filter_order": (
            "filter_overlength_before_max_examples"
        ),
        "max_sequence_length": int(
            cfg["max_sequence_length"]
        ),
        "raw_num_rows": len(raw_rows),
        "eligible_num_rows": len(
            eligible_rows
        ),
        "excluded_num_rows": len(
            excluded_rows
        ),
        "requested_max_examples": (
            bundle[
                "requested_max_examples"
            ]
        ),
        "selected_num_rows": len(rows),
        "selected_examples": [
            _selected_example_record(
                row,
                selected_index,
            )
            for selected_index, row
            in enumerate(rows)
        ],
        "excluded_examples": (
            excluded_rows
        ),
    }

    save_json(
        filter_manifest_path,
        preprocessing_manifest,
    )

    summary = {
        "run_name": run_name,
        "config": config_path,
        "dataset": str(
            bundle["dataset_path"]
        ),
        "adapter_output": str(
            output_spec
        ),
        "seed": int(cfg["seed"]),
        "raw_num_train_rows": len(
            raw_rows
        ),
        "eligible_num_train_rows": len(
            eligible_rows
        ),
        "excluded_num_train_rows": len(
            excluded_rows
        ),
        "num_train_rows": len(rows),
        "examples_processed": (
            run_examples
        ),
        "preprocessing_rule": (
            preprocessing_rule
        ),
        "preprocessing_manifest": str(
            filter_manifest_path.relative_to(
                repo_path(".")
            )
        ),
        "epochs": epochs,
        "batch_size": int(
            cfg["batch_size"]
        ),
        "gradient_accumulation_steps": (
            accumulation_steps
        ),
        "effective_batch_size": (
            int(cfg["batch_size"])
            * accumulation_steps
        ),
        "learning_rate": float(
            cfg["learning_rate"]
        ),
        "weight_decay": float(
            cfg.get(
                "weight_decay",
                0.0,
            )
        ),
        "beta": selected_beta,
        "max_sequence_length": int(
            cfg["max_sequence_length"]
        ),
        "max_grad_norm": (
            max_grad_norm
        ),
        "optimizer_updates": (
            update_step
        ),
        "total_parameters": (
            total_parameters
        ),
        "trainable_parameters": (
            trainable_count
        ),
        "mean_train_loss": (
            run_sums["loss"]
            / run_examples
        ),
        "mean_train_logit": (
            run_sums["logit_mean"]
            / run_examples
        ),
        "mean_train_policy_margin": (
            run_sums[
                "policy_margin_mean"
            ]
            / run_examples
        ),
        "mean_train_preference_accuracy": (
            run_sums[
                "preference_accuracy"
            ]
            / run_examples
        ),
        "wall_clock_seconds": (
            elapsed_seconds
        ),
        "peak_vram_gib": (
            peak_vram_gib
        ),
        "train_log": str(
            log_path.relative_to(
                repo_path(".")
            )
        ),
    }

    save_json(
        summary_path,
        summary,
    )

    print(
        f"Saved adapter to {output}",
        flush=True,
    )

    print(
        f"Saved training log to {log_path}",
        flush=True,
    )

    print(
        "Saved training summary to "
        f"{summary_path}",
        flush=True,
    )

    print(
        "Saved preprocessing manifest to "
        f"{filter_manifest_path}",
        flush=True,
    )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train a LoRA policy using Direct "
            "Preference Optimization."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/dpo.yaml",
    )

    parser.add_argument(
        "--run-name",
        default="standard",
    )

    parser.add_argument(
        "--dataset",
    )

    parser.add_argument(
        "--output",
    )

    parser.add_argument(
        "--beta",
        type=float,
    )

    parser.add_argument(
        "--max-examples",
        type=int,
    )

    args = parser.parse_args()

    run_training(
        config_path=args.config,
        run_name=args.run_name,
        dataset_path=args.dataset,
        output_path=args.output,
        beta=args.beta,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()