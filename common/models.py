from __future__ import annotations

import gc
from contextlib import contextmanager
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from common.data import repo_path


def resolve_dtype(name: str):
    name = str(name).lower()

    if name in {"bf16", "bfloat16"}:
        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            return torch.bfloat16

        return torch.float16

    if name in {"fp16", "float16", "half"}:
        return torch.float16

    return torch.float32


def load_tokenizer(
    model_id: str,
    padding_side: str = "left",
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side=padding_side,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = padding_side
    return tokenizer


def make_lora_config(cfg: dict) -> LoraConfig:
    lora_cfg = cfg["lora"]

    return LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(
            lora_cfg["target_modules"]
        ),
    )


def make_value_lora_config(
    cfg: dict,
) -> LoraConfig:
    lora_cfg = cfg.get(
        "value_lora",
        cfg["lora"],
    )

    return LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        task_type="SEQ_CLS",
        target_modules=list(
            lora_cfg["target_modules"]
        ),
        modules_to_save=["score"],
    )


def load_policy(
    cfg: dict,
    adapter_path: str | None = None,
    trainable: bool = False,
    fresh_lora: bool = False,
):
    dtype = resolve_dtype(
        cfg.get("dtype", "float16")
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    )

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )
    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    if adapter_path:
        model = PeftModel.from_pretrained(
            model,
            str(repo_path(adapter_path)),
            is_trainable=trainable,
        )

    elif fresh_lora:
        model = get_peft_model(
            model,
            make_lora_config(cfg),
        )

    if torch.cuda.is_available():
        model = model.cuda()

    if trainable:
        model.train()
        model.config.use_cache = False

        if hasattr(
            model,
            "gradient_checkpointing_enable",
        ):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={
                        "use_reentrant": False,
                    }
                )
            except TypeError:
                model.gradient_checkpointing_enable()

        if hasattr(
            model,
            "enable_input_require_grads",
        ):
            model.enable_input_require_grads()

    else:
        model.eval()

    return model


@contextmanager
def reference_mode(model):
    was_training = model.training
    model.eval()

    try:
        if isinstance(model, PeftModel):
            with model.disable_adapter():
                yield
        else:
            yield

    finally:
        if was_training:
            model.train()


def _quant_config(
    bits: int | None,
    dtype,
):
    if (
        bits is None
        or not torch.cuda.is_available()
    ):
        return None

    if bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    raise ValueError(
        f"Unsupported quantization bits: {bits}"
    )


def load_reward_model(cfg: dict):
    dtype = resolve_dtype(
        cfg.get("dtype", "float16")
    )

    quantization_config = _quant_config(
        (
            8
            if cfg.get(
                "quantize_frozen_models",
                True,
            )
            else None
        ),
        dtype,
    )

    kwargs = {
        "num_labels": 1,
        "low_cpu_mem_usage": True,
    }

    if quantization_config is not None:
        kwargs.update(
            {
                "quantization_config":
                    quantization_config,
                "device_map": "auto",
            }
        )
    else:
        kwargs["dtype"] = dtype

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            cfg["reward_model"],
            **kwargs,
        )
    )

    if (
        quantization_config is None
        and torch.cuda.is_available()
    ):
        model = model.cuda()

    # The tokenizer stored with the reward-model
    # repository was incompatible with the pinned
    # Transformers version during instructor
    # preparation. Use the configured canonical
    # tokenizer intentionally.
    tokenizer = load_tokenizer(
        cfg.get(
            "reward_tokenizer",
            cfg["base_model"],
        ),
        padding_side="left",
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, tokenizer


def load_value_model(
    cfg: dict,
    checkpoint: str,
    train_mode: str = "lora_head",
):
    dtype = resolve_dtype(
        cfg.get("dtype", "float16")
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            str(repo_path(checkpoint)),
            num_labels=1,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
    )

    tokenizer = load_tokenizer(
        cfg["base_model"]
    )
    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )
    model.config.use_cache = False

    if train_mode == "lora_head":
        # The released critic is a full merged
        # checkpoint. A fresh zero-initialized
        # LoRA adapter preserves its starting value
        # function while providing trainable
        # continuation capacity.
        model = get_peft_model(
            model,
            make_value_lora_config(cfg),
        )

    elif train_mode == "head_only":
        for name, parameter in (
            model.named_parameters()
        ):
            parameter.requires_grad_(
                "score" in name
                or "classifier" in name
            )

    elif train_mode == "frozen":
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        if torch.cuda.is_available():
            model = model.cuda()

        model.eval()
        return model

    elif train_mode == "full":
        pass

    else:
        raise ValueError(
            "Unknown value_train_mode="
            f"{train_mode!r}"
        )

    if torch.cuda.is_available():
        model = model.cuda()

    model.train()
    return model


def value_parameter_groups(
    model,
    lora_lr: float,
    head_lr: float,
):
    """Create optimizer groups for the critic."""

    lora_parameters = []
    head_parameters = []
    unexpected_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if "lora_" in name:
            lora_parameters.append(parameter)

        elif (
            "score" in name
            or "classifier" in name
        ):
            head_parameters.append(parameter)

        else:
            unexpected_parameters.append(
                (name, parameter)
            )

    if unexpected_parameters:
        names = ", ".join(
            name
            for name, _ in (
                unexpected_parameters[:8]
            )
        )

        raise RuntimeError(
            "Unexpected trainable critic "
            f"parameters: {names}"
        )

    groups = []

    if lora_parameters:
        groups.append(
            {
                "params": lora_parameters,
                "lr": float(lora_lr),
                "name": "critic_lora",
            }
        )

    if head_parameters:
        groups.append(
            {
                "params": head_parameters,
                "lr": float(head_lr),
                "name": "critic_head",
            }
        )

    if not groups:
        raise RuntimeError(
            "Critic has no trainable parameters."
        )

    return groups


def token_values(
    value_model,
    input_ids,
    attention_mask,
):
    backbone = getattr(
        value_model,
        value_model.base_model_prefix,
    )

    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states[-1]

    if hasattr(value_model, "score"):
        value_head = value_model.score

    elif hasattr(value_model, "classifier"):
        value_head = value_model.classifier

    else:
        raise RuntimeError(
            "Could not locate scalar value head."
        )

    return value_head(
        hidden_states
    ).squeeze(-1)


def trainable_parameters(model):
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def count_parameters(model):
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def clear_gpu(*objects):
    for obj in objects:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()