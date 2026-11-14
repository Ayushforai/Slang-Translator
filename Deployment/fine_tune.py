"""Deployment QLoRA training: Llama 3.2 1B, formal → slang only."""

import inspect
import math
import os
import sys
from pathlib import Path

# Kaggle "T4 x2" exposes two GPUs; device_map="auto" shards the model and Trainer crashes with
# "found one of them on device: cuda:1". QLoRA on 1B only needs one GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from slang_translator.env import load_env

load_env()

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "Dataa" / "train.jsonl"
VAL_PATH = ROOT / "Dataa" / "val.jsonl"
OUTPUT_DIR = ROOT / "models" / "slang_translator_llama_1b"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)


def _mixed_precision() -> tuple:
    """Return (compute_dtype, use_bf16, use_fp16).

    T4 on Colab: use float16 for 4-bit matmuls but **no** Trainer AMP (bf16/fp16
    grad scaler can crash at step 0 with NotImplementedError for BFloat16).
    """
    if not torch.cuda.is_available():
        return torch.float32, False, False
    gpu = torch.cuda.get_device_name(0).upper()
    fp16_gpus = ("T4", "TESLA T4", "GTX", "1650", "1660", "1070", "1080", "P100", "K80")
    force_fp16 = os.getenv("FORCE_FP16", "").lower() in ("1", "true", "yes")
    if force_fp16 or any(tag in gpu for tag in fp16_gpus):
        return torch.float16, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


def _training_arguments(total_steps: int, **kwargs):
    """Build TrainingArguments across Colab / local transformers version differences."""
    params = set(inspect.signature(TrainingArguments.__init__).parameters) - {"self"}

    if "eval_strategy" in kwargs and "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    ws = kwargs.get("warmup_steps")
    if ws is not None:
        if "warmup_steps" not in params and "warmup_ratio" in params and total_steps > 0:
            kwargs.pop("warmup_steps", None)
            kwargs["warmup_ratio"] = min(1.0, ws / total_steps)
        elif "warmup_steps" in params:
            kwargs["warmup_steps"] = ws
        else:
            kwargs.pop("warmup_steps", None)
    kwargs.pop("warmup_ratio", None)

    allowed = {k: v for k, v in kwargs.items() if k in params and v is not None}
    return TrainingArguments(**allowed)


def main():
    if not TRAIN_PATH.is_file():
        raise SystemExit("Missing Dataa/train.jsonl. Run: python -m slang_translator.cli prepare")

    print(f"Using fine_tune script: {Path(__file__).resolve()}")

    use_gpu = torch.cuda.is_available()
    compute_dtype, use_bf16, use_fp16 = _mixed_precision()
    if use_gpu:
        print(
            f"GPU available: {torch.cuda.get_device_name(0)} (using cuda:0 only, "
            f"dtype={'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp16-load/amp-off' if compute_dtype == torch.float16 else 'fp32'})"
        )
    else:
        print("GPU available: False")

    bnb_config = None
    if use_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    data_files = {"train": str(TRAIN_PATH)}
    if VAL_PATH.is_file():
        data_files["validation"] = str(VAL_PATH)
    dataset = load_dataset("json", data_files=data_files)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0} if use_gpu else None,
        token=hf_token,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_gpu:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_bs = 4 if use_gpu else 1
    grad_accum = 2
    epochs = 3
    train_n = len(dataset["train"])
    steps_per_epoch = max(1, math.ceil(train_n / (train_bs * grad_accum)))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * 0.03))

    has_val = "validation" in dataset
    training_arguments = _training_arguments(
        total_steps,
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=train_bs,
        gradient_accumulation_steps=grad_accum,
        eval_strategy="steps" if has_val else "no",
        eval_steps=100 if has_val else None,
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="paged_adamw_32bit" if use_gpu else "adamw_torch",
        report_to="none",
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
    )

    trainer_kwargs = dict(
        model=model,
        train_dataset=dataset["train"],
        args=training_arguments,
        processing_class=tokenizer,
    )
    if "validation" in dataset:
        trainer_kwargs["eval_dataset"] = dataset["validation"]

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    dest = OUTPUT_DIR / "final_checkpoint"
    trainer.save_model(str(dest))
    tokenizer.save_pretrained(str(dest))
    print(f"Saved adapter to {dest}")


if __name__ == "__main__":
    main()
