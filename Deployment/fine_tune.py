"""Deployment QLoRA training: Llama 3.2 1B, formal → slang only."""

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


def main():
    if not TRAIN_PATH.is_file():
        raise SystemExit("Missing Dataa/train.jsonl. Run: python -m slang_translator.cli prepare")

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"GPU available: {torch.cuda.get_device_name(0)} (using cuda:0 only)")
    else:
        print("GPU available: False")

    bnb_config = None
    if use_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
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
        torch_dtype=torch.bfloat16 if use_gpu else torch.float32,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_gpu:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_arguments = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4 if use_gpu else 1,
        per_device_eval_batch_size=4 if use_gpu else 1,
        gradient_accumulation_steps=2,
        eval_strategy="steps" if "validation" in dataset else "no",
        eval_steps=100 if "validation" in dataset else None,
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        bf16=use_gpu and torch.cuda.is_bf16_supported(),
        fp16=False,
        optim="paged_adamw_32bit" if use_gpu else "adamw_torch",
        report_to="none",
        load_best_model_at_end=bool("validation" in dataset),
        metric_for_best_model="eval_loss" if "validation" in dataset else None,
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
