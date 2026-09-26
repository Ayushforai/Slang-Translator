---
title: Formal to Slang
emoji: 🗣️
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Formal → Slang Rewriter

Rewrites **standard / formal English into modern slang**. It does **not** translate slang into formal English.

Live space: [ayushforai/slang-translator-web](https://huggingface.co/spaces/ayushforai/slang-translator-web)  
Adapter: [ayushforai/slang-translator-llama-1b](https://huggingface.co/ayushforai/slang-translator-llama-1b)

## Method

1. Parallel pairs: formal sentence ↔ slang/casual sentence.
2. Light cleaning that **keeps contractions and slang** on the target side.
3. Llama 3.2 instruct chat template (`Rewrite this in slang:`).
4. **QLoRA** SFT of `meta-llama/Llama-3.2-1B-Instruct` (LoRA rank 16 on attention + MLP).
5. Inference uses the chat template. If the model copies the input, a **lexical slang fallback** still shifts register.
6. Optional **TF-IDF + logistic regression** formality detector (routing for a future two-way model; the product always rewrites **toward slang**).

## Training (`Deployment/fine_tune.py`)

Training is **supervised fine-tuning (SFT)** with **QLoRA** on a frozen, 4-bit-quantized Llama 3.2 1B base. Only LoRA adapter weights are updated (~**11.27M** parameters, ~**0.9%** of the full model).

### Inputs and outputs

| Item | Path / value |
|---|---|
| Train split | `Dataa/train.jsonl` (field `text`: full Llama chat prompt + target slang) |
| Validation split | `Dataa/val.jsonl` (same format; used if present) |
| Base model | `meta-llama/Llama-3.2-1B-Instruct` (gated; needs `HUGGINGFACE_HUB_TOKEN`) |
| Checkpoints | `models/slang_translator_llama_1b/checkpoint-*` (every 100 steps) |
| Final adapter | `models/slang_translator_llama_1b/final_checkpoint/` |

Run after `python -m slang_translator.cli prepare`:

```bash
set HUGGINGFACE_HUB_TOKEN=...   # Windows
python Deployment/fine_tune.py
```

Recommended: **one GPU** with CUDA (Kaggle T4, Colab T4). CPU training is possible but very slow. On Kaggle **T4 x2**, the script pins **`CUDA_VISIBLE_DEVICES=0`** and uses `device_map={"": 0}` so the model stays on a single GPU (multi-GPU + `device_map="auto"` breaks the Trainer).

### QLoRA and LoRA settings

- **Quantization (GPU only):** 4-bit **NF4**, `bnb_4bit_compute_dtype=bfloat16`, double quant enabled.
- **LoRA:** `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `bias="none"`.
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Optimizer:** `paged_adamw_32bit` on GPU; `adamw_torch` on CPU.
- **Precision:** `bf16` when the GPU supports it; no fp16.

### Hyperparameters

| Setting | GPU | CPU |
|---|---|---|
| Epochs | 3 | 3 |
| `per_device_train_batch_size` | 4 | 1 |
| `per_device_eval_batch_size` | 4 | 1 |
| `gradient_accumulation_steps` | 2 | 2 |
| **Effective batch size** | **8** | **2** |
| Learning rate | `2e-4` | `2e-4` |
| LR schedule | cosine | cosine |
| Warmup | 3% of steps (`warmup_ratio=0.03`) | same |
| Weight decay | 0.01 | 0.01 |
| Max grad norm | 0.3 | 0.3 |

**Approximate steps per run** (with validation file present):

\[
\text{steps} = \left\lceil \frac{N_{\text{train}} \times \text{epochs}}{\text{batch} \times \text{grad\_accum}} \right\rceil
\]

Example: ~1,879 train rows → about **705** steps on GPU (3 epochs, effective batch 8). Step count changes if you re-run `prepare` and the train size changes.

### What gets logged during training

Hugging Face `Trainer` + TRL `SFTTrainer` write to the console (and to `trainer_state.json` inside each checkpoint folder). **`report_to="none"`** — no Weights & Biases unless you change it.

| Interval | What you see |
|---|---|
| **Every 10 steps** (`logging_steps=10`) | **`loss`** (train cross-entropy), **`mean_token_accuracy`**, **`learning_rate`**, **`epoch`**, and often **`entropy`** / **`grad_norm`** in the log history |
| **Every 100 steps** (`eval_steps=100`) | **`eval_loss`** on `Dataa/val.jsonl` (validation cross-entropy) |
| **Every 100 steps** (`save_steps=100`) | Checkpoint under `models/slang_translator_llama_1b/checkpoint-<step>/` |

After training completes, the best checkpoint (lowest **`eval_loss`**) is restored when validation is enabled (`load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`), then weights are saved again to **`final_checkpoint/`**.

### Interpreting metrics

- **`loss` / `eval_loss`:** Next-token prediction error on the formatted SFT strings. Lower is better. **`eval_loss`** is the honest generalization signal on the held-out val split; do not confuse train **`loss`** with test BLEU.
- **`mean_token_accuracy`:** Fraction of tokens where the model’s argmax matches the label (mostly on the training batch). High values late in training are normal; they do not prove slang quality on new sentences.
- **Checkpoints:** Use `final_checkpoint` for inference, or upload that folder to Hugging Face. Intermediate `checkpoint-*` dirs are useful if the session dies before the last step.

### Cloud training (Kaggle / Colab)

1. Enable **GPU** and **Internet**.
2. Store **`HUGGINGFACE_HUB_TOKEN`** in notebook secrets.
3. Clone this repo, `pip install transformers peft trl datasets accelerate bitsandbytes huggingface_hub`.
4. `python Deployment/fine_tune.py` — expect on the order of **20–60 minutes** on a T4 for ~700 steps.

## Honest metrics (from the published adapter run)

Logged in `checkpoint-2106/trainer_state.json` (train set, not a test BLEU):

| | Start (step 10) | End (step 2100 / 2106) |
|---|---|---|
| Cross-entropy loss | 4.75 | 0.267 (−94.4%) |
| Predictive entropy | 2.80 | 0.273 |
| Token accuracy | 33.9% | 89.5% |

Adapter size on disk: **11.27M LoRA parameters** (0.91% of Llama 3.2 1B, ~1.24B). Training used 4-bit NF4 when a GPU was available. Do not report 20.9M or 1.3B unless you re-count `print_trainable_parameters()` on a new run.

After changing data or prompts, retrain, then report **test** BLEU and slang-score from:

```bash
python -m slang_translator.cli prepare
python -m slang_translator.cli eval-baselines
python Scriptss/evaluate.py --lexical-only
# after GPU train:
python Scriptss/evaluate.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
python -m slang_translator.cli prepare
python -m slang_translator.cli train-detector
```

Train: see **[Training](#training-deploymentfine_tunepy)** above (`Deployment/fine_tune.py`).

Run the app:

```bash
python app.py
```

Open http://localhost:7860

## Project layout

- `slang_translator/` — preprocess, prompts, style score, detector, metrics, generation
- `Dataa/` — raw pairs, cleaned CSV, `train.jsonl` / `val.jsonl` / `test.csv`
- `Deployment/fine_tune.py` — QLoRA training on **train.jsonl only**
- `Scriptss/evaluate.py` — test-set BLEU, slang score, copy-rate
- `app.py` + `templates/index.html` — Flask UI

## Bidirectional (formal ↔ slang) — not in this product

To auto-detect register and convert **both** ways you would need:

1. **A router:** the TF-IDF detector already here, or a small classifier / LLM-as-judge. Mid-register sentences (“see you tomorrow”) are the failure mode.
2. **Two tasks in training:** same pairs reversed, with a different system prompt (`slang_to_formal`). Either one multi-task LoRA with a task prefix, or two adapters.
3. **Asymmetric data:** slang→formal is easier for Instruct models; formal→slang needs **much slangier** targets than office-casual paraphrases.
4. **Separate eval:** BLEU in the formal direction is meaningful; BLEU in the slang direction under-rewards valid slang that differs from the one reference.

Limitations: slang is many dialects; a wrong route makes text worse; reversing slang throws away tone; detectors trained on this CSV learn “corporate vs slightly casual,” not Twitter/Gen-Z.

## CV bullets you can defend

- Fine-tuned Llama 3.2 1B Instruct with QLoRA (rank 16, 11.3M trainable weights) for **formal → slang** rewriting.
- Train loss 4.75 → 0.267 (−94.4%) and entropy 2.80 → 0.273 over 2,106 steps; 89.5% train token accuracy (report as **training** dynamics).
- Flask + HTML/CSS app on Hugging Face Spaces; train/val/test split and BLEU/slang-score eval script.
