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

## Short intro

This project rewrites **standard or formal English into modern slang** (one direction only: formal → slang, not the reverse). It combines a **QLoRA–fine-tuned Llama 3.2 1B Instruct** adapter with a **Flask** web UI, optional **formality detection**, and **fallback** paths (CSV nearest-neighbor + lexical rules) when the LLM cannot load or copies the input.

**Live app:** [ayushforai/slang-translator-web](https://huggingface.co/spaces/ayushforai/slang-translator-web)  
**Adapter on Hugging Face:** [ayushforai/slang-translator-llama-1b](https://huggingface.co/ayushforai/slang-translator-llama-1b)

## Technologies used

| Area | Stack |
|------|--------|
| **Model** | `meta-llama/Llama-3.2-1B-Instruct`, PEFT LoRA, 4-bit QLoRA (GPU training) |
| **Training** | Hugging Face `transformers`, `trl` (SFT), `datasets`, `accelerate`, `bitsandbytes` |
| **App** | Flask, HTML/CSS/Bootstrap |
| **Deploy** | Docker, Hugging Face Spaces |
| **Data & ML utilities** | `pandas`, `scikit-learn` (TF-IDF + logistic formality detector), `joblib` |
| **Eval** | Custom BLEU / slang-score / copy-rate scripts |
| **Auth / Hub** | `huggingface_hub`, `python-dotenv` (`HUGGINGFACE_HUB_TOKEN` for gated Llama) |

## Features

- **Instruction-tuned rewriting** via Llama 3.2 chat template (`Rewrite this in slang:`).
- **LoRA adapter** (~11.27M trainable weights, ~0.91% of base) hosted on the Hub.
- **Train / val / test splits** from parallel CSVs (office-casual + Gen-Z pair files).
- **Formality detector** (TF-IDF + logistic regression) for register labels in the UI.
- **Layered inference fallback:** LLM → corpus match (`fallback_pairs.jsonl`) → regex lexical slang.
- **CLI:** `prepare`, `train-detector`, `eval-baselines`, `detect`.
- **Health endpoint** (`/health`) for Space and local checks.

## What users can do

- Type or paste **formal or neutral** sentences and get a **slang/casual rewrite**.
- Click **example phrases** in the UI to try common inputs.
- See **output register**, **mode** (`llm`, `corpus`, or `lexical`), and optional **corpus match score**.
- Run the same flow **locally** (with a valid HF token and enough RAM/CPU or GPU).
- (Developers) **Prepare data**, **train** the adapter on GPU, **evaluate** baselines, and **deploy** via Docker to Spaces.

## How to run the project

1. **Clone** the repo and create a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

2. **Configure Hugging Face** (required for Llama + adapter download):

```bash
copy .env.example .env          # Windows
# cp .env.example .env
```

Add your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Accept access for [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

3. **Prepare data** (splits + `fallback_pairs.jsonl`):

```bash
python -m slang_translator.cli prepare
python -m slang_translator.cli train-detector
```

4. **Run the web app:**

```bash
python app.py
```

Open **http://localhost:7860** (default port **7860**, overridable with `PORT`).

5. **Optional — train on GPU** (after `prepare`):

```bash
set HUGGINGFACE_HUB_TOKEN=...   # Windows
python Deployment/fine_tune.py
```

See [Training](#training-deploymentfine_tunepy) for Kaggle/Colab notes.

## Live demo video

<!-- Add your demo link or embed here -->

_TODO: link to screen recording (local use + Hugging Face Space)._

## Keyboard shortcuts

| Action | Shortcut |
|--------|----------|
| **Translate** | `Enter` (with focus in the text area) |
| **New line in input** | `Shift` + `Enter` |

## The process

1. **Data:** Merge `raw_data_fixed.csv` and Gen-Z CSVs under `Dataa/genz/` into cleaned parallel **formal ↔ slang** pairs (~2.5k unique rows).
2. **Cleaning:** Light normalization that **preserves contractions and slang** on the target side.
3. **Formatting:** Build Llama 3.2 instruct prompts; write `train.jsonl`, `val.jsonl`, `test.csv`, and `fallback_pairs.jsonl`.
4. **Training:** QLoRA SFT on frozen 4-bit Llama 3.2 1B (LoRA rank 16 on attention + MLP); save `final_checkpoint/` and publish adapter to the Hub.
5. **Inference:** Load base + adapter; generate with chat template; **enforce_slang** + corpus retrieval if output is too close to the input.
6. **Deploy:** Docker image on Hugging Face Spaces; `HUGGINGFACE_HUB_TOKEN` as a Space secret.

## What I learned

- **Parameter-efficient fine-tuning** (QLoRA) makes a 1B instruct model usable on consumer GPUs without full-weight updates.
- **Training loss and token accuracy** can look strong while **test BLEU** and human “slanginess” lag—evaluation must be split by metric type.
- **Gated models and Spaces** need valid tokens in secrets; a bad token forces fallback-only behavior.
- **Dataset labels** matter: many pairs are “corporate → casual,” not Gen-Z; the model and metrics reflect that ceiling.
- **Production-minded inference:** lazy model load, CPU-safe `device_map`, and CSV/rule fallbacks keep the demo usable when the LLM fails.

## Overall growth

- End-to-end ownership: **data pipeline → training → eval → Flask UI → Hub adapter → Spaces deploy**.
- Practiced **honest reporting** (identity BLEU floor, non-LLM fallback BLEU, training vs test metrics).
- Improved **debugging** across local CPU limits, multi-GPU training pitfalls, and remote runtime logs.

## How can it be improved

- **Richer slang targets** in training data (more Gen-Z, fewer mild paraphrases); retrain and re-evaluate.
- **GPU Space or Inference Endpoint** for reliable `via llm` latency and fewer OOMs on free CPU.
- **Stronger retrieval** (embeddings instead of TF-IDF cosine) or a small dual-encoder for fallback.
- **Human eval** or LLM-as-judge for register/slang quality beyond BLEU.
- **Optional bidirectional** mode (slang → formal) with a router and reversed pairs—see [Bidirectional (formal ↔ slang)](#bidirectional-formal--slang--not-in-this-product).
- **Sync Space with GitHub** or a one-command deploy script to avoid manual `hf upload`.

## Evaluation metrics

### Training dynamics (published adapter run, train split — not test BLEU)

Logged in `checkpoint-2106/trainer_state.json`:

| | Start (step 10) | End (step 2100 / 2106) |
|---|---|---|
| Cross-entropy loss | 4.75 | 0.267 (−94.4%) |
| Predictive entropy | 2.80 | 0.273 |
| Token accuracy | 33.9% | 89.5% |

Adapter: **11.27M** LoRA parameters (**~0.91%** of Llama 3.2 1B). Newer data splits use ~**1,879 / 236 / 236** train/val/test and ~**705** GPU steps (3 epochs, effective batch 8).

### Held-out test split (`Dataa/test.csv`, n = 236)

| Metric | Value | Meaning |
|--------|-------|---------|
| Identity corpus BLEU | **25.0** | Copy formal input → reference slang (baseline floor) |
| Lexical + CSV fallback corpus BLEU | **36.4** | Non-LLM rewrite path vs references |
| Formality detector (held-out) | **~96.7%** acc / macro-F1 | Formal vs slang **columns** in the corpus (register detection, not generation) |

Reproduce baselines:

```bash
python -m slang_translator.cli prepare
python -m slang_translator.cli eval-baselines
python Scriptss/evaluate.py --lexical-only
# with GPU + adapter loaded:
python Scriptss/evaluate.py
```

**Note:** Report adapter **test BLEU** only after `Scriptss/evaluate.py` with the LoRA loaded. Do not confuse train **loss** with slang quality on new sentences.

---

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

### Interpreting training logs

- **`loss` / `eval_loss`:** Next-token prediction error on the formatted SFT strings. Lower is better. **`eval_loss`** is the generalization signal on the val split.
- **`mean_token_accuracy`:** Argmax match on training batches; high late values are normal but do not prove slang quality on new sentences.
- **Checkpoints:** Use `final_checkpoint` for inference or Hub upload.

### Cloud training (Kaggle / Colab)

1. Enable **GPU** and **Internet**.
2. Store **`HUGGINGFACE_HUB_TOKEN`** in notebook secrets.
3. Clone this repo, `pip install -r requirements.txt` (or the training subset above).
4. `python Deployment/fine_tune.py` — expect on the order of **20–60 minutes** on a T4 for ~700 steps.

## Project layout

- `slang_translator/` — preprocess, prompts, style score, detector, retrieval fallback, metrics, generation
- `Dataa/` — raw pairs, cleaned CSV, `train.jsonl` / `val.jsonl` / `test.csv`, `fallback_pairs.jsonl`
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
