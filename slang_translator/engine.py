"""Generation helpers. Heavy ML imports stay inside functions so eval/preprocess can run without torch."""

from __future__ import annotations

import os

from .config import ADAPTER_FALLBACK, ADAPTER_REPO, BASE_MODEL
from .env import load_env
from .style import build_messages, enforce_slang, render_llama_prompt


def extract_assistant(decoded: str) -> str:
    if "assistant" in decoded:
        return decoded.split("assistant")[-1].strip()
    return decoded.strip()


class SlangEngine:
    def __init__(self, adapter: str | None = None, require_adapter: bool = True):
        import torch
        from huggingface_hub import login
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_env()
        token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        if token:
            login(token=token, add_to_git_credential=False)

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            token=token,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        last_error = None
        loaded = False
        candidates = []
        for repo in (adapter, ADAPTER_REPO, ADAPTER_FALLBACK):
            if not repo:
                continue
            candidates.append((repo, None))
            candidates.append((repo, "final_checkpoint"))
        seen = set()
        for repo, subfolder in candidates:
            key = (repo, subfolder)
            if key in seen:
                continue
            seen.add(key)
            try:
                kwargs = {"subfolder": subfolder} if subfolder else {}
                self.model = PeftModel.from_pretrained(self.model, repo, **kwargs)
                loaded = True
                self.adapter_id = f"{repo}/{subfolder}" if subfolder else repo
                break
            except Exception as exc:  # noqa: BLE001 — try the next location
                last_error = exc
        if require_adapter and not loaded:
            raise RuntimeError(f"Could not load LoRA adapter. Last error: {last_error}")
        self.adapter_loaded = loaded
        self.model.eval()

    def generate(self, text: str, reverse: bool = False, max_new_tokens: int = 96) -> str:
        import torch

        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                build_messages(text, reverse=reverse),
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = render_llama_prompt(text, reverse=reverse)

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.85,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        hyp = extract_assistant(decoded)
        if reverse:
            return hyp
        return enforce_slang(text, hyp)
