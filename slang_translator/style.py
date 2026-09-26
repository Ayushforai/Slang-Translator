import re
from typing import Optional

from .config import (
    INFORMAL_BOOST,
    LEXICAL_SLANG,
    MIN_SLANG_SCORE,
    SLANG_MARKERS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_REVERSE,
    USER_TEMPLATE,
    USER_TEMPLATE_REVERSE,
)


_CONTRACTION = re.compile(r"\b\w+n't\b|\b(?:i'm|you're|we're|they're|it's|that's|what's|who's|let's|y'all)\b", re.I)


def slang_score(text: str) -> float:
    """Heuristic 0–1 score: slang markers, contractions, missing polish."""
    if not text or not text.strip():
        return 0.0
    lowered = f" {text.lower()} "
    hits = sum(1 for m in SLANG_MARKERS if m in lowered)
    contractions = len(_CONTRACTION.findall(text))
    informal_punct = text.count("!") + text.count("...")
    length_penalty = 0.0
    # Title-case corporate sentences tend to be longer and more function-word heavy.
    tokens = re.findall(r"[A-Za-z']+", text)
    if not tokens:
        return 0.0
    raw = (hits * 2 + contractions + informal_punct) / max(len(tokens), 1)
    return min(1.0, raw + length_penalty)


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def too_similar(source: str, hypothesis: str, threshold: float = 0.82) -> bool:
    """True when the model barely changed the input."""
    if not hypothesis.strip():
        return True
    if hypothesis.strip().lower() == source.strip().lower():
        return True
    return jaccard(source, hypothesis) >= threshold and slang_score(hypothesis) <= slang_score(source) + 0.02


def lexical_slangify(text: str) -> str:
    """Deterministic fallback so outputs still move toward slang if the LM copies the input."""
    out = text.strip()
    for pattern, repl in LEXICAL_SLANG:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    if out:
        out = out[0].upper() + out[1:] if len(out) > 1 else out.upper()
    return out


def informal_boost(text: str) -> str:
    """Second pass for model outputs that are casual but still read as formal."""
    out = text.strip()
    for pattern, repl in INFORMAL_BOOST:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return lexical_slangify(out)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def enforce_slang(source: str, hypothesis: str) -> str:
    hyp = (hypothesis or "").strip()
    from_source = informal_boost(lexical_slangify(source))
    candidates: list[str] = [from_source]
    if hyp:
        from_hyp = informal_boost(lexical_slangify(hyp))
        if _normalize(from_hyp) != _normalize(source):
            candidates.append(from_hyp)
        if not too_similar(source, hyp):
            candidates.append(hyp)

    def rank(text: str) -> tuple:
        changed = int(_normalize(text) != _normalize(source))
        return (changed, slang_score(text))

    best = max(candidates, key=rank)
    if _normalize(best) == _normalize(source):
        best = from_source
    return best or from_source


def build_messages(text: str, reverse: bool = False, target: Optional[str] = None) -> list[dict]:
    system = SYSTEM_PROMPT_REVERSE if reverse else SYSTEM_PROMPT
    user = (USER_TEMPLATE_REVERSE if reverse else USER_TEMPLATE).format(text=text)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def render_llama_prompt(text: str, reverse: bool = False, target: Optional[str] = None) -> str:
    """Llama 3.2 instruct template (matches tokenizer.apply_chat_template)."""
    system = SYSTEM_PROMPT_REVERSE if reverse else SYSTEM_PROMPT
    user = (USER_TEMPLATE_REVERSE if reverse else USER_TEMPLATE).format(text=text)
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if target is not None:
        prompt += f"{target}<|eot_id|>"
    return prompt
