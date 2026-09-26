"""Nearest-neighbor fallback over the parallel CSV pairs.

This is not a trained generator. It copies the slang side of the closest
training example when the query is similar enough; otherwise it uses
regex lexical slang.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import CLEANED_CSV, FALLBACK_JSONL
from .preprocess import light_clean, load_all_training_pairs
from .style import _normalize, enforce_slang

_PUNCT = re.compile(r"[?.!,;:]+$")


def _key(text: str) -> str:
    return _PUNCT.sub("", _normalize(text))

# Below this cosine, retrieved slang is too unrelated to use.
MIN_COSINE = 0.82


def _pair_rows() -> list[tuple[str, str]]:
    if FALLBACK_JSONL.is_file():
        rows: list[tuple[str, str]] = []
        with FALLBACK_JSONL.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                formal, slang = rec.get("formal", ""), rec.get("slang", "")
                if formal and slang:
                    rows.append((formal, slang))
        if rows:
            return rows
    if CLEANED_CSV.is_file():
        import pandas as pd

        df = pd.read_csv(CLEANED_CSV)
        if {"formal", "slang"}.issubset(df.columns):
            return [
                (light_clean(f), light_clean(s))
                for f, s in zip(df["formal"], df["slang"])
                if isinstance(f, str) and isinstance(s, str) and f.strip() and s.strip()
            ]
    df = load_all_training_pairs()
    return list(zip(df["formal"].tolist(), df["slang"].tolist()))


def write_fallback_jsonl() -> int:
    """Materialize the lookup table from CSVs for deploy (Spaces, Docker)."""
    df = load_all_training_pairs()
    FALLBACK_JSONL.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with FALLBACK_JSONL.open("w", encoding="utf-8") as fh:
        for formal, slang in zip(df["formal"], df["slang"]):
            fh.write(json.dumps({"formal": formal, "slang": slang}, ensure_ascii=False) + "\n")
            n += 1
    return n


@lru_cache(maxsize=1)
def _index():
    pairs = _pair_rows()
    exact: dict[str, str] = {}
    docs: list[str] = []
    targets: list[str] = []
    for formal, slang in pairs:
        exact[_key(formal)] = slang
        exact.setdefault(_key(slang), slang)
        docs.append(formal)
        targets.append(slang)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
    matrix = vectorizer.fit_transform(docs)
    return exact, vectorizer, matrix, targets


def retrieve(text: str) -> dict | None:
    query = light_clean(text)
    if not query:
        return None
    exact, vectorizer, matrix, targets = _index()
    key = _key(query)
    if key in exact:
        return {"slang": exact[key], "score": 1.0, "match": "exact"}
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, matrix)[0]
    idx = int(sims.argmax())
    score = float(sims[idx])
    if score < MIN_COSINE:
        return None
    return {"slang": targets[idx], "score": score, "match": "nearest"}


def fallback_rewrite(text: str) -> dict:
    """Best non-LLM rewrite: corpus pair if close, else lexical rules."""
    hit = retrieve(text)
    lexical = enforce_slang(text, text)
    if hit:
        slang = hit["slang"]
        if hit["match"] == "exact" or hit["score"] >= MIN_COSINE:
            return {
                "text": slang,
                "mode": "corpus",
                "corpus_score": hit["score"],
                "match": hit["match"],
            }
    return {"text": lexical, "mode": "lexical", "corpus_score": None, "match": None}
