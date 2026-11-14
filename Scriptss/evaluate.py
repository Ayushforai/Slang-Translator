"""Evaluate a model or the lexical fallback on Dataa/test.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from slang_translator.config import TEST_CSV
from slang_translator.metrics import corpus_bleu, sentence_bleu
from slang_translator.style import enforce_slang, jaccard, slang_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexical-only", action="store_true", help="Do not load the LLM")
    parser.add_argument(
        "--adapter",
        default="",
        help="Local adapter dir or Hub repo. Default: Hub adapter in config.",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(TEST_CSV)
    if args.limit:
        df = df.head(args.limit)

    hyps = []
    if args.lexical_only:
        hyps = [enforce_slang(f, f) for f in df["formal"]]
    else:
        from slang_translator.engine import SlangEngine

        engine = SlangEngine(
            adapter=args.adapter or None,
            require_adapter=True,
        )
        hyps = [engine.generate(f) for f in df["formal"]]

    pairs = list(zip(hyps, df["slang"]))
    report = {
        "n": len(df),
        "corpus_bleu_vs_reference": corpus_bleu(pairs),
        "mean_sentence_bleu_vs_reference": sum(sentence_bleu(h, r) for h, r in pairs) / len(pairs),
        "mean_hypothesis_slang_score": sum(slang_score(h) for h in hyps) / len(hyps),
        "mean_source_slang_score": float(df["formal"].map(slang_score).mean()),
        "mean_reference_slang_score": float(df["slang"].map(slang_score).mean()),
        "mean_jaccard_to_source": sum(jaccard(s, h) for s, h in zip(df["formal"], hyps)) / len(hyps),
        "copy_rate": sum(h.strip().lower() == s.strip().lower() for h, s in zip(hyps, df["formal"])) / len(hyps),
        "examples": [
            {"src": df.iloc[i]["formal"], "hyp": hyps[i], "ref": df.iloc[i]["slang"]}
            for i in range(min(5, len(df)))
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
