"""Project CLI: preprocess, train detector, evaluate baselines."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import CLEANED_CSV, TEST_CSV
from .detector import FormalityDetector, train_detector
from .format import build_splits
from .metrics import corpus_bleu, sentence_bleu
from .preprocess import load_all_training_pairs, load_parallel
from .retrieval import write_fallback_jsonl
from .style import enforce_slang, slang_score


def cmd_prepare(_args) -> None:
    stats = build_splits()
    stats["fallback_pairs"] = write_fallback_jsonl()
    print(json.dumps(stats, indent=2))


def cmd_detector(_args) -> None:
    df = load_all_training_pairs()
    result = train_detector(df)
    print(json.dumps(result, indent=2))


def _eval_frame(df: pd.DataFrame) -> dict:
    identity = list(zip(df["formal"], df["slang"]))
    lexical = [(enforce_slang(f, f), s) for f, s in identity]
    oracle = list(zip(df["slang"], df["slang"]))
    return {
        "n": len(df),
        "source_slang_score": float(df["formal"].map(slang_score).mean()),
        "target_slang_score": float(df["slang"].map(slang_score).mean()),
        "identity_corpus_bleu": corpus_bleu(identity),
        "lexical_fallback_corpus_bleu": corpus_bleu(lexical),
        "oracle_corpus_bleu": corpus_bleu(oracle),
        "identity_mean_sentence_bleu": float(
            sum(sentence_bleu(f, s) for f, s in identity) / len(df)
        ),
    }


def cmd_eval(_args) -> None:
    path = TEST_CSV if TEST_CSV.is_file() else CLEANED_CSV
    df = pd.read_csv(path)
    if "formal" not in df.columns:
        df = load_parallel(path)
    print(json.dumps({"split": str(path), **_eval_frame(df)}, indent=2))


def cmd_detect_text(args) -> None:
    det = FormalityDetector()
    print(json.dumps(det.predict(args.text), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal → slang project tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="Clean data, Llama prompts, train/val/test splits, fallback JSONL")
    sub.add_parser("train-detector", help="Fit TF-IDF formality classifier")
    sub.add_parser("eval-baselines", help="BLEU / slang-score bounds on the test split")
    detect = sub.add_parser("detect", help="Classify one sentence as formal or informal")
    detect.add_argument("text")
    args = parser.parse_args()
    dispatch = {
        "prepare": cmd_prepare,
        "train-detector": cmd_detector,
        "eval-baselines": cmd_eval,
        "detect": cmd_detect_text,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
