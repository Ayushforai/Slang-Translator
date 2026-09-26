import re
import warnings
from pathlib import Path

import pandas as pd

_WS = re.compile(r"\s+")


def light_clean(text) -> str:
    """Keep slang intact: do not expand contractions or 'correct' spelling."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = _WS.sub(" ", text).strip()
    return text


def load_parallel(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    formal_col = (
        cols.get("formal_text")
        or cols.get("formal_text_cleaned")
        or cols.get("formal")
        or cols.get("normal")
        or cols.get("plain english")
    )
    informal_col = (
        cols.get("informal_text")
        or cols.get("informal_text_cleaned")
        or cols.get("slang")
        or cols.get("gen_z")
        or cols.get("gen-z slang")
        or cols.get("genz slang")
    )
    if not formal_col or not informal_col:
        raise ValueError(f"Expected formal/slang columns in {path}, got {list(df.columns)}")
    out = pd.DataFrame(
        {
            "formal": df[formal_col].map(light_clean),
            "slang": df[informal_col].map(light_clean),
        }
    )
    out = out[(out["formal"].str.len() > 0) & (out["slang"].str.len() > 0)]
    out = out.drop_duplicates(subset=["formal", "slang"]).reset_index(drop=True)
    return out


def iter_data_csv_paths() -> list[Path]:
    """All parallel-pair CSVs under Dataa/, excluding generated splits."""
    from .config import DATA_DIR, EXCLUDED_DATA_CSV_NAMES, GENZ_CSVS, RAW_CSV

    ordered: list[Path] = []
    seen: set[Path] = set()
    for path in (RAW_CSV, *GENZ_CSVS):
        if not path.is_file():
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    for path in sorted(DATA_DIR.rglob("*.csv")):
        if path.name in EXCLUDED_DATA_CSV_NAMES:
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered


def load_all_training_pairs() -> pd.DataFrame:
    """Merge every formal/slang CSV under Dataa/ (see iter_data_csv_paths)."""
    frames: list[pd.DataFrame] = []
    for path in iter_data_csv_paths():
        try:
            frames.append(load_parallel(path))
        except ValueError as exc:
            warnings.warn(f"Skipping {path}: {exc}", stacklevel=1)
    if not frames:
        raise FileNotFoundError(
            "No training CSVs found under Dataa/. Add pair files with formal/slang columns."
        )
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["formal", "slang"]).reset_index(drop=True)

