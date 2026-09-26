from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Dataa"
MODELS_DIR = ROOT / "models"

RAW_CSV = DATA_DIR / "raw_data_fixed.csv"
GENZ_DIR = DATA_DIR / "genz"
GENZ_CSVS = (
    GENZ_DIR / "genz_dataset.csv",
    GENZ_DIR / "gen_z_slangs_translation.csv",
)
# Outputs of prepare — must not be fed back in as training sources.
EXCLUDED_DATA_CSV_NAMES = frozenset({"cleaned_data.csv", "test.csv"})
CLEANED_CSV = DATA_DIR / "cleaned_data.csv"
TRAIN_JSONL = DATA_DIR / "train.jsonl"
VAL_JSONL = DATA_DIR / "val.jsonl"
TEST_CSV = DATA_DIR / "test.csv"
FORMATTED_JSONL = DATA_DIR / "formatted_dataset.jsonl"
DETECTOR_PATH = DATA_DIR / "formality_detector.joblib"
FALLBACK_JSONL = DATA_DIR / "fallback_pairs.jsonl"

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_REPO = "ayushforai/slang-translator-llama-1b"
ADAPTER_FALLBACK = "ayushnsfw/slang-translator-llama-1b"

TASK_FORMAL_TO_SLANG = "formal_to_slang"
TASK_SLANG_TO_FORMAL = "slang_to_formal"

SYSTEM_PROMPT = (
    "You are a slang rewriter. Convert standard or formal English into modern "
    "internet slang. Keep the original meaning. Use contractions, casual wording, "
    "and slang. Do not stay polite or formal. Output only the rewritten text."
)

SYSTEM_PROMPT_REVERSE = (
    "You are a formal rewriter. Convert slang or casual English into clear "
    "standard English. Keep the original meaning. Output only the rewritten text."
)

USER_TEMPLATE = "Rewrite this in slang:\n{text}"
USER_TEMPLATE_REVERSE = "Rewrite this in standard English:\n{text}"

SLANG_MARKERS = (
    "gonna",
    "wanna",
    "gotta",
    "kinda",
    "sorta",
    "lemme",
    "gimme",
    "ain't",
    "y'all",
    "yeah",
    "yep",
    "nah",
    "nope",
    "lol",
    "lmao",
    "omg",
    "idk",
    "tbh",
    "ngl",
    "lowkey",
    "highkey",
    "bruh",
    "bro",
    "dude",
    "fam",
    "lit",
    "dope",
    "vibe",
    "bet",
    "no cap",
    "sus",
    "hmu",
    "asap",
    "cuz",
    "tho",
    "thru",
    "imma",
    "outta",
    "hella",
    "my bad",
    "hang on",
    "messed up",
    "shot down",
    "kicks off",
    "bounce",
    "gonna",
)

# Longer phrases first. These run even when the LLM is down.
LEXICAL_SLANG = (
    (r"\blet['’]?s go to that place\b", "let's bounce to that spot"),
    (r"\blet us go to that place\b", "let's bounce to that spot"),
    (r"\blet['’]?s go to\b", "let's bounce to"),
    (r"\blet us go to\b", "let's bounce to"),
    (r"\blet['’]?s go\b", "let's bounce"),
    (r"\blet us go\b", "let's bounce"),
    (r"\bgo to that place\b", "hit that spot"),
    (r"\bthat place\b", "that spot"),
    (r"\bI would like to request your assistance\b", "I need you to come thru"),
    (r"\brequest your assistance\b", "need your help"),
    (r"\bthank you for your help\b", "thanks for coming thru"),
    (r"\bcould you please clarify this point\b", "can you break this down"),
    (r"\bplease wait a moment\b", "hang on a sec"),
    (r"\bwait a moment\b", "hang on a sec"),
    (r"\bgoing to\b", "gonna"),
    (r"\bwant to\b", "wanna"),
    (r"\bhave to\b", "gotta"),
    (r"\bhas to\b", "has gotta"),
    (r"\bgot to\b", "gotta"),
    (r"\bkind of\b", "kinda"),
    (r"\bsort of\b", "sorta"),
    (r"\blet me\b", "lemme"),
    (r"\bgive me\b", "gimme"),
    (r"\bbecause\b", "cuz"),
    (r"\byou all\b", "y'all"),
    (r"\bI am\b", "I'm"),
    (r"\bdo not\b", "don't"),
    (r"\bcannot\b", "can't"),
    (r"\bwill not\b", "won't"),
    (r"\bit is\b", "it's"),
    (r"\bthat is\b", "that's"),
    (r"\bwhat is\b", "what's"),
    (r"\bI would like to\b", "I wanna"),
    (r"\bwould like to\b", "wanna"),
    (r"\bcould you please\b", "can you"),
    (r"\bcould you\b", "can you"),
    (r"\bwould you\b", "can you"),
    (r"\bplease\s+", ""),
    (r"\bkindly\s+", ""),
    (r"\bthank you\b", "thanks"),
    (r"\bI apologize\b", "my bad"),
    (r"\bI am sorry\b", "my bad"),
    (r"\bI'm sorry\b", "my bad"),
)

# Extra pass when the model still sounds formal (applied after LEXICAL_SLANG).
INFORMAL_BOOST = (
    (r"\bclarify this point\b", "break this down"),
    (r"\bclarify\b", "spell out"),
    (r"\bexplain this\b", "break this down"),
    (r"\bexplain that\b", "break that down"),
    (r"\brequest your assistance\b", "need your help"),
    (r"\bassistance\b", "help"),
    (r"\bwait a moment\b", "hang on a sec"),
    (r"\ba moment\b", "a sec"),
    (r"\bthank you for your help\b", "thanks for the help"),
    (r"\bI would like to\b", "I wanna"),
    (r"\bplease\b", ""),
    (r"\bgo to\b", "hit"),
    (r"\bplace\b", "spot"),
)

# Outputs below this heuristic score are pushed harder toward slang.
MIN_SLANG_SCORE = 0.06
