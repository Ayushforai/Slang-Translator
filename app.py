import logging
import os
import threading

from flask import Flask, jsonify, render_template, request

from slang_translator.detector import FormalityDetector
from slang_translator.env import load_env
from slang_translator.style import enforce_slang, slang_score

load_env()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

_engine = None
_engine_error = None
_llm_disabled = False
_lock = threading.Lock()
_detector = FormalityDetector()


def get_engine():
    global _engine, _engine_error, _llm_disabled
    if _llm_disabled:
        raise RuntimeError(_engine_error or "LLM disabled")
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        if _llm_disabled:
            raise RuntimeError(_engine_error or "LLM disabled")
        try:
            from slang_translator.engine import SlangEngine

            _engine = SlangEngine(require_adapter=True)
            logger.info("Loaded adapter %s", _engine.adapter_id)
        except Exception as exc:  # noqa: BLE001
            _engine_error = str(exc)
            _llm_disabled = True
            logger.exception("Model failed to load")
            raise
    return _engine


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "llm_loaded": _engine is not None,
            "llm_error": _engine_error,
            "detector": _detector.path.is_file(),
        }
    )


@app.route("/translate", methods=["POST"])
def translate_api():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"success": False, "error": "Please provide text to rewrite into slang."}), 400

    style = _detector.predict(text)
    already_informal = style["label"] == "informal" and style["informal_prob"] >= 0.65

    mode = "llm"
    try:
        engine = get_engine()
        slang_text = engine.generate(text)
    except Exception:
        logger.warning("LLM unavailable, using lexical slang fallback")
        mode = "lexical"
        slang_text = enforce_slang(text, text)

    out_style = _detector.predict(slang_text)
    note = None
    if already_informal:
        note = "Input already looks informal; rewritten anyway toward slang."
    elif out_style["label"] == "formal" and slang_score(slang_text) < 0.12:
        note = (
            "Output still reads formal — training data uses mild casual English, not heavy Gen-Z slang. "
            "Retrain with stronger slang labels or use a GPU host for full model quality."
        )

    return jsonify(
        {
            "success": True,
            "informal": slang_text,
            "source_slang_score": slang_score(text),
            "output_slang_score": slang_score(slang_text),
            "detected_register": style["label"],
            "output_register": out_style["label"],
            "mode": mode,
            "already_informal": already_informal,
            "note": note,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.run(debug=False, host="0.0.0.0", port=port)
