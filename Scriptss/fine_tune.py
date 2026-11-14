"""QLoRA training for Llama 3.2 1B. Canonical script: Deployment/fine_tune.py."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "Deployment" / "fine_tune.py"),
        run_name="__main__",
    )
