"""Kaggle script-kernel entry point.

`src/` is shipped as a private Kaggle Dataset and mounted at SRC_DATASET.
This script puts it on the import path, fixes the working directory so the
relative paths in `src/config.py` land in the kernel's persisted output, and
runs each pipeline stage exactly as it runs locally.
"""

import os
import subprocess
import sys
from pathlib import Path

SRC_DATASET = "/kaggle/input/lstm-stock-prediction-src"

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ta", "yfinance"],
    check=True,
)

import datetime

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

sys.path.insert(0, SRC_DATASET)
env = {
    **os.environ,
    "PYTHONPATH": SRC_DATASET + os.pathsep + os.environ.get("PYTHONPATH", ""),
    "RUN_ID": run_id,
}

os.chdir("/kaggle/working")
Path("data").mkdir(exist_ok=True)

for stage in ("src.data_pipeline", "src.train", "src.evaluate"):
    print(f"\n=== {stage} ===", flush=True)
    subprocess.run([sys.executable, "-m", stage], check=True, env=env)
