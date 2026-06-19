"""Deploy the pipeline to Kaggle in one command.

Stages `src/` into a build directory alongside a `dataset-metadata.json`,
versions (or creates) the supporting Kaggle Dataset, then pushes the kernel.
The Kaggle username is read from `kernel-metadata.json` so it lives in one place.

Usage:
    uv run python scripts/deploy_kaggle.py -m "what changed"
    uv run python scripts/deploy_kaggle.py --dry-run
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNEL_METADATA = PROJECT_ROOT / "kernel-metadata.json"
SRC_DIR = PROJECT_ROOT / "src"
BUILD_DIR = PROJECT_ROOT / "build" / "kaggle_dataset"
DATASET_SLUG = "lstm-stock-prediction-src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--message",
        default=f"deploy {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        help="Version bump message for the Kaggle Dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage the build directory but skip the kaggle CLI calls.",
    )
    return parser.parse_args()


def read_kaggle_username() -> str:
    metadata = json.loads(KERNEL_METADATA.read_text())
    kernel_id = metadata["id"]
    username = kernel_id.split("/", 1)[0]
    if username == "YOUR_KAGGLE_USERNAME":
        sys.exit(
            f"Replace YOUR_KAGGLE_USERNAME in {KERNEL_METADATA.relative_to(PROJECT_ROOT)} "
            "with your Kaggle username before deploying."
        )
    return username


def stage_dataset(username: str) -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    shutil.copytree(
        SRC_DIR,
        BUILD_DIR / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    dataset_metadata = {
        "id": f"{username}/{DATASET_SLUG}",
        "title": "LSTM Stock Prediction — Source",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (BUILD_DIR / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2, ensure_ascii=False) + "\n"
    )


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, text=True, capture_output=True)


def version_or_create_dataset(message: str) -> None:
    version = run(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(BUILD_DIR),
            "-m",
            message,
            "--dir-mode",
            "zip",
        ]
    )
    if version.returncode == 0:
        sys.stdout.write(version.stdout)
        return

    sys.stdout.write(version.stdout)
    sys.stderr.write(version.stderr)
    print("\nversion failed — attempting first-time `datasets create`", flush=True)

    create = run(
        ["kaggle", "datasets", "create", "-p", str(BUILD_DIR), "--dir-mode", "zip"]
    )
    sys.stdout.write(create.stdout)
    sys.stderr.write(create.stderr)
    if create.returncode != 0:
        sys.exit(create.returncode)


def push_kernel() -> None:
    result = run(["kaggle", "kernels", "push", "-p", str(PROJECT_ROOT)])
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    args = parse_args()
    username = read_kaggle_username()
    stage_dataset(username)
    print(f"Staged {BUILD_DIR.relative_to(PROJECT_ROOT)}", flush=True)

    if args.dry_run:
        print("--dry-run set, skipping kaggle CLI calls.")
        return

    version_or_create_dataset(args.message)
    push_kernel()

    kernel_id = json.loads(KERNEL_METADATA.read_text())["id"]
    print(f"\nKernel pushed: https://www.kaggle.com/code/{kernel_id}")


if __name__ == "__main__":
    main()
