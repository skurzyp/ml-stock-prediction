import os
import subprocess
import datetime
import sys
from pathlib import Path


def main():
    run_id = os.environ.get("RUN_ID", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.environ["RUN_ID"] = run_id

    # We must set this before importing config, or just read the config after setting it
    # We will import config down here so that it uses the environment variables we just set
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config

    config.RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SETTINGS_PATH, "w") as f:
        f.write(f"RUN_ID={run_id}\n")
        f.write(f"TARGET_HORIZON={config.TARGET_HORIZON}\n")
        f.write(f"SEQ_LEN={config.SEQ_LEN}\n")
        f.write(f"LEARNING_RATE={config.LEARNING_RATE}\n")
        f.write(f"HIDDEN_SIZE={config.HIDDEN_SIZE}\n")
        f.write(f"DROPOUT={config.DROPOUT}\n")
        f.write(f"BATCH_SIZE={config.BATCH_SIZE}\n")
        f.write(f"NUM_LAYERS={config.NUM_LAYERS}\n")

    print(f"==========================================")
    print(f"Starting pipeline run: {run_id}")
    print(f"==========================================")

    for stage in ["src.data_pipeline", "src.train", "src.evaluate"]:
        print(f"\n=== {stage} ===")
        subprocess.run([sys.executable, "-m", stage], check=True)

    print(f"\nPipeline finished. Results saved to: data/runs/{run_id}")


if __name__ == "__main__":
    main()
