import os
import subprocess
import datetime
import sys


def run_experiment(name: str, env_vars: dict[str, str]):
    run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_exp_{name}"
    env = os.environ.copy()
    env.update(env_vars)
    env["RUN_ID"] = run_id

    print(f"\n==========================================")
    print(f"Running Experiment: {name}")
    print(f"Overrides: {env_vars}")
    print(f"==========================================")

    subprocess.run([sys.executable, "scripts/run_pipeline.py"], env=env, check=True)


def main():
    experiments = [
        {"name": "baseline", "env": {"TARGET_HORIZON": "1", "SEQ_LEN": "60"}},
        {
            "name": "changed_training",
            "env": {
                "TARGET_HORIZON": "1",
                "SEQ_LEN": "60",
                "LEARNING_RATE": "5e-4",
                "HIDDEN_SIZE": "256",
            },
        },
        {"name": "changed_data_30", "env": {"TARGET_HORIZON": "1", "SEQ_LEN": "30"}},
        {"name": "changed_data_5", "env": {"TARGET_HORIZON": "1", "SEQ_LEN": "5"}},
        {"name": "changed_data", "env": {"TARGET_HORIZON": "1", "SEQ_LEN": "120"}},
        {"name": "future_baseline", "env": {"TARGET_HORIZON": "5", "SEQ_LEN": "60"}},
        {
            "name": "future_changed_training",
            "env": {
                "TARGET_HORIZON": "5",
                "SEQ_LEN": "60",
                "LEARNING_RATE": "5e-4",
                "HIDDEN_SIZE": "256",
            },
        },
        {
            "name": "future_changed_data_30",
            "env": {"TARGET_HORIZON": "5", "SEQ_LEN": "30"},
        },
        {
            "name": "future_changed_data_5",
            "env": {"TARGET_HORIZON": "5", "SEQ_LEN": "5"},
        },
        {
            "name": "future_changed_data",
            "env": {"TARGET_HORIZON": "5", "SEQ_LEN": "120"},
        },
    ]

    for exp in experiments:
        run_experiment(exp["name"], exp["env"])


if __name__ == "__main__":
    main()
