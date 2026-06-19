import argparse
import sys
import subprocess
from pathlib import Path
import logging
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
import pandas as pd
from .dataset import chronological_split, fit_scaler, transform_split
from .strategy import ModelStrategy, PyTorchStrategy
from . import config, model_cnn, model_lstm

log = logging.getLogger(__name__)


# collect_predictions is now handled by ModelStrategy.evaluate


def print_metrics(
    name: str, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, f=None
) -> None:
    auc = roc_auc_score(labels, probs)
    output = f"\n{'='*40}\n  {name}\n{'='*40}\n"
    output += cast(
        str,
        classification_report(labels, preds, target_names=["Down (0)", "Up (1)"]),
    )
    output += f"\nROC-AUC: {auc:.4f}\n"
    print(output, end="")
    if f:
        f.write(output)


def plot_confusion_matrices(
    names: list[str],
    labels_list: list[np.ndarray],
    preds_list: list[np.ndarray],
    save_path: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name, labels, preds in zip(axes, names, labels_list, preds_list):
        cm = confusion_matrix(labels, preds)
        ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"]).plot(
            ax=ax, colorbar=False
        )
        ax.set_title(name)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        log.info("Saved confusion matrices to %s", save_path)
    plt.close()


def plot_roc_curves(
    names: list[str],
    labels_list: list[np.ndarray],
    probs_list: list[np.ndarray],
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=(6, 5))
    for name, labels, probs in zip(names, labels_list, probs_list):
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        log.info("Saved ROC curves to %s", save_path)
    plt.close()


def plot_training_history(
    histories: dict[str, dict[str, list[float]]],
    save_path: str | None = None,
) -> None:
    """Plot train/val loss and val accuracy curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=f"{name} train")
        axes[0].plot(epochs, history["val_loss"], linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, history["val_acc"], label=name)

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].set_title("Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        log.info("Saved training history to %s", save_path)
    plt.close()


def mcnemar_test(
    name_a: str,
    name_b: str,
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    f=None,
) -> None:
    """McNemar test — checks if two models make significantly different errors on the same samples."""
    a_correct = preds_a == labels
    b_correct = preds_b == labels

    # n01: A wrong, B correct  |  n10: A correct, B wrong
    n01 = int(np.sum(~a_correct & b_correct))
    n10 = int(np.sum(a_correct & ~b_correct))

    # McNemar statistic with continuity correction
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0.0
    pvalue = 1 - chi2_dist.cdf(statistic, df=1)

    output = f"\nMcNemar test: {name_a} vs {name_b}\n"
    output += f"  n01 ({name_b} better): {n01}  |  n10 ({name_a} better): {n10}\n"

    output += f"  chi2={statistic:.4f}  p={pvalue:.4f}\n"
    if pvalue < 0.05:
        better = name_b if n01 > n10 else name_a
        output += (
            f"  → Statistically significant difference (p<0.05). {better} is better.\n"
        )
    else:
        output += "  → No statistically significant difference (p≥0.05).\n"

    print(output, end="")
    if f:
        f.write(output)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def evaluate_strategy(
    strategy: ModelStrategy,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = fit_scaler(train_df, config.FEATURE_COLS)
    test_x = transform_split(test_df, scaler, config.FEATURE_COLS)
    test_y = cast(np.ndarray, test_df["target"].values)

    path = config.MODEL_PATH.replace(
        "best_model.pt", f"best_model_{strategy.name.lower()}.pt"
    )

    eval_data = strategy.prepare_eval_data(
        test_x, test_y, config.SEQ_LEN, config.BATCH_SIZE
    )
    labels, preds, probs = strategy.evaluate(eval_data, path, device)

    print(f"\n========================================")
    print(f"  {strategy.name}")
    print(f"========================================")
    print(
        classification_report(
            labels, preds, target_names=["Down (0)", "Up (1)"], zero_division=0  # type: ignore
        )
    )
    auc = roc_auc_score(labels, probs)
    print(f"ROC-AUC: {auc:.4f}")

    run_dir = Path(config.MODEL_PATH).parent
    plot_confusion_matrix(
        labels,
        preds,
        f"Confusion Matrix - {strategy.name}",
        str(run_dir / f"cm_{strategy.name.lower()}.png"),
    )

    # Save probs and labels for plotting later
    np.save(run_dir / f"probs_{strategy.name.lower()}.npy", probs)
    np.save(run_dir / f"labels_{strategy.name.lower()}.npy", labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ALL")
    args = parser.parse_args()

    df = pd.read_json(config.DATA_PATH, orient="records")
    train_df, val_df, test_df = chronological_split(
        df, config.TRAIN_RATIO, config.VAL_RATIO
    )
    n_features = len(config.FEATURE_COLS)

    strategies: list[ModelStrategy] = [
        PyTorchStrategy(
            "LSTM",
            lambda: model_lstm.build_model(
                n_features, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT
            ),
        ),
        PyTorchStrategy(
            "CNN",
            lambda: model_cnn.build_model(
                n_features, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT
            ),
        ),
    ]

    if args.model == "ALL":
        for m in ["LSTM", "CNN"]:
            subprocess.run(
                [sys.executable, "-m", "src.evaluate", "--model", m], check=True
            )

        # Plot aggregate ROC
        run_dir = Path(config.MODEL_PATH).parent
        plt.figure(figsize=(8, 6))
        for m in ["LSTM", "CNN"]:
            m_lower = m.lower()
            probs_path = run_dir / f"probs_{m_lower}.npy"
            labels_path = run_dir / f"labels_{m_lower}.npy"
            if probs_path.exists() and labels_path.exists():
                probs = np.load(probs_path)
                labels = np.load(labels_path)
                fpr, tpr, _ = roc_curve(labels, probs)
                auc = roc_auc_score(labels, probs)
                plt.plot(fpr, tpr, label=f"{m} (AUC = {auc:.4f})")

        plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(run_dir / "roc_curve_comparison.png")
        plt.close()
        return

    # Evaluate single model
    for strategy in strategies:
        if strategy.name == args.model:
            evaluate_strategy(strategy, train_df, val_df, test_df)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
