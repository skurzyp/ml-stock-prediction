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

log = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (true_labels, predicted_labels, predicted_probabilities)."""
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        logits = model(x_batch.to(device))
        y_batch = y_batch.view_as(logits)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(y_batch.cpu().numpy())
        all_probs.append(probs)

    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    preds = (probs >= 0.5).astype(int)
    return labels, preds, probs


def print_metrics(
    name: str, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray
) -> None:
    auc = roc_auc_score(labels, probs)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(classification_report(labels, preds, target_names=["Down (0)", "Up (1)"]))
    print(f"ROC-AUC: {auc:.4f}")


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
    plt.show()


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
    plt.show()


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
    plt.show()


def mcnemar_test(
    name_a: str,
    name_b: str,
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
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

    print(f"\nMcNemar test: {name_a} vs {name_b}")
    print(f"  n01 ({name_b} better): {n01}  |  n10 ({name_a} better): {n10}")
    print(f"  chi2={statistic:.4f}  p={pvalue:.4f}")
    if pvalue < 0.05:
        better = name_b if n01 > n10 else name_a
        print(f"  → Statistically significant difference (p<0.05). {better} is better.")
    else:
        print("  → No statistically significant difference (p≥0.05).")


if __name__ == "__main__":
    import pandas as pd
    from typing import cast as typing_cast

    from . import config, model_cnn, model_lstm
    from .dataset import (
        NASDAQSequenceDataset,
        build_dataloaders,
        chronological_split,
        fit_scaler,
        transform_split,
    )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = len(config.FEATURE_COLS)

    df = pd.read_json(config.DATA_PATH, orient="records")
    train_df, val_df, test_df = chronological_split(
        df, config.TRAIN_RATIO, config.VAL_RATIO
    )
    scaler = fit_scaler(train_df, config.FEATURE_COLS)
    test_x = transform_split(test_df, scaler, config.FEATURE_COLS)
    test_y = typing_cast(np.ndarray, test_df["target"].values)
    test_ds = NASDAQSequenceDataset(test_x, test_y, config.SEQ_LEN)
    _, _, test_dl = build_dataloaders(
        test_ds, test_ds, test_ds, config.BATCH_SIZE  # val/train slots unused
    )

    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    histories: dict[str, dict[str, list[float]]] = {}

    for name, builder in [
        ("LSTM", model_lstm.build_model),
        ("CNN", model_cnn.build_model),
    ]:
        path = config.MODEL_PATH.replace(".pt", f"_{name.lower()}.pt")
        model = builder(
            n_features, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        labels, preds, probs = collect_predictions(model, test_dl, device)
        results[name] = (labels, preds, probs)
        print_metrics(name, labels, preds, probs)

    names = list(results.keys())
    labels_list = [results[n][0] for n in names]
    preds_list = [results[n][1] for n in names]
    probs_list = [results[n][2] for n in names]

    plot_confusion_matrices(
        names, labels_list, preds_list, save_path="data/confusion_matrices.png"
    )
    plot_roc_curves(names, labels_list, probs_list, save_path="data/roc_curves.png")

    mcnemar_test("LSTM", "CNN", labels_list[0], preds_list[0], preds_list[1])
