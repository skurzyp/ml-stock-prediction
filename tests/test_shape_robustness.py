import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os

from src.strategy import PyTorchStrategy


class MockModel(nn.Module):
    """A mock model that returns pre-determined logits regardless of input."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        # We need at least one parameter with requires_grad to run optimizer steps in training tests
        self.param = nn.Parameter(torch.tensor([0.0]))
        self.logits = logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return the predefined logits, batch-matched
        # Adding self.param * 0 ensures gradients flow through the parameter
        return self.logits[: x.size(0)] + self.param * 0


def test_evaluate_accuracy_calculation():
    logits = torch.tensor([2.0, -2.0, 0.5, -0.5])
    model = MockModel(logits)
    criterion = nn.BCEWithLogitsLoss()
    strategy = PyTorchStrategy("Mock", lambda: model)

    y_1d = torch.tensor([1.0, 1.0, 0.0, 0.0])
    X = torch.zeros(4, 5)

    ds_1d = TensorDataset(X, y_1d)
    dl_1d = DataLoader(ds_1d, batch_size=4)

    val_loss_1d, val_acc_1d = strategy._run_evaluate(
        model, dl_1d, criterion, torch.device("cpu")
    )
    assert val_acc_1d == 0.5

    y_2d = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    ds_2d = TensorDataset(X, y_2d)
    dl_2d = DataLoader(ds_2d, batch_size=4)

    val_loss_2d, val_acc_2d = strategy._run_evaluate(
        model, dl_2d, criterion, torch.device("cpu")
    )
    assert val_acc_2d == 0.5
    assert val_loss_2d == val_loss_1d


def test_collect_predictions_shape_robustness():
    logits = torch.tensor([2.0, -2.0, 0.5, -0.5])
    model = MockModel(logits)
    X = torch.zeros(4, 5)

    strategy = PyTorchStrategy("Mock", lambda: MockModel(logits))

    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    torch.save(model.state_dict(), path)

    try:
        # 1D targets
        y_1d = torch.tensor([1.0, 1.0, 0.0, 0.0])
        dl_1d = DataLoader(TensorDataset(X, y_1d), batch_size=4)
        labels_1d, preds_1d, probs_1d = strategy.evaluate(
            dl_1d, path, torch.device("cpu")
        )

        assert labels_1d.shape == (4,)
        assert preds_1d.shape == (4,)
        assert probs_1d.shape == (4,)
        np.testing.assert_array_equal(preds_1d, [1, 0, 1, 0])

        # 2D targets
        y_2d = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
        dl_2d = DataLoader(TensorDataset(X, y_2d), batch_size=4)
        labels_2d, preds_2d, probs_2d = strategy.evaluate(
            dl_2d, path, torch.device("cpu")
        )

        assert labels_2d.shape == (4,)
        assert preds_2d.shape == (4,)
        assert probs_2d.shape == (4,)
        np.testing.assert_array_equal(labels_2d, [1, 1, 0, 0])
        np.testing.assert_array_equal(preds_2d, [1, 0, 1, 0])
    finally:
        os.remove(path)
