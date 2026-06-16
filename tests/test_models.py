import pytest
import torch

from src.model_cnn import CNNClassifier, build_model as build_cnn
from src.model_lstm import LSTMClassifier, build_model as build_lstm

BATCH = 8
SEQ_LEN = 60
N_FEATURES = 12
HIDDEN = 64
LAYERS = 2
DROPOUT = 0.0  # deterministic forward pass in tests


def _dummy(batch: int = BATCH) -> torch.Tensor:
    return torch.randn(batch, SEQ_LEN, N_FEATURES)


# --- shared contract: both models must satisfy these ---


@pytest.mark.parametrize("build_fn", [build_lstm, build_cnn])
def test_output_shape(build_fn):
    model = build_fn(N_FEATURES, HIDDEN, LAYERS, DROPOUT)
    out = model(_dummy())
    assert out.shape == (BATCH,)


@pytest.mark.parametrize("build_fn", [build_lstm, build_cnn])
def test_output_dtype(build_fn):
    model = build_fn(N_FEATURES, HIDDEN, LAYERS, DROPOUT)
    out = model(_dummy())
    assert out.dtype == torch.float32


@pytest.mark.parametrize("build_fn", [build_lstm, build_cnn])
def test_output_is_raw_logit(build_fn):
    """Output must be unbounded logits — verify by checking model has no Sigmoid in its modules."""
    model = build_fn(N_FEATURES, HIDDEN, LAYERS, DROPOUT)
    sigmoid_modules = [m for m in model.modules() if isinstance(m, torch.nn.Sigmoid)]
    assert (
        len(sigmoid_modules) == 0
    ), "model contains Sigmoid — apply it outside the model"


@pytest.mark.parametrize("build_fn", [build_lstm, build_cnn])
def test_gradients_flow(build_fn):
    model = build_fn(N_FEATURES, HIDDEN, LAYERS, DROPOUT)
    out = model(_dummy())
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(not torch.all(g == 0) for g in grads)


@pytest.mark.parametrize("build_fn", [build_lstm, build_cnn])
def test_batch_size_one(build_fn):
    model = build_fn(N_FEATURES, HIDDEN, LAYERS, DROPOUT)
    model.eval()
    out = model(_dummy(batch=1))
    assert out.shape == (1,)


@pytest.mark.parametrize("build_fn", [build_lstm, build_cnn])
def test_eval_mode_deterministic(build_fn):
    """Same input must produce same output in eval mode (dropout off)."""
    model = build_fn(N_FEATURES, HIDDEN, LAYERS, dropout=0.5)
    model.eval()
    x = _dummy()
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)


@pytest.mark.parametrize("build_fn", [build_lstm, build_cnn])
def test_single_layer(build_fn):
    """num_layers=1 must not raise (edge case for LSTM dropout guard)."""
    model = build_fn(N_FEATURES, HIDDEN, num_layers=1, dropout=DROPOUT)
    out = model(_dummy())
    assert out.shape == (BATCH,)


# --- model-specific ---


def test_lstm_returns_correct_type():
    model = build_lstm(N_FEATURES, HIDDEN, LAYERS, DROPOUT)
    assert isinstance(model, LSTMClassifier)


def test_cnn_returns_correct_type():
    model = build_cnn(N_FEATURES, HIDDEN, LAYERS, DROPOUT)
    assert isinstance(model, CNNClassifier)


def test_cnn_batchnorm_switches_mode():
    """BatchNorm behaves differently in train vs eval — verify model respects the switch."""
    model = build_cnn(N_FEATURES, HIDDEN, LAYERS, dropout=0.0)
    x = _dummy()
    model.train()
    out_train = model(x)
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    # outputs must differ (running stats vs batch stats) — not identical
    assert not torch.allclose(out_train, out_eval)
