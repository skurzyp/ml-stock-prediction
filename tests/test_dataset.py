import numpy as np
import pandas as pd
import pytest
import torch

from src.config import FEATURE_COLS
from src.dataset import (
    NASDAQSequenceDataset,
    build_dataloaders,
    chronological_split,
    fit_scaler,
    transform_split,
)

SEQ_LEN = 60
N_ROWS = 300
N_FEATURES = len(FEATURE_COLS)


def _make_pipeline_df(n: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data: dict[str, np.ndarray] = {col: rng.standard_normal(n) for col in FEATURE_COLS}
    data["target"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


@pytest.fixture()
def pipeline_df():
    return _make_pipeline_df()


@pytest.fixture()
def splits(pipeline_df):
    return chronological_split(pipeline_df, 0.70, 0.15)


@pytest.fixture()
def scaler_and_arrays(splits):
    train_df, val_df, test_df = splits
    scaler = fit_scaler(train_df, FEATURE_COLS)
    train_x = transform_split(train_df, scaler, FEATURE_COLS)
    val_x = transform_split(val_df, scaler, FEATURE_COLS)
    test_x = transform_split(test_df, scaler, FEATURE_COLS)
    return scaler, train_df, val_df, test_df, train_x, val_x, test_x


# --- chronological_split ---


def test_chronological_split_sizes(pipeline_df, splits):
    train_df, val_df, test_df = splits
    assert len(train_df) + len(val_df) + len(test_df) == len(pipeline_df)


def test_chronological_split_no_overlap(pipeline_df, splits):
    train_df, val_df, test_df = splits
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_chronological_split_order(pipeline_df, splits):
    train_df, val_df, test_df = splits
    assert train_df.index[-1] < val_df.index[0]
    assert val_df.index[-1] < test_df.index[0]


# --- fit_scaler / transform_split ---


def test_fit_scaler_train_only(scaler_and_arrays):
    scaler, train_df, *_ = scaler_and_arrays
    expected_mean = train_df[FEATURE_COLS].values.mean(axis=0)
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-5)


def test_transform_split_shape(scaler_and_arrays):
    _, train_df, _, _, train_x, _, _ = scaler_and_arrays
    assert train_x.shape == (len(train_df), N_FEATURES)


def test_transform_split_dtype(scaler_and_arrays):
    _, _, _, _, train_x, _, _ = scaler_and_arrays
    assert train_x.dtype == np.float32


# --- NASDAQSequenceDataset ---


def _make_dataset(n: int = N_ROWS) -> NASDAQSequenceDataset:
    rng = np.random.default_rng(1)
    features = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    targets = rng.integers(0, 2, n).astype(np.float32)
    return NASDAQSequenceDataset(features, targets, SEQ_LEN)


def test_dataset_length():
    ds = _make_dataset()
    assert len(ds) == N_ROWS - SEQ_LEN


def test_dataset_item_shape():
    ds = _make_dataset()
    x, y = ds[0]
    assert x.shape == (SEQ_LEN, N_FEATURES)
    assert y.shape == ()


def test_dataset_item_dtype():
    ds = _make_dataset()
    x, y = ds[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


# --- build_dataloaders ---


def test_dataloaders_batch_shape():
    batch_size = 16
    train_ds = _make_dataset()
    val_ds = _make_dataset(100)
    test_ds = _make_dataset(100)
    train_dl, val_dl, test_dl = build_dataloaders(train_ds, val_ds, test_ds, batch_size)
    x_batch, y_batch = next(iter(train_dl))
    assert x_batch.shape == (batch_size, SEQ_LEN, N_FEATURES)
    assert y_batch.shape == (batch_size,)
