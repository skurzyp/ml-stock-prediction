import numpy as np
import pandas as pd

from src.data_pipeline import add_technical_indicators, clean_data, create_target


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 10000 + rng.standard_normal(n).cumsum() * 50
    return pd.DataFrame(
        {
            "Open": close - rng.uniform(0, 20, n),
            "High": close + rng.uniform(0, 40, n),
            "Low": close - rng.uniform(0, 40, n),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )


EXPECTED_INDICATOR_COLS = {
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_high",
    "bb_low",
    "bb_mid",
    "bb_pband",
    "sma_20",
    "sma_50",
    "atr_14",
}


def test_add_technical_indicators_columns():
    df = add_technical_indicators(_make_ohlcv())
    assert EXPECTED_INDICATOR_COLS.issubset(set(df.columns))


def test_add_technical_indicators_drops_ohlcv():
    df = add_technical_indicators(_make_ohlcv())
    for col in ("Open", "High", "Low", "Volume"):
        assert col not in df.columns


def test_create_target_values():
    df = add_technical_indicators(_make_ohlcv())
    df = create_target(df)
    assert set(df["target"].unique()).issubset({0, 1})


def test_create_target_length():
    raw = _make_ohlcv()
    before = add_technical_indicators(raw)
    after = create_target(before.copy())
    assert len(after) == len(before) - 1


def test_clean_data_drops_nans():
    df = add_technical_indicators(_make_ohlcv())
    df = create_target(df)
    dirty = df.copy()
    dirty.iloc[0, 0] = float("nan")
    clean = clean_data(dirty)
    assert clean.isna().sum().sum() == 0
    assert len(clean) < len(dirty)
