import logging
from pathlib import Path
from typing import cast

import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from . import config

log = logging.getLogger(__name__)


def download_raw_data(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")
    assert isinstance(df, pd.DataFrame)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = cast(pd.DataFrame, df[["Open", "High", "Low", "Close", "Volume"]])
    log.info("Downloaded %d rows for %s", len(df), ticker)
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = cast(pd.Series, df["Close"])
    high = cast(pd.Series, df["High"])
    low = cast(pd.Series, df["Low"])

    df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()

    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb = BollingerBands(close=close, window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_pband"] = bb.bollinger_pband()

    df["sma_20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(close=close, window=50).sma_indicator()

    df["atr_14"] = AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    return cast(pd.DataFrame, df.drop(columns=["Open", "High", "Low", "Volume"]))


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = cast(pd.DataFrame, df.iloc[:-1])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = cast(pd.DataFrame, df.dropna())
    log.info(
        "Dropped %d rows with NaN (indicator warmup). %d rows remaining.",
        before - len(df),
        len(df),
    )
    return df


def save_to_json(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", date_format="iso")
    log.info("Saved %d rows to %s", len(df), path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    df = download_raw_data(config.TICKER, config.START_DATE, config.END_DATE)
    df = add_technical_indicators(df)
    df = create_target(df)
    df = clean_data(df)
    save_to_json(df, config.DATA_PATH)
