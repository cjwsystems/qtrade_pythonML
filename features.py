# features.py
import pandas as pd
import numpy as np
import ta


def build_features(df: pd.DataFrame):
    """
    Expects df with columns:
      ['Open', 'High', 'Low', 'Close', 'Volume', 'return', 'target']
    Returns:
      df_feat, X, y, feature_cols
    """
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]
    ret = df["return"]

    # Technical features
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["sma_10"] = close.rolling(10).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_ratio_10_50"] = df["sma_10"] / df["sma_50"]

    df["vol_10"] = ret.rolling(10).std()
    df["vol_20"] = ret.rolling(20).std()

    # Price structure
    df["high_low_range"] = (high - low) / close
    df["close_open_range"] = (close - open_) / open_

    # Volume z-score
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    df["volume_zscore"] = (volume - vol_mean) / vol_std

# --- NEW: Market Regime Features ---
    # Long-term trend: 200-day moving average
    df["sma_200"] = close.rolling(200).mean()
    df["trend_200"] = (close / df["sma_200"]) - 1  # % above/below 200d MA

    # Medium-term volatility: 60-day annualized realized vol
    df["ann_vol_60"] = ret.rolling(60).std() * np.sqrt(252)

    # Define a simple risk-on regime:
    # Continuous regime score between 0 and 1
    trend_score = np.clip((df["trend_200"] + 0.10) / 0.20, 0, 1)
    vol_score = np.clip((0.40 - df["ann_vol_60"]) / 0.40, 0, 1)

    df["regime_weight"] = 0.5 * trend_score + 0.5 * vol_score

    # Drop NaNs from rolling windows
    df = df.dropna()

    feature_cols = [
        "rsi_14", "sma_10", "sma_50", "sma_ratio_10_50",
        "vol_10", "vol_20",
        "high_low_range", "close_open_range",
        "volume_zscore",
        # You *can* let the model see regime features too:
        "trend_200", "ann_vol_60",
    ]

    X = df[feature_cols].copy()
    y = df["target"].copy()

    return df, X, y, feature_cols
