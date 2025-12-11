import yfinance as yf
import pandas as pd
import numpy as np
import ta

def ensure_series(col):
        if isinstance(col, pd.DataFrame):
            return col.iloc[:, 0]
        return col

def load_price_data(symbol="SPY", start="2010-01-01"):
    df = yf.download(symbol, start=start)
    df = df.dropna()

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(symbol, axis=1, level=1)
        except Exception:
            df = df.groupby(level=0, axis=1).first()

    close  = ensure_series(df["Close"])
    high   = ensure_series(df["High"])
    low    = ensure_series(df["Low"])
    open_  = ensure_series(df["Open"])
    volume = ensure_series(df["Volume"])

    # Daily returns
    df["return"] = close.pct_change()

    # === NEW: 5-day forward return target (REGRESSION) ===
    # future_5d_ret = (price in 5 days / price today) - 1
    df["fwd_5d_ret"] = close.pct_change(5).shift(-5)

    # Technical features
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["sma_10"] = close.rolling(10).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_ratio_10_50"] = df["sma_10"] / df["sma_50"]

    df["vol_10"] = df["return"].rolling(10).std()
    df["vol_20"] = df["return"].rolling(20).std()

    df["high_low_range"] = (high - low) / close
    df["close_open_range"] = (close - open_) / open_

    vol_mean = volume.rolling(20).mean()
    vol_std  = volume.rolling(20).std()
    df["volume_zscore"] = (volume - vol_mean) / vol_std

    # Trend/regime helper features (you already had similar ones)
    df["sma_200"]   = close.rolling(200).mean()
    df["trend_200"] = (close / df["sma_200"]) - 1.0
    df["sma_50"]    = close.rolling(50).mean()
    df["ann_vol_60"] = df["return"].rolling(60).std() * np.sqrt(252)

    
    # Simple regime weight: high in calm uptrends, low otherwise
    bull = (df["trend_200"] > 0) & (df["ann_vol_60"] < 0.25)
    df["regime_weight"] = np.where(bull, 1.0, 0.3)
    

    # Drop NaNs from indicators & forward returns
    df = df.dropna()

    feature_cols = [
        "rsi_14", "sma_10", "sma_50", "sma_ratio_10_50",
        "vol_10", "vol_20", "high_low_range",
        "close_open_range", "volume_zscore",
        "trend_200", "ann_vol_60"
    ]

    X = df[feature_cols].copy()
    y = df["fwd_5d_ret"].copy()   # <=== continuous target

    return df, X, y, feature_cols

def build_joint_dataset(symbols, start="2010-01-01"):
    """
    Build a joint (stacked) dataset across multiple symbols for a single
    multi-asset regression model.

    Returns:
      joint_df: DataFrame with columns: feature_cols, 'target', 'symbol', 'date'
      dfs:      dict[symbol] -> original df (aligned to common index)
      Xs:       dict[symbol] -> feature DataFrame (aligned)
      ys:       dict[symbol] -> target Series (aligned)
      feature_cols: list of feature names (same for all symbols)
    """
    dfs = {}
    Xs = {}
    ys = {}
    feature_cols = None

    # 1) Load per-asset data
    for sym in symbols:
        df_sym, X_sym, y_sym, feat_cols = load_price_data(sym, start)
        dfs[sym] = df_sym
        Xs[sym] = X_sym
        ys[sym] = y_sym
        if feature_cols is None:
            feature_cols = feat_cols
        else:
            # sanity check: all assets have the same feature_cols ordering
            if feature_cols != feat_cols:
                raise ValueError(f"Feature columns mismatch for {sym}")

    # 2) Align all assets on a common date index
    common_index = None
    for sym in symbols:
        idx = dfs[sym].index
        common_index = idx if common_index is None else common_index.intersection(idx)

    for sym in symbols:
        dfs[sym] = dfs[sym].loc[common_index]
        Xs[sym] = Xs[sym].loc[common_index]
        ys[sym] = ys[sym].loc[common_index]

    # 3) Build stacked joint DataFrame
    rows = []
    for sym in symbols:
        df_feat = Xs[sym].copy()
        df_feat["target"] = ys[sym].values
        df_feat["symbol"] = sym
        df_feat["date"] = pd.to_datetime(dfs[sym].index)  # ensure datetime
        rows.append(df_feat)

    joint_df = pd.concat(rows, ignore_index=True)

    return joint_df, dfs, Xs, ys, feature_cols

