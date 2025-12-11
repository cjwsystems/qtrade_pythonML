# data.py
import yfinance as yf
import pandas as pd
from features import build_features


def _ensure_series(col):
    import pandas as pd
    if isinstance(col, pd.DataFrame):
        return col.iloc[:, 0]
    return col


def load_price_data(symbol: str = "SPY", start: str = "2010-01-01"):
    """
    Download OHLCV, create return/target, add features.
    Returns:
      df, X, y, feature_cols
    """
    df = yf.download(symbol, start=start)
    df = df.dropna()

    # Handle possible MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(symbol, axis=1, level=1)
        except Exception:
            df = df.groupby(level=0, axis=1).first()

    close = _ensure_series(df["Close"])
    high = _ensure_series(df["High"])
    low = _ensure_series(df["Low"])
    open_ = _ensure_series(df["Open"])
    volume = _ensure_series(df["Volume"])

    # Rebuild a clean base frame
    base = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    '''
    # Basic returns and target
    base["return"] = base["Close"].pct_change()
    base["target"] = (base["return"].shift(-1) > 0).astype(int)
    '''
    # Basic daily returns
    base["return"] = base["Close"].pct_change()

    # 5-day forward return (compound)
    future_5d_return = (1 + base["return"]).rolling(5).apply(
        lambda x: x.prod(), raw=True
    ) - 1

    # Shift so today's features predict future 5-day outcome
    base["target"] = (future_5d_return.shift(-5) > 0).astype(int)

    # Drop the initial NaNs
    base = base.dropna()

    # Add ML features
    df_feat, X, y, feature_cols = build_features(base)
    return df_feat, X, y, feature_cols
