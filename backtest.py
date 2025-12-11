# backtest.py
import numpy as np
import pandas as pd


def backtest_long_only(df_slice, prob, threshold=0.6, cost_bp=1.0):
    s = df_slice.copy()
    s["prob"] = prob
    s["signal"] = (s["prob"] > threshold).astype(int)  # 1 = long, 0 = flat

    # Base position from ML signal
    s["position"] = s["signal"].shift(1).fillna(0)

    # --- NEW: Apply regime filter if available ---
    if "regime_risk_on" in s.columns:
        # Use yesterday's regime to decide if we're allowed to be in the market
        if "regime_weight" in s.columns:
            weight = s["regime_weight"].shift(1).fillna(1.0)
            s["position"] = s["position"] * weight

    # Returns
    s["strategy_gross"] = s["position"] * s["return"]

    # Transaction costs on position changes
    s["trade"] = s["position"].diff().abs().fillna(s["position"].abs())
    cost = cost_bp / 10000.0
    s["cost"] = s["trade"] * cost

    s["strategy_net"] = s["strategy_gross"] - s["cost"]

    s["equity"] = (1 + s["strategy_net"]).cumprod()
    s["buy_hold"] = (1 + s["return"]).cumprod()

    total_ret = s["equity"].iloc[-1] - 1
    bh_ret = s["buy_hold"].iloc[-1] - 1

    ann_factor = 252
    ann_ret = (1 + total_ret) ** (ann_factor / len(s)) - 1
    ann_vol = s["strategy_net"].std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (s["equity"].cummax() - s["equity"]).max()

    return {
        "total_return": total_ret,
        "buy_hold_return": bh_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "curve": s[["equity", "buy_hold"]],
    }

def backtest_prob_weighted(
    df_slice,
    prob,
    base_threshold=0.55,
    target_vol=0.15,
    cost_bp=1.0,
):
    s = df_slice.copy()
    s["prob"] = prob

    # Position between 0 and 1 based on probability above threshold
    s["raw_signal"] = np.clip(
        (s["prob"] - base_threshold) / (1 - base_threshold),
        0,
        1,
    )

    # Volatility scaling (use past 20-day vol)
    s["vol_20"] = s["return"].rolling(20).std()
    s["vol_20"] = s["vol_20"].replace(0, np.nan).ffill()

    s["vol_target"] = target_vol / (s["vol_20"] * np.sqrt(252))
    s["vol_target"] = s["vol_target"].clip(upper=1.0)

    # Base position from ML confidence & vol scaling
    s["position"] = (s["raw_signal"] * s["vol_target"]).shift(1).fillna(0)

    # --- NEW: Apply regime filter if available ---
    if "regime_risk_on" in s.columns:
        if "regime_weight" in s.columns:
            weight = s["regime_weight"].shift(1).fillna(1.0)
            s["position"] = s["position"] * weight


    s["strategy_gross"] = s["position"] * s["return"]

    s["trade"] = s["position"].diff().abs().fillna(s["position"].abs())
    cost = cost_bp / 10000.0
    s["cost"] = s["trade"] * cost

    s["strategy_net"] = s["strategy_gross"] - s["cost"]

    s["equity"] = (1 + s["strategy_net"]).cumprod()
    s["buy_hold"] = (1 + s["return"]).cumprod()

    total_ret = s["equity"].iloc[-1] - 1
    ann_factor = 252
    ann_ret = (1 + total_ret) ** (ann_factor / len(s)) - 1
    ann_vol = s["strategy_net"].std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (s["equity"].cummax() - s["equity"]).max()

    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "curve": s[["equity", "buy_hold"]],
    }

def walk_forward(
    df,
    X,
    y,
    model_factory,
    start_year=2015,
    end_year=None,
    mode="long_only",
    threshold=0.6,
    base_threshold=0.55,
    target_vol=0.15,
    cost_bp=1.0,
):
    """
    Expanding-window walk-forward backtest.
    mode: 'long_only' or 'prob_weighted'
    """
    if end_year is None:
        end_year = int(df.index.year.max())

    records = []
    equity_all = []

    years = range(start_year, end_year + 1)
    for year in years:
        train_mask = df.index.year < year
        trade_mask = df.index.year == year

        if trade_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_trade = X[trade_mask]
        df_trade = df[trade_mask]

        model = model_factory()
        model.fit(X_train, y_train)
        p_trade = model.predict_proba(X_trade)[:, 1]

        if mode == "long_only":
            stats = backtest_long_only(
                df_trade,
                p_trade,
                threshold=threshold,
                cost_bp=cost_bp,
            )
        elif mode == "prob_weighted":
            stats = backtest_prob_weighted(
                df_trade,
                p_trade,
                base_threshold=base_threshold,
                target_vol=target_vol,
                cost_bp=cost_bp,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        records.append({
            "year": year,
            "ann_return": stats["ann_return"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
        })

        equity_all.append(stats["curve"]["equity"])

    equity_concat = pd.concat(equity_all)
    records_df = pd.DataFrame(records)
    return records_df, equity_concat

def backtest_portfolio(
    dfs_test: dict,
    probs_test: dict,
    thresholds: dict,
    symbols: list,
    *,
    score_smoothing_window: int = 10,
    temperature: float = 0.05,
    cash_floor: float = 0.20,
    max_weight_per_asset: float = 0.50,
):
    """
    Multi-asset long-only portfolio backtest with:
      - Smoothed ML scores
      - Regime weighting
      - Softmax allocation
      - Cash floor and per-asset caps

    dfs_test:     dict[symbol] -> df slice for TEST period (must share same index)
                  Each df must have columns: 'return', 'regime_weight'
    probs_test:   dict[symbol] -> 1D array of probabilities (aligned with dfs_test[symbol])
    thresholds:   dict[symbol] -> float threshold per asset
    symbols:      list of symbols to include

    Parameters (kwargs):
      score_smoothing_window: rolling window (days) for score smoothing
      temperature: softmax temperature (smaller -> more concentrated)
      cash_floor: minimum fraction of portfolio always kept in cash
      max_weight_per_asset: cap on any single asset's weight (before normalization)

    Returns:
      dict with:
        'total_return', 'ann_return', 'ann_vol', 'sharpe',
        'max_drawdown', 'equity', 'weights', 'cash'
    """
    # Use the index from the first asset (we'll assume all are aligned after run_portfolio_experiment)
    test_index = next(iter(dfs_test.values())).index

    # Build returns DataFrame: rows = dates, cols = symbols
    returns = pd.DataFrame(
        {sym: dfs_test[sym]["return"].values for sym in symbols},
        index=test_index,
    )

    # --- 1) Build raw scores per asset (ML + regime) ---
    score_frames = {}
    for sym in symbols:
        df_s = dfs_test[sym]
        p_s = np.asarray(probs_test[sym])

        if len(p_s) != len(df_s):
            raise ValueError(f"Length mismatch for {sym}: probs={len(p_s)}, df={len(df_s)}")

        thr = thresholds[sym]

        # Base ML score: how far above threshold?
        base_score = np.maximum(p_s - thr, 0.0)

        # Yesterday's regime_weight to avoid lookahead
        regime = df_s["regime_weight"].shift(1).fillna(df_s["regime_weight"].iloc[0]).values

        raw_score = base_score * regime
        score_frames[sym] = raw_score

    scores = pd.DataFrame(score_frames, index=test_index)

    # --- 2) Smooth scores over time to reduce choppiness ---
    scores_smooth = scores.rolling(window=score_smoothing_window, min_periods=1).mean()

    # --- 3) Convert scores into weights via softmax with caps and cash floor ---
    weights_list = []
    cash_list = []

    for dt, row in scores_smooth.iterrows():
        s = row.values.astype(float)

        # If all scores are zero or negative -> all cash
        if np.all(s <= 0):
            w = np.zeros_like(s)
            cash = 1.0
        else:
            # Scale by temperature for softmax
            x = s / temperature

            # Numerical stability: subtract max
            x = x - np.max(x)

            exps = np.exp(x)
            softmax_raw = exps / exps.sum()

            # Cap any single asset
            w_capped = np.minimum(softmax_raw, max_weight_per_asset)

            total = w_capped.sum()
            if total > 0:
                investable = 1.0 - cash_floor
                w = w_capped / total * investable
                cash = 1.0 - w.sum()
            else:
                w = np.zeros_like(s)
                cash = 1.0

        weights_list.append(w)
        cash_list.append(cash)

    weights = pd.DataFrame(
        weights_list,
        index=test_index,
        columns=symbols,
    )
    cash_series = pd.Series(cash_list, index=test_index, name="cash")

    # --- 4) Portfolio returns & equity ---
    # Cash assumed to earn 0% for now
    port_ret = (weights * returns).sum(axis=1)

    equity = (1 + port_ret).cumprod()

    # --- 5) Metrics ---
    total_ret = equity.iloc[-1] - 1
    ann_factor = 252
    ann_ret = (1 + total_ret) ** (ann_factor / len(equity)) - 1
    ann_vol = port_ret.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (equity.cummax() - equity).max()

    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "equity": equity,
        "weights": weights,
        "cash": cash_series,
    }

def backtest_portfolio(
    dfs_test: dict,
    probs_test: dict,
    thresholds: dict,
    symbols: list,
    *,
    score_smoothing_window: int = 10,
    cash_floor: float = 0.20,
    max_long_weight_per_asset: float = 0.50,   # default cap (overridden per asset below)
    max_short_weight_per_asset: float = 0.05,  # small per-asset shorts
    short_gross_fraction: float = 0.10,        # total short gross <= 10% of equity
    # Volatility targeting params
    target_vol_annual: float = 0.10,
    vol_lookback: int = 60,
    min_leverage: float = 0.0,
    max_leverage: float = 2.0,
    # Short signal parameters
    short_threshold_global: float = 0.45,      # require p < 0.45 for shorts
):
    """
    Multi-asset LONG/SHORT portfolio backtest with:
      - Smoothed ML scores
      - Regime weighting
      - Asymmetric long & short thresholds (asset-specific for longs)
      - Relaxed long trend gating (OR logic)
      - Trend-confirmed, macro-gated shorts
      - Per-asset long caps (smaller for TLT)
      - Cash floor and per-asset long/short caps
      - Portfolio-level volatility targeting (via leverage scaling)

    dfs_test:     dict[symbol] -> df slice for TEST period (must share same index)
                  Each df must have columns:
                    'return', 'regime_weight', 'trend_200', 'sma_50', 'Close'
                  For SPY: also 'ann_vol_60' (for macro regime).
    probs_test:   dict[symbol] -> 1D array of probabilities (aligned with dfs_test[symbol])
    thresholds:   dict[symbol] -> float long-threshold per asset (will be relaxed for some)
    symbols:      list of symbols to include
    """
    # Use index from the first asset (assumed aligned)
    test_index = next(iter(dfs_test.values())).index

    # Per-asset returns
    returns = pd.DataFrame(
        {sym: dfs_test[sym]["return"].values for sym in symbols},
        index=test_index,
    )

    # --- Per-asset long caps (TLT gets a smaller cap) ---
    per_asset_long_cap = {
        "SPY": 0.50,
        "QQQ": 0.50,
        "TLT": 0.20,   # smaller max long on TLT
        "GLD": 0.50,
    }
    long_caps_array = np.array([
        per_asset_long_cap.get(sym, max_long_weight_per_asset) for sym in symbols
    ])

    # --- Portfolio-level macro regime (from SPY) for gating shorts & TLT longs ---
    if (
        "SPY" in dfs_test
        and "trend_200" in dfs_test["SPY"].columns
        and "ann_vol_60" in dfs_test["SPY"].columns
    ):
        spy = dfs_test["SPY"]
        # Risk-off if SPY significantly below 200d MA or vol is high
        macro_risk_off = (
            (spy["trend_200"] < -0.02) | (spy["ann_vol_60"] > 0.25)
        ).astype(float)
        macro_risk_off = macro_risk_off.shift(1).reindex(test_index).fillna(0.0)
    else:
        macro_risk_off = pd.Series(0.0, index=test_index)

    macro_risk_on = 1.0 - macro_risk_off

    # --- 1) Build raw signed scores per asset (long - short) ---
    score_frames = {}
    for sym in symbols:
        df_s = dfs_test[sym]
        p_s = np.asarray(probs_test[sym])

        if len(p_s) != len(df_s):
            raise ValueError(f"Length mismatch for {sym}: probs={len(p_s)}, df={len(df_s)}")

        # Base long threshold from validation
        base_long_thr = thresholds[sym]
        # Asset-specific relaxing of long thresholds
        if sym == "SPY":
            long_thr = min(base_long_thr, 0.55)   # make SPY easier to go long
        elif sym == "GLD":
            long_thr = min(base_long_thr, 0.50)   # GLD also easier to go long
        else:
            long_thr = base_long_thr

        short_thr = short_threshold_global      # global short threshold (strict)

        # Base long/short components
        long_score = np.maximum(p_s - long_thr, 0.0)      # bullish
        short_score = np.maximum(short_thr - p_s, 0.0)    # bearish

        # Regime weight (lagged) per asset
        regime_w = df_s["regime_weight"].shift(1).fillna(df_s["regime_weight"].iloc[0]).values

        # Long trend confirmation (RELAXED):
        # For longs, allow if price above 50d OR 200d trend positive.
        if "sma_50" in df_s.columns and "trend_200" in df_s.columns and "Close" in df_s.columns:
            close = df_s["Close"].values
            sma50 = df_s["sma_50"].values
            trend200 = df_s["trend_200"].values
            long_trend_mask = ((close > sma50) | (trend200 > 0.0)).astype(float)
        else:
            long_trend_mask = np.ones_like(long_score, dtype=float)

        # Short trend confirmation (unchanged):
        if "sma_50" in df_s.columns and "trend_200" in df_s.columns and "Close" in df_s.columns:
            short_trend_mask = ((close < sma50) & (trend200 < 0.0)).astype(float)
        else:
            short_trend_mask = np.zeros_like(short_score, dtype=float)

        # Macro risk arrays
        macro_off_arr = macro_risk_off.values
        macro_on_arr = macro_risk_on.values

        # Apply regime & trend gating:
        # Longs: scaled by regime, require relaxed long uptrend
        long_score = long_score * regime_w * long_trend_mask

        # TLT-specific extra gating for longs:
        # TLT longs only allowed when macro is not in bad risk-off regime.
        if sym == "TLT":
            long_score = long_score * macro_on_arr

        # Shorts: require regime, short trend confirmation, and macro risk-off
        short_score = short_score * regime_w * short_trend_mask * macro_off_arr

        # Signed score: positive = net bullish, negative = net bearish
        signed_score = long_score - short_score
        score_frames[sym] = signed_score

    scores = pd.DataFrame(score_frames, index=test_index)

    # --- 2) Smooth scores over time ---
    scores_smooth = scores.rolling(window=score_smoothing_window, min_periods=1).mean()

    # --- 3) Convert scores into long & short weights with per-asset caps + cash floor ---
    weights_list = []
    cash_list = []

    investable_long = 1.0 - cash_floor
    investable_short = investable_long * short_gross_fraction

    for dt, row in scores_smooth.iterrows():
        s = row.values.astype(float)

        pos = np.maximum(s, 0.0)   # bullish scores
        neg = np.maximum(-s, 0.0)  # bearish scores (magnitude)

        if pos.sum() == 0 and neg.sum() == 0:
            w_long = np.zeros_like(s)
            w_short = np.zeros_like(s)
            cash = 1.0
        else:
            # --- Long side ---
            if pos.sum() > 0:
                long_raw = pos / pos.sum() * investable_long
            else:
                long_raw = np.zeros_like(s)

            # Apply per-asset long caps
            w_long = np.minimum(long_raw, long_caps_array)

            # --- Short side (small overlay) ---
            if neg.sum() > 0 and investable_short > 0:
                short_raw = neg / neg.sum() * investable_short
            else:
                short_raw = np.zeros_like(s)

            w_short = np.minimum(short_raw, max_short_weight_per_asset)

            # Cash is whatever long side doesn't use (shorts funded from longs)
            cash = 1.0 - w_long.sum()
            if cash < 0:
                cash = 0.0

        w = w_long - w_short
        weights_list.append(w)
        cash_list.append(cash)

    weights = pd.DataFrame(
        weights_list,
        index=test_index,
        columns=symbols,
    )
    cash_series = pd.Series(cash_list, index=test_index, name="cash")

    # --- 4) Base portfolio returns (before vol targeting) ---
    base_port_ret = (weights * returns).sum(axis=1)

    # --- 5) Volatility targeting via leverage scaling ---
    leverage_list = []
    final_port_ret = []

    for i, dt in enumerate(test_index):
        if i < vol_lookback:
            lev = 1.0  # warmup
        else:
            window = base_port_ret.iloc[i - vol_lookback : i]
            realized_vol = window.std() * np.sqrt(252)

            if realized_vol > 0:
                lev = target_vol_annual / realized_vol
            else:
                lev = 1.0

            lev = float(np.clip(lev, min_leverage, max_leverage))

        leverage_list.append(lev)
        final_port_ret.append(base_port_ret.iloc[i] * lev)

    leverage_series = pd.Series(leverage_list, index=test_index, name="leverage")
    port_ret = pd.Series(final_port_ret, index=test_index, name="port_ret")

    # --- 6) Equity curve & metrics ---
    equity = (1 + port_ret).cumprod()

    total_ret = equity.iloc[-1] - 1
    ann_factor = 252
    ann_ret = (1 + total_ret) ** (ann_factor / len(equity)) - 1
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (equity.cummax() - equity).max()

    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "equity": equity,
        "weights": weights,
        "cash": cash_series,
        "leverage": leverage_series,
    }
