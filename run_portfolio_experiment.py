# run_portfolio_experiment.py
import numpy as np
import matplotlib.pyplot as plt

from data import load_price_data
from models import get_xgb_classifier
from backtest import backtest_long_only, backtest_portfolio


def main():
    symbols = ["SPY", "QQQ", "TLT", "GLD"]
    start_date = "2010-01-01"

    # 1. Load per-asset data
    dfs = {}
    Xs = {}
    ys = {}
    feature_cols_map = {}

    for sym in symbols:
        df_sym, X_sym, y_sym, feature_cols = load_price_data(sym, start_date)
        dfs[sym] = df_sym
        Xs[sym] = X_sym
        ys[sym] = y_sym
        feature_cols_map[sym] = feature_cols

    # 2. Align on common date index
    common_index = None
    for sym in symbols:
        idx = dfs[sym].index
        common_index = idx if common_index is None else common_index.intersection(idx)

    for sym in symbols:
        dfs[sym] = dfs[sym].loc[common_index]
        Xs[sym] = Xs[sym].loc[common_index]
        ys[sym] = ys[sym].loc[common_index]

    dates = common_index

    # 3. Train/valid/test split on common timeline
    train_end = "2018-12-31"
    valid_end = "2021-12-31"

    train_mask = (dates <= train_end)
    valid_mask = (dates > train_end) & (dates <= valid_end)
    test_mask = (dates > valid_end)

    # Containers
    models = {}
    p_valid_dict = {}
    p_test_dict = {}
    thresholds = {}

    # 4. Train one model per asset, tune per-asset thresholds on validation
    for sym in symbols:
        df_sym = dfs[sym]
        X_sym = Xs[sym]
        y_sym = ys[sym]

        X_train, y_train = X_sym[train_mask], y_sym[train_mask]
        X_valid, y_valid = X_sym[valid_mask], y_sym[valid_mask]
        X_test, y_test = X_sym[test_mask], y_sym[test_mask]

        df_valid = df_sym.loc[valid_mask]
        df_test = df_sym.loc[test_mask]

        model = get_xgb_classifier()
        model.fit(X_train, y_train)

        models[sym] = model

        p_valid = model.predict_proba(X_valid)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        p_valid_dict[sym] = p_valid
        p_test_dict[sym] = p_test

        # Threshold search for this asset using single-asset backtest
        best_thr = None
        best_sharpe = -np.inf

        for thr in np.arange(0.50, 0.70, 0.02):
            stats = backtest_long_only(df_valid, p_valid, threshold=thr, cost_bp=1.0)
            if stats["sharpe"] > best_sharpe:
                best_sharpe = stats["sharpe"]
                best_thr = thr

        if best_thr is None:
            best_thr = 0.55

        thresholds[sym] = best_thr
        print(f"[{sym}] Best threshold: {best_thr:.2f} | Sharpe (valid): {best_sharpe:.3f}")

    # 5. Build test slices dict to feed the portfolio backtester
    dfs_test = {sym: dfs[sym].loc[test_mask] for sym in symbols}

    # 6. Run the portfolio backtest (multi-asset)
    portfolio_stats = backtest_portfolio(
        dfs_test=dfs_test,
        probs_test=p_test_dict,
        thresholds=thresholds,
        symbols=symbols,
    )

    print("\n=== Multi-Asset Portfolio (Test Period) ===")
    print("Total return:", portfolio_stats["total_return"])
    print("Annualized return:", portfolio_stats["ann_return"])
    print("Annualized vol:", portfolio_stats["ann_vol"])
    print("Sharpe:", portfolio_stats["sharpe"])
    print("Max drawdown:", portfolio_stats["max_drawdown"])

    # 7. Plot portfolio equity
    portfolio_stats["equity"].plot(figsize=(12, 6))
    plt.title("Multi-Asset ML Portfolio Equity (Test)")
    plt.ylabel("Equity (start = 1.0)")
    plt.show()

    # 8. (Optional) Inspect weights behavior over time
    weights_30 = portfolio_stats["weights"].rolling(30).mean()
    ax = weights_30.plot(figsize=(12, 6))
    plt.title("30-Day Avg Portfolio Weights (Test Period)")
    plt.ylabel("Weight")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
