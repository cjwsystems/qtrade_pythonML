# run_experiment.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from data import load_price_data
from models import get_xgb_classifier
from backtest import (
    backtest_long_only,
    backtest_prob_weighted,
    walk_forward,
)


def main():
    symbol = "SPY"
    start_date = "2010-01-01"

    # 1. Load data & features
    df, X, y, feature_cols = load_price_data(symbol, start_date)
    dates = df.index

    # 2. Define train/valid/test splits
    train_end = "2018-12-31"
    valid_end = "2021-12-31"

    train_mask = (dates <= train_end)
    valid_mask = (dates > train_end) & (dates <= valid_end)
    test_mask = (dates > valid_end)

    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    df_valid = df[valid_mask]
    df_test = df[test_mask]

    # 3. Train static XGBoost model
    model = get_xgb_classifier()
    model.fit(X_train, y_train)

    p_train = model.predict_proba(X_train)[:, 1]
    p_valid = model.predict_proba(X_valid)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    # Quick classification report on validation
    pred_valid = (p_valid > 0.5).astype(int)
    print("Validation classification report (0.5 threshold):")
    print(classification_report(y_valid, pred_valid))

    # 4. Threshold search on validation (long-only)
    best_thr = None
    best_sharpe = -np.inf

    for thr in np.arange(0.50, 0.80, 0.02):
        stats = backtest_long_only(df_valid, p_valid, threshold=thr, cost_bp=1.0)
        if stats["sharpe"] > best_sharpe:
            best_sharpe = stats["sharpe"]
            best_thr = thr

    print(f"Best threshold on validation: {best_thr:.2f} | Sharpe: {best_sharpe:.3f}")

    # 5. Static backtest on test set (long-only)
    test_stats_long = backtest_long_only(df_test, p_test, threshold=best_thr, cost_bp=1.0)
    print("\n=== Static Test Results (Long-Only) ===")
    print("Total return:", test_stats_long["total_return"])
    print("Buy & hold return:", test_stats_long["buy_hold_return"])
    print("Annualized return:", test_stats_long["ann_return"])
    print("Annualized vol:", test_stats_long["ann_vol"])
    print("Sharpe:", test_stats_long["sharpe"])
    print("Max drawdown:", test_stats_long["max_drawdown"])

    # 6. Static backtest on test set (probability-weighted)
    test_stats_weighted = backtest_prob_weighted(
        df_test,
        p_test,
        base_threshold=best_thr,
        target_vol=0.15,
        cost_bp=1.0,
    )
    print("\n=== Static Test Results (Prob-Weighted) ===")
    print("Total return:", test_stats_weighted["total_return"])
    print("Annualized return:", test_stats_weighted["ann_return"])
    print("Annualized vol:", test_stats_weighted["ann_vol"])
    print("Sharpe:", test_stats_weighted["sharpe"])
    print("Max drawdown:", test_stats_weighted["max_drawdown"])

    # Plot static test equity curves
    ax = test_stats_long["curve"].rename(columns={
        "equity": "long_only_equity",
        "buy_hold": "buy_hold",
    }).plot(figsize=(12, 6))
    test_stats_weighted["curve"]["equity"].rename("prob_weighted_equity").plot(ax=ax)
    plt.title(f"{symbol} - Static Test: ML Strategy vs Buy & Hold")
    plt.show()

    # 7. Walk-forward (expanding window) – long-only
    model_factory = get_xgb_classifier  # function that returns a new model

    wf_records_long, wf_equity_long = walk_forward(
        df,
        X,
        y,
        model_factory=model_factory,
        start_year=2015,
        end_year=int(df.index.year.max()),
        mode="long_only",
        threshold=best_thr,
        cost_bp=1.0,
    )

    print("\n=== Walk-Forward Yearly Metrics (Long-Only) ===")
    print(wf_records_long)

    # 8. Walk-forward (expanding window) – probability-weighted
    wf_records_weighted, wf_equity_weighted = walk_forward(
        df,
        X,
        y,
        model_factory=model_factory,
        start_year=2015,
        end_year=int(df.index.year.max()),
        mode="prob_weighted",
        base_threshold=best_thr,
        target_vol=0.15,
        cost_bp=1.0,
    )

    print("\n=== Walk-Forward Yearly Metrics (Prob-Weighted) ===")
    print(wf_records_weighted)

    # 9. Plot walk-forward equity curves vs buy & hold
    # Build a buy & hold curve across full period
    buy_hold_full = (1 + df["return"]).cumprod()
    wf_df = wf_equity_long.to_frame(name="wf_long_only_equity")
    wf_df["wf_prob_weighted_equity"] = wf_equity_weighted.reindex(wf_df.index, method="ffill")
    wf_df["buy_hold"] = buy_hold_full.reindex(wf_df.index, method="ffill")

    wf_df.plot(figsize=(12, 6))
    plt.title(f"{symbol} - Walk-Forward Equity Curves")
    plt.show()

if __name__ == "__main__":
    main()
