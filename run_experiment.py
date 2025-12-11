# run_experiment.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from data import load_price_data
from models import get_xgb_regressor
from backtest import backtest_long_only

def main():
    symbol = "SPY"
    df, X, y, feature_cols = load_price_data(symbol, "2010-01-01")

    dates = df.index

    train_end = "2018-12-31"
    valid_end = "2021-12-31"

    train_mask = (dates <= train_end)
    valid_mask = (dates > train_end) & (dates <= valid_end)
    test_mask  = (dates > valid_end)

    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    df_valid = df[valid_mask]
    df_test  = df[test_mask]

    # === Train regression model ===
    model = get_xgb_regressor()
    model.fit(X_train, y_train)

    # Predictions (5-day forward returns)
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test  = model.predict(X_test)

    # Basic sanity checks
    print("Train R^2:", r2_score(y_train, pred_train))
    print("Valid R^2:", r2_score(y_valid, pred_valid))
    print("Pred valid stats:",
          "mean", pred_valid.mean(),
          "std", pred_valid.std())

    # Correlation between predicted & realized returns on validation
    corr = np.corrcoef(pred_valid, y_valid)[0, 1]
    print("Valid corr(pred, actual):", corr)

    # === Threshold search on validation ===
    # We want: go long when predicted 5d return > threshold
    # We'll search over quantiles of predicted returns.
    q_grid = [0.40, 0.50, 0.60, 0.70, 0.80]
    thr_candidates = np.quantile(pred_valid, q_grid)

    best_thr = None
    best_sharpe = -np.inf

    print("\nThreshold scan on validation:")
    for thr in thr_candidates:
        stats = backtest_long_only(df_valid, pred_valid, threshold=thr, cost_bp=1.0)
        print(f"  thr={thr:.5f} | Sharpe={stats['sharpe']:.3f}")
        if stats["sharpe"] > best_sharpe:
            best_sharpe = stats["sharpe"]
            best_thr = thr

    print("\nBest validation threshold:", best_thr, "Sharpe:", best_sharpe)

    # === Test set backtest ===
    test_stats = backtest_long_only(df_test, pred_test, threshold=best_thr, cost_bp=1.0)

    print("\n=== Static Test Results (Long-Only, Regression) ===")
    print("Total return:",       test_stats["total_return"])
    print("Buy & hold return:",  test_stats["buy_hold_return"])
    print("Annualized return:",  test_stats["ann_return"])
    print("Annualized vol:",     test_stats["ann_vol"])
    print("Sharpe:",             test_stats["sharpe"])
    print("Max drawdown:",       test_stats["max_drawdown"])

    # Plot equity vs buy & hold
    test_stats["curve"].plot(figsize=(12, 6))
    plt.title(f"{symbol} Regression ML Strategy vs Buy & Hold (Test)")
    plt.ylabel("Equity (start=1)")
    plt.show()

if __name__ == "__main__":
    main()
