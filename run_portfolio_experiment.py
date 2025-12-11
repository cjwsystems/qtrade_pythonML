import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import build_joint_dataset
from models import get_xgb_regressor
from backtest import backtest_long_only, backtest_portfolio


def main():
    symbols = ["SPY", "QQQ", "TLT", "GLD"]
    start_date = "2010-01-01"

    # === Build joint dataset for a single multi-asset regressor ===
    joint_df, dfs, Xs, ys, feature_cols = build_joint_dataset(symbols, start=start_date)

    # One-hot encode the asset identity
    asset_dummies = pd.get_dummies(joint_df["symbol"], prefix="asset")

    # Full feature matrix: base features + asset dummies
    X_joint = pd.concat([joint_df[feature_cols], asset_dummies], axis=1)
    y_joint = joint_df["target"].values
    dates_joint = joint_df["date"]

    # Time splits based on date
    train_end = pd.to_datetime("2018-12-31")
    valid_end = pd.to_datetime("2021-12-31")

    train_mask = dates_joint <= train_end
    valid_mask = (dates_joint > train_end) & (dates_joint <= valid_end)
    test_mask  = dates_joint > valid_end

    X_train = X_joint[train_mask]
    y_train = y_joint[train_mask]

    X_valid = X_joint[valid_mask]
    y_valid = y_joint[valid_mask]

    X_test = X_joint[test_mask]
    y_test = y_joint[test_mask]

    # === Train a single joint regressor on all assets ===
    model = get_xgb_regressor()
    model.fit(X_train, y_train)

    # (Optional) sanity check on the whole validation slice
    pred_valid_all = model.predict(X_valid)
    corr_all = np.corrcoef(pred_valid_all, y_valid)[0, 1]
    print("Joint valid corr(pred, actual):", corr_all)

    # === Per-asset thresholds using the joint model ===
    preds_valid = {}
    preds_test = {}
    thresholds = {}

    # We will re-run the model only as needed for each asset subset
    for sym in symbols:
        df_sym = dfs[sym]
        X_sym = Xs[sym]
        y_sym = ys[sym]

        # Rebuild the one-hot columns for this asset's X
        sym_mask = (joint_df["symbol"] == sym)
        # Indices for this symbol
        dates_sym = joint_df.loc[sym_mask, "date"].values
        # We know X_sym is aligned to common_index, same as dates_sym sorted, but
        # easier: construct a new X_sym_joint using the same transformation

        # Build X for this symbol alone
        asset_dummy_sym = pd.get_dummies(
            pd.Series(sym, index=X_sym.index),
            prefix="asset"
        )
        # Ensure same dummy columns as in X_joint (fill missing with 0)
        for col in asset_dummies.columns:
            if col not in asset_dummy_sym.columns:
                asset_dummy_sym[col] = 0.0
        asset_dummy_sym = asset_dummy_sym[asset_dummies.columns]

        X_sym_joint = pd.concat([X_sym[feature_cols], asset_dummy_sym], axis=1)

        # Date masks at the asset level
        dates_idx = df_sym.index
        train_mask_sym = dates_idx <= train_end
        valid_mask_sym = (dates_idx > train_end) & (dates_idx <= valid_end)
        test_mask_sym  = dates_idx > valid_end

        X_valid_sym = X_sym_joint.loc[valid_mask_sym]
        X_test_sym  = X_sym_joint.loc[test_mask_sym]

        pred_valid_sym = model.predict(X_valid_sym)
        pred_test_sym  = model.predict(X_test_sym)

        preds_valid[sym] = pred_valid_sym
        preds_test[sym]  = pred_test_sym

        df_valid_sym = df_sym.loc[valid_mask_sym]

        # Threshold search (per asset) on validation using backtest_long_only
        q_grid = [0.40, 0.50, 0.60, 0.70, 0.80]
        thr_candidates = np.quantile(pred_valid_sym, q_grid)

        best_thr = None
        best_sharpe = -np.inf
        for thr in thr_candidates:
            stats = backtest_long_only(df_valid_sym, pred_valid_sym, threshold=thr, cost_bp=1.0)
            if stats["sharpe"] > best_sharpe:
                best_sharpe = stats["sharpe"]
                best_thr = thr

        thresholds[sym] = best_thr
        print(f"[{sym}] Best regression threshold: {best_thr:.5f} | Sharpe (valid): {best_sharpe:.3f}")

    # === Multi-asset portfolio backtest on TEST period ===
    dates_common = dfs[symbols[0]].index  # all aligned
    test_mask_dates = dates_common > valid_end

    dfs_test = {sym: dfs[sym].loc[test_mask_dates] for sym in symbols}

    portfolio_stats = backtest_portfolio(
        dfs_test=dfs_test,
        probs_test=preds_test,   # predicted returns as scores
        thresholds=thresholds,
        symbols=symbols,
    )

    print("\n=== Multi-Asset Joint Regression Portfolio (Test Period) ===")
    print("Total return:",      portfolio_stats["total_return"])
    print("Annualized return:", portfolio_stats["ann_return"])
    print("Annualized vol:",    portfolio_stats["ann_vol"])
    print("Sharpe:",            portfolio_stats["sharpe"])
    print("Max drawdown:",      portfolio_stats["max_drawdown"])

    # Equity curve
    portfolio_stats["equity"].plot(figsize=(12, 6))
    plt.title("Multi-Asset Joint Regression Portfolio Equity (Test)")
    plt.ylabel("Equity (start=1)")
    plt.show()

    # 30-day smoothed weights for readability
    weights_30 = portfolio_stats["weights"].rolling(30).mean()
    weights_30.plot(figsize=(12, 6))
    plt.title("30-Day Avg Portfolio Weights (Joint Regression)")
    plt.ylabel("Weight")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
