# models.py
from xgboost import XGBClassifier


def get_xgb_classifier():
    """
    Returns a configured XGBClassifier instance.
    """
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
