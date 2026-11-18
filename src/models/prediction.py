"""SVM model predicting whether a customer buys again within 90 days of a cutoff date.

Target (precise definition): for a cutoff date T and a customer with at least one
purchase before T, label = 1 if that customer places at least one order in
[T, T + 90 days), else 0. Features use only transactions before T.

Leakage control: snapshots are split chronologically (config.py). Every training
label window closes before the first validation cutoff, and every validation
label window closes before the test cutoff.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from src.analytics.rfm import customer_features
from src.config import PREDICTION_HORIZON_DAYS, RANDOM_SEED, get_logger

log = get_logger(__name__)

MODEL_FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value", "unique_products",
                  "tenure_days", "avg_days_between_orders", "orders_last_90d", "revenue_last_90d"]
# Baseline rule a marketer might use without a model: "bought in the last 90 days => will return".
BASELINE_RECENCY_DAYS = 90
SVM_PARAM_GRID = [{"C": c, "gamma": g} for c in (0.1, 0.3, 1.0) for g in (0.005, 0.01, 0.02, 0.05)]
# SVM training time grows ~quadratically, so cap the training sample.
MAX_TRAIN_ROWS = 15_000
BOOTSTRAP_ROUNDS = 1_000
ACTION_TIERS = [(0.7, "Likely to return: upsell"), (0.4, "Undecided: nudge"),
                (0.0, "Unlikely to return: win-back")]


def build_snapshot(transactions: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Features as of `cutoff` and the 90-day repeat-purchase label."""
    cutoff_ts = pd.Timestamp(cutoff)
    window_end = cutoff_ts + pd.Timedelta(days=PREDICTION_HORIZON_DAYS)
    if window_end > transactions["invoice_date"].max():
        raise ValueError(f"Label window for {cutoff} extends past the end of the data")
    features = customer_features(transactions, cutoff_ts)
    in_window = transactions["invoice_date"].between(cutoff_ts, window_end, inclusive="left")
    buyers = set(transactions.loc[in_window, "customer_id"])
    return features.assign(cutoff=cutoff_ts,
                           label=features["customer_id"].isin(buyers).astype(int))


def build_dataset(transactions: pd.DataFrame, cutoffs: list[str]) -> pd.DataFrame:
    """Stack snapshots for several cutoffs into one modelling table."""
    return pd.concat([build_snapshot(transactions, c) for c in cutoffs], ignore_index=True)


def make_svm(C: float = 1.0, gamma="scale", probability: bool = False) -> Pipeline:
    """log1p + scaling + RBF SVM. Class weights offset label imbalance."""
    return Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced",
                    probability=probability, random_state=RANDOM_SEED)),
    ])


def _sample(train: pd.DataFrame) -> pd.DataFrame:
    if len(train) <= MAX_TRAIN_ROWS:
        return train
    return train.sample(MAX_TRAIN_ROWS, random_state=RANDOM_SEED)


def tune_svm(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Grid-search C and gamma by validation ROC-AUC. Returns (best params, all results)."""
    sample = _sample(train)
    results = []
    for params in SVM_PARAM_GRID:
        started = time.perf_counter()
        model = make_svm(**params).fit(sample[MODEL_FEATURES], sample["label"])
        scores = model.decision_function(validation[MODEL_FEATURES])
        auc = roc_auc_score(validation["label"], scores)
        results.append({**params, "gamma": str(params["gamma"]), "val_roc_auc": auc})
        log.info("SVM %s -> val AUC %.4f (%.1fs)", params, auc, time.perf_counter() - started)
    table = pd.DataFrame(results)
    best = SVM_PARAM_GRID[int(table["val_roc_auc"].idxmax())]
    return best, table


def best_threshold(labels: pd.Series, probabilities: np.ndarray) -> float:
    """Probability cutoff that maximises F1 on the validation set."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(labels, probabilities >= t, zero_division=0) for t in grid]
    return round(float(grid[int(np.argmax(scores))]), 2)


def evaluate(labels: pd.Series, scores: np.ndarray, predictions: np.ndarray) -> dict:
    """Classification metrics for one model on one dataset."""
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def auc_difference_ci(labels: pd.Series, model_scores: np.ndarray, baseline_scores: np.ndarray,
                      n_boot: int = BOOTSTRAP_ROUNDS) -> dict:
    """Bootstrap 95% confidence interval for ROC-AUC(model) - ROC-AUC(baseline)."""
    rng = np.random.default_rng(RANDOM_SEED)
    y = labels.to_numpy()
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], model_scores[idx]) - roc_auc_score(y[idx], baseline_scores[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(np.mean(diffs)), "ci95_low": float(low), "ci95_high": float(high)}


def baseline_predict(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Rule baseline. Scores are negative recency so ROC-AUC is comparable."""
    scores = -data["recency_days"].to_numpy()
    predictions = (data["recency_days"] <= BASELINE_RECENCY_DAYS).astype(int).to_numpy()
    return scores, predictions


def train_and_evaluate(train: pd.DataFrame, validation: pd.DataFrame,
                       test: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Tune on validation, then report baseline vs SVM on the untouched test set."""
    best_params, tuning = tune_svm(train, validation)
    sample = _sample(train)
    model = make_svm(**best_params, probability=True).fit(sample[MODEL_FEATURES], sample["label"])
    threshold = best_threshold(validation["label"],
                               model.predict_proba(validation[MODEL_FEATURES])[:, 1])

    test_proba = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    base_scores, base_predictions = baseline_predict(test)
    report = {
        "target": f"Customer places >= 1 order within {PREDICTION_HORIZON_DAYS} days after cutoff",
        "rows": {"train": len(train), "train_sampled": len(sample),
                 "validation": len(validation), "test": len(test)},
        "positive_rate": {"train": float(train["label"].mean()),
                          "validation": float(validation["label"].mean()),
                          "test": float(test["label"].mean())},
        "best_params": {**best_params, "gamma": str(best_params["gamma"])},
        "tuning": tuning.to_dict(orient="records"),
        "threshold": threshold,
        "test": {
            "baseline": evaluate(test["label"], base_scores, base_predictions),
            "svm": evaluate(test["label"], test_proba, (test_proba >= threshold).astype(int)),
            "roc_auc_gain_vs_baseline": auc_difference_ci(test["label"], test_proba, base_scores),
        },
    }
    return model, report


def action_tier(probability: float) -> str:
    """Translate a probability into a recommended marketing action."""
    return next(label for floor, label in ACTION_TIERS if probability >= floor)


def score_customers(model: Pipeline, transactions: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Repeat-purchase probability for every known customer as of `as_of`."""
    features = customer_features(transactions, as_of)
    probability = model.predict_proba(features[MODEL_FEATURES])[:, 1]
    return (features.assign(purchase_probability=probability,
                            action=[action_tier(p) for p in probability])
            .sort_values("purchase_probability", ascending=False)
            .reset_index(drop=True))
