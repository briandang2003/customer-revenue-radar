"""Customer features (RFM and behaviour) and rule-based RFM segments.

`customer_features` only looks at transactions strictly before `as_of`,
which is what lets the prediction model reuse it without data leakage.
"""

import numpy as np
import pandas as pd

RECENT_WINDOW_DAYS = 90
NEW_CUSTOMER_TENURE_DAYS = 90

SEGMENT_DESCRIPTIONS = {
    "High-value": "Bought recently, buy often and spend the most. Protect and reward them.",
    "Loyal": "Buy often and are still active, but spend less than the top tier. Upsell them.",
    "New": "First purchase within the last 90 days. Onboard them toward a second order.",
    "At-risk": "Used to buy often or spend a lot, but haven't bought in a while. Win them back now.",
    "Inactive": "Bought rarely, spent little, and not recently. Low-cost reactivation only.",
    "Regular": "Average on every dimension. Keep engaged with standard marketing.",
}


def customer_features(transactions: pd.DataFrame, as_of: str | pd.Timestamp) -> pd.DataFrame:
    """One row per customer describing their behaviour before `as_of`."""
    as_of = pd.Timestamp(as_of)
    history = transactions[transactions["invoice_date"] < as_of]
    if history.empty:
        raise ValueError(f"No transactions before {as_of.date()}")

    orders = (history.groupby(["customer_id", "invoice"], as_index=False)
              .agg(order_date=("invoice_date", "min"), order_value=("revenue", "sum")))
    recent_cutoff = as_of - pd.Timedelta(days=RECENT_WINDOW_DAYS)
    recent = orders[orders["order_date"] >= recent_cutoff]

    features = orders.groupby("customer_id").agg(
        first_purchase=("order_date", "min"),
        last_purchase=("order_date", "max"),
        frequency=("invoice", "nunique"),
        monetary=("order_value", "sum"),
    )
    features["recency_days"] = (as_of - features["last_purchase"]).dt.days
    features["tenure_days"] = (as_of - features["first_purchase"]).dt.days
    features["avg_order_value"] = features["monetary"] / features["frequency"]
    span_days = (features["last_purchase"] - features["first_purchase"]).dt.days
    features["avg_days_between_orders"] = np.where(
        features["frequency"] > 1, span_days / (features["frequency"] - 1), np.nan)
    features["unique_products"] = history.groupby("customer_id")["stock_code"].nunique()
    features["orders_last_90d"] = recent.groupby("customer_id")["invoice"].nunique()
    features["revenue_last_90d"] = recent.groupby("customer_id")["order_value"].sum()
    features["country"] = history.groupby("customer_id")["country"].agg(lambda s: s.mode().iat[0])

    filled = features.fillna({"orders_last_90d": 0, "revenue_last_90d": 0.0})
    # Customers with one order have no gap; use their tenure as a conservative stand-in.
    filled["avg_days_between_orders"] = filled["avg_days_between_orders"].fillna(filled["tenure_days"])
    return filled.drop(columns=["first_purchase", "last_purchase"]).reset_index()


def _quintile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Score 1-5 by percentile rank. Tied values always share a score
    (so bins can be unequal, e.g. all one-order customers get the same F score)."""
    percentile = values.rank(method="average", pct=True, ascending=higher_is_better)
    return np.ceil(percentile * 5).clip(1, 5).astype(int)


def _segment(row: pd.Series) -> str:
    r, f, m = row["r_score"], row["f_score"], row["m_score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "High-value"
    if f >= 4 and r >= 3:
        return "Loyal"
    if row["tenure_days"] <= NEW_CUSTOMER_TENURE_DAYS:
        return "New"
    if r <= 2 and (f >= 3 or m >= 3):
        return "At-risk"
    if r <= 2:
        return "Inactive"
    return "Regular"


def rfm_segments(features: pd.DataFrame) -> pd.DataFrame:
    """Add R, F, M quintile scores and a named segment to customer features."""
    scored = features.assign(
        r_score=_quintile(features["recency_days"], higher_is_better=False),
        f_score=_quintile(features["frequency"]),
        m_score=_quintile(features["monetary"]),
    )
    return scored.assign(
        rfm_score=scored["r_score"] + scored["f_score"] + scored["m_score"],
        segment=scored.apply(_segment, axis=1),
    )


def segment_summary(customers: pd.DataFrame, by: str = "segment") -> pd.DataFrame:
    """Size and average behaviour of each segment, with revenue share."""
    summary = customers.groupby(by).agg(
        customers=("customer_id", "count"),
        avg_recency_days=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        total_revenue=("monetary", "sum"),
    )
    summary["revenue_share"] = summary["total_revenue"] / summary["total_revenue"].sum()
    summary["customer_share"] = summary["customers"] / summary["customers"].sum()
    return summary.sort_values("total_revenue", ascending=False).reset_index()
