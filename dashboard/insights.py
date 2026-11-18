"""Turn computed results into plain-English retention strategies and recommendations.

Every number in the text is computed from the data passed in, never hard-coded.
"""

import pandas as pd

RETENTION_PLAYBOOK = {
    "High-value": "VIP treatment: early access, account manager, loyalty perks. Never discount-blast them.",
    "Loyal": "Cross-sell complementary products and volume-tier pricing to grow basket size.",
    "New": "Welcome series and a time-limited second-order incentive within 30 days.",
    "At-risk": "Personal win-back outreach now: 'we miss you' offer on products they used to buy.",
    "Inactive": "One low-cost reactivation email; suppress from paid campaigns if no response.",
    "Regular": "Standard newsletter and seasonal campaigns; test small incentives to move them up.",
}


def _money(value: float) -> str:
    return f"£{value:,.0f}"


def recommendations(summary: pd.DataFrame, predictions: pd.DataFrame,
                    monthly: pd.DataFrame, top_country: pd.Series) -> str:
    """Markdown recommendations derived from segment, prediction and revenue results."""
    seg = summary.set_index("segment")
    high, at_risk = seg.loc["High-value"], seg.loc["At-risk"]
    at_risk_proba = predictions.loc[predictions["segment"] == "At-risk", "purchase_probability"]
    likely = predictions[predictions["action"].str.startswith("Likely")]
    peak = monthly.loc[monthly["revenue"].idxmax()]
    country_share = top_country["revenue"] / monthly["revenue"].sum()

    return f"""
### 1. Protect the High-value segment first
High-value customers are **{high['customer_share']:.0%} of customers but {high['revenue_share']:.0%} of revenue**
({_money(high['total_revenue'])}). Losing even a few of them costs more than any acquisition campaign can
replace. *Action:* VIP programme, priority service, no generic discounting.

### 2. Win back At-risk customers while they are still reachable
**{int(at_risk['customers']):,} At-risk customers** historically spent {_money(at_risk['total_revenue'])}
({at_risk['revenue_share']:.0%} of revenue) but last bought **{at_risk['avg_recency_days']:.0f} days ago**
on average, and their median predicted chance of returning unprompted is only
**{at_risk_proba.median():.0%}**. *Action:* rank them by predicted probability (Retention tab) and
contact the top of the list first; they are the cheapest to recover.

### 3. Use the model to spend marketing budget where it changes behaviour
**{len(likely):,} customers** are likely to buy anyway (≥70% probability): send them upsell content,
not discounts. Reserve incentives for the *Undecided* tier, where a nudge can change the outcome.

### 4. Plan inventory and campaigns around seasonality
Revenue peaks in **{peak['month']}** ({_money(peak['revenue'])}). The Q4 gift season drives the year,
so stock, staffing and campaigns should ramp up from September.

### 5. Reduce concentration risk
**{top_country['country']}** generates **{country_share:.0%}** of revenue. Growing the strongest
international markets lowers dependence on a single economy.

*Caveat:* predictions are probabilities estimated from past behaviour, not guarantees. Validate any
campaign with a holdout group before scaling it.
"""
