"""Plotly figure builders for the dashboard. Each takes a DataFrame and returns a figure."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.models.prediction import ACTION_TIERS

TEMPLATE = "plotly_white"
PRIMARY = "#2563eb"
SECONDARY = "#f59e0b"
SEGMENT_COLORS = {"High-value": "#2563eb", "Loyal": "#16a34a", "New": "#06b6d4",
                  "Regular": "#94a3b8", "At-risk": "#f59e0b", "Inactive": "#dc2626"}
WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def _style(fig: go.Figure, title: str) -> go.Figure:
    return fig.update_layout(template=TEMPLATE, title=title, margin=dict(l=20, r=20, t=50, b=20),
                             legend=dict(orientation="h", y=-0.2))


def monthly_trend(monthly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=monthly["month"], y=monthly["orders"], name="Orders",
                marker_color="#cbd5e1", opacity=0.45, yaxis="y2")
    fig.add_scatter(x=monthly["month"], y=monthly["revenue"], name="Revenue ($)",
                    mode="lines+markers", line=dict(color=PRIMARY, width=3))
    fig.update_layout(yaxis=dict(title="Revenue ($)"),
                      yaxis2=dict(title="Orders", overlaying="y", side="right", showgrid=False))
    return _style(fig, "Monthly revenue and orders")


def country_bar(countries: pd.DataFrame) -> go.Figure:
    fig = px.bar(countries.sort_values("revenue"), x="revenue", y="country", orientation="h",
                 color_discrete_sequence=[PRIMARY], labels={"revenue": "Revenue ($)", "country": ""})
    return _style(fig, "Revenue by country")


def product_bar(products: pd.DataFrame) -> go.Figure:
    data = products.assign(label=products["description"].str.slice(0, 32))
    fig = px.bar(data.sort_values("revenue"), x="revenue", y="label", orientation="h",
                 color_discrete_sequence=[SECONDARY], hover_data=["stock_code", "units"],
                 labels={"revenue": "Revenue ($)", "label": ""})
    return _style(fig, "Top products by revenue")


def weekday_hour_heatmap(data: pd.DataFrame) -> go.Figure:
    grid = data.pivot_table(index="weekday", columns="hour", values="revenue", fill_value=0)
    fig = px.imshow(grid, color_continuous_scale="Blues", aspect="auto",
                    labels={"x": "Hour of day", "y": "", "color": "Revenue ($)"},
                    y=[WEEKDAYS[d] for d in grid.index])
    return _style(fig, "When customers buy (revenue by weekday and hour)")


def segment_share(summary: pd.DataFrame) -> go.Figure:
    long = summary.melt(id_vars="segment", value_vars=["customer_share", "revenue_share"],
                        var_name="measure", value_name="share")
    long["measure"] = long["measure"].map({"customer_share": "% of customers",
                                           "revenue_share": "% of revenue"})
    fig = px.bar(long, x="segment", y="share", color="measure", barmode="group",
                 color_discrete_sequence=["#94a3b8", PRIMARY], labels={"segment": "", "share": ""})
    fig.update_yaxes(tickformat=".0%")
    return _style(fig, "RFM segments: share of customers vs share of revenue")


def k_selection(scores: pd.DataFrame, chosen_k: int) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=scores["k"], y=scores["inertia"], name="Inertia (elbow)",
                    mode="lines+markers", line=dict(color="#94a3b8"))
    fig.add_scatter(x=scores["k"], y=scores["silhouette"], name="Silhouette",
                    mode="lines+markers", line=dict(color=PRIMARY, width=3), yaxis="y2")
    fig.add_vline(x=chosen_k, line_dash="dash", line_color=SECONDARY,
                  annotation_text=f"chosen k={chosen_k}")
    fig.update_layout(xaxis_title="Number of clusters (k)", yaxis=dict(title="Inertia"),
                      yaxis2=dict(title="Silhouette", overlaying="y", side="right", showgrid=False))
    return _style(fig, "Choosing k: elbow method and silhouette score")


def pca_scatter(customers: pd.DataFrame, explained: list[float], color: str) -> go.Figure:
    colors = SEGMENT_COLORS if color == "segment" else None
    fig = px.scatter(customers, x="pc1", y="pc2", color=color, opacity=0.6,
                     color_discrete_map=colors,
                     hover_data=["customer_id", "recency_days", "frequency", "monetary"],
                     labels={"pc1": f"PC1 ({explained[0]:.0%} of variance)",
                             "pc2": f"PC2 ({explained[1]:.0%} of variance)"})
    fig.update_traces(marker=dict(size=5))
    return _style(fig, f"Customers in 2-D (PCA), coloured by {color.replace('_', ' ')}")


def confusion_heatmap(matrix: dict, title: str) -> go.Figure:
    values = [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]]
    fig = px.imshow(values, text_auto=True, color_continuous_scale="Blues",
                    x=["Predicted: no purchase", "Predicted: purchase"],
                    y=["Actual: no purchase", "Actual: purchase"])
    fig.update_coloraxes(showscale=False)
    return _style(fig, title)


def metric_comparison(metrics: pd.DataFrame) -> go.Figure:
    fig = px.bar(metrics, x="metric", y="value", color="model", barmode="group", text_auto=".3f",
                 color_discrete_sequence=["#94a3b8", PRIMARY], labels={"metric": "", "value": ""})
    fig.update_yaxes(range=[0, 1])
    return _style(fig, "Test set: rule baseline vs SVM")


def probability_histogram(predictions: pd.DataFrame) -> go.Figure:
    tier_colors = dict(zip([label for _, label in ACTION_TIERS], ["#16a34a", SECONDARY, "#dc2626"]))
    fig = px.histogram(predictions, x="purchase_probability", color="action", nbins=40,
                       color_discrete_map=tier_colors,
                       category_orders={"action": list(tier_colors)},
                       labels={"purchase_probability": "Predicted probability of buying in next 90 days"})
    return _style(fig, "Distribution of predicted repeat-purchase probability")
