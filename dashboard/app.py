"""Customer Revenue Radar — interactive Gradio dashboard.

Run the pipeline first (`python -m src.pipeline`), then:  python -m dashboard.app
"""

import json

import gradio as gr
import pandas as pd

from dashboard import charts
from dashboard.insights import RETENTION_PLAYBOOK, recommendations
from src.analytics import eda
from src.analytics.rfm import SEGMENT_DESCRIPTIONS, segment_summary
from src.config import (CLEANING_REPORT, CLUSTER_REPORT, CUSTOMERS_PARQUET, PREDICTION_REPORT,
                        PREDICTIONS_PARQUET, TEST_CUTOFFS, TRAIN_CUTOFFS, VALIDATION_CUTOFFS)

CUSTOMER_COLUMNS = ["customer_id", "country", "segment", "recency_days", "frequency", "monetary"]
METRICS = ["precision", "recall", "f1", "roc_auc"]
TOP_N_CUSTOMERS = 200


def load_artifacts() -> dict:
    """Load every pipeline output once at start-up."""
    paths = [CUSTOMERS_PARQUET, PREDICTIONS_PARQUET, CLUSTER_REPORT, PREDICTION_REPORT, CLEANING_REPORT]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing {missing}. Run `python -m src.pipeline` first.")
    return {
        "customers": pd.read_parquet(CUSTOMERS_PARQUET),
        "predictions": pd.read_parquet(PREDICTIONS_PARQUET),
        "clusters": json.loads(CLUSTER_REPORT.read_text()),
        "model": json.loads(PREDICTION_REPORT.read_text()),
        "cleaning": json.loads(CLEANING_REPORT.read_text()),
    }


def kpi_cards(k: dict) -> str:
    cards = [("Revenue", f"${k['revenue']:,.0f}"), ("Orders", f"{k['orders']:,}"),
             ("Avg order value", f"${k['avg_order_value']:,.2f}"), ("Customers", f"{k['customers']:,}")]
    items = "".join(f"<div class='kpi'><span>{label}</span><b>{value}</b></div>" for label, value in cards)
    return f"<div class='kpis'>{items}</div>"


def months_between(first: str, last: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(first[:7], last[:7], freq="M")]


def revenue_view(start_month: str, end_month: str, country: str):
    """Recompute every Revenue Intelligence output for the selected filters."""
    if start_month > end_month:
        raise gr.Error("Start month must be on or before end month.")
    start = f"{start_month}-01"
    end = (pd.Period(end_month, freq="M").end_time).strftime("%Y-%m-%d")
    return (kpi_cards(eda.kpis(start, end, country)),
            charts.monthly_trend(eda.monthly_revenue(start, end, country)),
            charts.country_bar(eda.revenue_by_country(start, end)),
            charts.product_bar(eda.top_products(start, end, country)),
            charts.weekday_hour_heatmap(eda.weekday_hour_revenue(start, end, country)))


def metrics_table(model_report: dict) -> pd.DataFrame:
    labels = {"baseline": "Rule baseline (bought in last 90 days)", "svm": "SVM"}
    rows = [{"model": label, "metric": metric, "value": round(model_report["test"][name][metric], 3)}
            for name, label in labels.items() for metric in METRICS]
    return pd.DataFrame(rows)


def priority_list(predictions: pd.DataFrame, action: str, segment: str) -> pd.DataFrame:
    view = predictions
    if action != "All":
        view = view[view["action"] == action]
    if segment != "All":
        view = view[view["segment"] == segment]
    columns = ["customer_id", "purchase_probability", "action", *CUSTOMER_COLUMNS[1:]]
    return view[columns].head(TOP_N_CUSTOMERS).round({"purchase_probability": 3, "monetary": 0})


def segment_customers(predictions: pd.DataFrame, segment: str) -> pd.DataFrame:
    view = predictions[predictions["segment"] == segment].sort_values("monetary", ascending=False)
    columns = [*CUSTOMER_COLUMNS, "purchase_probability", "action"]
    return view[columns].head(TOP_N_CUSTOMERS).round({"purchase_probability": 3, "monetary": 0})


def build_app(data: dict) -> gr.Blocks:
    customers, predictions = data["customers"], data["predictions"]
    clusters, model = data["clusters"], data["model"]
    first, last = eda.date_range()
    months = months_between(first, last)
    summary = segment_summary(customers)
    cluster_profile = segment_summary(customers, by="cluster_name")
    monthly = eda.monthly_revenue()
    explained = clusters["pca_explained_variance"]
    svm, base = model["test"]["svm"], model["test"]["baseline"]

    with gr.Blocks(title="Customer Revenue Radar") as app:
        gr.Markdown(f"# Customer Revenue Radar\nUK online gift retailer · {first} to {last} · "
                    f"{data['cleaning']['clean_rows']:,} cleaned sales lines")

        with gr.Tab("Executive Overview"):
            gr.HTML(kpi_cards(eda.kpis()))
            gr.Plot(charts.monthly_trend(monthly))
            gr.Plot(charts.segment_share(summary))

        with gr.Tab("Revenue Intelligence"):
            with gr.Row():
                start = gr.Dropdown(months, value=months[0], label="From month")
                end = gr.Dropdown(months, value=months[-1], label="To month")
                country = gr.Dropdown([eda.ALL_COUNTRIES, *eda.countries()],
                                      value=eda.ALL_COUNTRIES, label="Country")
            kpi_html = gr.HTML()
            trend = gr.Plot()
            with gr.Row():
                country_plot, product_plot = gr.Plot(), gr.Plot()
            heatmap = gr.Plot()
            outputs = [kpi_html, trend, country_plot, product_plot, heatmap]
            for control in (start, end, country):
                control.change(revenue_view, [start, end, country], outputs)
            app.load(revenue_view, [start, end, country], outputs)

        with gr.Tab("Customer Segmentation"):
            gr.Markdown("### RFM segments\nCustomers scored 1–5 on **R**ecency, **F**requency "
                        "and **M**onetary value, then grouped by business rules.")
            gr.Dataframe(summary.round(2), interactive=False)
            gr.Markdown("\n".join(f"- **{s}** — {d}" for s, d in SEGMENT_DESCRIPTIONS.items()))
            gr.Markdown(f"### K-Means clusters\nk = **{clusters['chosen_k']}** chosen by "
                        f"{clusters['selection_rule']}.")
            gr.Plot(charts.k_selection(pd.DataFrame(clusters["k_scores"]), clusters["chosen_k"]))
            gr.Dataframe(cluster_profile.round(2), interactive=False)
            gr.Markdown(f"### PCA view\nTwo components keep **{sum(explained):.0%}** of the "
                        "variation in 7 behavioural features, so each dot is a customer placed "
                        "by overall behaviour.")
            with gr.Row():
                gr.Plot(charts.pca_scatter(customers, explained, "cluster_name"))
                gr.Plot(charts.pca_scatter(customers, explained, "segment"))

        with gr.Tab("Purchase Predictions"):
            gr.Markdown(f"**Target:** {model['target']}. Chronological split: train cutoffs "
                        f"{TRAIN_CUTOFFS[0]} → {TRAIN_CUTOFFS[-1]}, validation "
                        f"{', '.join(VALIDATION_CUTOFFS)}, test {', '.join(TEST_CUTOFFS)}. "
                        f"Decision threshold {model['threshold']:.2f} (tuned on validation).")
            gr.Plot(charts.metric_comparison(metrics_table(model)))
            with gr.Row():
                gr.Plot(charts.confusion_heatmap(base["confusion_matrix"], "Rule baseline"))
                gr.Plot(charts.confusion_heatmap(svm["confusion_matrix"], "SVM"))
            gr.Plot(charts.probability_histogram(predictions))
            gr.Markdown(f"### Prioritised customer list (top {TOP_N_CUSTOMERS})")
            with gr.Row():
                action = gr.Dropdown(["All", *predictions["action"].unique()], value="All",
                                     label="Recommended action")
                segment = gr.Dropdown(["All", *SEGMENT_DESCRIPTIONS], value="All", label="Segment")
            table = gr.Dataframe(priority_list(predictions, "All", "All"), interactive=False)
            for control in (action, segment):
                control.change(lambda a, s: priority_list(predictions, a, s), [action, segment], table)

        with gr.Tab("Retention Intelligence"):
            for name in ("At-risk", "Inactive", "High-value"):
                row = summary.set_index("segment").loc[name]
                gr.Markdown(f"### {name} · {int(row['customers']):,} customers · "
                            f"${row['total_revenue']:,.0f} lifetime revenue\n"
                            f"**Strategy:** {RETENTION_PLAYBOOK[name]}")
                gr.Dataframe(segment_customers(predictions, name), interactive=False,
                             max_height=300)
            gr.Markdown("### Playbook for every segment\n" + "\n".join(
                f"- **{s}** — {p}" for s, p in RETENTION_PLAYBOOK.items()))

        with gr.Tab("Business Recommendations"):
            top_country = eda.revenue_by_country(top_n=1).iloc[0]
            gr.Markdown(recommendations(summary, predictions, monthly, top_country))

    return app


CSS = """
.kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px;}
.kpi {border:1px solid var(--border-color-primary); border-radius:12px; padding:14px;}
.kpi span {display:block; font-size:.85rem; opacity:.7;}
.kpi b {font-size:1.6rem;}
"""

if __name__ == "__main__":
    build_app(load_artifacts()).launch(theme=gr.themes.Soft(), css=CSS)
