"""Revenue analytics written as SQL against the SQLite transactions table.

Every function accepts optional date and country filters so the dashboard
and notebooks share exactly the same numbers.
"""

import pandas as pd

from src.data.database import query

ALL_COUNTRIES = "All countries"


def _where(start: str | None, end: str | None, country: str | None) -> tuple[str, tuple]:
    """Build a parameterised WHERE clause for the common filters."""
    clauses, params = [], []
    if start:
        clauses.append("invoice_date >= ?")
        params.append(f"{start} 00:00:00")
    if end:
        clauses.append("invoice_date <= ?")
        params.append(f"{end} 23:59:59")
    if country and country != ALL_COUNTRIES:
        clauses.append("country = ?")
        params.append(country)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, tuple(params)


def kpis(start=None, end=None, country=None) -> dict:
    """Headline numbers: revenue, orders, customers, average order value."""
    where, params = _where(start, end, country)
    row = query(f"""
        SELECT ROUND(SUM(revenue), 2)          AS revenue,
               COUNT(DISTINCT invoice)         AS orders,
               COUNT(DISTINCT customer_id)     AS customers,
               SUM(quantity)                   AS units
        FROM transactions {where}""", params).iloc[0]
    orders = int(row["orders"] or 0)
    revenue = float(row["revenue"] or 0.0)
    return {
        "revenue": revenue,
        "orders": orders,
        "customers": int(row["customers"] or 0),
        "units": int(row["units"] or 0),
        "avg_order_value": revenue / orders if orders else 0.0,
    }


def monthly_revenue(start=None, end=None, country=None) -> pd.DataFrame:
    """Revenue, orders, customers and AOV per calendar month."""
    where, params = _where(start, end, country)
    df = query(f"""
        SELECT SUBSTR(invoice_date, 1, 7)   AS month,
               SUM(revenue)                 AS revenue,
               COUNT(DISTINCT invoice)      AS orders,
               COUNT(DISTINCT customer_id)  AS customers
        FROM transactions {where}
        GROUP BY month ORDER BY month""", params)
    return df.assign(avg_order_value=df["revenue"] / df["orders"])


def revenue_by_country(start=None, end=None, top_n: int = 15) -> pd.DataFrame:
    """Top countries by revenue."""
    where, params = _where(start, end, None)
    return query(f"""
        SELECT country, SUM(revenue) AS revenue,
               COUNT(DISTINCT invoice) AS orders,
               COUNT(DISTINCT customer_id) AS customers
        FROM transactions {where}
        GROUP BY country ORDER BY revenue DESC LIMIT ?""", params + (top_n,))


def top_products(start=None, end=None, country=None, top_n: int = 15) -> pd.DataFrame:
    """Best-selling products by revenue."""
    where, params = _where(start, end, country)
    return query(f"""
        SELECT stock_code, MAX(description) AS description,
               SUM(revenue) AS revenue, SUM(quantity) AS units,
               COUNT(DISTINCT invoice) AS orders
        FROM transactions {where}
        GROUP BY stock_code ORDER BY revenue DESC LIMIT ?""", params + (top_n,))


def weekday_hour_revenue(start=None, end=None, country=None) -> pd.DataFrame:
    """Revenue by weekday (0=Sunday) and hour, for seasonality heatmaps."""
    where, params = _where(start, end, country)
    return query(f"""
        SELECT CAST(STRFTIME('%w', invoice_date) AS INTEGER) AS weekday,
               CAST(STRFTIME('%H', invoice_date) AS INTEGER) AS hour,
               SUM(revenue) AS revenue
        FROM transactions {where}
        GROUP BY weekday, hour""", params)


def orders_per_customer() -> pd.DataFrame:
    """Distribution of how many orders each customer placed."""
    return query("""
        SELECT orders, COUNT(*) AS customers FROM (
            SELECT customer_id, COUNT(DISTINCT invoice) AS orders
            FROM transactions GROUP BY customer_id)
        GROUP BY orders ORDER BY orders""")


def countries() -> list[str]:
    """All countries, largest revenue first, for filter dropdowns."""
    return query("SELECT country FROM transactions GROUP BY country "
                 "ORDER BY SUM(revenue) DESC")["country"].tolist()


def date_range() -> tuple[str, str]:
    """First and last transaction dates as YYYY-MM-DD strings."""
    row = query("SELECT MIN(invoice_date) AS lo, MAX(invoice_date) AS hi FROM transactions").iloc[0]
    return row["lo"][:10], row["hi"][:10]
