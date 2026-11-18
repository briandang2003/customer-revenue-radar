"""SQLite storage for clean transactions, plus SQL helpers used by the analytics layer."""

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DATABASE_PATH

TABLE = "transactions"


def write_transactions(clean: pd.DataFrame, db_path: Path = DATABASE_PATH) -> None:
    """Replace the transactions table with the clean data and index common filters."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    table = clean.assign(invoice_date=clean["invoice_date"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    with sqlite3.connect(db_path) as conn:
        table.to_sql(TABLE, conn, if_exists="replace", index=False)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_customer ON {TABLE}(customer_id)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_date ON {TABLE}(invoice_date)")


def query(sql: str, params: tuple = (), db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Run a parameterised read-only query and return a DataFrame."""
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found. Run `python -m src.pipeline` first.")
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(sql, conn, params=params)
