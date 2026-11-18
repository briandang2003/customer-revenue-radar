"""Shared fixtures: a small synthetic raw dataset, its cleaned form, and a temp SQLite db."""

import functools

import numpy as np
import pandas as pd
import pytest

from src.analytics import eda
from src.data import database
from src.data.clean import clean_transactions

COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate",
           "Price", "Customer ID", "Country"]
START, END = pd.Timestamp("2010-01-01"), pd.Timestamp("2011-06-30")
COUNTRIES = ["United Kingdom", "United Kingdom", "France", "Germany"]
PRODUCTS = ["85123A", "22423", "47566", "84879", "21212"]

# Hand-checked customer used by the RFM tests.
KNOWN_CUSTOMER = 99999


def _random_rows(rng: np.random.Generator) -> list[list]:
    rows, invoice = [], 500000
    span_seconds = int((END - START).total_seconds())
    for i in range(20):
        customer = 12000 + i
        country = COUNTRIES[i % len(COUNTRIES)]
        for _ in range(rng.integers(1, 9)):
            invoice += 1
            date = START + pd.Timedelta(seconds=int(rng.integers(0, span_seconds)))
            for _ in range(rng.integers(1, 4)):
                rows.append([str(invoice), str(rng.choice(PRODUCTS)), "ITEM",
                             int(rng.integers(1, 11)), date,
                             float(rng.choice([0.85, 1.25, 2.95, 4.5])), float(customer), country])
    return rows


def _known_rows() -> list[list]:
    c = float(KNOWN_CUSTOMER)
    return [
        ["600001", "22423", "A", 2, pd.Timestamp("2010-03-01 10:00"), 5.0, c, "France"],
        ["600001", "47566", "B", 1, pd.Timestamp("2010-03-01 10:00"), 3.0, c, "France"],
        ["600002", "22423", "A", 4, pd.Timestamp("2010-06-01 12:00"), 2.5, c, "France"],
        ["600003", "84879", "C", 1, pd.Timestamp("2011-01-15 09:00"), 9.0, c, "France"],
    ]


def _dirty_rows(first_row: list) -> list[list]:
    date = pd.Timestamp("2010-05-05 11:00")
    return [
        list(first_row),  # exact duplicate
        ["C700001", "22423", "A", 1, date, 2.0, 12000.0, "France"],      # cancellation
        ["700002", "22423", "A", -3, date, 2.0, 12000.0, "France"],      # negative quantity
        ["700003", "22423", "A", 1, date, 0.0, 12000.0, "France"],       # zero price
        ["700004", "POST", "POSTAGE", 1, date, 18.0, 12000.0, "France"],  # non-product code
        ["700005", "22423", "A", 1, date, 2.0, np.nan, "France"],        # missing customer id
    ]


@pytest.fixture
def raw() -> pd.DataFrame:
    rows = _random_rows(np.random.default_rng(42)) + _known_rows()
    return pd.DataFrame(rows + _dirty_rows(rows[0]), columns=COLUMNS)


@pytest.fixture
def clean(raw) -> pd.DataFrame:
    return clean_transactions(raw)[0]


@pytest.fixture
def sqlite_db(clean, tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    database.write_transactions(clean, db_path)
    monkeypatch.setattr(eda, "query", functools.partial(database.query, db_path=db_path))
    return db_path
