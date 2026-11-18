"""Clean raw transactions into an analysis-ready sales table.

Every rule below is counted in the returned report so each cleaning
decision is documented with its impact.
"""

import pandas as pd

from src.config import GBP_TO_USD, get_logger

log = get_logger(__name__)

# Stock codes that are fees or adjustments, not products.
NON_PRODUCT_CODES = {"POST", "DOT", "M", "C2", "BANK CHARGES", "AMAZONFEE", "CRUK",
                     "D", "S", "PADS", "B", "ADJUST", "ADJUST2", "TEST001", "TEST002"}

COLUMN_NAMES = {"Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
                "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "price",
                "Customer ID": "customer_id", "Country": "country"}


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean sales lines, report of rows removed per rule)."""
    df = raw.rename(columns=COLUMN_NAMES)
    report = {"raw_rows": len(df)}

    rules = [
        # The two Excel sheets overlap (1-9 Dec 2010), creating exact duplicates.
        ("exact_duplicates", df.duplicated()),
        # Invoices starting with "C" are cancellations/returns; "A" are bad-debt adjustments.
        ("cancellations_and_adjustments", ~df["invoice"].str.match(r"^\d")),
        ("non_positive_quantity", df["quantity"] <= 0),
        ("non_positive_price", df["price"] <= 0),
        ("non_product_codes", df["stock_code"].str.upper().isin(NON_PRODUCT_CODES)),
        # Customer-level analysis needs an ID; guest checkouts can't be tracked.
        ("missing_customer_id", df["customer_id"].isna()),
    ]
    # Sales later reversed by a matching cancellation (same customer, product and quantity).
    rules.insert(2, ("cancelled_sales", _cancelled_sales(df)))
    keep = pd.Series(True, index=df.index)
    for name, drop_mask in rules:
        newly_dropped = keep & drop_mask
        report[name] = int(newly_dropped.sum())
        keep = keep & ~drop_mask

    clean = df.loc[keep].assign(
        customer_id=lambda d: d["customer_id"].astype("int64"),
        description=lambda d: d["description"].fillna("").str.strip(),
        price=lambda d: d["price"] * GBP_TO_USD,  # all money is reported in USD
        revenue=lambda d: (d["quantity"] * d["price"]).round(2),
    ).reset_index(drop=True)

    report["clean_rows"] = len(clean)
    report["customers"] = int(clean["customer_id"].nunique())
    report["invoices"] = int(clean["invoice"].nunique())
    validate_clean(clean)
    log.info("Cleaning kept %d of %d rows", report["clean_rows"], report["raw_rows"])
    return clean, report


def _cancelled_sales(df: pd.DataFrame) -> pd.Series:
    """Flag sale lines that a later "C" invoice reverses exactly.

    Without this, an order that was cancelled minutes later (e.g. 80,995 units of one
    product) would still count as revenue once the cancellation row is dropped.
    """
    keys = ["customer_id", "stock_code", "quantity"]
    is_cancel = df["invoice"].str.startswith("C")
    cancels = df[is_cancel & df["customer_id"].notna()].drop_duplicates()
    cancels = (cancels.assign(quantity=-cancels["quantity"])[keys + ["invoice_date"]]
               .rename(columns={"invoice_date": "cancel_date"}).reset_index(names="cancel_id"))
    sales = df[~is_cancel].reset_index().merge(cancels, on=keys, how="inner")
    sales = sales[sales["invoice_date"] <= sales["cancel_date"]]
    # Each cancellation reverses only the most recent matching sale before it.
    matched = sales.sort_values("invoice_date").groupby("cancel_id")["index"].last()
    return pd.Series(df.index.isin(matched), index=df.index)


def validate_clean(clean: pd.DataFrame) -> None:
    """Raise ValueError if the cleaned table breaks any data contract."""
    checks = {
        "quantity must be positive": (clean["quantity"] > 0).all(),
        "price must be positive": (clean["price"] > 0).all(),
        "customer_id must be present": clean["customer_id"].notna().all(),
        "no cancellation invoices": clean["invoice"].str.match(r"^\d").all(),
        "no exact duplicates": not clean.duplicated().any(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Clean data failed validation: {failed}")
