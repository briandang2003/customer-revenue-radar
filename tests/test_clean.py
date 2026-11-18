import pandas as pd
import pytest

from src.data.clean import clean_transactions, validate_clean

RULES = ["exact_duplicates", "cancellations_and_adjustments", "non_positive_quantity",
         "non_positive_price", "non_product_codes", "missing_customer_id"]


def test_each_dirty_row_type_is_removed_exactly_once(raw):
    # Act
    _, report = clean_transactions(raw)

    # Assert
    assert {rule: report[rule] for rule in RULES} == {rule: 1 for rule in RULES}


def test_report_row_counts_add_up(raw):
    # Act
    clean, report = clean_transactions(raw)

    # Assert
    removed = sum(report[r] for r in RULES) + report["cancelled_sales"]
    assert report["raw_rows"] - removed == report["clean_rows"] == len(clean)


def test_clean_table_has_no_dirty_rows(clean):
    # Assert
    assert (clean["quantity"] > 0).all()
    assert (clean["price"] > 0).all()
    assert clean["customer_id"].notna().all()
    assert not clean["invoice"].str.startswith("C").any()
    assert "POST" not in set(clean["stock_code"])
    assert not clean.duplicated().any()


def test_revenue_is_quantity_times_price(clean):
    # Assert
    expected = (clean["quantity"] * clean["price"]).round(2)
    pd.testing.assert_series_equal(clean["revenue"], expected, check_names=False)


def test_validate_clean_raises_on_negative_quantity(clean):
    # Arrange
    bad = clean.assign(quantity=-clean["quantity"])

    # Act / Assert
    with pytest.raises(ValueError, match="quantity must be positive"):
        validate_clean(bad)


def test_validate_clean_raises_on_cancellation_invoice(clean):
    # Arrange
    bad = clean.assign(invoice="C" + clean["invoice"])

    # Act / Assert
    with pytest.raises(ValueError, match="no cancellation invoices"):
        validate_clean(bad)


def test_cancelled_sale_is_removed_but_earlier_identical_sale_is_kept():
    # Arrange: two identical purchases, then a cancellation of one of them
    row = {"StockCode": "85123A", "Description": "HEART", "Quantity": 6, "Price": 2.5,
           "Customer ID": 12345.0, "Country": "United Kingdom"}
    raw = pd.DataFrame([
        {**row, "Invoice": "500001", "InvoiceDate": pd.Timestamp("2010-01-01 10:00")},
        {**row, "Invoice": "500002", "InvoiceDate": pd.Timestamp("2010-02-01 10:00")},
        {**row, "Invoice": "C500003", "Quantity": -6, "InvoiceDate": pd.Timestamp("2010-02-01 10:05")},
    ])

    # Act
    clean, report = clean_transactions(raw)

    # Assert: only the most recent matching sale is reversed
    assert report["cancelled_sales"] == 1
    assert clean["invoice"].tolist() == ["500001"]
