import pytest

from src.analytics import eda


def test_kpis_match_pandas(clean, sqlite_db):
    # Act
    result = eda.kpis()

    # Assert
    assert result["revenue"] == pytest.approx(clean["revenue"].sum())
    assert result["orders"] == clean["invoice"].nunique()
    assert result["customers"] == clean["customer_id"].nunique()
    assert result["units"] == clean["quantity"].sum()


def test_kpis_country_filter(clean, sqlite_db):
    # Arrange
    france = clean[clean["country"] == "France"]

    # Act
    result = eda.kpis(country="France")

    # Assert
    assert result["revenue"] == pytest.approx(france["revenue"].sum())
    assert result["customers"] == france["customer_id"].nunique()


def test_monthly_revenue_sums_to_total(clean, sqlite_db):
    # Act
    monthly = eda.monthly_revenue()

    # Assert
    assert monthly["revenue"].sum() == pytest.approx(eda.kpis()["revenue"])
    assert monthly["month"].is_monotonic_increasing
