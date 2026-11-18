import pandas as pd

from src.analytics.rfm import SEGMENT_DESCRIPTIONS, customer_features, rfm_segments

KNOWN_CUSTOMER = 99999  # hand-built customer in conftest.py

AS_OF = "2010-07-01"


def test_rfm_values_for_hand_checked_customer(clean):
    # Act
    features = customer_features(clean, AS_OF).set_index("customer_id")
    known = features.loc[KNOWN_CUSTOMER]

    # Assert: orders on 2010-03-01 (2*5 + 1*3) and 2010-06-01 12:00 (4*2.5)
    assert known["recency_days"] == 29
    assert known["frequency"] == 2
    assert known["monetary"] == 23.0
    assert known["unique_products"] == 2
    assert known["country"] == "France"


def test_features_ignore_transactions_on_or_after_as_of(clean):
    # Arrange
    as_of = pd.Timestamp(AS_OF)
    history_only = clean[clean["invoice_date"] < as_of]

    # Act
    full = customer_features(clean, as_of)
    truncated = customer_features(history_only, as_of)

    # Assert
    pd.testing.assert_frame_equal(full, truncated)


def test_rfm_scores_are_between_one_and_five(clean):
    # Act
    scored = rfm_segments(customer_features(clean, "2011-07-01"))

    # Assert
    for column in ["r_score", "f_score", "m_score"]:
        assert scored[column].between(1, 5).all()


def test_every_customer_gets_a_known_segment(clean):
    # Act
    scored = rfm_segments(customer_features(clean, "2011-07-01"))

    # Assert
    assert scored["segment"].isin(SEGMENT_DESCRIPTIONS.keys()).all()
    assert len(scored) == clean["customer_id"].nunique()
