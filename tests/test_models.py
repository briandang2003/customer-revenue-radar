import pandas as pd
import pytest

from src import config
from src.analytics.rfm import customer_features
from src.models.clustering import CLUSTER_FEATURES, choose_k, evaluate_k, fit_kmeans, pca_projection
from src.models.prediction import action_tier, build_snapshot

CUTOFF = pd.Timestamp("2011-03-01")
HORIZON = pd.Timedelta(days=config.PREDICTION_HORIZON_DAYS)
SMALL_K_RANGE = range(2, 6)


@pytest.fixture
def features(clean):
    return customer_features(clean, "2011-07-01")


def test_choose_k_returns_actionable_k_within_range(features):
    # Act
    k = choose_k(evaluate_k(features, SMALL_K_RANGE))

    # Assert
    assert 3 <= k <= max(SMALL_K_RANGE)


def test_fit_kmeans_labels_every_customer(features):
    # Act
    model = fit_kmeans(features, 3)
    labels = model.predict(features[CLUSTER_FEATURES])

    # Assert
    assert len(labels) == len(features)
    assert set(labels) <= {0, 1, 2}


def test_pca_projection_returns_two_components(features):
    # Act
    coords, explained = pca_projection(features)

    # Assert
    assert list(coords.columns) == ["pc1", "pc2"]
    assert len(coords) == len(features)
    assert ((explained >= 0) & (explained <= 1)).all()


def test_snapshot_features_do_not_use_future_transactions(clean):
    # Act
    snapshot = build_snapshot(clean, CUTOFF)
    past_only = customer_features(clean[clean["invoice_date"] < CUTOFF], CUTOFF)

    # Assert
    pd.testing.assert_frame_equal(snapshot[past_only.columns], past_only)


def test_snapshot_label_is_one_iff_customer_buys_in_window(clean):
    # Arrange
    in_window = clean["invoice_date"].between(CUTOFF, CUTOFF + HORIZON, inclusive="left")
    buyers = set(clean.loc[in_window, "customer_id"])

    # Act
    snapshot = build_snapshot(clean, CUTOFF)

    # Assert
    expected = snapshot["customer_id"].isin(buyers).astype(int)
    assert (snapshot["label"] == expected).all()
    assert 0 < snapshot["label"].sum() < len(snapshot)


def test_snapshot_raises_when_label_window_passes_data_end(clean):
    # Arrange
    late_cutoff = clean["invoice_date"].max() - pd.Timedelta(days=30)

    # Act / Assert
    with pytest.raises(ValueError, match="extends past the end"):
        build_snapshot(clean, late_cutoff)


@pytest.mark.parametrize("probability, action", [
    (0.95, "Likely to return: upsell"),
    (0.7, "Likely to return: upsell"),
    (0.5, "Undecided: nudge"),
    (0.1, "Unlikely to return: win-back"),
])
def test_action_tier_maps_probability_to_action(probability, action):
    assert action_tier(probability) == action


def test_config_cutoffs_leave_no_label_overlap_between_splits():
    # Arrange
    train = pd.to_datetime(config.TRAIN_CUTOFFS)
    validation = pd.to_datetime(config.VALIDATION_CUTOFFS)
    test = pd.to_datetime(config.TEST_CUTOFFS)

    # Assert
    assert train.max() + HORIZON <= validation.min()
    assert validation.max() + HORIZON <= test.min()
