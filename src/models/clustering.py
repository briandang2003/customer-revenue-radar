"""K-Means customer segmentation with data-driven choice of k, plus PCA for 2-D views."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import CLUSTER_K_RANGE, RANDOM_SEED, get_logger

log = get_logger(__name__)

CLUSTER_FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value",
                    "unique_products", "tenure_days", "avg_days_between_orders"]
# k=2 only separates "good" from "bad" customers, which RFM scores already do.
MIN_ACTIONABLE_K = 3


def _preprocessor() -> Pipeline:
    """log1p tames the heavy right skew of spend and frequency, then standardise."""
    return make_pipeline(FunctionTransformer(np.log1p), StandardScaler())


def evaluate_k(features: pd.DataFrame, k_range=CLUSTER_K_RANGE) -> pd.DataFrame:
    """Inertia (elbow method) and silhouette score for each candidate k."""
    scaled = _preprocessor().fit_transform(features[CLUSTER_FEATURES])
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(scaled)
        rows.append({"k": k, "inertia": model.inertia_,
                     "silhouette": silhouette_score(scaled, model.labels_)})
    return pd.DataFrame(rows)


def choose_k(scores: pd.DataFrame) -> int:
    """Pick the actionable k (>= 3) with the highest silhouette score."""
    candidates = scores[scores["k"] >= MIN_ACTIONABLE_K]
    return int(candidates.loc[candidates["silhouette"].idxmax(), "k"])


def fit_kmeans(features: pd.DataFrame, k: int) -> Pipeline:
    """Fit preprocessing + K-Means as a single persistable pipeline."""
    model = Pipeline([*_preprocessor().steps,
                      ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED))])
    return model.fit(features[CLUSTER_FEATURES])


def pca_projection(features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Project scaled customer features onto 2 principal components."""
    scaled = _preprocessor().fit_transform(features[CLUSTER_FEATURES])
    pca = PCA(n_components=2, random_state=RANDOM_SEED).fit(scaled)
    coords = pd.DataFrame(pca.transform(scaled), columns=["pc1", "pc2"], index=features.index)
    return coords, pca.explained_variance_ratio_


def pca_loadings(features: pd.DataFrame) -> pd.DataFrame:
    """How strongly each original feature contributes to each component."""
    scaled = _preprocessor().fit_transform(features[CLUSTER_FEATURES])
    pca = PCA(n_components=2, random_state=RANDOM_SEED).fit(scaled)
    return pd.DataFrame(pca.components_.T, index=CLUSTER_FEATURES, columns=["pc1", "pc2"])


def name_clusters(customers: pd.DataFrame) -> dict[int, str]:
    """Describe each cluster relative to the median customer, e.g. 'Recent · Frequent · High-spend'."""
    medians = customers[["recency_days", "frequency", "monetary"]].median()
    profile = customers.groupby("cluster")[["recency_days", "frequency", "monetary"]].median()
    names = {}
    for cluster, row in profile.iterrows():
        recency = "Recent" if row["recency_days"] <= medians["recency_days"] else "Lapsed"
        frequency = "Frequent" if row["frequency"] > medians["frequency"] else "Occasional"
        spend = "High-spend" if row["monetary"] > medians["monetary"] else "Low-spend"
        names[int(cluster)] = f"{recency} · {frequency} · {spend}"
    return names
