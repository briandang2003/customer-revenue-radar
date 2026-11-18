"""End-to-end pipeline: raw data -> clean SQLite table -> RFM -> K-Means/PCA -> SVM.

Run with:  python -m src.pipeline
"""

import json

import joblib
import numpy as np

from src.analytics.rfm import customer_features, rfm_segments
from src.config import (CLEAN_PARQUET, CLEANING_REPORT, CLUSTER_REPORT, CUSTOMERS_PARQUET,
                        KMEANS_MODEL, MODELS_DIR, PREDICTION_REPORT, PREDICTIONS_PARQUET,
                        PROCESSED_DIR, RANDOM_SEED, SCORING_DATE, SVM_MODEL, TEST_CUTOFFS,
                        TRAIN_CUTOFFS, VALIDATION_CUTOFFS, get_logger)
from src.data.clean import clean_transactions
from src.data.database import write_transactions
from src.data.ingest import load_raw
from src.models import clustering, prediction

log = get_logger("pipeline")


def _save_json(data: dict, path) -> None:
    path.write_text(json.dumps(data, indent=2, default=float))


def run_cleaning():
    clean, report = clean_transactions(load_raw())
    clean.to_parquet(CLEAN_PARQUET, index=False)
    write_transactions(clean)
    _save_json(report, CLEANING_REPORT)
    return clean


def run_segmentation(clean):
    """RFM segments + K-Means clusters + PCA coordinates for every customer."""
    customers = rfm_segments(customer_features(clean, SCORING_DATE))
    scores = clustering.evaluate_k(customers)
    k = clustering.choose_k(scores)
    model = clustering.fit_kmeans(customers, k)
    coords, explained = clustering.pca_projection(customers)
    customers = customers.assign(cluster=model.predict(customers[clustering.CLUSTER_FEATURES]),
                                 pc1=coords["pc1"], pc2=coords["pc2"])
    names = clustering.name_clusters(customers)
    customers = customers.assign(
        cluster_name=customers["cluster"].map(lambda c: f"Cluster {c}: {names[c]}"))

    joblib.dump(model, KMEANS_MODEL)
    _save_json({
        "chosen_k": k,
        "selection_rule": "highest silhouette score among k >= 3",
        "k_scores": scores.to_dict(orient="records"),
        "pca_explained_variance": explained.tolist(),
        "pca_loadings": clustering.pca_loadings(customers).to_dict(),
        "cluster_names": {str(c): n for c, n in names.items()},
    }, CLUSTER_REPORT)
    log.info("Chose k=%d; PCA explains %.1f%% of variance", k, 100 * explained.sum())
    return customers


def run_prediction(clean):
    train = prediction.build_dataset(clean, TRAIN_CUTOFFS)
    validation = prediction.build_dataset(clean, VALIDATION_CUTOFFS)
    test = prediction.build_dataset(clean, TEST_CUTOFFS)
    model, report = prediction.train_and_evaluate(train, validation, test)
    joblib.dump(model, SVM_MODEL)
    _save_json(report, PREDICTION_REPORT)
    log.info("Test ROC-AUC  baseline=%.3f  svm=%.3f",
             report["test"]["baseline"]["roc_auc"], report["test"]["svm"]["roc_auc"])
    return prediction.score_customers(model, clean, SCORING_DATE)


def main() -> None:
    np.random.seed(RANDOM_SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    clean = run_cleaning()
    customers = run_segmentation(clean)
    scored = run_prediction(clean)
    customers.to_parquet(CUSTOMERS_PARQUET, index=False)
    scored.merge(customers[["customer_id", "segment", "cluster_name"]], on="customer_id") \
        .to_parquet(PREDICTIONS_PARQUET, index=False)
    log.info("Pipeline complete")


if __name__ == "__main__":
    main()
