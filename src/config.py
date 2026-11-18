"""Central configuration: paths, constants, and the global random seed."""

import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_ZIP = RAW_DIR / "online_retail_ii.zip"
RAW_XLSX = RAW_DIR / "online_retail_II.xlsx"
RAW_PARQUET = RAW_DIR / "online_retail_ii.parquet"

CLEAN_PARQUET = PROCESSED_DIR / "transactions.parquet"
DATABASE_PATH = PROCESSED_DIR / "retail.db"
CUSTOMERS_PARQUET = PROCESSED_DIR / "customers.parquet"
PREDICTIONS_PARQUET = PROCESSED_DIR / "predictions.parquet"
CLEANING_REPORT = PROCESSED_DIR / "cleaning_report.json"
CLUSTER_REPORT = MODELS_DIR / "clustering_report.json"
PREDICTION_REPORT = MODELS_DIR / "prediction_report.json"
KMEANS_MODEL = MODELS_DIR / "kmeans.joblib"
SVM_MODEL = MODELS_DIR / "svm.joblib"

RANDOM_SEED = 42

# The raw prices are in British pounds. Everything downstream is reported in US dollars,
# converted at one fixed rate: roughly the average GBP->USD rate over Dec 2009 - Dec 2011.
GBP_TO_USD = 1.58

# Clustering: candidate numbers of clusters to evaluate.
CLUSTER_K_RANGE = range(2, 11)

# Purchase prediction: will a customer buy again within this many days of a cutoff date?
PREDICTION_HORIZON_DAYS = 90
# Monthly snapshot cutoffs, split chronologically. Every label window
# (cutoff + 90 days) ends before the next split's first cutoff, so no
# future information leaks backwards. Data ends 2011-12-09.
TRAIN_CUTOFFS = ["2010-06-01", "2010-07-01", "2010-08-01", "2010-09-01", "2010-10-01",
                 "2010-11-01", "2010-12-01", "2011-01-01", "2011-02-01"]
VALIDATION_CUTOFFS = ["2011-06-01"]
TEST_CUTOFFS = ["2011-09-01"]
# Scoring date for live predictions: the day after the last transaction.
SCORING_DATE = "2011-12-10"


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a consistent format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
