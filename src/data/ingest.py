"""Download the UCI Online Retail II dataset and load it into a DataFrame."""

import urllib.request
import zipfile

import pandas as pd

from src.config import DATASET_URL, RAW_DIR, RAW_PARQUET, RAW_XLSX, RAW_ZIP, get_logger

log = get_logger(__name__)

RAW_COLUMNS = ["Invoice", "StockCode", "Description", "Quantity",
               "InvoiceDate", "Price", "Customer ID", "Country"]


def download_dataset() -> None:
    """Download and unzip the dataset unless the Excel file already exists."""
    if RAW_XLSX.exists():
        log.info("Raw dataset already present: %s", RAW_XLSX)
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", DATASET_URL)
    try:
        urllib.request.urlretrieve(DATASET_URL, RAW_ZIP)
    except OSError as error:
        raise RuntimeError(f"Could not download dataset from {DATASET_URL}") from error
    with zipfile.ZipFile(RAW_ZIP) as archive:
        archive.extractall(RAW_DIR)
    if not RAW_XLSX.exists():
        raise FileNotFoundError(f"Expected {RAW_XLSX.name} inside {RAW_ZIP.name}")


def load_raw() -> pd.DataFrame:
    """Return both yearly sheets stacked into one DataFrame.

    Reading the Excel file takes ~30s, so the result is cached as Parquet.
    """
    if RAW_PARQUET.exists():
        return pd.read_parquet(RAW_PARQUET)
    download_dataset()
    log.info("Reading Excel workbook (slow, cached afterwards)")
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None,
                           dtype={"Invoice": str, "StockCode": str, "Description": str})
    raw = pd.concat(sheets.values(), ignore_index=True)
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw data is missing columns: {sorted(missing)}")
    raw.to_parquet(RAW_PARQUET, index=False)
    log.info("Loaded %d raw rows", len(raw))
    return raw
