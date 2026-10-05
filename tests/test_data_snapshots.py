"""Offline provenance checks for the fixed teaching-data snapshots."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
MANIFEST = json.loads((DATA / "manifest.json").read_text())


@pytest.mark.parametrize("entry", MANIFEST, ids=lambda entry: entry["file"])
def test_snapshots_match_recorded_bytes_and_schema(entry):
    path = DATA / entry["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    frame = pd.read_csv(path)
    assert len(frame) == entry["rows"]
    assert list(frame.columns) == entry["columns"]
    numeric = frame.select_dtypes(include="number")
    assert np.isfinite(numeric.to_numpy()).all()
    assert entry["source_url"].startswith("https://")


@pytest.mark.parametrize("entry", [item for item in MANIFEST if "ticker" in item],
                         ids=lambda entry: entry["ticker"])
def test_new_market_snapshots_have_valid_dates_and_levels(entry):
    frame = pd.read_csv(DATA / entry["file"], parse_dates=["Date"])
    dates = frame["Date"]
    assert dates.is_unique and dates.is_monotonic_increasing
    assert dates.min().strftime("%Y-%m-%d") == entry["start"]
    assert dates.max().strftime("%Y-%m-%d") == entry["end"]
    assert dates.min() >= pd.Timestamp("2017-01-01")
    assert dates.max() <= pd.Timestamp("2025-12-31")
    assert (frame["Close"] > 0).all()
    assert entry["missing_close_rows_removed"] >= 0


def test_comparable_market_and_currency_metadata():
    indexed = {entry.get("ticker"): entry for entry in MANIFEST if "ticker" in entry}
    assert indexed["^GSPC"]["series_type"] == "price index"
    assert indexed["^GDAXIP"]["series_type"] == "price index"
    assert indexed["^GSPC"]["units"] == "USD"
    assert indexed["^GDAXIP"]["units"] == "EUR"
    assert indexed["^GDAXIP"]["provider_name"].split()[-1] == "K"
    assert indexed["EURUSD=X"]["units"] == "USD per EUR"
    dax = pd.read_csv(DATA / indexed["^GDAXIP"]["file"])
    assert dax["Close"].iloc[0] == pytest.approx(5638.490234375)


def test_shared_price_calendar_is_the_documented_fixed_sample():
    files = ["sp500_prices_2017_2025.csv", "dax_price_prices_2017_2025.csv", "eurusd_2017_2025.csv"]
    calendars = [set(pd.read_csv(DATA / name)["Date"]) for name in files]
    shared = sorted(set.intersection(*calendars))
    assert len(shared) == 2217
    assert shared[0] == "2017-01-03"
    assert shared[-1] == "2025-12-30"
    # The intersection is deliberately smaller than either full equity calendar.
    assert len(shared) < min(len(calendar) for calendar in calendars[:2])
