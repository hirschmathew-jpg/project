"""Shared data layer used by all four panels.

Every panel gets its data through two helpers:

* ``load_fred(name, series_ids, start)`` - economic data from FRED (St. Louis Fed)
* ``load_prices(name, tickers, start)`` - market prices from Yahoo Finance (yfinance)

Both read a CSV snapshot from the ``data/`` folder first. That keeps the
dashboard fast, lets it run with no internet connection, and means the numbers
don't change between when we hand it in and when it gets marked. Passing
``refresh=True`` downloads fresh data and overwrites the snapshot. If the
download fails we fall back to the old snapshot instead of crashing.

FRED access works without an API key (public CSV download). If a
``FRED_API_KEY`` environment variable is set, the official FRED API is used
instead (see Session 6 of the module).
"""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_TIMEOUT = 30  # seconds


class DataUnavailableError(RuntimeError):
    """Raised when data can't be downloaded and no snapshot exists either."""


# ---------------------------------------------------------------------------
# Snapshot files (data/<name>.csv)
# ---------------------------------------------------------------------------

def snapshot_path(name: str) -> Path:
    """Where the CSV snapshot for dataset ``name`` lives."""
    return DATA_DIR / f"{name}.csv"


def read_snapshot(name: str) -> pd.DataFrame:
    """Read a saved snapshot as a DataFrame with a sorted DatetimeIndex."""
    df = pd.read_csv(snapshot_path(name), index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "date"
    return df.sort_index()


def write_snapshot(name: str, df: pd.DataFrame) -> Path:
    """Save ``df`` to ``data/<name>.csv`` and return the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(name)
    out = df.copy()
    out.index.name = "date"
    out.to_csv(path, float_format="%.8g", date_format="%Y-%m-%d")
    return path


def load_dataset(name: str, fetch: Callable[[], pd.DataFrame], refresh: bool = False) -> pd.DataFrame:
    """Return dataset ``name`` from its snapshot, downloading it when needed.

    ``fetch`` is a function that downloads the data. It's only called when
    ``refresh`` is True or when no snapshot exists yet.
    """
    path = snapshot_path(name)
    if refresh or not path.exists():
        try:
            df = fetch()
            if df.empty:
                raise DataUnavailableError(f"download for '{name}' returned no rows")
            write_snapshot(name, df)
        except Exception as exc:  # network down, API change, rate limit...
            if path.exists():
                logger.warning("Refreshing '%s' failed (%s); using saved snapshot.", name, exc)
            else:
                raise DataUnavailableError(
                    f"Could not download '{name}' and there is no saved snapshot in {DATA_DIR}."
                ) from exc
    return read_snapshot(name)


# ---------------------------------------------------------------------------
# FRED (economic data)
# ---------------------------------------------------------------------------

def _fred_series_csv(series_id: str, start: str) -> pd.Series:
    """Download one FRED series from the public CSV endpoint (no key needed)."""
    resp = requests.get(
        FRED_CSV_URL, params={"id": series_id, "cosd": start}, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))
    # First column is the date (FRED has called it both DATE and observation_date)
    dates = pd.to_datetime(raw.iloc[:, 0])
    values = pd.to_numeric(raw.iloc[:, 1], errors="coerce")  # "." means missing
    return pd.Series(values.to_numpy(), index=dates, name=series_id)


def _fred_series_api(series_id: str, start: str, api_key: str) -> pd.Series:
    """Download one FRED series through the official API (needs FRED_API_KEY)."""
    resp = requests.get(
        FRED_API_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    observations = resp.json()["observations"]
    dates = pd.to_datetime([obs["date"] for obs in observations])
    values = pd.to_numeric(pd.Series([obs["value"] for obs in observations]), errors="coerce")
    return pd.Series(values.to_numpy(), index=dates, name=series_id)


def fetch_fred(series_ids: Iterable[str], start: str = "1990-01-01") -> pd.DataFrame:
    """Download several FRED series into one DataFrame (one column per series)."""
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    columns = []
    for series_id in series_ids:
        if api_key:
            columns.append(_fred_series_api(series_id, start, api_key))
        else:
            columns.append(_fred_series_csv(series_id, start))
    df = pd.concat(columns, axis=1).sort_index()
    df.index.name = "date"
    return df.dropna(how="all")


def load_fred(name: str, series_ids: Iterable[str], start: str = "1990-01-01",
              refresh: bool = False) -> pd.DataFrame:
    """FRED series as a DataFrame, read from ``data/<name>.csv`` when available."""
    series_ids = list(series_ids)
    df = load_dataset(name, lambda: fetch_fred(series_ids, start), refresh=refresh)
    missing = [s for s in series_ids if s not in df.columns]
    if missing:
        raise DataUnavailableError(f"Snapshot '{name}' is missing columns {missing}; refresh the data.")
    return df[series_ids]


# ---------------------------------------------------------------------------
# Yahoo Finance (market prices)
# ---------------------------------------------------------------------------

def fetch_prices(tickers: Iterable[str], start: str = "2010-01-01") -> pd.DataFrame:
    """Download daily adjusted close prices (dividends and splits included)."""
    import yfinance as yf  # imported here because it's slow to import

    tickers = list(tickers)
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise DataUnavailableError(f"Yahoo Finance returned no data for {tickers}")
    close = raw["Close"]
    if isinstance(close, pd.Series):  # happens with a single ticker
        close = close.to_frame(tickers[0])
    close = close.reindex(columns=tickers)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close.index.name = "date"
    return close.dropna(how="all")


def load_prices(name: str, tickers: Iterable[str], start: str = "2010-01-01",
                refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted close prices, read from ``data/<name>.csv`` when available."""
    tickers = list(tickers)
    df = load_dataset(name, lambda: fetch_prices(tickers, start), refresh=refresh)
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise DataUnavailableError(f"Snapshot '{name}' is missing tickers {missing}; refresh the data.")
    return df[tickers]


# ---------------------------------------------------------------------------
# US recessions (shared by several panels for grey shading)
# ---------------------------------------------------------------------------

def load_recessions(refresh: bool = False) -> pd.Series:
    """Monthly NBER recession indicator (1 = recession month) from FRED series USREC."""
    return load_fred("recessions", ["USREC"], start="1960-01-01", refresh=refresh)["USREC"]


def recession_periods(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Turn a 0/1 monthly recession flag into a list of (start, end) dates.

    ``end`` is the first month *after* the recession, so a shaded band from
    start to end covers every recession month completely.
    """
    flag = flag.dropna().astype(int)
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    for date, value in flag.items():
        if value == 1 and start is None:
            start = date
        elif value == 0 and start is not None:
            periods.append((start, date))
            start = None
    if start is not None:  # recession still going at the end of the data
        periods.append((start, flag.index[-1] + pd.offsets.MonthBegin(1)))
    return periods


def filter_dates(df: pd.DataFrame | pd.Series, start=None, end=None):
    """Rows of ``df`` between ``start`` and ``end`` (inclusive, either can be None)."""
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df
