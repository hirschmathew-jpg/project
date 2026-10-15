"""Tests for the shared data layer (no internet needed)."""

import pandas as pd
import pytest

from dashboard import data


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    return tmp_path


def sample_frame():
    index = pd.date_range("2024-01-01", periods=3, freq="MS")
    return pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, None, 6.0]}, index=index)


def test_snapshot_round_trip(tmp_data_dir):
    data.write_snapshot("demo", sample_frame())
    loaded = data.read_snapshot("demo")
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert list(loaded.columns) == ["A", "B"]
    assert loaded.loc["2024-02-01", "A"] == 2.0
    assert pd.isna(loaded.loc["2024-02-01", "B"])


def test_load_dataset_downloads_when_no_snapshot(tmp_data_dir):
    calls = []

    def fetch():
        calls.append(1)
        return sample_frame()

    df = data.load_dataset("demo", fetch)
    assert len(calls) == 1
    assert (tmp_data_dir / "demo.csv").exists()
    assert len(df) == 3

    # Second call uses the snapshot, no download
    data.load_dataset("demo", fetch)
    assert len(calls) == 1


def test_load_dataset_falls_back_to_snapshot_when_download_fails(tmp_data_dir):
    data.write_snapshot("demo", sample_frame())

    def broken_fetch():
        raise ConnectionError("no internet")

    df = data.load_dataset("demo", broken_fetch, refresh=True)
    assert len(df) == 3


def test_load_dataset_raises_without_snapshot_or_internet(tmp_data_dir):
    def broken_fetch():
        raise ConnectionError("no internet")

    with pytest.raises(data.DataUnavailableError):
        data.load_dataset("demo", broken_fetch)


def test_fetch_fred_parses_public_csv(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    class FakeResponse:
        text = "observation_date,FEDFUNDS\n2024-01-01,5.33\n2024-02-01,.\n2024-03-01,5.33\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse())
    df = data.fetch_fred(["FEDFUNDS"], start="2024-01-01")
    assert list(df.columns) == ["FEDFUNDS"]
    assert df.loc["2024-01-01", "FEDFUNDS"] == 5.33
    # "." is FRED's missing value; a date with nothing in any column is dropped
    assert pd.Timestamp("2024-02-01") not in df.index
    assert len(df) == 2


def test_fetch_fred_uses_api_when_key_set(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "abc123")
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"observations": [{"date": "2024-01-01", "value": "3.1"},
                                     {"date": "2024-02-01", "value": "."}]}

    def fake_get(url, params, timeout):
        seen.update(params)
        return FakeResponse()

    monkeypatch.setattr(data.requests, "get", fake_get)
    df = data.fetch_fred(["CPIAUCSL"])
    assert seen["api_key"] == "abc123"
    assert df.loc["2024-01-01", "CPIAUCSL"] == 3.1
    assert len(df) == 1


def test_fetch_prices_keeps_finished_days_only(monkeypatch):
    """Yahoo's row for today is a price during trading, not a close, so it is dropped."""
    import sys
    import types

    today = pd.Timestamp.now().normalize()
    days = pd.DatetimeIndex([today - pd.Timedelta(days=2), today - pd.Timedelta(days=1), today])
    columns = pd.MultiIndex.from_product([["Close", "Open"], ["SPY", "AGG"]])
    raw = pd.DataFrame(1.0, index=days, columns=columns)

    fake_yfinance = types.ModuleType("yfinance")
    fake_yfinance.download = lambda *args, **kwargs: raw
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    close = data.fetch_prices(["SPY", "AGG"])
    assert list(close.columns) == ["SPY", "AGG"]
    assert list(close.index) == list(days[:2])


def test_recession_periods():
    index = pd.date_range("2020-01-01", periods=8, freq="MS")
    flag = pd.Series([0, 1, 1, 0, 0, 1, 1, 1], index=index)
    periods = data.recession_periods(flag)
    assert periods[0] == (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-01"))
    # Recession still running at the end: band runs to the month after the last data point
    assert periods[1] == (pd.Timestamp("2020-06-01"), pd.Timestamp("2020-09-01"))


def test_recession_periods_none():
    index = pd.date_range("2020-01-01", periods=4, freq="MS")
    assert data.recession_periods(pd.Series([0, 0, 0, 0], index=index)) == []


RECESSIONS = [(pd.Timestamp("2001-03-01"), pd.Timestamp("2001-12-01")),
              (pd.Timestamp("2007-12-01"), pd.Timestamp("2009-07-01")),
              (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-05-01"))]


def test_recessions_in_window():
    window = data.recessions_in_window(RECESSIONS, pd.Timestamp("2009-01-01"), pd.Timestamp("2019-12-01"))
    assert window == [RECESSIONS[1]]  # only the 2008-09 recession overlaps
    both = data.recessions_in_window(RECESSIONS, "2001-06-01", "2008-06-01")  # strings work too
    assert both == RECESSIONS[:2]
    assert data.recessions_in_window(RECESSIONS, "2010-01-01", "2019-12-31") == []
    assert data.recessions_in_window([], "2000-01-01", "2026-01-01") == []


def test_recessions_touching_the_window_edge_are_left_out():
    # They would only give a band of zero width (and a legend entry for nothing)
    # COVID recession ends (2020-05-01) exactly where the window starts
    assert data.recessions_in_window(RECESSIONS, "2020-05-01", "2021-12-31") == []
    # 2001 recession starts exactly where the window ends
    assert data.recessions_in_window(RECESSIONS, "1999-01-01", "2001-03-01") == []
    # One day more on either side and the recession is in
    assert data.recessions_in_window(RECESSIONS, "2020-04-30", "2021-12-31") == [RECESSIONS[2]]
    assert data.recessions_in_window(RECESSIONS, "1999-01-01", "2001-03-02") == [RECESSIONS[0]]


def test_filter_dates():
    df = sample_frame()
    assert len(data.filter_dates(df, "2024-02-01", None)) == 2
    assert len(data.filter_dates(df, None, "2024-01-31")) == 1
    assert len(data.filter_dates(df, "2024-01-01", "2024-03-01")) == 3


def test_snapshot_end_dates(tmp_data_dir):
    data.write_snapshot("demo", sample_frame())
    # A forecast row far ahead with only one of three columns filled is ignored
    index = pd.to_datetime(["2024-01-01", "2024-02-01", "2030-01-01"])
    forecast = pd.DataFrame({"A": [1.0, 2.0, None], "B": [1.0, 2.0, None], "C": [5.0, 5.0, 4.5]}, index=index)
    data.write_snapshot("with_forecast", forecast)
    (tmp_data_dir / "empty.csv").write_text("date,A\n")
    assert data.snapshot_end_dates() == {"demo": pd.Timestamp("2024-03-01"),
                                         "with_forecast": pd.Timestamp("2024-02-01")}


def test_every_shipped_snapshot_has_dates():
    """Every CSV in data/ is a dated snapshot (no stray files)."""
    end_dates = data.snapshot_end_dates()
    csv_files = sorted(path.stem for path in data.DATA_DIR.glob("*.csv"))
    assert sorted(end_dates) == csv_files
    assert "recessions" in end_dates


def test_recessions_snapshot_is_shipped():
    """The repo must include recession dates so the dashboard works offline."""
    rec = data.load_recessions()
    assert rec.isin([0, 1]).all()
    assert rec.loc["2020-04-01"] == 1  # COVID recession
