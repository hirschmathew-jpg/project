"""Tests for the shared chart theme (no internet needed)."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from dashboard import theme

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_chart_background_matches_the_streamlit_page():
    """Charts sit flat on the page only if both files use the same colour."""
    config = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text()
    assert f'backgroundColor = "{theme.PAGE_BACKGROUND}"' in config


def test_style_figure_sets_the_background_on_the_figure_itself():
    # Streamlit ignores the template's background and would paint its own grey
    fig = theme.style_figure(go.Figure())
    assert fig.layout.plot_bgcolor == theme.PLOT_BACKGROUND
    assert fig.layout.paper_bgcolor == theme.PLOT_BACKGROUND
    assert fig.layout.height == theme.CHART_HEIGHT


def test_legend_rows():
    assert theme.legend_rows([]) == 1
    assert theme.legend_rows(["Median", "Start"]) == 1
    four_long_entries = ["Fed funds rate", "Rule: Taylor (1993)", "Core PCE inflation", "US recession"]
    assert theme.legend_rows(four_long_entries) == 2
    assert theme.legend_rows(four_long_entries, available_width=900) == 1


def test_legend_top_margin_grows_one_row_at_a_time():
    one_row = theme.legend_top_margin(["A", "B"])
    two_rows = theme.legend_top_margin(["A long legend entry"] * 4)
    assert one_row == theme.TOOLBAR_HEIGHT + theme.LEGEND_ROW_HEIGHT
    assert two_rows == one_row + theme.LEGEND_ROW_HEIGHT


def test_legend_names_skips_hidden_entries():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1], y=[1], name="Shown"))
    fig.add_trace(go.Scatter(x=[1], y=[1], name="Hidden", showlegend=False))
    assert theme.legend_names(fig) == ["Shown"]


def test_recession_bands_skip_recessions_that_only_touch_the_window():
    covid = (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-05-01"))
    fig = theme.add_recession_bands(go.Figure(), [covid], start="2020-05-01", end="2021-12-31")
    assert len(fig.layout.shapes) == 0  # no band of zero width
    fig = theme.add_recession_bands(go.Figure(), [covid], start="2020-03-01", end="2021-12-31")
    assert len(fig.layout.shapes) == 1
    band = fig.layout.shapes[0]
    assert pd.Timestamp(band.x0) == pd.Timestamp("2020-03-01")  # clipped to the window
    assert band.fillcolor == theme.RECESSION_FILL


def test_recession_legend_entry_is_an_empty_grey_square():
    fig = theme.add_recession_legend_entry(go.Figure())
    entry = fig.data[0]
    assert entry.name == "US recession"
    assert list(entry.x) == [None]  # draws nothing on the chart, only in the legend
    assert entry.marker.symbol == "square"
    assert entry.marker.color == theme.RECESSION_FILL
    assert theme.legend_names(fig) == ["US recession"]
