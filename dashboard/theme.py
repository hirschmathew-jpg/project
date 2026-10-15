"""Shared chart styling so the four panels look like one dashboard.

All colours come from one colour-blind-safe palette. Categorical colours are
used in a fixed order (BLUE first, then ORANGE, AQUA, ...) and a series keeps
its colour even when other series are hidden by a control.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Categorical colours, in the order they should be assigned
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES

# Ink and chrome
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
NEUTRAL = "#f0efec"

# Chart background. It MUST stay the same as backgroundColor in .streamlit/config.toml,
# so the charts sit flat on the page. (If we left it transparent, Streamlit would fill
# the plot area with secondaryBackgroundColor, a darker grey that hides the recession
# bands.) Use PLOT_BACKGROUND for anything drawn "in the chart's own colour", e.g. a
# label background that hides a line running behind it.
PAGE_BACKGROUND = "#f9f9f7"
PLOT_BACKGROUND = PAGE_BACKGROUND

# Positive / negative (diverging) colours, e.g. bars above/below zero
POSITIVE = BLUE
NEGATIVE = RED

# Grey band used to shade recessions
RECESSION_FILL = "rgba(137, 135, 129, 0.16)"

# Single-hue ramp for magnitude (light = small, dark = big)
SEQUENTIAL_SCALE = [
    [0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1.0, "#0d366b"],
]
# Red <-> grey <-> blue for values that can be negative or positive (e.g. correlations)
DIVERGING_SCALE = [
    [0.0, "#b62f2f"], [0.25, "#ec9a99"], [0.5, NEUTRAL], [0.75, "#86b6ef"], [1.0, "#1c5cab"],
]

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
CHART_HEIGHT = 400

# Plotly toolbar (appears at the top right when the mouse is over a chart):
# keep zoom, pan, reset and download, drop the clutter
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines",
                               "zoomIn2d", "zoomOut2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

# Space at the top of a chart for a legend placed above the plot (see legend_top_margin)
TOOLBAR_HEIGHT = 36      # px kept free at the very top for the toolbar
LEGEND_ROW_HEIGHT = 22   # px per row of legend entries
# Narrowest plot area (px) a panel gets in the 2 x 2 grid (a window about 1200 px wide;
# below that app.py stacks the panels in one column, so they get wider again)
NARROWEST_PLOT_WIDTH = 430

pio.templates["dashboard"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, size=13, color=INK_SECONDARY),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
        colorway=SERIES,
        paper_bgcolor=PLOT_BACKGROUND,
        plot_bgcolor=PLOT_BACKGROUND,
        # automargin: Plotly grows the margins to fit the tick labels and axis titles,
        # so nothing is cut off even though the default margins below are small
        xaxis=dict(
            showgrid=False, zeroline=False, showline=True, linecolor=AXIS,
            ticks="outside", tickcolor=AXIS, title=dict(font=dict(size=12, color=INK_MUTED)),
            automargin=True,
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
            title=dict(font=dict(size=12, color=INK_MUTED)), automargin=True,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=INK_SECONDARY),
        ),
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
        margin=dict(l=8, r=8, t=36, b=8),
        hovermode="x unified",
    )
)


def style_figure(fig: go.Figure, *, height: int = CHART_HEIGHT, title: str | None = None,
                 x_title: str | None = None, y_title: str | None = None,
                 hovermode: str | bool = "x unified") -> go.Figure:
    """Apply the shared dashboard look to a Plotly figure and return it."""
    fig.update_layout(template="dashboard", height=height, hovermode=hovermode)
    # The background colours are set on the figure itself as well as in the template:
    # Streamlit only looks at the figure's own layout and would otherwise paint its grey in
    fig.update_layout(paper_bgcolor=PLOT_BACKGROUND, plot_bgcolor=PLOT_BACKGROUND)
    if title:
        fig.update_layout(title=dict(text=title))
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    return fig


def legend_rows(entry_names: list[str], available_width: int = NARROWEST_PLOT_WIDTH) -> int:
    """Estimate how many rows a horizontal legend with these entries needs.

    Each entry takes about 30 px (symbol + gap) plus 6.5 px per character at
    the legend's 12 px font. Entries go on the same row until the row is full.
    """
    rows = 1
    row_width = 0.0
    for name in entry_names:
        entry_width = 30 + 6.5 * len(name)
        if row_width > 0 and row_width + entry_width > available_width:
            rows += 1          # this entry starts a new row
            row_width = 0.0
        row_width += entry_width
    return rows


def legend_top_margin(entry_names: list[str]) -> int:
    """Top margin (px) for a chart whose legend sits above the plot.

    Plotly's toolbar appears at the top right when the mouse is over a chart.
    The margin leaves room for the toolbar AND every legend row below it, so
    the toolbar never covers a legend entry, even in the narrow 2 x 2 grid.
    """
    return TOOLBAR_HEIGHT + LEGEND_ROW_HEIGHT * legend_rows(entry_names)


def legend_names(fig: go.Figure) -> list[str]:
    """Names of the traces that appear in the figure's legend."""
    names = []
    for trace in fig.data:
        if trace.showlegend is not False and trace.name:
            names.append(trace.name)
    return names


def add_recession_bands(fig: go.Figure, periods, start=None, end=None, row=None, col=None) -> go.Figure:
    """Shade recession periods in grey (only the ones overlapping start..end).

    A recession that only touches the edge of the window is skipped, because
    it would be a band of zero width.
    """
    lo = pd.Timestamp(start) if start is not None else None
    hi = pd.Timestamp(end) if end is not None else None
    for rec_start, rec_end in periods:
        if (hi is not None and rec_start >= hi) or (lo is not None and rec_end <= lo):
            continue
        x0 = max(rec_start, lo) if lo is not None else rec_start
        x1 = min(rec_end, hi) if hi is not None else rec_end
        kwargs = {"row": row, "col": col} if row is not None else {}
        fig.add_vrect(x0=x0, x1=x1, fillcolor=RECESSION_FILL, line_width=0, layer="below", **kwargs)
    return fig


def add_recession_legend_entry(fig: go.Figure) -> go.Figure:
    """Add a "US recession" entry (a grey square) to the legend.

    Shaded bands are shapes, and shapes never appear in a Plotly legend. An
    empty trace (no points) with a square marker in the band colour gives the
    legend an entry without drawing anything on the chart. Only call this when
    at least one band was drawn, so the legend never names something that
    isn't there.
    """
    fig.add_trace(go.Scatter(
        x=[None], y=[None], name="US recession", mode="markers", hoverinfo="skip",
        marker=dict(symbol="square", size=12, color=RECESSION_FILL, line=dict(color=AXIS, width=1)),
    ))
    return fig
