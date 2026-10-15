"""Shared chart styling so the four panels look like one dashboard.

All colours come from one colour-blind-safe palette. Categorical colours are
used in a fixed order (BLUE first, then ORANGE, AQUA, ...) and a series keeps
its colour even when other series are hidden by a control.
"""

from __future__ import annotations

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

# Plotly toolbar: keep zoom/pan/download, drop the clutter
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

pio.templates["dashboard"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, size=13, color=INK_SECONDARY),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
        colorway=SERIES,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False, zeroline=False, showline=True, linecolor=AXIS,
            ticks="outside", tickcolor=AXIS, title=dict(font=dict(size=12, color=INK_MUTED)),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
            title=dict(font=dict(size=12, color=INK_MUTED)),
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
    if title:
        fig.update_layout(title=dict(text=title))
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        fig.update_yaxes(title_text=y_title)
    return fig


def add_recession_bands(fig: go.Figure, periods, start=None, end=None, row=None, col=None) -> go.Figure:
    """Shade recession periods in grey (only the ones overlapping start..end)."""
    import pandas as pd

    lo = pd.Timestamp(start) if start is not None else None
    hi = pd.Timestamp(end) if end is not None else None
    for rec_start, rec_end in periods:
        if (hi is not None and rec_start > hi) or (lo is not None and rec_end < lo):
            continue
        x0 = max(rec_start, lo) if lo is not None else rec_start
        x1 = min(rec_end, hi) if hi is not None else rec_end
        kwargs = {"row": row, "col": col} if row is not None else {}
        fig.add_vrect(x0=x0, x1=x1, fillcolor=RECESSION_FILL, line_width=0, layer="below", **kwargs)
    return fig
