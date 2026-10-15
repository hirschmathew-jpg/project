"""Inflation, Interest Rates & Markets - a 4-panel economics/finance dashboard.

Run it with:

    streamlit run app.py

Each of the four panels lives in its own module in dashboard/panels/ and is
owned by one team member. This file only lays them out in a 2 x 2 grid and
adds the shared header and sidebar.
"""

from __future__ import annotations

import logging

import streamlit as st

from dashboard.data import load_recessions
from dashboard.panels import PANELS

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Inflation, Rates & Markets",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def refresh_all_data() -> list[str]:
    """Re-download every panel's data. Returns a list of error messages (empty = all good)."""
    errors = []
    for panel in PANELS:
        try:
            panel.refresh_data()
        except Exception as exc:
            errors.append(f"Panel {panel.NUMBER} ({panel.TITLE}): {exc}")
    try:
        load_recessions(refresh=True)
    except Exception as exc:
        errors.append(f"Recession dates: {exc}")
    st.cache_data.clear()
    return errors


def render_sidebar() -> None:
    with st.sidebar:
        st.header("About this dashboard")
        st.markdown(
            "Four linked views of the same story: how inflation, interest rates and "
            "markets have moved together, and what that means for an investor. "
            "Every chart has its own controls, so you can change the time period, "
            "the model assumptions or the scenario and see the result straight away."
        )

        st.subheader("Team")
        for panel in PANELS:
            st.markdown(f"**Panel {panel.NUMBER} · {panel.TITLE}**  \n{panel.OWNER}")

        st.subheader("Data")
        st.caption(
            "Charts load from the CSV snapshots in the `data/` folder, so the dashboard "
            "works offline. Click below to download the latest numbers from FRED and "
            "Yahoo Finance."
        )
        if st.button("Refresh data", key="refresh_data", icon="🔄"):
            with st.spinner("Downloading the latest data..."):
                errors = refresh_all_data()
            if errors:
                st.warning("Some downloads failed, so the saved data was kept:\n\n- " + "\n- ".join(errors))
            else:
                st.success("All data refreshed.")

        st.caption(
            "Sources: Federal Reserve Bank of St. Louis (FRED), Yahoo Finance via yfinance, "
            "NBER recession dates. For education only, not investment advice."
        )


def render_panel(panel) -> None:
    """Draw one panel in a bordered card. If it fails, show an error but keep the rest of the dashboard running."""
    with st.container(border=True):
        try:
            panel.render()
        except Exception as exc:
            logger.exception("Panel %s failed", panel.SLUG)
            st.error(f"Panel {panel.NUMBER} ({panel.TITLE}) couldn't be drawn: {exc}")


def main() -> None:
    render_sidebar()

    st.title("Inflation, Interest Rates & Markets")
    st.markdown(
        "**1** Did the Fed keep up with inflation? → **2** What did the bond market expect? "
        "→ **3** Which stocks won and lost? → **4** What could it mean for a portfolio?"
    )

    for row_start in range(0, len(PANELS), 2):
        columns = st.columns(2, gap="medium")
        for column, panel in zip(columns, PANELS[row_start:row_start + 2]):
            with column:
                render_panel(panel)

    st.caption(
        "MIS20080 Introduction to Programming · Group dashboard project · "
        "Built with Python, pandas, NumPy, Plotly and Streamlit."
    )


main()
