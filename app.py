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

from dashboard.data import load_recessions, snapshot_end_dates
from dashboard.panels import PANELS

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Inflation, Rates & Markets",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Streamlit always puts st.columns side by side (until the window is phone-sized).
# Below about 1200 px the 2 x 2 grid gets too narrow for the panels' controls, so this
# CSS stacks the four panels in one column instead. The open sidebar takes about 300 px,
# so with the sidebar open the panels are stacked below 1550 px. (Newer Streamlit marks an
# open sidebar with data-collapsed="false", version 1.40 with aria-expanded="true".)
# The CSS only touches the grid rows, i.e. the rows that contain a panel marker
# (see render_panel), not the rows of controls inside a panel.
PAGE_CSS = """
<style>
@media (max-width: 1199px) {
  div[data-testid="stHorizontalBlock"]:has(.dashboard-panel) {
    flex-direction: column;
  }
  div[data-testid="stHorizontalBlock"]:has(.dashboard-panel) > div {
    width: 100% !important;
    min-width: 100% !important;
    flex: 1 1 100% !important;
  }
}
@media (max-width: 1549px) {
  body:has([data-testid="stSidebar"][data-collapsed="false"]) div[data-testid="stHorizontalBlock"]:has(.dashboard-panel),
  body:has([data-testid="stSidebar"][aria-expanded="true"]) div[data-testid="stHorizontalBlock"]:has(.dashboard-panel) {
    flex-direction: column;
  }
  body:has([data-testid="stSidebar"][data-collapsed="false"]) div[data-testid="stHorizontalBlock"]:has(.dashboard-panel) > div,
  body:has([data-testid="stSidebar"][aria-expanded="true"]) div[data-testid="stHorizontalBlock"]:has(.dashboard-panel) > div {
    width: 100% !important;
    min-width: 100% !important;
    flex: 1 1 100% !important;
  }
}
/* The marker itself takes no space */
div[data-testid="stElementContainer"]:has(.dashboard-panel),
div.element-container:has(.dashboard-panel) {
  display: none;
}
</style>
"""


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


@st.cache_data(show_spinner=False)
def cached_snapshot_dates() -> dict:
    """Last date in each saved CSV in data/ (cached; cleared when the data is refreshed)."""
    return snapshot_end_dates()


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
        try:
            end_dates = cached_snapshot_dates()
        except Exception:  # unreadable files: the panels show their own messages
            end_dates = {}
        if end_dates:
            lines = []
            for name, last_date in end_dates.items():
                if last_date.day == 1:  # monthly FRED data is dated the 1st: show "Sep 2026"
                    when = f"{last_date:%b %Y}"
                else:
                    when = f"{last_date.day} {last_date:%b %Y}"
                lines.append(f"`{name}.csv` · up to {when}")
            st.caption("Saved data:  \n" + "  \n".join(lines))

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


def render_header() -> None:
    """Title, the story in one sentence, and the four panel questions in reading order."""
    st.title("Inflation, Interest Rates & Markets")
    st.markdown(
        "How did the inflation surge after 2021, and the Federal Reserve's response, play out "
        "in bonds, stocks and portfolios? Each panel answers one question. Read them in order, "
        "then change the controls to test the story."
    )
    questions = []
    for panel in PANELS:
        questions.append(f"**{panel.NUMBER}** {panel.QUESTION}")
    st.markdown(" → ".join(questions))


def render_panel(panel) -> None:
    """Draw one panel in a bordered card. If it fails, show an error but keep the rest of the dashboard running."""
    with st.container(border=True):
        # Invisible marker that PAGE_CSS uses to find the grid rows
        st.markdown('<span class="dashboard-panel"></span>', unsafe_allow_html=True)
        try:
            panel.render()
        except Exception as exc:
            logger.exception("Panel %s failed", panel.SLUG)
            st.error(f"Panel {panel.NUMBER} ({panel.TITLE}) couldn't be drawn: {exc}")


def main() -> None:
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    render_sidebar()
    render_header()

    # Two rows of two panels: [1, 2] then [3, 4]
    for row_start in range(0, len(PANELS), 2):
        columns = st.columns(2, gap="medium")
        for column, panel in zip(columns, PANELS[row_start:row_start + 2]):
            with column:
                render_panel(panel)

    st.caption(
        "MIS20080 Introduction to Programming · Group dashboard project · "
        "Built with Python, pandas, NumPy, Plotly and Streamlit. "
        "Open the sidebar (top left) for the team, the data sources and a data refresh button."
    )


main()
