"""Small Streamlit building blocks every panel uses, so all four panels share
the same layout: header -> controls -> key numbers -> chart -> insight -> details.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.theme import PLOTLY_CONFIG


def panel_header(number: int, title: str, owner: str, question: str) -> None:
    """Panel title, who owns it, and the question the chart answers."""
    st.markdown(f"#### {number} · {title}")
    st.caption(f"**{question}** · Panel owner: {owner}")


def show_chart(fig: go.Figure, key: str) -> None:
    """Draw a Plotly figure using our own theme (not Streamlit's)."""
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


def insight(text: str) -> None:
    """One or two sentences telling the story the chart shows right now."""
    st.info(text, icon="💡")


def details(method_markdown: str, data: pd.DataFrame, key: str, file_name: str) -> None:
    """Expanders with the method/sources and a table view of the plotted data."""
    with st.expander("Method & data sources"):
        st.markdown(method_markdown)
    with st.expander("View the data behind this chart"):
        st.dataframe(data, height=240)
        st.download_button(
            "Download as CSV",
            data=data.to_csv().encode("utf-8"),
            file_name=file_name,
            mime="text/csv",
            key=f"{key}_download",
        )


def no_data_message(what: str, exc: Exception) -> None:
    """Friendly message when a panel's data can't be loaded."""
    st.warning(
        f"{what} data isn't available ({exc}). Run `python scripts/update_data.py` "
        "with an internet connection to download it."
    )
