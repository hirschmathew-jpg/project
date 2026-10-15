"""Panel 3 - Sector Winners & Losers.

Placeholder: the panel owner replaces this module with the real panel.
"""

from __future__ import annotations

import streamlit as st

from dashboard import ui

NUMBER = 3
SLUG = "sectors"
TITLE = "Sector Winners & Losers"
OWNER = "Team member 3"  # panel owner: put your name here
QUESTION = "Which parts of the stock market won and lost?"


def refresh_data() -> None:
    """Download fresh data for this panel (used by the refresh button and scripts/update_data.py)."""


def render() -> None:
    """Draw the whole panel inside the current Streamlit container."""
    ui.panel_header(NUMBER, TITLE, OWNER, QUESTION)
    st.write("Coming soon.")
