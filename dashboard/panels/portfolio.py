"""Panel 4 - Portfolio Scenario Lab.

Placeholder: the panel owner replaces this module with the real panel.
"""

from __future__ import annotations

import streamlit as st

from dashboard import ui

NUMBER = 4
SLUG = "portfolio"
TITLE = "Portfolio Scenario Lab"
OWNER = "Team member 4"  # panel owner: put your name here
QUESTION = "What could this mean for an investor's portfolio?"


def refresh_data() -> None:
    """Download fresh data for this panel (used by the refresh button and scripts/update_data.py)."""


def render() -> None:
    """Draw the whole panel inside the current Streamlit container."""
    ui.panel_header(NUMBER, TITLE, OWNER, QUESTION)
    st.write("Coming soon.")
