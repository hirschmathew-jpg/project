"""Panel 1 - Inflation vs the Fed.

Placeholder: the panel owner replaces this module with the real panel.
"""

from __future__ import annotations

import streamlit as st

from dashboard import ui

NUMBER = 1
SLUG = "inflation"
TITLE = "Inflation vs the Fed"
OWNER = "Team member 1"  # panel owner: put your name here
QUESTION = "Did interest rates keep up with inflation?"


def refresh_data() -> None:
    """Download fresh data for this panel (used by the refresh button and scripts/update_data.py)."""


def render() -> None:
    """Draw the whole panel inside the current Streamlit container."""
    ui.panel_header(NUMBER, TITLE, OWNER, QUESTION)
    st.write("Coming soon.")
