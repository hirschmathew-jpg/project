"""Panel 2 - The Yield Curve.

Placeholder: the panel owner replaces this module with the real panel.
"""

from __future__ import annotations

import streamlit as st

from dashboard import ui

NUMBER = 2
SLUG = "yield_curve"
TITLE = "The Yield Curve"
OWNER = "Team member 2"  # panel owner: put your name here
QUESTION = "What is the bond market signalling about recessions?"


def refresh_data() -> None:
    """Download fresh data for this panel (used by the refresh button and scripts/update_data.py)."""


def render() -> None:
    """Draw the whole panel inside the current Streamlit container."""
    ui.panel_header(NUMBER, TITLE, OWNER, QUESTION)
    st.write("Coming soon.")
