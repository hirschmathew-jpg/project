"""Work on one panel on its own, at roughly the width it gets in the dashboard.

    streamlit run scripts/preview_panel.py -- inflation
    streamlit run scripts/preview_panel.py -- yield_curve
    streamlit run scripts/preview_panel.py -- sectors
    streamlit run scripts/preview_panel.py -- portfolio

You can also switch panels with the URL, e.g. http://localhost:8501/?panel=sectors
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from dashboard.panels import PANELS_BY_SLUG  # noqa: E402

st.set_page_config(page_title="Panel preview", layout="centered")

slug = st.query_params.get("panel") or (sys.argv[1] if len(sys.argv) > 1 else "inflation")
if slug not in PANELS_BY_SLUG:
    st.error(f"Unknown panel '{slug}'. Choose one of: {', '.join(PANELS_BY_SLUG)}")
    st.stop()

with st.container(border=True):
    PANELS_BY_SLUG[slug].render()
