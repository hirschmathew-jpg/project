"""Smoke test: run the whole dashboard headless and check every panel draws without errors."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard.panels import PANELS

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def run_app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_app_runs_without_errors():
    at = run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_every_panel_is_shown():
    at = run_app()
    headings = " ".join(md.value for md in at.markdown)
    for panel in PANELS:
        assert panel.TITLE in headings


def test_panel_contract():
    for panel in PANELS:
        for attr in ("NUMBER", "SLUG", "TITLE", "OWNER", "QUESTION"):
            assert getattr(panel, attr), f"{panel.__name__} is missing {attr}"
        assert callable(panel.render)
        assert callable(panel.refresh_data)
    assert [p.NUMBER for p in PANELS] == [1, 2, 3, 4]
