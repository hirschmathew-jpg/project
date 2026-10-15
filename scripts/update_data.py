"""Download the latest data for every panel and save it in data/.

    python scripts/update_data.py

Needs an internet connection. If a download fails, the old snapshot is kept.
Set the FRED_API_KEY environment variable to use the official FRED API.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.data import load_recessions  # noqa: E402
from dashboard.panels import PANELS  # noqa: E402


def main() -> int:
    failures = 0
    jobs = [(f"Panel {p.NUMBER}: {p.TITLE}", p.refresh_data) for p in PANELS]
    jobs.append(("Recession dates (USREC)", lambda: load_recessions(refresh=True)))
    for label, job in jobs:
        try:
            job()
            print(f"OK      {label}")
        except Exception as exc:
            failures += 1
            print(f"FAILED  {label}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
