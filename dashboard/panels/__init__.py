"""The four dashboard panels, one per team member.

Every panel module follows the same small contract so app.py can treat them
the same way:

* ``NUMBER``, ``SLUG``, ``TITLE``, ``OWNER``, ``QUESTION`` - constants
* ``refresh_data()`` - download fresh data and update the snapshot in data/
* ``render()``       - draw the panel (controls + chart) in the current container
"""

from dashboard.panels import inflation, portfolio, sectors, yield_curve

PANELS = [inflation, yield_curve, sectors, portfolio]
PANELS_BY_SLUG = {panel.SLUG: panel for panel in PANELS}
