"""Ask Cosmos (on the Spark, through an SSH tunnel on localhost:8765) what is on the table.

Always optional: any failure or timeout returns None and the policy carries on with its colour pipeline.
"""

from __future__ import annotations

import base64
import json
import urllib.request

import cv2

URL = "http://127.0.0.1:8765/understand"


def understand(rgb, instruction: str, timeout: float = 20.0) -> dict | None:
    """{"objects": [...], "pick_colour": str | None, "raw": str}, or None if Cosmos is unreachable."""
    try:
        ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        body = json.dumps({"image_b64": base64.b64encode(png.tobytes()).decode(), "instruction": instruction})
        req = urllib.request.Request(URL, body.encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        return out if "error" not in out else None
    except Exception:  # noqa: BLE001  # tunnel down, server loading, timeout: the policy does not need it
        return None
