"""Ask the held/empty verifier on the Spark (C-RADIOv2-B + head, through the SSH tunnel on localhost:8767) about a fingertip crop.

Always optional: any failure or timeout returns None and the policy carries on without the check (and skips the server for two minutes).
"""
from __future__ import annotations

import base64
import json
import time
import urllib.request

import cv2

URL = "http://127.0.0.1:8767/held"
_DOWN_UNTIL = [0.0]


def p_held(rgb, timeout: float = 3.0) -> float | None:
    """P(an object is held between the fingertips) for a 240x240 RGB crop, or None if the verifier is unreachable."""
    if time.time() < _DOWN_UNTIL[0]:
        return None
    try:
        ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        body = json.dumps({"image_b64": base64.b64encode(png.tobytes()).decode()}).encode()
        req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return float(json.loads(r.read())["p_held"])
    except Exception:  # noqa: BLE001  # tunnel down, server busy, timeout
        _DOWN_UNTIL[0] = time.time() + 120.0
        return None
