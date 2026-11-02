"""Ask the open-vocabulary detector (on the Spark, through an SSH tunnel on localhost:8766) to find named objects.

Always optional: any failure or timeout returns None and the policy carries on with its colour pipeline.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request

import cv2

URL = "http://127.0.0.1:8766/detect"


_DOWN_UNTIL = [0.0]  # circuit breaker: after one failure or timeout, skip the server for 2 minutes (a hung tunnel must not stall a rollout)


def find(rgb, queries: list[str], threshold: float = 0.2, timeout: float = 8.0, region=None) -> list[dict] | None:
    """[{"label", "score", "box": [x0, y0, x1, y1]}] in the image's pixels, or None if the detector is unreachable."""
    if time.time() < _DOWN_UNTIL[0]:
        return None
    try:
        ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return None
        payload = {"image_b64": base64.b64encode(png.tobytes()).decode(), "queries": queries, "threshold": threshold}
        if region is not None:
            payload["region"] = list(region)
        body = json.dumps(payload)
        req = urllib.request.Request(URL, body.encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        return out["detections"] if "detections" in out else None
    except Exception:  # noqa: BLE001
        _DOWN_UNTIL[0] = time.time() + 120.0  # tunnel down, server loading, timeout: the policy does not need it
        return None
