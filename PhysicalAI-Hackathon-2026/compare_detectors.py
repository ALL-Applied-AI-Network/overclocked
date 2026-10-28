"""Offline check: the open-vocabulary detector against the colour pipeline on the saved table frames in debug/.

    DETECT_URL=http://127.0.0.1:8767/detect uv run compare_detectors.py [debug/*_raw.png ...]

For each frame it runs Policy.look() both ways (the real code path, minus the arm) and compares what each would pick.
"""

import glob
import math
import os
import sys

import cv2

import detect_client
import policy as P

detect_client.URL = os.environ.get("DETECT_URL", detect_client.URL)
frames = sys.argv[1:] or sorted(glob.glob("debug/*_raw.png"))
agree = open_only = colour_only = both_none = 0
for f in frames:
    rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (1080, 1920):
        continue
    picks = {}
    for mode in ("colour", "open_vocab"):
        P.DETECTOR = mode
        pol = P.Policy()
        pol.instruction = "Put the cube into the yellow bin."
        pol._save_debug = lambda *a, **k: None   # never write debug files from a comparison
        pol._ask_cosmos = lambda *a, **k: None
        pol.log = lambda *a, **k: None
        pol.placed = 0
        picks[mode] = pol.look({"images": {"table": rgb}})
    c, o = picks["colour"], picks["open_vocab"]
    if c and o:
        d = math.dist(c["xy_mm"], o["xy_mm"])
        agree += d < 10
        print(f"{os.path.basename(f)}: colour {c['colour']} ({c['xy_mm'][0]:.0f}, {c['xy_mm'][1]:.0f}); "
              f"open-vocab {o['colour']} ({o['xy_mm'][0]:.0f}, {o['xy_mm'][1]:.0f}) score {o['iou']:.2f}; {d:.1f} mm apart"
              + ("" if d < 10 else "   <-- DIFFERENT"))
    elif c:
        colour_only += 1
        print(f"{os.path.basename(f)}: colour {c['colour']} ({c['xy_mm'][0]:.0f}, {c['xy_mm'][1]:.0f}); open-vocab NOTHING   <-- MISSED")
    elif o:
        open_only += 1
        print(f"{os.path.basename(f)}: colour nothing; open-vocab {o['colour']} ({o['xy_mm'][0]:.0f}, {o['xy_mm'][1]:.0f}) score {o['iou']:.2f}   <-- EXTRA")
    else:
        both_none += 1
print(f"\n{len(frames)} frames: agree {agree}, open-vocab missed {colour_only}, open-vocab extra {open_only}, both none {both_none}")
