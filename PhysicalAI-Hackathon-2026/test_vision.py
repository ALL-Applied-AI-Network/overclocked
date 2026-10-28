"""Check cube_vision against known cube poses.

    uv run test_vision.py                                    # synthetic frames rendered with the real table camera
    uv run test_vision.py --image world_camera.png --gt ground_truth.json [--size 1920x1080]
                                                             # a frame from the v0.5 twin (same camera calibration)

Prints each cube's position error (mm) and yaw error (deg, modulo 90), and exits non-zero if any cube is missed or
off by more than --max-mm / --max-deg. Writes an overlay next to the image for a look.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, load, project
from cube_vision import cube_corners, detect_cubes, draw, square_yaw

RGB = {"yellow": (235, 190, 30), "blue": (30, 70, 200), "red": (200, 30, 30)}


def render(cubes, size=(1920, 1080), seed=0) -> np.ndarray:
    """A grey table with shaded cubes, a yellow bin and noise, through the real table camera."""
    cam = load()["table"]
    rng = np.random.default_rng(seed)
    w, h = size
    s = cam["size"][0] / w
    img = np.full((h, w, 3), (205, 203, 198), np.uint8)
    bin_xy = np.array([0.51, 0.11])  # the default scene's bin, 50 mm square, 55 mm tall
    for z in (0.0, 0.055):
        sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.025 + bin_xy
        pts = project(cam, [[x, y, SURFACE_Z + z] for x, y in sq]) / s
        cv2.fillConvexPoly(img, np.round(pts).astype(np.int32), (240, 200, 20))
    for c in cubes:
        corners = cube_corners(np.array(c["xy_mm"]) / 1000, c["yaw_deg"])
        uv = project(cam, corners) / s
        base = np.array(RGB[c["colour"]], float)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):  # side faces, darker
            face = uv[[a, b, b + 4, a + 4]]
            cv2.fillConvexPoly(img, np.round(face * 16).astype(np.int32), (base * 0.7).tolist(), shift=4)
        cv2.fillConvexPoly(img, np.round(uv[4:] * 16).astype(np.int32), base.tolist(), shift=4)  # top face
    img = cv2.GaussianBlur(img, (3, 3), 0.7)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def score(found, truth, max_mm, max_deg) -> bool:
    ok = True
    used = set()
    for t in truth:
        cands = [(np.hypot(*np.subtract(f["xy_mm"], t["xy_mm"])), k) for k, f in enumerate(found)
                 if f["colour"] == t["colour"] and k not in used]
        if not cands or min(cands)[0] > 15:
            print(f"  MISSED  {t['name']:<10} {t['colour']:<6} at {np.round(t['xy_mm'], 1)}")
            ok = False
            continue
        d, k = min(cands)
        used.add(k)
        f = found[k]
        dyaw = abs(square_yaw(f["yaw_deg"] - t["yaw_deg"]))
        good = d <= max_mm and dyaw <= max_deg
        ok &= good
        print(f"  {'ok  ' if good else 'BAD '}    {t['name']:<10} {t['colour']:<6} pos err {d:5.2f} mm   "
              f"yaw err {dyaw:4.1f} deg   iou {f['iou']:.2f}")
    extra = [f for k, f in enumerate(found) if k not in used]
    for f in extra:
        print(f"  EXTRA   {f['colour']:<6} at {np.round(f['xy_mm'], 1)} iou {f['iou']:.2f}")
    return ok and not extra


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image")
    p.add_argument("--gt")
    p.add_argument("--size", default="1920x1080", help="resize the frame to this, as IMAGE_SIZE would")
    p.add_argument("--max-mm", type=float, default=1.0)
    p.add_argument("--max-deg", type=float, default=4.0)
    args = p.parse_args()
    size = tuple(int(v) for v in args.size.split("x"))
    ok = True
    if args.image:
        rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        gt = json.loads(Path(args.gt).read_text())
        truth = [{"name": o["name"], "colour": o["colour"], "xy_mm": np.array(o["xyz"][:2]) * 1000,
                  "yaw_deg": o["yaw_deg"]} for o in gt["objects"] if o["category"] == "cube"]
        cubes = detect_cubes(rgb)
        print(f"{args.image} at {size[0]}x{size[1]}: {len(cubes)} detections, {len(truth)} cubes")
        ok = score(cubes, truth, args.max_mm, args.max_deg)
        out = Path(args.image).with_name(Path(args.image).stem + f"_detect_{size[0]}.png")
        cv2.imwrite(str(out), draw(rgb, cubes))
        print(f"  overlay: {out}")
    else:
        rng = np.random.default_rng(1)
        for trial in range(3):
            truth = []
            for k, colour in enumerate(["yellow", "yellow", "blue", "blue", "red", "red"]):
                truth.append({"name": f"cube{k}", "colour": colour,
                              "xy_mm": np.array([rng.uniform(330, 480), -180 + 60 * k + rng.uniform(-10, 10)]),
                              "yaw_deg": float(rng.uniform(-45, 45))})
            rgb = render(truth, size, seed=trial)
            cubes = detect_cubes(rgb)
            print(f"synthetic frame {trial} at {size[0]}x{size[1]}: {len(cubes)} detections")
            ok &= score(cubes, truth, args.max_mm, args.max_deg)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
