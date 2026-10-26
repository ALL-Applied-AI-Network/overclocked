"""Find small coloured cubes on the work surface with the table camera (numpy + OpenCV only).

    from cube_vision import detect_cubes
    cubes = detect_cubes(obs["images"]["table"])   # [{"colour", "xy_mm", "yaw_deg", "iou", "px"}, ...]

Pipeline: HSV colour gate -> connected components -> keep blobs whose area matches a cube's projected size at that
spot -> fit the cube's projected 3D silhouette (x, y, yaw) to the blob. The fit models the side faces the camera
sees, so the result is the cube's centre, not the blob centroid (which leans toward the camera).

Positions are x, y in mm in UF0 (= the base frame's x, y), the frame of obs["tcp_pose"] and placement.yaml.
Yaw is in degrees in (-45, 45]: a square looks the same every 90 degrees.
"""
from __future__ import annotations

import cv2
import numpy as np

from calibration.geometry import SURFACE_Z, load, pixel_to_base, project

# Hue (OpenCV 0-179), minimum saturation and value. Starting points from the v0.5 twin; retune on real frames.
COLOUR_GATES = {
    "yellow": {"hue": [(14, 42)], "s": 65, "v": 45},
    "blue": {"hue": [(95, 135)], "s": 55, "v": 35},
    "red": {"hue": [(0, 10), (170, 179)], "s": 70, "v": 40},
}
CUBE_EDGE_M = 0.010
REACH_MM = 540.0  # stay inside the ~550 mm reach the cell enforces
TCP_BOX_XY = ((150.0, 650.0), (-450.0, 450.0))


def square_yaw(deg: float) -> float:
    return (float(deg) + 45.0) % 90.0 - 45.0


def colour_mask(hsv: np.ndarray, colour: str) -> np.ndarray:
    gate = COLOUR_GATES[colour]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue = np.zeros(h.shape, bool)
    for lo, hi in gate["hue"]:
        hue |= (h >= lo) & (h <= hi)
    return (hue & (s >= gate["s"]) & (v >= gate["v"])).astype(np.uint8)


def cube_corners(xy_m, yaw_deg: float, edge: float = CUBE_EDGE_M, surface: float = SURFACE_Z) -> np.ndarray:
    """The cube's 8 corners in the base frame (m), resting on the surface."""
    t = np.radians(yaw_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * edge / 2 @ rot.T + np.asarray(xy_m)
    return np.array([[x, y, surface + z] for z in (0.0, edge) for x, y in sq])


def silhouette_px(cam: dict, xy_m, yaw_deg: float, scale: float, edge: float = CUBE_EDGE_M) -> np.ndarray:
    """The cube's projected outline (convex hull) in image pixels; scale = native width / image width."""
    uv = project(cam, cube_corners(xy_m, yaw_deg, edge)) / scale
    return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)


def _fill(hull: np.ndarray, shape, origin) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    pts = np.round((hull - origin) * 16).astype(np.int32)  # 4 bits of sub-pixel precision
    cv2.fillConvexPoly(m, pts, 1, lineType=cv2.LINE_8, shift=4)
    return m


def top_face_px(cam: dict, xy_m, yaw_deg: float, scale: float, edge: float = CUBE_EDGE_M) -> np.ndarray:
    """The cube's top face, projected, in image pixels."""
    return project(cam, cube_corners(xy_m, yaw_deg, edge)[4:]) / scale


def bright_part(blob: np.ndarray, value: np.ndarray) -> np.ndarray | None:
    """The lit top face inside a blob: its pixels brighter than Otsu's split of the blob's V channel."""
    vals = value[blob > 0]
    if vals.size < 8 or np.ptp(vals) < 12:
        return None  # no usable contrast between the top and the sides
    t, _ = cv2.threshold(vals.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((value > t) & (blob > 0)).astype(np.uint8)


TOP_WEIGHT = 1.0  # how much the top face counts against the whole silhouette in the fit


def fit_pose(observed: np.ndarray, origin, cam: dict, xy0_m, scale: float, edge: float = CUBE_EDGE_M,
             top: np.ndarray | None = None):
    """Grid search (x, y, yaw) maximising silhouette IoU with the observed blob (plus, with top, the top face's
    IoU with the blob's bright part, which carries most of the yaw); coarse then fine. Returns the silhouette IoU."""
    obs = observed.astype(bool)
    top = None if top is None else top.astype(bool)

    def _iou(pred, ref):
        union = np.logical_or(pred, ref).sum()
        return np.logical_and(pred, ref).sum() / union if union else 0.0

    def sil(xy, yaw):
        return _iou(_fill(silhouette_px(cam, xy, yaw, scale, edge), obs.shape, origin).astype(bool), obs)

    def iou(xy, yaw):
        s = sil(xy, yaw)
        if top is not None:
            s += TOP_WEIGHT * _iou(_fill(top_face_px(cam, xy, yaw, scale, edge), obs.shape, origin).astype(bool), top)
        return s

    best = (-1.0, np.asarray(xy0_m, float), 0.0)
    offs = np.arange(-0.003, 0.003 + 1e-9, 0.001)  # coarse: +-3 mm, every 5 deg of the 90 deg period
    for dx in offs:
        for dy in offs:
            xy = np.asarray(xy0_m) + [dx, dy]
            for yaw in np.arange(-45, 45, 5.0):
                s = iou(xy, yaw)
                if s > best[0]:
                    best = (s, xy, yaw)
    # refine around the coarse optimum
    for span, step, yspan, ystep in ((0.0012, 0.0004, 4.0, 1.0), (0.0004, 0.0001, 1.0, 0.25)):
        s0, xy0, yaw0 = best
        offs = np.arange(-span, span + 1e-9, step)
        for dx in offs:
            for dy in offs:
                for yaw in yaw0 + np.arange(-yspan, yspan + 1e-9, ystep):
                    s = iou(xy0 + [dx, dy], yaw)
                    if s > best[0]:
                        best = (s, xy0 + [dx, dy], yaw)
    return sil(best[1], best[2]), best[1], best[2]


def detect_cubes(rgb: np.ndarray, colours=("yellow", "blue", "red"), cams: dict | None = None,
                 edge: float = CUBE_EDGE_M, min_iou: float = 0.55) -> list[dict]:
    """All cubes of the given colours in a table-camera RGB image (any size; the calibration is 1920x1080)."""
    cams = cams or load()
    cam = cams["table"]
    H, W = rgb.shape[:2]
    scale = cam["size"][0] / W
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    found = []
    for colour in colours:
        mask = colour_mask(hsv, colour)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 4:
                continue
            u, v = cents[i] * scale
            xy = pixel_to_base(cam, (u, v), z=SURFACE_Z + edge / 2)[:2]
            x_mm, y_mm = xy * 1000
            if not (TCP_BOX_XY[0][0] <= x_mm <= TCP_BOX_XY[0][1] and TCP_BOX_XY[1][0] <= y_mm <= TCP_BOX_XY[1][1]
                    and np.hypot(x_mm, y_mm) <= REACH_MM):
                continue
            hull = silhouette_px(cam, xy, 0.0, scale, edge)
            expected = cv2.contourArea(hull.astype(np.float32))
            if not 0.35 * expected <= area <= 2.5 * expected:
                continue  # a bin, the tape, a screw, glare: not one cube
            x0, y0, w, h = stats[i, :4]
            pad = int(np.ceil(6 / scale)) + 3
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(W, stats[i, 0] + w + pad), min(H, stats[i, 1] + h + pad)
            blob = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
            top = bright_part(blob, hsv[y0:y1, x0:x1, 2])
            score, fxy, fyaw = fit_pose(blob, np.array([x0, y0], float), cam, xy, scale, edge, top)
            if score < min_iou:
                continue
            found.append({"colour": colour, "xy_mm": (float(fxy[0] * 1000), float(fxy[1] * 1000)),
                          "yaw_deg": square_yaw(fyaw), "iou": float(score), "px": (float(u), float(v)),
                          "area_px": int(area)})
    return sorted(found, key=lambda c: -c["iou"])


def draw(rgb: np.ndarray, cubes: list[dict], cams: dict | None = None, edge: float = CUBE_EDGE_M) -> np.ndarray:
    """A BGR copy of the image with each detection's fitted outline and label, for debugging."""
    cams = cams or load()
    cam = cams["table"]
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    scale = cam["size"][0] / rgb.shape[1]
    for k, c in enumerate(cubes):
        hull = silhouette_px(cam, np.array(c["xy_mm"]) / 1000, c["yaw_deg"], scale, edge)
        cv2.polylines(out, [np.round(hull).astype(np.int32)], True, (0, 255, 0), 1)
        u, v = np.array(c["px"]) / scale
        cv2.putText(out, f"{k}:{c['colour'][0]} {c['yaw_deg']:+.0f} {c['iou']:.2f}", (int(u) + 8, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return out
