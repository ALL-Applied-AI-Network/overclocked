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

from calibration.geometry import SURFACE_Z, flange_pose, load, pixel_to_base, project

# Hue (OpenCV 0-179), minimum saturation and value. Starting points from the v0.5 twin; retune on real frames.
COLOUR_GATES = {
    "yellow": {"hue": [(14, 42)], "s": 65, "v": 45},
    "blue": {"hue": [(95, 135)], "s": 55, "v": 35},
    "red": {"hue": [(0, 10), (170, 179)], "s": 70, "v": 40},
}
# Minimum silhouette fit to accept a blob. Yellow is strict: the gripper's brass connectors pass the yellow gate.
# Blue and red have no such look-alike on the cell, and a real blue cube on the table fitted only 0.48 at 1920x1080.
# False = what ran on 26 Oct, when the open-vocabulary detector grasped cubes (the saturated-or-dark mask only). True adds the
# shadow-aware object_mask, built later for grey objects such as batteries (policy_battery_experiments.py).
USE_OBJECT_MASK = False
ANY_SAT, ANY_DARK = 55, 130  # colour_mask("any"): saturation or darkness that the white table never reaches
# Measured from the table camera a 10 mm cube is 12.6-17 mm across (its side faces and shadow count): median 15.4, aspect <= 1.46.
ELONGATED_ASPECT = 1.6   # long/short side from which an object has one right way round for the jaws
MAX_GRASP_SHORT_MM = 26.0  # a measured short side wider than this will not fit the ~12 mm jaws. 27 real cubes measured 12-19.7 mm
# (median 15.8) with object_mask at dist 30. Lying AA batteries (14.5 mm) measured 20-24 mm in table_blobs and the fingers went
# below the top of one without a fault (27 Oct), so the limit is 26, not the 22 that would have refused half of them
MIN_IOU = {"yellow": 0.75, "blue": 0.40, "red": 0.40}
BIN_MID_M = 0.025  # half the bin's wall height: where the yellow blob's centroid is
CUBE_EDGE_M = 0.010
REACH_MM = 540.0  # stay inside the ~550 mm reach the cell enforces
TCP_BOX_XY = ((150.0, 650.0), (-450.0, 450.0))


def square_yaw(deg: float) -> float:
    return (float(deg) + 45.0) % 90.0 - 45.0


def colour_mask(hsv: np.ndarray, colour: str) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if colour == "any":  # whatever is not the white table: saturated or dark
        return ((s >= ANY_SAT) | (v < ANY_DARK)).astype(np.uint8)
    gate = COLOUR_GATES[colour]
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
                 edge: float = CUBE_EDGE_M, min_iou: float | None = None) -> list[dict]:
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
            if score < (min_iou if min_iou is not None else MIN_IOU[colour]):
                continue
            found.append({"colour": colour, "xy_mm": (float(fxy[0] * 1000), float(fxy[1] * 1000)),
                          "yaw_deg": square_yaw(fyaw), "iou": float(score), "px": (float(u), float(v)),
                          "area_px": int(area)})
    return sorted(found, key=lambda c: -c["iou"])


def detect_bin(rgb: np.ndarray, cams: dict | None = None) -> tuple[float, float] | None:
    """The yellow bin's (x, y) mm on the table in a table-camera RGB image, or None.

    The bin is a near-square yellow blob of ~5000-7000 px at 1920x1080, filling ~85% of its rotated bounding box. The arm, its base and the table-edge
    tape are also yellow but far larger or mostly empty (15-55% filled), and the cubes are far smaller.
    """
    cams = cams or load()
    cam = cams["table"]
    H, W = rgb.shape[:2]
    scale = cam["size"][0] / W
    mask = colour_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), "yellow")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    for i in range(1, n):
        x0, y0, w, h, area = stats[i]
        area_full = area * scale ** 2
        if not (3500 <= area_full <= 12000):
            continue
        pts = cv2.findNonZero((labels[y0:y0 + h, x0:x0 + w] == i).astype(np.uint8)) + np.array([x0, y0])
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)  # the rotated box: fill does not depend on how the bin is turned
        if rw < 1 or rh < 1 or area / (rw * rh) < 0.6 or not 0.7 <= rw / rh <= 1.4:
            continue
        xy = pixel_to_base(cam, tuple(cents[i] * scale), z=SURFACE_Z + BIN_MID_M)[:2] * 1000  # the blob's centroid is mid-wall
        if not (TCP_BOX_XY[0][0] <= xy[0] <= TCP_BOX_XY[0][1] and TCP_BOX_XY[1][0] <= xy[1] <= TCP_BOX_XY[1][1]
                and np.hypot(*xy) <= REACH_MM + 10.0):  # the drop point may sit at the cell's ~550 mm limit
            continue
        if best is None or area > best[0]:
            best = (area, (float(xy[0]), float(xy[1])))
    return best[1] if best else None


def object_mask(rgb: np.ndarray, box, ring: int = 14, dist: float = 30.0) -> np.ndarray:
    """What differs from the table inside box (xa, ya, xb, yb): pixels whose Lab colour is `dist` away from the table's
    colour, taken as the median of a ring just outside the box. Catches grey and silver objects that the saturated-or-dark
    mask of colour_mask("any") misses."""
    xa, ya, xb, yb = box
    H, W = rgb.shape[:2]
    X0, Y0, X1, Y1 = max(0, xa - ring), max(0, ya - ring), min(W, xb + ring), min(H, yb + ring)
    lab = cv2.cvtColor(rgb[Y0:Y1, X0:X1], cv2.COLOR_RGB2LAB).astype(np.float32)
    inner = np.zeros(lab.shape[:2], bool)
    inner[ya - Y0:yb - Y0, xa - X0:xb - X0] = True
    table = np.median(lab[~inner], axis=0) if (~inner).any() else np.array([200.0, 128.0, 128.0])
    d = np.linalg.norm(lab - table, axis=2)[ya - Y0:yb - Y0, xa - X0:xb - X0]
    return (d > dist).astype(np.uint8)


def table_blobs(rgb: np.ndarray, cams: dict | None = None, dist: float = 24.0, height_mm: float = 8.0) -> list[dict]:
    """Everything on the table that is not the table, whatever it is: blobs whose Lab colour differs from the local table
    colour (a big median blur at quarter resolution, so uneven light does not matter). Each is returned like
    objects_from_detections' targets, with colour "any" and a score of 0; only blobs a gripper could plausibly hold
    (a few mm to ~40 mm) inside the arm's reach come back."""
    cams = cams or load()
    cam = cams["table"]
    H, W = rgb.shape[:2]
    scale = cam["size"][0] / W
    q = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(q, 31)  # 31 px at quarter size ~ 120 px: wider than any object, so the blur is just the table
    diff = np.linalg.norm(q.astype(np.float32) - bg.astype(np.float32), axis=2)
    diff_full = cv2.resize(diff, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = (diff_full > dist).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    z = SURFACE_Z + height_mm / 1000
    out = []
    for i in range(1, n):
        x0, y0, w, h, area = stats[i]
        if area < 30:
            continue
        c = cents[i] * scale
        a = pixel_to_base(cam, tuple(c), z=z)[:2] * 1000
        b = pixel_to_base(cam, tuple(c + [10, 0]), z=z)[:2] * 1000
        mm_px = np.hypot(*(b - a)) / 10
        pts = np.column_stack(np.nonzero(labels[y0:y0 + h, x0:x0 + w] == i))[:, ::-1].astype(np.float32)
        (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
        if rw < rh:
            ang, rw, rh = ang + 90.0, rh, rw
        long_mm, short_mm = float(rw * mm_px * scale), float(rh * mm_px * scale)
        if not (4 <= short_mm <= 30 and long_mm <= 70):  # only the short side has to fit the jaws; the bin is dropped by position
            continue
        xy = a
        if not (TCP_BOX_XY[0][0] <= xy[0] <= TCP_BOX_XY[0][1] and TCP_BOX_XY[1][0] <= xy[1] <= TCP_BOX_XY[1][1]
                and np.hypot(*xy) <= REACH_MM):
            continue
        t = np.radians(ang)
        p1 = pixel_to_base(cam, tuple(c + 20 * np.array([np.cos(t), np.sin(t)])), z=z)[:2] * 1000
        axis = (np.degrees(np.arctan2(p1[1] - a[1], p1[0] - a[0])) + 90.0) % 180.0 - 90.0
        elongated = long_mm / max(short_mm, 1e-6) >= ELONGATED_ASPECT
        contrast = float(diff_full[y0:y0 + h, x0:x0 + w][labels[y0:y0 + h, x0:x0 + w] == i].mean())  # a pencil mark is faint
        out.append({"colour": "any", "label": "table blob", "contrast": contrast, "xy_mm": (float(xy[0]), float(xy[1])),
                    "yaw_deg": float(axis if elongated else square_yaw(axis)), "iou": 0.0, "px": (float(c[0]), float(c[1])),
                    "size_mm": (long_mm, short_mm), "long_mm": long_mm, "short_mm": short_mm, "elongated": bool(elongated),
                    "area_px": int(area)})
    return out


def blob_colour(rgb: np.ndarray, blob: dict, half: int = 24, share: float = 0.4) -> str:
    """The colour gate ("blue", "red", "yellow") that most of a table blob's pixels fall in, else "any". A coloured object
    then goes through the colour-masked wrist refinement, which centres it ~1.5 mm better than the generic mask (the
    generic mask counts shadow); only an object of no known colour uses the generic one."""
    u, v = int(blob["px"][0]), int(blob["px"][1])
    H, W = rgb.shape[:2]
    crop = cv2.cvtColor(rgb[max(0, v - half):min(H, v + half), max(0, u - half):min(W, u + half)], cv2.COLOR_RGB2HSV)
    best = max(COLOUR_GATES, key=lambda c: int(colour_mask(crop, c).sum()))
    return best if colour_mask(crop, best).sum() >= share * blob["area_px"] else "any"


def objects_from_detections(rgb: np.ndarray, dets: list[dict], cams: dict | None = None, height_mm: float = 8.0,
                            max_side_mm: float = 40.0) -> list[dict]:
    """Grasp targets from open-vocabulary boxes ({"label", "score", "box"} in the image's pixels).

    The box gives the object; the pixels inside it that are not table give its centre and yaw. Boxes that are not
    small enough to be graspable, or that lie outside the arm's reach, are dropped.
    """
    cams = cams or load()
    cam = cams["table"]
    H, W = rgb.shape[:2]
    scale = cam["size"][0] / W
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask_all = colour_mask(hsv, "any")
    z = SURFACE_Z + height_mm / 1000
    out = []
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) * scale
        a = pixel_to_base(cam, tuple(c), z=z)[:2] * 1000
        b = pixel_to_base(cam, tuple(c + [10, 0]), z=z)[:2] * 1000
        mm_px = np.hypot(*(b - a)) / 10
        w_mm, h_mm = (x1 - x0) * scale * mm_px, (y1 - y0) * scale * mm_px
        if max(w_mm, h_mm) > max_side_mm or min(w_mm, h_mm) < 4:
            continue
        pad = 4
        xa, ya, xb, yb = max(0, int(x0) - pad), max(0, int(y0) - pad), min(W, int(x1) + pad), min(H, int(y1) + pad)
        obj = (object_mask(rgb, (xa, ya, xb, yb)) | mask_all[ya:yb, xa:xb]) if USE_OBJECT_MASK else mask_all[ya:yb, xa:xb]
        n, labels, stats, cents = cv2.connectedComponentsWithStats(obj, connectivity=8)
        yaw, long_mm, short_mm, elongated = 0.0, max(w_mm, h_mm), min(w_mm, h_mm), False
        if n > 1:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[i, cv2.CC_STAT_AREA] >= 20:  # a real blob: its centre and its rectangle's angle
                c = (cents[i] + [xa, ya]) * scale
                pts = np.column_stack(np.nonzero(labels == i))[:, ::-1].astype(np.float32)
                (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
                if rw < rh:  # OpenCV's angle is that of the "width" side: make it the long side
                    ang += 90.0
                    rw, rh = rh, rw
                t = np.radians(ang)
                p0 = pixel_to_base(cam, tuple(c), z=z)[:2]
                p1 = pixel_to_base(cam, tuple(c + 20 * np.array([np.cos(t), np.sin(t)])), z=z)[:2]
                axis = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))  # the long side, in the table frame
                axis = (axis + 90.0) % 180.0 - 90.0
                long_mm, short_mm = float(rw * mm_px * scale), float(rh * mm_px * scale)
                aspect = long_mm / max(short_mm, 1e-6)
                elongated = aspect >= ELONGATED_ASPECT
                yaw = axis if elongated else square_yaw(axis)  # a long object has one right way round, a square has four
        xy = pixel_to_base(cam, tuple(c), z=z)[:2] * 1000
        if not (TCP_BOX_XY[0][0] <= xy[0] <= TCP_BOX_XY[0][1] and TCP_BOX_XY[1][0] <= xy[1] <= TCP_BOX_XY[1][1]
                and np.hypot(*xy) <= REACH_MM):
            continue
        out.append({"colour": "any", "label": d["label"], "xy_mm": (float(xy[0]), float(xy[1])), "yaw_deg": float(yaw),
                    "iou": float(d["score"]), "px": (float(c[0]), float(c[1])), "size_mm": (float(w_mm), float(h_mm)),
                    "long_mm": float(long_mm), "short_mm": float(short_mm), "elongated": bool(elongated)})
    return sorted(out, key=lambda o: -o["iou"])


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


# WRIST_SHIFT="2.3,13.4": the wrist camera's calibration error on 27 Oct, in the flange's own frame (mm: along, across the jaws). Two independent fits
# of the wrist-vs-table gap over that day's picks gave (+2.3, +13.4) and (+1.8, +12.7); a constant in the robot's frame fitted worse (6.7 mm
# against 3.7 mm), and the gap turns with the wrist angle. wrist_locate subtracts it, turned by the wrist angle R = J1 + J6. Off unless set.
_ws = [float(v) for v in __import__("os").environ.get("WRIST_SHIFT", "").replace(",", " ").split()]
WRIST_SHIFT = tuple(_ws[:2]) if len(_ws) >= 2 else None


def wrist_r_deg(joints) -> float:
    """The tool's rotation R (deg, what obs["tcp_pose"] reports) from the joints: R = J1 + J6 on this cell."""
    return float(joints[0]) + float(joints[5])


def wrist_locate(rgb: np.ndarray, joints, colour: str, guess_xy_mm, cams: dict | None = None,
                 edge: float = CUBE_EDGE_M, max_shift_mm: float = 25.0) -> dict | None:
    """The cube seen by the wrist camera while hovering over it: x, y (mm, UF0) of its top face and its yaw.
    joints: obs["joints"] from the same observation. None if no plausible blob of that colour is near the guess."""
    cams = cams or load()
    w = cams["wrist"]
    cam = dict(w)
    cam["T"] = flange_pose(joints) @ w["T"]
    H, W = rgb.shape[:2]
    scale = w["size"][0] / W
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = colour_mask(hsv, colour)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    top_z = SURFACE_Z + edge

    def to_table(uv):
        xd, yd = (np.asarray(uv, float) * scale - np.array(w["size"]) / 2) / w["fx"]
        x, y = xd, yd
        for _ in range(10):
            d = 1 + w["k1"] * (x * x + y * y)
            x, y = xd / d, yd / d
        ray = cam["T"][:3, :3] @ np.array([x, y, 1.0])
        o = cam["T"][:3, 3]
        return (o + ray * (top_z - o[2]) / ray[2])[:2] * 1000

    best = None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 50:
            continue
        xy = to_table(cents[i])
        shift = float(np.hypot(*(xy - np.asarray(guess_xy_mm))))
        if shift > max_shift_mm:
            continue
        if best is None or shift < best[0]:
            best = (shift, i, xy)
    if best is None:
        return None
    _, i, xy = best
    pts = np.column_stack(np.nonzero(labels == i))[:, ::-1].astype(np.float32)
    (cx, cy), (bw, bh), ang = cv2.minAreaRect(pts)
    t = np.radians(ang)
    a = to_table((cx, cy))
    b = to_table((cx + 20 * np.cos(t), cy + 20 * np.sin(t)))  # along one rectangle side, mapped onto the table
    yaw = square_yaw(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
    side_mm = np.hypot(*(to_table((cx + bw / 2, cy)) - to_table((cx - bw / 2, cy))))
    if WRIST_SHIFT is not None:  # take out the camera's own shift, which turns with the wrist
        r = np.radians(wrist_r_deg(joints))
        xy = xy - np.array([WRIST_SHIFT[0] * np.cos(r) - WRIST_SHIFT[1] * np.sin(r), WRIST_SHIFT[0] * np.sin(r) + WRIST_SHIFT[1] * np.cos(r)])
    return {"xy_mm": (float(xy[0]), float(xy[1])), "yaw_deg": float(yaw), "area_px": int(stats[i, 4]),
            "side_mm": float(side_mm)}
