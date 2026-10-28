"""Offline simulation for the wrist camera: ray-trace the real wrist colour and depth camera models over a table with objects,
so (A) the production colour pipeline (cube_vision.wrist_locate) and (B) a depth pipeline can be tested against known ground truth,
with no robot. Separate from the policy: nothing here is imported by policy.py.

    uv run depth_sim.py bias       # how far the colour blob's centre is from the cube's true top-face centre, by hover pose
    uv run depth_sim.py depth      # the depth pipeline (centre, height, width, yaw) under the calibration's noise, with holes
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from calibration.geometry import SURFACE_Z, depth_to_base, flange_pose, load
from cube_vision import wrist_locate

# Hover poses from the real log of 26-27 Oct (joints printed by the policy at the converged wrist view): the arm over a cube,
# R = wrist angle, tcp = fingertip x, y (mm). The cube is put exactly under the tool axis.
HOVER_POSES = [
    ((-16.4, 46.6, 11.4, 0.0, -54.8, 18.5), (419.7, -123.6)),
    ((-17.3, 45.8, 10.1, 0.0, -54.2, 22.1), (414.0, -128.6)),
    ((-15.7, 47.1, 12.3, 0.0, -55.2, 16.7), (423.4, -119.3)),
    ((16.8, 54.6, 25.6, 0.0, -61.0, -23.8), (455.0, 137.4)),
    ((-10.5, 45.9, 10.2, 0.0, -54.3, 14.9), (425.9, -79.1)),
]


@dataclass
class Box:
    """A box standing on the table: centre (m), yaw (deg), footprint lx x ly (m), height (m)."""
    x: float
    y: float
    yaw: float
    lx: float
    ly: float
    h: float
    kind: str = "box"  # "box" or "cylinder" (lying, axis along the box's x, radius = ly / 2, length lx)


def _to_box_frame(o, d, b: Box):
    c, s = np.cos(np.radians(b.yaw)), np.sin(np.radians(b.yaw))
    R = np.array([[c, s], [-s, c]])  # world -> box, about z
    o2 = np.array([o[0] - b.x, o[1] - b.y]) @ R.T
    d2 = d[:, :2] @ R.T
    return o2, d2


def trace(o: np.ndarray, dirs: np.ndarray, boxes: list[Box]) -> tuple[np.ndarray, np.ndarray]:
    """Ray origin o (3,), directions (N, 3) (unnormalised: t is the camera-axis depth when dirs[:, 2] is the camera's z share).
    Returns the hit parameter t (N,) (inf = none) and the id of what was hit (0 table, i+1 box i)."""
    t = np.full(len(dirs), np.inf)
    hit = np.zeros(len(dirs), int)
    tt = (SURFACE_Z - o[2]) / dirs[:, 2]  # the table plane
    ok = tt > 0
    t[ok] = tt[ok]
    for i, b in enumerate(boxes, 1):
        o2, d2 = _to_box_frame(o, dirs, b)
        z0, z1 = SURFACE_Z, SURFACE_Z + b.h
        if b.kind == "box":
            lo = np.array([-b.lx / 2, -b.ly / 2, z0]); hi = np.array([b.lx / 2, b.ly / 2, z1])
            oo = np.array([o2[0], o2[1], o[2]]); dd = np.column_stack([d2, dirs[:, 2]])
            with np.errstate(divide="ignore", invalid="ignore"):
                ta = (lo - oo) / dd; tb = (hi - oo) / dd
            tn = np.minimum(ta, tb).max(1); tf = np.maximum(ta, tb).min(1)
            m = (tn <= tf) & (tn > 0) & (tn < t)
            t[m] = tn[m]; hit[m] = i
        else:  # lying cylinder along x, axis at height z0 + r
            r = b.ly / 2; zc = z0 + r
            a = d2[:, 1] ** 2 + dirs[:, 2] ** 2
            bq = 2 * (o2[1] * d2[:, 1] + (o[2] - zc) * dirs[:, 2])
            cq = o2[1] ** 2 + (o[2] - zc) ** 2 - r * r
            disc = bq * bq - 4 * a * cq
            with np.errstate(invalid="ignore", divide="ignore"):
                th = (-bq - np.sqrt(disc)) / (2 * a)
            px = o2[0] + th * d2[:, 0]
            m = (disc > 0) & (th > 0) & (th < t) & (np.abs(px) <= b.lx / 2)
            t[m] = th[m]; hit[m] = i
    return t, hit


def _camera_dirs_depth(cam, step: int = 1):
    v, u = np.mgrid[0:cam["size"][1]:step, 0:cam["size"][0]:step]
    return np.column_stack([((u - cam["cx"]) / cam["fx"]).ravel(), ((v - cam["cy"]) / cam["fy"]).ravel(), np.ones(u.size)]), u.shape


def render_depth(cams, joints, boxes, rng, sigma_mm=3.0, hole_table=0.0, hole_obj=0.0):
    """The wrist depth image (uint16 mm, 0 = none) as the real camera would deliver it, with Gaussian noise (sigma_mm: the
    calibration's rms_plane_mm is 3.0) and random holes (the white table has no texture, silver objects reflect)."""
    cam = cams["wrist_depth"]
    T = flange_pose(joints) @ cam["T"]
    dirs_cam, shape = _camera_dirs_depth(cam)
    dirs = dirs_cam @ T[:3, :3].T
    t, hit = trace(T[:3, 3], dirs, boxes)
    depth = t * 1000.0
    depth[~np.isfinite(depth)] = 0
    depth = depth + rng.normal(0, sigma_mm, depth.shape) * (depth > 0)
    drop = np.where(hit == 0, rng.random(depth.shape) < hole_table, rng.random(depth.shape) < hole_obj)
    depth[drop] = 0
    return np.clip(depth, 0, 65535).astype(np.uint16).reshape(shape)


def render_colour(cams, joints, boxes, step: int = 1):
    """The wrist colour image: white table, blue objects, from the colour camera's model (pinhole + k1, as wrist_locate undoes it)."""
    w = cams["wrist"]
    T = flange_pose(joints) @ w["T"]
    v, u = np.mgrid[0:w["size"][1]:step, 0:w["size"][0]:step]
    xd, yd = (u - w["size"][0] / 2) / w["fx"], (v - w["size"][1] / 2) / w["fx"]
    x, y = xd.copy(), yd.copy()
    for _ in range(10):  # the same fixed-point undistortion wrist_locate uses
        d = 1 + w["k1"] * (x * x + y * y)
        x, y = xd / d, yd / d
    dirs = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)]) @ T[:3, :3].T
    _, hit = trace(T[:3, 3], dirs, boxes)
    img = np.full(hit.shape + (3,), 215, np.uint8)
    img[hit > 0] = (30, 80, 200)  # RGB blue, inside colour_mask("blue")
    return img.reshape(u.shape + (3,))


# ---- the depth pipeline under test ------------------------------------------------------------------------------------------

def locate_from_depth(depth, joints, cams, guess_xy_mm, radius_mm=45.0, min_height_mm=3.5):
    """The object nearest guess_xy_mm in a wrist depth image: top-face centre, height, footprint (long, short), long-axis yaw.
    The table plane is fitted from the depth itself (robust), so a few mm of calibration error in the surface do not matter."""
    pts, _ = depth_to_base(cams["wrist_depth"], depth, stride=2, with_pixels=True, joints=joints)
    pts = pts * 1000.0
    if len(pts) < 500:
        return None
    gx, gy = guess_xy_mm
    far = np.hypot(pts[:, 0] - gx, pts[:, 1] - gy) > radius_mm
    plane = np.array([0.0, 0.0, SURFACE_Z * 1000.0])  # z = a x + b y + c, started at the calibrated surface
    sel = far & (np.abs(pts[:, 2] - plane[2]) < 15)
    for _ in range(4):  # iterate a clipped least-squares plane over the table points
        if sel.sum() < 200:
            break
        A = np.column_stack([pts[sel, 0], pts[sel, 1], np.ones(sel.sum())])
        plane = np.linalg.lstsq(A, pts[sel, 2], rcond=None)[0]
        res = pts[:, 2] - np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))]) @ plane
        sel = far & (np.abs(res) < 6)
    height = pts[:, 2] - np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))]) @ plane
    near = np.hypot(pts[:, 0] - gx, pts[:, 1] - gy) < radius_mm
    sigma = 1.4826 * float(np.median(np.abs(height[far & (np.abs(height) < 15)]))) if (far & (np.abs(height) < 15)).sum() > 200 else 3.0
    obj = near & (height > max(min_height_mm, 2.5 * sigma)) & (height < 80)  # the threshold follows the measured noise
    if obj.sum() < 15:
        return None
    # the object is the cluster of cells (1 mm bins) nearest the guess
    cell = np.floor(pts[obj, :2]).astype(int)
    keys = {tuple(c): i for i, c in enumerate(cell)}
    from scipy import ndimage  # noqa: PLC0415  (only the experiments need it)
    lo = cell.min(0); grid = np.zeros(tuple(cell.max(0) - lo + 3), bool)
    grid[tuple((cell - lo + 1).T)] = True
    grid_c = ndimage.binary_opening(ndimage.binary_closing(grid, iterations=2), iterations=1)  # drop isolated noise cells
    lab, n = ndimage.label(grid_c)
    if n == 0:
        return None
    centres = ndimage.center_of_mass(grid, lab, range(1, n + 1))
    best = min(range(n), key=lambda i: np.hypot(centres[i][0] + lo[0] - 1 - gx, centres[i][1] + lo[1] - 1 - gy))
    inlab = lab[tuple((cell - lo + 1).T)] == best + 1
    if inlab.sum() < 15:
        return None
    P = pts[obj][inlab]; H = height[obj][inlab]
    top_h = float(np.percentile(H, 90))
    top = H >= top_h - 2.5
    if top.sum() < 8:
        return None
    centre = np.median(P[top][:, :2], axis=0)
    import cv2  # noqa: PLC0415
    (_, _), (rw, rh), ang = cv2.minAreaRect(P[:, :2].astype(np.float32))
    if rw < rh:
        ang += 90.0; rw, rh = rh, rw
    return {"xy_mm": (float(centre[0]), float(centre[1])), "height_mm": top_h, "long_mm": float(rw), "short_mm": float(rh),
            "yaw_deg": float((ang + 90.0) % 180.0 - 90.0), "n_points": int(len(P))}


# ---- experiments ------------------------------------------------------------------------------------------------------------

def experiment_bias(cams):
    """A cube exactly under the tool axis, seen by the colour camera: where does the production wrist_locate put it?"""
    print("COLOUR BLOB vs TRUE TOP-FACE CENTRE (cube centred on the tool axis; error = estimate - truth, mm)")
    print("  hover pose (tcp x, y)    R     10 mm cube: dx, dy      14.5 mm cylinder: dx, dy      (blob estimate; wrist_locate assumes a 10 mm top plane)")
    for joints, (x, y) in HOVER_POSES:
        out = []
        for b in (Box(x / 1000, y / 1000, 0.0, 0.010, 0.010, 0.010), Box(x / 1000, y / 1000, 0.0, 0.050, 0.0145, 0.0145, "cylinder")):
            img = render_colour(cams, joints, [b])
            loc = wrist_locate(img, list(joints), "blue", (x, y), cams, edge=b.h, max_shift_mm=40)
            out.append("none" if loc is None else f"{loc['xy_mm'][0] - x:+5.1f}, {loc['xy_mm'][1] - y:+5.1f}")
        print(f"  ({x:6.1f}, {y:7.1f})   {joints[5] - 0:+6.1f}      {out[0]:16s}      {out[1]}")


def experiment_depth(cams, trials=12, sigma=3.0, hole_table=0.0, hole_obj=0.0, seed=0):
    rng = np.random.default_rng(seed)
    print(f"DEPTH PIPELINE (noise {sigma} mm, table holes {hole_table:.0%}, object holes {hole_obj:.0%}); error = estimate - truth")
    print("  object                    found   centre err (mm)       height err   long err   short err   yaw err (deg)")
    objs = {"10 mm cube": lambda yaw, x, y: Box(x, y, yaw, 0.010, 0.010, 0.010),
            "AA battery 50x14.5": lambda yaw, x, y: Box(x, y, yaw, 0.050, 0.0145, 0.0145, "cylinder"),
            "bolt-like 40x10x8": lambda yaw, x, y: Box(x, y, yaw, 0.040, 0.010, 0.008)}
    for name, mk in objs.items():
        rows = []
        for k in range(trials):
            joints, (x, y) = HOVER_POSES[k % len(HOVER_POSES)]
            yaw = float(rng.uniform(-80, 80))
            b = mk(yaw, x / 1000, y / 1000)
            depth = render_depth(cams, joints, [b], rng, sigma, hole_table, hole_obj)
            r = locate_from_depth(depth, list(joints), cams, (x, y))
            if r is None:
                rows.append(None); continue
            dyaw = (r["yaw_deg"] - ((yaw + 90) % 180 - 90) + 90) % 180 - 90 if b.lx != b.ly else float("nan")
            rows.append((np.hypot(r["xy_mm"][0] - x, r["xy_mm"][1] - y), r["height_mm"] - b.h * 1000,
                         r["long_mm"] - b.lx * 1000, r["short_mm"] - b.ly * 1000, dyaw))
        ok = [r for r in rows if r]
        if not ok:
            print(f"  {name:24s} 0/{trials}"); continue
        a = np.array(ok)
        print(f"  {name:24s} {len(ok):2d}/{trials}   {a[:,0].mean():4.1f} (max {a[:,0].max():4.1f})        {a[:,1].mean():+5.1f}       {a[:,2].mean():+5.1f}      {a[:,3].mean():+5.1f}      "
              + (f"{np.nanmean(np.abs(a[:,4])):4.1f}" if not np.all(np.isnan(a[:,4])) else "  -"))


if __name__ == "__main__":
    cams = load()
    what = sys.argv[1] if len(sys.argv) > 1 else "bias"
    if what == "bias":
        experiment_bias(cams)
    else:
        experiment_depth(cams)
        experiment_depth(cams, hole_table=0.5, hole_obj=0.3)
        experiment_depth(cams, sigma=1.0)
