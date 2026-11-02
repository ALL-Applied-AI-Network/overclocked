"""Self-calibration of the table camera from our own trials. Offline: the policy never fits anything while the arm moves.

What it learns: the table camera's position error, as a linear function of where the cube is. The target is where the wrist camera then sees the cube
(first view, taken right above the table estimate). Every logged cube is one example, so it needs no extra trials and no labels.

    uv run python calib_learn.py fit        # leave-one-out on all pairs; promote the candidate only if it beats the champion
    uv run python calib_learn.py report     # the numbers: first-view error with the model against what it would have been without it
    uv run python calib_learn.py check      # roll back automatically if the model made the recent first views worse
    uv run python calib_learn.py rollback   # restore the previous champion

It writes only calibration_model.json (+ .prev.json, calibration_candidate.json). Motion, limits and gates are not touched.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "verifier_data" / "calib_pairs.jsonl"
DESCENTS = HERE / "verifier_data" / "descents.jsonl"
MODEL = HERE / "calibration_model.json"
PREV = HERE / "calibration_model.prev.json"
CAND = HERE / "calibration_candidate.json"
HISTORY_LOG = HERE / "logs" / "slot_table_loop.log"  # 30 Oct colour-cube trials, before the logger existed (positions parsed from the text log)
RIDGE = 1.0
MIN_PAIRS = 12        # fewer than this: no model
MARGIN_MM = 0.5       # a candidate must beat the champion by this on leave-one-out
CHECK_LAST = 6        # `check` looks at this many most recent applied first views
CHECK_WORSE_MM = 1.0


def feats(xy: np.ndarray) -> np.ndarray:
    xy = np.atleast_2d(xy)
    return np.c_[np.ones(len(xy)), xy[:, 0] / 100.0, xy[:, 1] / 100.0]


def fit_w(xy: np.ndarray, d: np.ndarray) -> np.ndarray:
    F = feats(xy)
    pen = RIDGE * np.diag([0.0, 1.0, 1.0])  # the constant is free, the slopes are shrunk a little
    return np.linalg.solve(F.T @ F + pen, F.T @ d)


def loo_errors(xy: np.ndarray, d: np.ndarray, kind: str) -> np.ndarray:
    out = []
    for i in range(len(xy)):
        m = np.arange(len(xy)) != i
        if kind == "zero":
            p = np.zeros(2)
        elif kind == "const":
            p = d[m].mean(axis=0)
        else:
            p = (feats(xy[i]) @ fit_w(xy[m], d[m]))[0]
        out.append(np.linalg.norm(d[i] - p))
    return np.array(out)


def history_pairs() -> list[dict]:
    """Rows from the old text log: 'pick blue at (x, y)' then the first 'wrist view: cube at (x, y)'."""
    if not HISTORY_LOG.exists():
        return []
    pick = re.compile(r"\[policy\] pick (blue|yellow|red) at \(([-\d.]+), ([-\d.]+)\)")
    view = re.compile(r"wrist view: cube at \(([-\d.]+), ([-\d.]+)\)")
    rows, cur = [], None
    for line in HISTORY_LOG.read_text(errors="ignore").splitlines():
        m = pick.search(line)
        if m:
            cur = (float(m[2]), float(m[3]))
            continue
        m = view.search(line)
        if m and cur:
            rows.append({"pre_xy": list(cur), "corr": [0.0, 0.0], "wrist_xy": [float(m[1]), float(m[2])], "applied": False, "source": "log"})
            cur = None
    return rows


def load_pairs(history: bool = True) -> list[dict]:
    rows = []
    if PAIRS.exists():
        rows = [json.loads(line) for line in PAIRS.read_text().splitlines() if line.strip()]
    return (history_pairs() if history else []) + rows


def training_set(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    xy = np.array([r["pre_xy"] for r in rows], float)
    d = np.array([[r["wrist_xy"][0] - r["pre_xy"][0], r["wrist_xy"][1] - r["pre_xy"][1]] for r in rows], float)
    return xy, d


def model_json(W: np.ndarray, xy: np.ndarray, version: int, n: int, stats: dict) -> dict:
    pad = 20.0  # no extrapolation beyond 20 mm outside the positions it saw
    return {"version": version, "n": n, "W": W.tolist(), "bounds": [float(xy[:, 0].min() - pad), float(xy[:, 0].max() + pad),
                                                                      float(xy[:, 1].min() - pad), float(xy[:, 1].max() + pad)],
            "ridge": RIDGE, "features": "1, x/100, y/100 (mm, UF0, table estimate with TABLE_XY_OFFSET); output = dx, dy mm", "stats": stats}


def predict(model: dict | None, xy: np.ndarray) -> np.ndarray:
    if not model:
        return np.zeros((len(xy), 2))
    x0, x1, y0, y1 = model["bounds"]
    out = np.clip(feats(xy) @ np.asarray(model["W"], float), -20.0, 20.0)
    inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    return out * inside[:, None]


def cmd_fit() -> int:
    rows = load_pairs()
    if len(rows) < MIN_PAIRS:
        print(f"only {len(rows)} pairs, need {MIN_PAIRS}: not fitting")
        return 1
    xy, d = training_set(rows)
    errs = {k: loo_errors(xy, d, k) for k in ("zero", "const", "linear")}
    champion = json.loads(MODEL.read_text()) if MODEL.exists() else None
    champ_err = np.linalg.norm(d - predict(champion, xy), axis=1)  # the champion replayed on every pair (it has seen some: this favours it)
    print(f"{len(rows)} pairs ({sum(r.get('source') == 'log' for r in rows)} from the 30 Oct text log)")
    for k, e in errs.items():
        print(f"  leave-one-out first-view error, {k:6s}: mean {e.mean():.2f}  median {np.median(e):.2f}  p90 {np.percentile(e, 90):.2f} mm")
    print(f"  champion replayed (v{champion['version'] if champion else '-'}):          mean {champ_err.mean():.2f} mm")
    W = fit_w(xy, d)
    stats = {"loo_mean_mm": float(errs["linear"].mean()), "loo_zero_mm": float(errs["zero"].mean()), "loo_const_mm": float(errs["const"].mean())}
    version = (champion["version"] + 1) if champion else 1
    cand = model_json(W, xy, version, len(rows), stats)
    CAND.write_text(json.dumps(cand, indent=1))
    best_other = min(errs["zero"].mean(), errs["const"].mean(), champ_err.mean() if champion else 1e9)
    if errs["linear"].mean() <= best_other - MARGIN_MM:
        if MODEL.exists():
            shutil.copy(MODEL, PREV)
        MODEL.write_text(json.dumps(cand, indent=1))
        print(f"PROMOTED candidate v{version}: LOO {errs['linear'].mean():.2f} mm vs best alternative {best_other:.2f} mm")
        print("slopes (mm per 100 mm):", np.round(W[1:], 2).tolist(), " constant:", np.round(W[0], 2).tolist())
    else:
        print(f"candidate NOT promoted: LOO {errs['linear'].mean():.2f} mm is not {MARGIN_MM} mm better than {best_other:.2f} mm")
    return 0


def applied_stats(rows: list[dict]) -> dict:
    """Per row: error the arm started with (wrist - hover) and the error it would have had with no model (wrist - pre)."""
    app = [r for r in rows if r.get("applied") and r.get("source") != "log"]
    out = {"n": len(app)}
    if not app:
        return out
    wo = np.array([np.linalg.norm(np.array(r["wrist_xy"]) - np.array(r["pre_xy"])) for r in app])
    wi = np.array([np.linalg.norm(np.array(r["wrist_xy"]) - np.array(r["hover_xy"])) for r in app])
    out.update(without_mean=float(wo.mean()), with_mean=float(wi.mean()), without_median=float(np.median(wo)), with_median=float(np.median(wi)),
               better=int((wi < wo).sum()), versions=sorted({r.get("version") for r in app if r.get("version") is not None}))
    return out


def cmd_report() -> int:
    rows = load_pairs(history=False)
    s = applied_stats(rows)
    base = [r for r in rows if not r.get("applied")]
    print(f"logged first views: {len(rows)}  (model off: {len(base)}, model on: {s['n']})")
    if base:
        e = [np.linalg.norm(np.array(r["wrist_xy"]) - np.array(r["pre_xy"])) for r in base]
        print(f"  model off: table-to-wrist gap  mean {np.mean(e):.1f}  median {np.median(e):.1f} mm  (n={len(e)})")
    if s["n"]:
        print(f"  model on : gap at the first view  mean {s['with_mean']:.1f}  median {s['with_median']:.1f} mm (n={s['n']}, versions {s['versions']})")
        print(f"  same cubes, had the model been off: mean {s['without_mean']:.1f}  median {s['without_median']:.1f} mm; "
              f"the model's start was closer on {s['better']}/{s['n']}")
    if DESCENTS.exists():
        dl = [json.loads(line) for line in DESCENTS.read_text().splitlines() if line.strip()]
        for label, sel in (("model off", lambda r: r.get("calib_version") is None), ("model on", lambda r: r.get("calib_version") is not None)):
            f = [r["wrist_fixes"] for r in dl if "wrist_fixes" in r and r.get("phrase") == "cube" and sel(r)]
            if f:
                print(f"  re-hovers per cube, {label}: {np.mean(f):.2f} (n={len(f)})")
    if MODEL.exists():
        m = json.loads(MODEL.read_text())
        print(f"  champion: v{m['version']} fitted on {m['n']} pairs, leave-one-out {m['stats']['loo_mean_mm']:.2f} mm")
    return 0


def cmd_check() -> int:
    s = applied_stats([r for r in load_pairs(history=False) if r.get("applied")][-CHECK_LAST:])
    if s["n"] < CHECK_LAST:
        print(f"{s['n']} applied first views, need {CHECK_LAST}: nothing to check")
        return 0
    print(f"last {s['n']} applied: with the model {s['with_mean']:.1f} mm, without {s['without_mean']:.1f} mm")
    if s["with_mean"] > s["without_mean"] + CHECK_WORSE_MM:
        print("the model made it worse: rolling back")
        return cmd_rollback()
    print("keeping the champion")
    return 0


def cmd_rollback() -> int:
    if PREV.exists():
        shutil.copy(PREV, MODEL)
        PREV.unlink()
        print("restored the previous champion")
    elif MODEL.exists():
        MODEL.rename(HERE / "calibration_model.rolled_back.json")
        print("no previous champion: the model is switched off (file renamed)")
    return 0


if __name__ == "__main__":
    cmds = {"fit": cmd_fit, "report": cmd_report, "check": cmd_check, "rollback": cmd_rollback}
    sys.exit(cmds.get(sys.argv[1] if len(sys.argv) > 1 else "report", cmd_report)())
