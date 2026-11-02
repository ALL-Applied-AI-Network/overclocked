"""Find every grasp in the cell's recorded runs (manipbench datasets) and say how it ended.

    uv run --with pyarrow --with pandas python mine_lifts.py

A grasp = the gripper command goes 0 -> 1. Its outcome, from the recorded TCP track alone:
  released_high   the gripper opened again high above the table (z > -295 mm UF0): it carried something to the bin (or carried nothing)
  ended_after     the episode ended before that: a hit, a stop, or the operator ended it
Hits and stops are told apart by the cell's end_reason. Frames for each grasp (wrist, 1.2 s after the close) go to verifier_data/mined/.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/.cache/huggingface/lerobot/manipbench")
rows = []
for run in sorted(glob.glob(f"{ROOT}/r*")):
    meta = json.load(open(f"{run}/manipbench.json"))
    for ep in meta:
        i = ep["episode_index"]
        f = f"{run}/data/chunk-000/file-{i:03d}.parquet"
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        act = np.stack(d["action"].to_numpy()); tcp = np.stack(d["observation.tcp_pose"].to_numpy())
        g = act[:, 6] if act.shape[1] > 6 else None
        if g is None or ep.get("action_kind") != "tcp":
            continue
        close = [k for k in range(1, len(g)) if g[k] > 0.5 and g[k - 1] <= 0.5]
        for c in close:
            after = [k for k in range(c + 1, len(g)) if g[k] <= 0.5]
            rel = next((k for k in after if tcp[k][2] > -295.0), None)
            rows.append({"run": os.path.basename(run), "episode": i, "trial": ep["trial_id"][-3:], "end": ep["end_reason"], "close_frame": c,
                         "n_frames": len(g), "grasp_z": round(float(tcp[c][2]), 1), "grasp_xy": [round(float(tcp[c][0]), 1), round(float(tcp[c][1]), 1)],
                         "R": round(float(tcp[c][5]), 1), "outcome": "released_high" if rel else "ended_after"})
df = pd.DataFrame(rows)
print(len(df), "grasps in", df[["run", "episode"]].drop_duplicates().shape[0], "episodes")
print(df.groupby(["run", "outcome"]).size().unstack(fill_value=0).to_string())
df.to_json("verifier_data/mined_grasps.json", orient="records", indent=1)
