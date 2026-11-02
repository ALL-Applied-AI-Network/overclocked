"""One-class test: learn only what an EMPTY gripper looks like (post-release crops, known empty); everything else = holding something.
Train only on post-release empties; test on lift / pre-release crops (held vs real empty). The same grasp's own post-release crop is never used (no leakage)."""
import numpy as np, json
z = np.load("features_v2.npz"); y = z["y"]; g = z["g"]; kind = z["kind"]
def norm(F): return F / np.linalg.norm(F, axis=1, keepdims=True)
def auc(s, yy):
    pos, neg = s[yy == 1], s[yy == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if len(pos) and len(neg) else float("nan")
for name in ("c_radio", "dinov2_small"):
    F = norm(z[name]); post = kind == "post"
    out = {}
    for k_nn in (1, 3):
        score = np.zeros(len(y))
        for i in range(len(y)):
            ref = post & (g != g[i])               # empties from OTHER grasps only
            d = 1 - F[ref] @ F[i]                  # cosine distance to each empty-gripper example
            score[i] = np.sort(d)[:k_nn].mean()    # larger = less like an empty gripper = more likely holding something
        # decision threshold from the empties alone: the 95th percentile of how far a held-out EMPTY (post-release) crop sits from the other empties
        thr = np.percentile(score[post], 95)
        res = {}
        for kd in ("lift", "pre", "all_real"):
            mk = (kind != "post") if kd == "all_real" else (kind == kd)
            pred = score > thr
            res[kd] = {"auc": round(auc(score[mk], y[mk]), 3), "held_ok": f"{int((pred & (y == 1) & mk).sum())}/{int(((y == 1) & mk).sum())}", "false_pass": f"{int((pred & (y == 0) & mk).sum())}/{int(((y == 0) & mk).sum())}"}
        res["post_called_empty"] = f"{int(((score <= thr) & post).sum())}/{int(post.sum())}"
        out[f"k={k_nn}"] = res
        print(name, f"k={k_nn}", json.dumps(res), flush=True)
    json.dump(out, open(f"results_oneclass_{name}.json", "w"), indent=1)
