"""Held/empty verifier v2: crops of three kinds per grasp (lift above the table, just before release over the bin, just after release), frozen backbone + logistic head.
Leave-one-GRASP-out: all crops of a grasp are held out together, so no near-duplicate leaks into training."""
import json, os, time
import numpy as np, torch, cv2
from transformers import AutoModel
D = "data"; dev = "cuda"
lab = json.load(open(f"{D}/side_labels.json")); pre = json.load(open(f"{D}/prerelease_index.json")); post = json.load(open(f"{D}/postrelease_index.json"))
items = []  # (grasp, kind, label 1=held 0=empty, path)
for k, v in lab.items():
    if v["label"] in ("held", "empty"):
        p = f"{D}/side_crops_lifted/{os.path.basename(v['crop'])}"
        if os.path.exists(p): items.append((int(k), "lift", 1 if v["label"] == "held" else 0, p))
for m in pre:
    p = f"{D}/side_crops_prerelease/{m['file']}"
    if os.path.exists(p): items.append((m["k"], "pre", 1 if m["label"] == "held" else 0, p))
for m in post:
    p = f"{D}/side_crops_postrelease/{m['file']}"
    if os.path.exists(p): items.append((m["k"], "post", 0, p))
g = np.array([i[0] for i in items]); kind = np.array([i[1] for i in items]); y = np.array([i[2] for i in items])
print(len(items), "crops;", {kd: (int(((kind == kd) & (y == 1)).sum()), int(((kind == kd) & (y == 0)).sum())) for kd in ("lift", "pre", "post")}, "(held, empty)", flush=True)
imgs = [cv2.cvtColor(cv2.imread(i[3]), cv2.COLOR_BGR2RGB) for i in items]
def prep(im, mean=None, std=None):
    x = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    if mean is not None: x = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1))
mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
feats = {}
with torch.no_grad():
    for name, loader in (("c-radio", lambda: AutoModel.from_pretrained("nvidia/C-RADIOv2-B", trust_remote_code=True)), ("dinov2-small", lambda: AutoModel.from_pretrained("facebook/dinov2-small"))):
        m = loader().eval().to(dev); fs = []
        for im in imgs:
            x = prep(im, *(() if name == "c-radio" else (mean, std))).unsqueeze(0).to(dev); o = m(x)
            fs.append((o[0] if name == "c-radio" else o.pooler_output).float().flatten().cpu().numpy())
        feats[name] = np.stack(fs); del m; torch.cuda.empty_cache()
def logreg(X, t, l2):
    X = torch.tensor(X, dtype=torch.float32); t = torch.tensor(t, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    wts = torch.where(t > 0.5, 0.5 / t.mean(), 0.5 / (1 - t.mean()))
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=200)
    def cl():
        opt.zero_grad(); z = X @ w + b
        loss = (wts * torch.nn.functional.binary_cross_entropy_with_logits(z, t, reduction="none")).mean() + l2 * (w ** 2).sum() / len(t); loss.backward(); return loss
    opt.step(cl); return w.detach(), b.detach()
def logo(F, l2):
    s = np.zeros(len(y))
    for gi in np.unique(g):
        te = g == gi; tr = ~te; mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        w, b = logreg((F[tr] - mu) / sd, y[tr], l2)
        s[te] = (torch.tensor((F[te] - mu) / sd, dtype=torch.float32) @ w + b).numpy()
    return s
def auc(s, yy):
    pos, neg = s[yy == 1], s[yy == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if len(pos) and len(neg) else float("nan")
def rule(im):
    h, w = im.shape[:2]; c = im[h // 2 - 22:h // 2 + 30, w // 2 - 22:w // 2 + 22]; hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV)
    return float(((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 60)).sum() + (hsv[:, :, 2] < 55).sum())
rs = np.array([rule(im) for im in imgs])
def report(name, s, thr):
    pred = s > thr; out = {}
    for kd in ("lift", "pre", "post", "all"):
        mk = np.ones(len(y), bool) if kd == "all" else kind == kd
        out[kd] = {"held_ok": f"{int((pred & (y == 1) & mk).sum())}/{int(((y == 1) & mk).sum())}", "false_pass": f"{int((pred & (y == 0) & mk).sum())}/{int(((y == 0) & mk).sum())}", "auc": round(auc(s[mk], y[mk]), 3)}
    print(name, json.dumps(out), flush=True); return out
res = {"n": len(y)}
for name in feats:
    for l2 in (1.0, 10.0):
        res[f"{name}_l2={l2}"] = report(f"{name} l2={l2}", logo(feats[name], l2), 0.0)
# colour rule with a threshold chosen on the crops of all OTHER grasps (leave-one-grasp-out), no peeking
sc = np.zeros(len(y)); thr_used = []
for gi in np.unique(g):
    te = g == gi; tr = ~te; cand = np.unique(rs[tr]); best = max(cand, key=lambda t: ((rs[tr] > t) == (y[tr] == 1)).mean()); sc[te] = rs[te] - best; thr_used.append(float(best))
res["colour_rule"] = report("colour rule (threshold from other grasps)", sc, 0.0)
json.dump(res, open("results_verifier_v2.json", "w"), indent=1); np.savez("features_v2.npz", y=y, g=g, kind=kind, **{k.replace("-", "_"): v for k, v in feats.items()}); print("DONE", flush=True)
