"""Held/empty verifier on side-camera crops: frozen backbone + small logistic head, leave-one-out, against a colour rule."""
import json, os, sys, time
import numpy as np, torch, cv2
from transformers import AutoModel
D = "data"; dev = "cuda"
lab = json.load(open(f"{D}/side_labels.json"))
items = [(k, v["label"], f"{D}/side_crops_lifted/{os.path.basename(v['crop'])}") for k, v in lab.items() if v["label"] in ("held", "empty")]
items = [(k, l, p) for k, l, p in items if os.path.exists(p)]
y = np.array([1 if l == "held" else 0 for _, l, _ in items]); print(len(items), "crops:", int(y.sum()), "held,", int((1 - y).sum()), "empty", flush=True)
imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for _, _, p in items]

def prep(im, size, mean=None, std=None):
    x = cv2.resize(im, (size, size), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    if mean is not None: x = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1))

feats, lat = {}, {}
with torch.no_grad():
    for name, loader in (("c-radio", lambda: AutoModel.from_pretrained("nvidia/C-RADIOv2-B", trust_remote_code=True)),
                         ("dinov2-small", lambda: AutoModel.from_pretrained("facebook/dinov2-small"))):
        m = loader().eval().to(dev)
        mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
        fs = []
        for im in imgs:
            x = prep(im, 224, None if name == "c-radio" else mean, None if name == "c-radio" else std).unsqueeze(0).to(dev)
            out = m(x)
            f = out[0] if name == "c-radio" else out.pooler_output
            fs.append(f.float().flatten().cpu().numpy())
        feats[name] = np.stack(fs)
        x = prep(imgs[0], 224, None if name == "c-radio" else mean, None if name == "c-radio" else std).unsqueeze(0).to(dev)
        for _ in range(5): m(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(30): m(x)
        torch.cuda.synchronize(); lat[name] = (time.time() - t) / 30 * 1000
        print(name, "features", feats[name].shape, "backbone latency %.1f ms (fp32, torch, batch 1)" % lat[name], flush=True)
        del m; torch.cuda.empty_cache()

def logreg(X, t, l2=1.0, iters=200):
    X = torch.tensor(X, dtype=torch.float32); t = torch.tensor(t, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    pw = torch.tensor([(1 - t).sum() / max(t.sum(), 1)] ) if False else None
    wts = torch.where(t > 0.5, 0.5 / t.mean(), 0.5 / (1 - t.mean()))  # balanced classes
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters)
    def closure():
        opt.zero_grad(); z = X @ w + b
        loss = (wts * torch.nn.functional.binary_cross_entropy_with_logits(z, t, reduction="none")).mean() + l2 * (w ** 2).sum() / len(t)
        loss.backward(); return loss
    opt.step(closure); return w.detach(), b.detach()

def loo(F, l2):
    scores = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        w, b = logreg((F[tr] - mu) / sd, y[tr], l2)
        scores[i] = float(((torch.tensor((F[i] - mu) / sd, dtype=torch.float32) @ w) + b))
    return scores

def auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))

res = {"n": len(y), "held": int(y.sum()), "empty": int((1 - y).sum()), "latency_ms": lat}
for name in feats:
    for l2 in (1.0, 10.0):
        s = loo(feats[name], l2); pred = s > 0
        res[f"{name}_l2={l2}"] = {"auc": round(auc(s, y), 3), "held_correct": int((pred & (y == 1)).sum()), "held_missed": int((~pred & (y == 1)).sum()),
                                   "false_pass(empty called held)": int((pred & (y == 0)).sum()), "empty_correct": int((~pred & (y == 0)).sum())}
        print(name, "l2", l2, res[f"{name}_l2={l2}"], flush=True)

# colour rule: saturated or dark pixels in a small window right at the fingertip (the crop is centred on the projected tip)
def rule(im):
    h, w = im.shape[:2]; c = im[h // 2 - 22:h // 2 + 30, w // 2 - 22:w // 2 + 22]
    hsv = cv2.cvtColor(c, cv2.COLOR_RGB2HSV); return float(((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 60)).sum() + ((hsv[:, :, 2] < 55)).sum())
rs = np.array([rule(im) for im in imgs]); res["colour_rule"] = {"auc": round(auc(rs, y), 3), "held_median": float(np.median(rs[y == 1])), "empty_median": float(np.median(rs[y == 0]))}
best = max(((auc(rs, y), t) for t in np.unique(rs)), key=lambda a: a[0]); print("colour rule", res["colour_rule"], flush=True)
json.dump(res, open("results_verifier.json", "w"), indent=1)
np.savez("features.npz", y=y, keys=np.array([k for k, _, _ in items]), **{k.replace('-', '_'): v for k, v in feats.items()})
print("DONE", flush=True)
