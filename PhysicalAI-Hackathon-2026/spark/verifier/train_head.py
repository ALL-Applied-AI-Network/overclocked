import numpy as np, torch, json
z = np.load("features_v2.npz"); F = z["c_radio"]; y = z["y"]; g = z["g"]; kind = z["kind"]
def logreg(X, t, l2=1.0):
    X = torch.tensor(X, dtype=torch.float32); t = torch.tensor(t, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    wts = torch.where(t > 0.5, 0.5 / t.mean(), 0.5 / (1 - t.mean())); opt = torch.optim.LBFGS([w, b], lr=1, max_iter=300)
    def cl():
        opt.zero_grad(); zz = X @ w + b
        loss = (wts * torch.nn.functional.binary_cross_entropy_with_logits(zz, t, reduction="none")).mean() + l2 * (w ** 2).sum() / len(t); loss.backward(); return loss
    opt.step(cl); return w.detach().numpy(), b.detach().numpy()
s = np.zeros(len(y))
for gi in np.unique(g):
    te = g == gi; tr = ~te; mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6; w, b = logreg((F[tr] - mu) / sd, y[tr]); s[te] = ((F[te] - mu) / sd) @ w + b
p = 1 / (1 + np.exp(-s)); real_empty = (y == 0) & (kind != "post")
print("threshold on P(held): held kept | false pass on real empties (lift+pre, %d) | false pass on all empties (%d)" % (real_empty.sum(), (y == 0).sum()))
for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    print("  %.1f: %d/%d | %d/%d | %d/%d" % (t, ((p >= t) & (y == 1)).sum(), (y == 1).sum(), ((p >= t) & real_empty).sum(), real_empty.sum(), ((p >= t) & (y == 0)).sum(), (y == 0).sum()))
mu, sd = F.mean(0), F.std(0) + 1e-6; w, b = logreg((F - mu) / sd, y)
np.savez("head_cradio.npz", mu=mu, sd=sd, w=w, b=b); print("head saved:", w.shape)
