"""Does the head generalise to an object type it never saw? Train with every battery grasp (and its post-release crop) removed; test on the battery crops."""
import numpy as np, torch
z = np.load("features_v2.npz"); y = z["y"]; g = z["g"]; kind = z["kind"]; F = z["c_radio"]
batteries = {21, 22, 25}   # the three held-battery lifts in the labelled set (by eye): all their crops are held out
def logreg(X, t, l2=1.0):
    X = torch.tensor(X, dtype=torch.float32); t = torch.tensor(t, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    wts = torch.where(t > 0.5, 0.5 / t.mean(), 0.5 / (1 - t.mean())); opt = torch.optim.LBFGS([w, b], lr=1, max_iter=300)
    def cl():
        opt.zero_grad(); zz = X @ w + b
        loss = (wts * torch.nn.functional.binary_cross_entropy_with_logits(zz, t, reduction="none")).mean() + l2 * (w ** 2).sum() / len(t); loss.backward(); return loss
    opt.step(cl); return w.detach().numpy(), b.detach().numpy()
te = np.isin(g, list(batteries)); tr = ~te
mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6; w, b = logreg((F[tr] - mu) / sd, y[tr])
p = 1 / (1 + np.exp(-(((F - mu) / sd) @ w + b)))
for kd in ("lift", "pre", "post"):
    m = te & (kind == kd)
    print(f"unseen battery, {kd:4s} crops: n={int(m.sum())}, labelled held: {int((y[m]==1).sum())}, called held at P>=0.5: {int(((p>=0.5)&m&(y==1)).sum())}; P(held) = {np.round(p[m],2).tolist()}")
print("training set: %d crops (%d held, %d empty)" % (tr.sum(), (y[tr] == 1).sum(), (y[tr] == 0).sum()))
