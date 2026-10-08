"""Node classifiers used inside the tree.

A node is a small neural network that maps the (trunk) feature vector to a logit
(binary nodes) or to K logits (multiclass nodes).

Training objective of a binary node -- the neural generalisation of eqs. (1)-(3) of the
tree-type network note:

    minimise  sum_i  c_i * q_i,     q_i = max(0, 1 - (2 y_i - 1) f(x_i))

i.e. the (weighted) sum of hinge slacks; ``f`` is an MLP instead of an affine map.
"Don't care" samples (category X in Table II) are simply left out of the node's data.

``IsoNode`` is a closed-form two-layer threshold network that isolates a set of positive
points from a set of negative points on the unit sphere. It is the high-dimensional analogue
of the hand-built hyperplanes of Fig. 4 and guarantees termination: on L2-normalised features
every training point is an extreme point, hence linearly separable from all the others.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNode(nn.Module):
    def __init__(self, d, h, out=1):
        super().__init__()
        self.d, self.h, self.out = d, h, out
        if h > 0:
            self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, out))
        else:  # a single (linear threshold) neuron, exactly as in the note
            self.net = nn.Linear(d, out)
        self.kind = "mlp" if h > 0 else "linear"
        self.cols = None  # optional (start, end) column block of the input this node reads

    def forward(self, x):
        if getattr(self, "cols", None) is not None:
            x = x[:, self.cols[0]:self.cols[1]]
        z = self.net(x)
        return z.squeeze(1) if self.out == 1 else z

    def hidden(self, x):
        if getattr(self, "cols", None) is not None:
            x = x[:, self.cols[0]:self.cols[1]]
        return F.relu(self.net[0](x)) if self.h > 0 else x

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class IsoNode(nn.Module):
    """OR of spherical caps: fires iff <x/|x|, c_j> >= tau_j for some unit j."""

    def __init__(self, C, tau):
        super().__init__()
        self.register_buffer("C", torch.as_tensor(C, dtype=torch.float32))
        self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
        self.kind = "iso"
        self.out = 1
        self.h = len(tau)

    def forward(self, x):
        xn = F.normalize(x, dim=1)
        s = xn @ self.C.T - self.tau  # n, units
        # logit: positive iff any unit fires. Scale so it is a usable margin.
        return 10.0 * s.max(1).values

    def n_params(self):
        return self.C.numel() + self.tau.numel()


def fit_iso(X, y):
    """Greedy cover of positives by spherical caps that exclude every negative."""
    Xn = F.normalize(X, dim=1)
    pos = torch.where(y > 0.5)[0]
    neg = torch.where(y <= 0.5)[0]
    Pn, Nn = Xn[pos], Xn[neg]
    covered = torch.zeros(len(pos), dtype=torch.bool)
    C, T = [], []
    while not covered.all():
        i = torch.where(~covered)[0][0]
        c = Pn[i]
        max_neg = (Nn @ c).max().item() if len(neg) else -1.0
        if max_neg >= 1 - 1e-6:  # duplicate point with conflicting label: cannot be separated
            covered[i] = True
            continue
        tau = 0.5 * (max_neg + 1.0)  # max-margin threshold between the cap and the negatives
        C.append(c)
        T.append(tau)
        covered |= (Pn @ c) >= tau
    if not C:
        C, T = [Xn[0] * 0], [2.0]
    return IsoNode(torch.stack(C), torch.tensor(T))


def _weights(y, mode):
    n = len(y)
    npos = y.sum().item()
    nneg = n - npos
    if npos == 0 or nneg == 0 or mode == "none":
        return torch.ones(n)
    if mode == "balanced":
        wp, wn = n / (2 * npos), n / (2 * nneg)
    elif mode == "sqrt":
        r = math.sqrt(nneg / npos)
        wp, wn = r, 1.0
    else:
        raise ValueError(mode)
    w = torch.where(y > 0.5, torch.tensor(wp), torch.tensor(wn))
    return w * n / w.sum()


def fit_binary(X, y, h=32, epochs=400, lr=1e-2, wd=0.0, loss="hinge", pos_weight="sqrt",
               batch=8192, seed=0, init=None, check_every=25):
    """Train a binary node until it fits its data (zero errors) or the epoch budget ends.

    X: float tensor (n,d); y: float tensor of {0,1}. Returns (model, logits on X).
    """
    torch.manual_seed(seed)
    model = init if init is not None else MLPNode(X.shape[1], h, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    w = _weights(y, pos_weight)
    s = 2 * y - 1
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n) if n > batch else None
        for b in range(0, n, batch):
            idx = perm[b:b + batch] if perm is not None else slice(None)
            z = model(X[idx])
            if loss == "hinge":
                l = (w[idx] * F.relu(1 - s[idx] * z)).mean()
            elif loss == "sqhinge":
                l = (w[idx] * F.relu(1 - s[idx] * z) ** 2).mean()
            else:
                l = (w[idx] * F.binary_cross_entropy_with_logits(z, y[idx], reduction="none")).mean()
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        if (ep + 1) % check_every == 0:
            with torch.no_grad():
                err = ((model(X) >= 0).float() != y).sum().item()
            if err == 0:
                break
    with torch.no_grad():
        z = model(X)
    return model, z


def fit_multiclass(X, y, k, h=64, epochs=200, lr=1e-2, wd=0.0, batch=8192, seed=0,
                   check_every=25, label_smoothing=0.0):
    torch.manual_seed(seed)
    model = MLPNode(X.shape[1], h, k)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            l = F.cross_entropy(model(X[idx]), y[idx], label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
        if (ep + 1) % check_every == 0:
            with torch.no_grad():
                if (model(X).argmax(1) != y).sum().item() == 0:
                    break
    with torch.no_grad():
        z = model(X)
    return model, z
