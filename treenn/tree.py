"""Tree-type networks whose nodes are small neural networks.

Binary tree (faithful generalisation of the note)
-------------------------------------------------
For a node with logit z(x) and data with labels y in {0,1}, the four categories of Table I are

    C1: z<0, y=0      C2: z>=0, y=1      C3: z<0, y=1 (missed)      C4: z>=0, y=0 (false alarm)

Child A must output 1 on C3 and 0 on C1 and C4 (C2 = don't care);
child B must output 1 on C4 and 0 on C2 and C3 (C1 = don't care)   -- Table II.
The node's corrected net input is   z(x) + w_A * A(x) + w_B * B(x),   w_A >= 0, w_B <= 0   (eq. 9).

Given perfect children, the LP (6)-(9) over (w_A, w_B) has the closed-form minimal solution

    w_A = max_{i in C3} (delta - z_i),        w_B = - max_{i in C4} (delta + z_i),

which is what we use by default; ``resolve='lp'`` instead re-solves the whole output layer of the
node together with w_A, w_B by linear programming exactly as in (6)-(9) (optionally with an
L1 penalty on the weights to minimise the number of non-zero weights).

Multiclass trees
----------------
* ``OvRForest``      : K binary trees (one-vs-rest).
* ``CorrectorTree``  : a K-way root; for every class k with errors, a binary "detector" child D_k
                       that fires on misclassified samples of class k and is silent on every sample
                       of another class (correct samples of class k are don't-care). This reduces
                       exactly to Table II when K=2. Logits: z_k(x) + w_k D_k(x).
* ``GatedTree``      : a K-way node, an error-detector child E (binary tree) and a re-classifier
                       child R (a GatedTree trained only on the node's errors).
                       Output = R(x) if E(x) fires else node(x). Two children per node regardless of K.
"""
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn.functional as F

from .node import MLPNode, IsoNode, fit_binary, fit_multiclass, fit_iso


@dataclass
class GrowConfig:
    h: int = 32                  # hidden width of binary nodes (0 = single linear neuron)
    h_root: int = 64             # hidden width of multiclass (K-way) nodes
    epochs: int = 300
    root_epochs: int = 100
    lr: float = 1e-2
    wd: float = 0.0
    loss: str = "hinge"
    pos_weight: str = "auto"     # auto: balanced for linear nodes, sqrt-balanced for MLP nodes
    max_depth: int = 10          # depth after which closed-form isolation nodes are used
    iso_below: int = 0           # use an isolation node directly if a child has <= this many positives
    fp_ratio: float = 3.0        # fallback to isolation if a child makes > fp_ratio * npos errors
    delta: float = 1.0           # margin used for closed-form connection weights
    resolve: str = "closed"      # 'closed' or 'lp'
    lp_l1: float = 0.0           # L1 weight penalty inside the LP (minimise #weights)
    seed: int = 0
    verbose: bool = False
    blocks: list = None          # [(start,end), ...]: per-node choice among trunk feature blocks


class Recorder:
    def __init__(self):
        self.nodes = []
        self.t0 = time.time()

    def add(self, **kw):
        self.nodes.append(kw)
        return kw


# --------------------------------------------------------------------------------------
# Binary tree
# --------------------------------------------------------------------------------------
class BNode:
    def __init__(self, net, depth, path):
        self.net, self.depth, self.path = net, depth, path
        self.A = self.B = None
        self.wA = 0.0
        self.wB = 0.0

    def logit(self, X):
        z = self.net(X)
        if self.A is not None:
            z = z + self.wA * self.A.fire(X)
        if self.B is not None:
            z = z + self.wB * self.B.fire(X)
        return z

    def fire(self, X):
        return (self.logit(X) >= 0).float()

    def walk(self):
        yield self
        for c in (self.A, self.B):
            if c is not None:
                yield from c.walk()


def _lp_resolve(node, X, y, a, b, l1):
    """Re-solve the node's output layer + (w_A, w_B) with the LP (6)-(9)."""
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, hstack, identity
    net = node.net
    with torch.no_grad():
        H = net.hidden(X).numpy().astype(np.float64)
    n, h = H.shape
    s = (2 * y.numpy() - 1).astype(np.float64)
    a = a.numpy().astype(np.float64)
    b = b.numpy().astype(np.float64)
    # variables: u (h, free) c (free) wA>=0 wB<=0 q>=0 ; u split as u+ - u- for the L1 term
    feats = np.concatenate([H, np.ones((n, 1)), a[:, None], b[:, None]], 1) * s[:, None]
    m = h + 3
    # -(s*(feat.v)) - q <= -1
    A_ub = hstack([csr_matrix(-feats), csr_matrix(feats[:, :h + 1]), -identity(n, format="csr")])
    # columns: [v (m) ; u- c- (h+1) ; q (n)]   where v = [u+, c+, wA, wB]
    cost = np.concatenate([np.full(h, l1), [0, 0, 0], np.full(h, l1), [0], np.ones(n)])
    bounds = [(0, None)] * h + [(0, None), (0, None), (None, 0)] + [(0, None)] * (h + 1) + [(0, None)] * n
    res = linprog(cost, A_ub=A_ub, b_ub=-np.ones(n), bounds=bounds, method="highs")
    if res.status != 0:
        return None
    v = res.x
    u = v[:h] - v[m:m + h]
    c = v[h] - v[m + h]
    wA, wB = v[h + 1], v[h + 2]
    slack = v[m + h + 1:].sum()
    last = net.net[-1] if net.h > 0 else net.net
    with torch.no_grad():
        if net.h > 0:
            last.weight.copy_(torch.tensor(u, dtype=torch.float32).view(1, -1))
            last.bias.fill_(float(c))
        else:
            last.weight.copy_(torch.tensor(u, dtype=torch.float32).view(1, -1))
            last.bias.fill_(float(c))
    return float(wA), float(wB), float(slack), int((np.abs(u) > 1e-9).sum())


def grow_binary(X, idx, y, cfg, rec, depth=0, path="R", budget_pos=None):
    """Grow a binary tree on samples X[idx] with labels y (float {0,1}, aligned with idx).

    Returns a BNode whose ``fire`` is correct on every sample of X[idx] (barring exact duplicates
    with conflicting labels).
    """
    Xs = X[idx]
    n, npos = len(idx), int(y.sum().item())
    use_iso = depth >= cfg.max_depth or (depth > 0 and npos <= cfg.iso_below)
    reason = ("depth" if depth >= cfg.max_depth else "forced") if use_iso else None
    mlp_tp = mlp_err = None
    if not use_iso:
        pw = cfg.pos_weight if cfg.pos_weight != "auto" else ("balanced" if cfg.h == 0 else "sqrt")
        best = None
        for blk in (cfg.blocks or [None]):
            Xb = Xs if blk is None else Xs[:, blk[0]:blk[1]]
            net_b, z_b = fit_binary(Xb, y, h=cfg.h, epochs=cfg.epochs, lr=cfg.lr, wd=cfg.wd,
                                    loss=cfg.loss, pos_weight=pw, seed=cfg.seed + depth)
            net_b.cols = blk
            e_b = int(((z_b >= 0).float() != y).sum().item())
            if best is None or e_b < best[2]:
                best = (net_b, z_b, e_b)
        net, z = best[0], best[1]
        pred = z >= 0
        err = int((pred.float() != y).sum().item())
        tp = int((pred & (y > 0.5)).sum().item())
        # progress test: a child exists to fix `npos` samples. If it fixes none of them (the
        # trivial all-zero solution, typical for "inside-out" child problems) or it creates more
        # errors than `fp_ratio` x what it was asked to fix, fall back to an isolation node.
        mlp_tp, mlp_err = tp, err
        if depth > 0 and (tp == 0 or err > cfg.fp_ratio * npos):
            use_iso = True
            reason = "trivial" if tp == 0 else "too_many_errors"
    if use_iso:
        net = fit_iso(Xs, y)
        with torch.no_grad():
            z = net(Xs)
        pred = z >= 0
        err = int((pred.float() != y).sum().item())
    node = BNode(net, depth, path)
    node.idx = idx.to(torch.int32)      # which training samples this node was trained on
    node.ypos = (y > 0.5)               # its targets (True = must fire)
    yb = y > 0.5
    C3 = (~pred) & yb
    C4 = pred & (~yb)
    info = rec.add(path=path, depth=depth, kind=net.kind, h=int(getattr(net, "h", 0)),
                   cols=list(net.cols) if getattr(net, "cols", None) is not None else None,
                   n=n, npos=npos, err=err, n_c3=int(C3.sum()), n_c4=int(C4.sum()),
                   params=int(net.n_params()), fallback=reason, mlp_tp=mlp_tp, mlp_err=mlp_err)
    if cfg.verbose:
        print(f"{'  ' * depth}{path}: kind={net.kind} n={n} pos={npos} C3={int(C3.sum())} "
              f"C4={int(C4.sum())} t={time.time() - rec.t0:.0f}s", flush=True)
    a = torch.zeros(n)
    b = torch.zeros(n)
    if C3.any():  # child A: 1 on C3, 0 on C1 and C4, C2 don't care
        m = ~(pred & yb)
        node.A = grow_binary(X, idx[m], C3[m].float(), cfg, rec, depth + 1, path + ".A")
        with torch.no_grad():
            a = node.A.fire(Xs)
    if C4.any():  # child B: 1 on C4, 0 on C2 and C3, C1 don't care
        m = ~((~pred) & (~yb))
        node.B = grow_binary(X, idx[m], C4[m].float(), cfg, rec, depth + 1, path + ".B")
        with torch.no_grad():
            b = node.B.fire(Xs)
    if node.A is not None or node.B is not None:
        done = False
        if cfg.resolve == "lp" and isinstance(net, MLPNode):
            last = net.net[-1] if net.h > 0 else net.net
            saved = (last.weight.detach().clone(), last.bias.detach().clone())
            out = _lp_resolve(node, Xs, y, a, b, cfg.lp_l1)
            if out is not None:
                node.wA, node.wB, slack, nnz = out
                info.update(lp_slack=slack, lp_nnz=nnz)
                done = True
                with torch.no_grad():
                    ok = bool((((node.net(Xs) + node.wA * a + node.wB * b) >= 0).float() == y).all())
                if not ok:
                    # Non-zero slack: with the l1 term the LP may accept a few errors in exchange for
                    # smaller weights. Keep the guarantee: restore the node and use the closed form.
                    with torch.no_grad():
                        last.weight.copy_(saved[0])
                        last.bias.copy_(saved[1])
                    info.update(lp_fallback=True)
                    node.wA = node.wB = 0.0
                    done = False
        if not done:
            if node.A is not None:
                node.wA = float((cfg.delta - z[C3]).max().item())
            if node.B is not None:
                node.wB = -float((cfg.delta + z[C4]).max().item())
        info.update(wA=node.wA, wB=node.wB)
    return node


class BinaryTree:
    def __init__(self, cfg=None):
        self.cfg = cfg or GrowConfig()

    def fit(self, X, y):
        self.rec = Recorder()
        idx = torch.arange(len(X))
        self.root = grow_binary(X, idx, torch.as_tensor(y).float(), self.cfg, self.rec)
        return self

    @torch.no_grad()
    def logit(self, X):
        return self.root.logit(X)

    @torch.no_grad()
    def predict(self, X):
        return (self.logit(X) >= 0).long()

    @torch.no_grad()
    def root_predict(self, X):
        return (self.root.net(X) >= 0).long()

    def nodes(self):
        return list(self.root.walk())


# --------------------------------------------------------------------------------------
# Multiclass trees
# --------------------------------------------------------------------------------------
class OvRForest:
    def __init__(self, k, cfg=None):
        self.k, self.cfg = k, cfg or GrowConfig()

    def fit(self, X, y):
        self.rec = Recorder()
        y = torch.as_tensor(y)
        self.trees = []
        for c in range(self.k):
            idx = torch.arange(len(X))
            self.trees.append(grow_binary(X, idx, (y == c).float(), self.cfg, self.rec,
                                          0, f"T{c}"))
        return self

    @torch.no_grad()
    def logits(self, X):
        return torch.stack([t.logit(X) for t in self.trees], 1)

    @torch.no_grad()
    def predict(self, X):
        return self.logits(X).argmax(1)

    @torch.no_grad()
    def root_predict(self, X):
        return torch.stack([t.net(X) for t in self.trees], 1).argmax(1)

    def nodes(self):
        return [n for t in self.trees for n in t.walk()]


class CorrectorTree:
    def __init__(self, k, cfg=None):
        self.k, self.cfg = k, cfg or GrowConfig()

    def fit(self, X, y, root=None):
        cfg = self.cfg
        self.rec = Recorder()
        y = torch.as_tensor(y)
        if root is None:
            best = None
            for blk in (cfg.blocks or [None]):
                Xb = X if blk is None else X[:, blk[0]:blk[1]]
                r, Zb = fit_multiclass(Xb, y, self.k, h=cfg.h_root, epochs=cfg.root_epochs,
                                       lr=cfg.lr, wd=cfg.wd, seed=cfg.seed)
                r.cols = blk
                e = int((Zb.argmax(1) != y).sum())
                if best is None or e < best[2]:
                    best = (r, Zb, e)
            root, Z = best[0], best[1]
        else:
            with torch.no_grad():
                Z = root(X)
        self.root = root
        wrong = Z.argmax(1) != y
        self.rec.add(path="R", depth=0, kind="mlp-k", h=cfg.h_root, n=len(X), npos=-1,
                     cols=list(root.cols) if getattr(root, "cols", None) is not None else None,
                     err=int(wrong.sum()), params=int(root.n_params()))
        if cfg.verbose:
            print(f"root: errors {int(wrong.sum())}/{len(X)}", flush=True)
        self.det, self.w = {}, {}
        for c in range(self.k):
            pos = wrong & (y == c)
            if not pos.any():
                continue
            m = pos | (y != c)
            idx = torch.where(m)[0]
            self.det[c] = grow_binary(X, idx, pos[m].float(), cfg, self.rec, 1, f"D{c}")
            need = Z[pos].max(1).values - Z[pos][:, c]
            self.w[c] = float(need.max().item() + cfg.delta)
            for r in self.rec.nodes:
                if r["path"] == f"D{c}":
                    r["w_conn"] = self.w[c]
        return self

    @torch.no_grad()
    def logits(self, X):
        Z = self.root(X).clone()
        for c, d in self.det.items():
            Z[:, c] += self.w[c] * d.fire(X)
        return Z

    @torch.no_grad()
    def predict(self, X):
        return self.logits(X).argmax(1)

    @torch.no_grad()
    def root_predict(self, X):
        return self.root(X).argmax(1)

    def nodes(self):
        return [n for d in self.det.values() for n in d.walk()]


class ProtoNode(torch.nn.Module):
    """Winner-take-all RBF layer (1-NN on the sphere) -- fallback leaf for the gated tree."""

    def __init__(self, X, y, k):
        super().__init__()
        self.register_buffer("C", F.normalize(X, dim=1))
        self.register_buffer("lab", F.one_hot(y, k).float())
        self.kind, self.h = "proto", len(X)

    def forward(self, X):
        s = F.normalize(X, dim=1) @ self.C.T
        return self.lab[s.argmax(1)] * 10

    def n_params(self):
        return self.C.numel() + len(self.C)


class GNode:
    def __init__(self, net, depth, path):
        self.net, self.depth, self.path = net, depth, path
        self.E = self.R = None

    def predict(self, X):
        p = self.net(X).argmax(1)
        if self.E is not None:
            e = self.E.fire(X).bool()
            if e.any():
                p[e] = self.R.predict(X[e])
        return p


def grow_gated(X, idx, y, k, cfg, rec, depth=0, path="G"):
    ys = y[idx]
    if depth > 0 and len(idx) <= 2:
        net = ProtoNode(X[idx], ys, k)
        Z = net(X[idx])
    else:
        net, Z = fit_multiclass(X[idx], ys, k, h=cfg.h_root, epochs=cfg.root_epochs if depth == 0
                                else cfg.epochs, lr=cfg.lr, wd=cfg.wd, seed=cfg.seed + depth)
    wrong = Z.argmax(1) != ys
    if wrong.all() and depth > 0:  # no progress: memorise with a prototype layer
        net = ProtoNode(X[idx], ys, k)
        wrong = net(X[idx]).argmax(1) != ys
    node = GNode(net, depth, path)
    rec.add(path=path, depth=depth, kind=getattr(net, "kind", "mlp-k"), h=int(net.h), n=len(idx),
            npos=-1, err=int(wrong.sum()), params=int(net.n_params()))
    if cfg.verbose:
        print(f"{'  ' * depth}{path}: n={len(idx)} err={int(wrong.sum())}", flush=True)
    if wrong.any():
        node.E = grow_binary(X, idx, wrong.float(), cfg, rec, depth + 1, path + ".E")
        node.R = grow_gated(X, idx[wrong], y, k, cfg, rec, depth + 1, path + ".R")
    return node


class GatedTree:
    def __init__(self, k, cfg=None):
        self.k, self.cfg = k, cfg or GrowConfig()

    def fit(self, X, y):
        self.rec = Recorder()
        self.root = grow_gated(X, torch.arange(len(X)), torch.as_tensor(y), self.k, self.cfg, self.rec)
        return self

    @torch.no_grad()
    def predict(self, X):
        return self.root.predict(X)

    @torch.no_grad()
    def root_predict(self, X):
        return self.root.net(X).argmax(1)

    def nodes(self):
        """All binary nodes (the error detectors and their children), depth first."""
        out, stack = [], [self.root]
        while stack:
            g = stack.pop()
            if g.E is not None:
                out.extend(g.E.walk())
            if g.R is not None:
                stack.append(g.R)
        return out


# --------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------
def summarise(model, Xtr, ytr, Xte, yte):
    ytr = torch.as_tensor(ytr)
    yte = torch.as_tensor(yte)
    nodes = model.rec.nodes
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    rtr, rte = model.root_predict(Xtr), model.root_predict(Xte)
    changed = pte != rte
    out = dict(
        train_acc=float((ptr == ytr).float().mean()),
        test_acc=float((pte == yte).float().mean()),
        root_train_acc=float((rtr == ytr).float().mean()),
        root_test_acc=float((rte == yte).float().mean()),
        n_nodes=len(nodes),
        n_iso=sum(1 for n in nodes if n["kind"] in ("iso", "proto")),
        params=sum(n["params"] for n in nodes),
        max_depth=max(n["depth"] for n in nodes),
        test_changed=int(changed.sum()),
        test_changed_fixed=int(((pte == yte) & changed).sum()),
        test_changed_broken=int(((rte == yte) & changed).sum()),
        secs=time.time() - model.rec.t0,
    )
    return out
