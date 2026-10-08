"""Minimising the size of a grown tree.

1. ``prune_weights``  -- decision-preserving magnitude pruning. Every MLP node is pruned to the
   highest sparsity at which, after a short masked fine-tune, it still makes *exactly the same
   decisions* on its own training samples. Because the node's decisions define its children's
   training sets (Table II), the whole tree stays 100% correct on the training set, and the
   connection weights are simply recomputed in closed form.

2. ``subtree_curve``  -- cost-complexity pruning: repeatedly remove the subtree that fixes the
   fewest training samples per parameter and record train/test accuracy against size.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from .node import MLPNode
from .tree import BinaryTree, CorrectorTree, OvRForest


def _layers(net):
    return [m for m in net.modules() if isinstance(m, torch.nn.Linear)]


def _nnz(net):
    return sum(int((l.weight != 0).sum()) + l.bias.numel() for l in _layers(net))


def _try_sparsity(net, X, target, multiclass, sparsity, epochs, lr):
    """Return a pruned copy that preserves decisions, or None."""
    cand = copy.deepcopy(net)
    lin = _layers(cand)
    W = torch.cat([l.weight.abs().flatten() for l in lin])
    k = int(sparsity * W.numel())
    if k <= 0:
        return cand
    thr = W.kthvalue(k).values
    masks = [(l.weight.abs() > thr).float() for l in lin]
    with torch.no_grad():
        for l, m in zip(lin, masks):
            l.weight.mul_(m)
    opt = torch.optim.Adam(cand.parameters(), lr=lr)
    s = 2 * target.float() - 1 if not multiclass else None
    for ep in range(epochs):
        with torch.no_grad():
            z = cand(X)
            ok = (z.argmax(1) == target) if multiclass else ((z >= 0) == target)
        if ok.all():
            return cand
        z = cand(X)
        if multiclass:
            loss = F.cross_entropy(z, target)
        else:
            loss = F.relu(1 - s * z).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for l, m in zip(lin, masks):
                l.weight.mul_(m)
    with torch.no_grad():
        z = cand(X)
        ok = (z.argmax(1) == target) if multiclass else ((z >= 0) == target)
    return cand if ok.all() else None


def prune_net(net, X, multiclass, levels=(0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99), epochs=150, lr=3e-3):
    with torch.no_grad():
        z = net(X)
        target = z.argmax(1) if multiclass else (z >= 0)
    best, best_s = net, 0.0
    for s in levels:
        c = _try_sparsity(net, X, target, multiclass, s, epochs, lr)
        if c is None:
            break
        best, best_s = c, s
    return best, best_s


def _recompute_binary(node, X, delta=1.0):
    with torch.no_grad():
        Xs = X[node.idx.long()]
        z = node.net(Xs)
        pred = z >= 0
        C3 = (~pred) & node.ypos
        C4 = pred & (~node.ypos)
        if node.A is not None:
            node.wA = float((delta - z[C3]).max())
        if node.B is not None:
            node.wB = -float((delta + z[C4]).max())


def prune_weights(model, X, y, verbose=False, **kw):
    """Decision-preserving pruning of every MLP node of a tree (in place on a deep copy)."""
    model = copy.deepcopy(model)
    y = torch.as_tensor(y)
    before = after = 0
    rows = []
    bnodes = model.nodes()
    if isinstance(model, CorrectorTree):
        b = _nnz(model.root)
        model.root, s = prune_net(model.root, X, True, **kw)
        a = _nnz(model.root)
        before, after = before + b, after + a
        rows.append(dict(path="R", depth=0, before=b, after=a, sparsity=s))
        with torch.no_grad():
            Z = model.root(X)
            wrong = Z.argmax(1) != y
            for c in model.det:
                pos = wrong & (y == c)
                model.w[c] = float((Z[pos].max(1).values - Z[pos][:, c]).max() + 1.0)
    # children first (deepest first) is not required: each node's decisions are preserved
    for node in bnodes:
        if not isinstance(node.net, MLPNode):
            p = node.net.n_params()
            before, after = before + p, after + p
            continue
        Xs = X[node.idx.long()]
        b = _nnz(node.net)
        node.net, s = prune_net(node.net, Xs, False, **kw)
        a = _nnz(node.net)
        before, after = before + b, after + a
        rows.append(dict(path=node.path, depth=node.depth, before=b, after=a, sparsity=s))
        if verbose:
            print(rows[-1], flush=True)
    for node in bnodes:
        _recompute_binary(node, X)
    return model, dict(before=before, after=after, rows=rows)


def input_features_used(model):
    """Number of distinct input features with a non-zero first-layer weight, per node."""
    out = []
    nets = [n.net for n in model.nodes()]
    if isinstance(model, CorrectorTree):
        nets = [model.root] + nets
    for net in nets:
        if isinstance(net, MLPNode):
            W = _layers(net)[0].weight
            out.append(int((W != 0).any(0).sum()))
    return out


def subtree_curve(model, Xtr, ytr, Xte, yte):
    """Greedy cost-complexity pruning of binary subtrees. Returns list of (params, train, test)."""
    model = copy.deepcopy(model)
    ytr, yte = torch.as_tensor(ytr), torch.as_tensor(yte)

    def size(n):
        return sum(m.net.n_params() for m in n.walk())

    def evaluate():
        return (float((model.predict(Xtr) == ytr).float().mean()),
                float((model.predict(Xte) == yte).float().mean()))

    def root_params():
        if isinstance(model, CorrectorTree):
            return model.root.n_params()
        return 0

    def all_params():
        return root_params() + sum(m.net.n_params() for m in model.nodes())

    curve = [dict(params=all_params(), nodes=len(model.nodes()), **dict(zip(["train", "test"], evaluate())))]
    while True:
        # candidate prunes: any child pointer (A/B of a binary node or a detector of the root)
        cands = []
        for node in model.nodes():
            for side in ("A", "B"):
                c = getattr(node, side)
                if c is not None:
                    cands.append((node, side, c))
        if isinstance(model, CorrectorTree):
            for k, d in model.det.items():
                cands.append((model, k, d))
        if not cands:
            break
        # score = samples fixed (positives of the child) per parameter of the subtree
        best = min(cands, key=lambda t: (int(t[2].ypos.sum()) / size(t[2])))
        parent, side, child = best
        if isinstance(side, str):
            setattr(parent, side, None)
        else:
            del model.det[side]
        tr, te = evaluate()
        curve.append(dict(params=all_params(), nodes=len(model.nodes()), train=tr, test=te))
    return curve
