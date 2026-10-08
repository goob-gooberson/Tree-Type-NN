"""Model-wise double descent: plain MLP vs tree-of-MLPs as the node width grows.

For every width H we train
  * a single MLP (d -> H -> K), full-batch Adam, fixed long budget, no regularisation;
  * a CorrectorTree that uses *that same MLP* as its root and binary nodes of width H,
    grown until it interpolates the (possibly noisy) training set.
and record train/test error, test cross-entropy, parameter count, and (with label noise)
how many of the corrupted labels were memorised.

    python scripts/run_double_descent.py --dataset cifar10 --feats resnet --n_train 4000 --noise 0.2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from treenn.experiment import load_feats
from treenn.node import MLPNode
from treenn.tree import CorrectorTree, GrowConfig, summarise


def train_mlp(X, y, k, H, epochs, lr, seed, Xte, yte, log_every=0):
    torch.manual_seed(seed)
    m = MLPNode(X.shape[1], H, k)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    curve = []
    for ep in range(epochs):
        loss = F.cross_entropy(m(X), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_every and (ep + 1) % log_every == 0:
            with torch.no_grad():
                curve.append(dict(epoch=ep + 1, train_err=float((m(X).argmax(1) != y).float().mean()),
                                  test_err=float((m(Xte).argmax(1) != yte).float().mean())))
    return m, curve


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--feats", default="resnet")
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--widths", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_tree", action="store_true")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    d = load_feats(a.dataset, a.feats, a.pca, a.n_train, a.noise, a.seed)
    X, y, Xte, yte, k = d["Xtr"], d["ytr"], d["Xte"], d["yte"], d["k"]
    flipped = torch.as_tensor(d["flipped"])
    out = a.out or f"results/dd/{a.dataset}_{a.feats}_n{a.n_train}_noise{a.noise}_s{a.seed}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["model"], r["H"]))
    for H in [int(w) for w in a.widths.split(",")]:
        if ("mlp", H) in done and (a.skip_tree or ("tree", H) in done):
            print(f"H={H}: already in {out}, skipped", flush=True)
            continue
        t0 = time.time()
        m, _ = train_mlp(X, y, k, H, a.epochs, a.lr, a.seed, Xte, yte)
        if ("mlp", H) not in done:
            with torch.no_grad():
                ptr, Zte = m(X).argmax(1), m(Xte)
            r = dict(model="mlp", H=H, params=m.n_params(),
                     train_err=float((ptr != y).float().mean()),
                     test_err=float((Zte.argmax(1) != yte).float().mean()),
                     test_loss=float(F.cross_entropy(Zte, yte)),
                     noisy_memorised=float((ptr[flipped] == y[flipped]).float().mean()) if flipped.any() else None,
                     secs=time.time() - t0)
            print(r, flush=True)
            open(out, "a").write(json.dumps(r) + "\n")
        if not a.skip_tree and ("tree", H) not in done:
            cfg = GrowConfig(h=H, h_root=H, root_epochs=a.epochs // 4, epochs=300, seed=a.seed)
            t = CorrectorTree(k, cfg).fit(X, y, root=m)  # same trained MLP as the root
            s = summarise(t, X, y, Xte, yte)
            ptr = t.predict(X)
            r = dict(model="tree", H=H, params=s["params"], n_nodes=s["n_nodes"], n_iso=s["n_iso"],
                     max_depth=s["max_depth"], train_err=1 - s["train_acc"], test_err=1 - s["test_acc"],
                     root_test_err=1 - s["root_test_acc"], root_train_err=1 - s["root_train_acc"],
                     test_changed=s["test_changed"], fixed=s["test_changed_fixed"],
                     broken=s["test_changed_broken"],
                     noisy_memorised=float((ptr[flipped] == y[flipped]).float().mean()) if flipped.any() else None,
                     secs=s["secs"])
            print(r, flush=True)
            open(out, "a").write(json.dumps(r) + "\n")
