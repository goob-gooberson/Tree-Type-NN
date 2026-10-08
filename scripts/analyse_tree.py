"""Per-node analysis of a saved tree: who does each node memorise, and what does it do at test time?

For every binary node we report its depth, number of positives, the fraction of its positives whose
training label was corrupted (when the run used label noise), how often the node's *own* network
fires on its training data and on the test set, and -- for the children of the root -- how many test
predictions their corrections fix or break.

    python scripts/analyse_tree.py --tree results/trees/<tag>.pt
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from treenn.experiment import load_feats
from treenn.tree import CorrectorTree, BinaryTree

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    ck = torch.load(a.tree, weights_only=False)
    args, model = ck["args"], ck["model"]
    tag = os.path.basename(a.tree).replace(".pt", "")
    d = load_feats(args["dataset"], args["feats"], args["pca"], args["n_train"], args["noise"], args["seed"],
                   separate=args.get("select_trunk", False), subset=args.get("subset"))
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    flipped = torch.as_tensor(d["flipped"])
    rows = []
    with torch.no_grad():
        for n in model.nodes():
            idx = n.idx.long()
            pos = idx[n.ypos]
            own_tr = (n.net(Xtr[idx]) >= 0).float().mean().item()
            own_te = (n.net(Xte) >= 0).float().mean().item()
            sub_te = n.fire(Xte).mean().item()
            rows.append(dict(path=n.path, depth=n.depth, kind=n.net.kind, n=len(idx), npos=len(pos),
                             noisy_frac_pos=float(flipped[pos].float().mean()) if flipped.any() and len(pos) else None,
                             own_fire_train=own_tr, own_fire_test=own_te, subtree_fire_test=sub_te,
                             cols=list(n.net.cols) if getattr(n.net, "cols", None) is not None else None))
        out = dict(tag=tag, nodes=rows, base_noise=float(flipped.float().mean()))
        if isinstance(model, CorrectorTree):
            Z0 = model.root(Xte)
            p0 = Z0.argmax(1)
            per = []
            for c, det in model.det.items():
                Z = Z0.clone()
                Z[:, c] += model.w[c] * det.fire(Xte)
                p = Z.argmax(1)
                ch = p != p0
                per.append(dict(cls=c, changed=int(ch.sum()), fixed=int((ch & (p == yte)).sum()),
                                broken=int((ch & (p0 == yte)).sum()), fires=int(det.fire(Xte).sum())))
            out["detectors"] = per
            if flipped.any():
                with torch.no_grad():
                    wrong_root = model.root(Xtr).argmax(1) != ytr
                out["noisy_frac_root_errors"] = float(flipped[wrong_root].float().mean())
                out["root_memorised_noisy"] = float((~wrong_root[flipped]).float().mean())
    json.dump(out, open(f"results/analysis_{tag}.json", "w"), indent=1)
    by_depth = {}
    for r in rows:
        by_depth.setdefault(r["depth"], []).append(r)
    for dpt, rs in sorted(by_depth.items()):
        nf = [r["noisy_frac_pos"] for r in rs if r["noisy_frac_pos"] is not None]
        print(dpt, len(rs), "pos", sum(r["npos"] for r in rs),
              "noisy", round(float(np.mean(nf)), 3) if nf else None,
              "own_fire_test", round(float(np.mean([r["own_fire_test"] for r in rs])), 4))
    if "detectors" in out:
        print(out["detectors"])
