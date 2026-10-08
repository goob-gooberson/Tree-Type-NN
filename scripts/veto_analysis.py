"""How do nodes recognise their positives? For every MLP node of a tree, measure the fraction of its
positives (and negatives) on which *all* of its hidden units are silent, i.e. the node fires purely by
its output bias ("fire unless vetoed"), together with how often the node fires on test images.

    python scripts/veto_analysis.py results/trees/<tag>.pt [...]
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from treenn.experiment import load_feats

if __name__ == "__main__":
    out = {}
    for path in sys.argv[1:]:
        ck = torch.load(path, weights_only=False)
        m, a = ck["model"], ck["args"]
        d = load_feats(a["dataset"], a["feats"], a["pca"], a["n_train"], a["noise"], a["seed"],
                       separate=a.get("select_trunk", False), subset=a.get("subset"))
        X, Xte = d["Xtr"], d["Xte"]
        rows = []
        with torch.no_grad():
            for n in m.nodes():
                if n.net.kind != "mlp":
                    continue
                idx = n.idx.long()
                pos, neg = idx[n.ypos], idx[~n.ypos]
                hp, hn = n.net.hidden(X[pos]), n.net.hidden(X[neg])
                rows.append(dict(path=n.path, depth=n.depth, npos=len(pos), nneg=len(neg),
                                 silent_pos=float((hp.sum(1) == 0).float().mean()) if len(pos) else None,
                                 silent_neg=float((hn.sum(1) == 0).float().mean()) if len(neg) else None,
                                 bias=float(n.net.net[-1].bias),
                                 fire_test=float((n.net(Xte) >= 0).float().mean())))
        tag = os.path.basename(path).replace(".pt", "")
        out[tag] = rows
        print(tag, flush=True)
    os.makedirs("results", exist_ok=True)
    old = json.load(open("results/veto.json")) if os.path.exists("results/veto.json") else {}
    old.update(out)
    json.dump(old, open("results/veto.json", "w"), indent=1)
