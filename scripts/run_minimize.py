"""Minimise the size of a grown tree.

    python scripts/run_minimize.py --tree results/trees/cifar10_resnet_corrector_h32_n0.0.pt
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from treenn.experiment import load_feats
from treenn.minimize import prune_weights, subtree_curve, input_features_used

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    ap.add_argument("--levels", default="0.5,0.8,0.9,0.95,0.98,0.99")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--skip_prune", action="store_true")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    ck = torch.load(a.tree, weights_only=False)
    args, model = ck["args"], ck["model"]
    tag = os.path.basename(a.tree).replace(".pt", "")
    d = load_feats(args["dataset"], args["feats"], args["pca"], args["n_train"], args["noise"], args["seed"],
                   separate=args.get("select_trunk", False), subset=args.get("subset"))
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    out = dict(tag=tag)
    t0 = time.time()
    out["subtree_curve"] = subtree_curve(model, Xtr, ytr, Xte, yte)
    print("subtree curve", len(out["subtree_curve"]), f"{time.time() - t0:.0f}s", flush=True)
    if not a.skip_prune:
        pm, info = prune_weights(model, Xtr, ytr, verbose=True,
                                 levels=tuple(float(x) for x in a.levels.split(",")), epochs=a.epochs)
        out["prune"] = dict(
            nnz_before=info["before"], nnz_after=info["after"], rows=info["rows"],
            train_acc=float((pm.predict(Xtr) == ytr).float().mean()),
            test_acc=float((pm.predict(Xte) == yte).float().mean()),
            test_acc_before=float((model.predict(Xte) == yte).float().mean()),
            features_used_before=input_features_used(model), features_used_after=input_features_used(pm),
            secs=time.time() - t0)
        print({k: v for k, v in out["prune"].items() if k != "rows"}, flush=True)
    json.dump(out, open(f"results/minimize_{tag}.json", "w"), indent=1)
