"""Grow a tree of small neural networks on cached trunk features.

    python scripts/run_tree.py --dataset cifar10 --feats resnet --model corrector --h 32
    python scripts/run_tree.py --dataset catsdogs --feats cnn --model binary
"""
import argparse
import os
import sys
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from treenn.experiment import load_feats, build, fit_and_summarise, save_json
from treenn.tree import GrowConfig

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--feats", default="resnet")
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--model", default="corrector")
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--h_root", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--root_epochs", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--max_depth", type=int, default=10)
    ap.add_argument("--iso_below", type=int, default=0)
    ap.add_argument("--pos_weight", default="auto")
    ap.add_argument("--fp_ratio", type=float, default=3.0)
    ap.add_argument("--loss", default="hinge")
    ap.add_argument("--resolve", default="closed")
    ap.add_argument("--lp_l1", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--subset", default=None, help="half_a | half_b (halves used by train_trunk --split half)")
    ap.add_argument("--select_trunk", action="store_true",
                    help="with several --feats: each node picks one trunk's features (per-trunk PCA)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    d = load_feats(a.dataset, a.feats, a.pca, a.n_train, a.noise, a.seed, separate=a.select_trunk,
                   subset=a.subset)
    cfg = GrowConfig(h=a.h, h_root=a.h_root, epochs=a.epochs, root_epochs=a.root_epochs, wd=a.wd,
                     max_depth=a.max_depth, iso_below=a.iso_below, fp_ratio=a.fp_ratio, pos_weight=a.pos_weight,
                     loss=a.loss, resolve=a.resolve, lp_l1=a.lp_l1, seed=a.seed, verbose=a.verbose,
                     blocks=d.get("blocks"))
    model = build(a.model, d["k"], cfg)
    s = fit_and_summarise(model, d)
    tag = a.tag or (f"{a.dataset}_{a.feats.replace(',', '+')}{'_sel' if a.select_trunk else ''}"
                     f"_{a.model}_h{a.h}_n{a.noise}")
    print(tag, s, flush=True)
    save_json(dict(args=vars(a), cfg=asdict(cfg), summary=s, nodes=model.rec.nodes),
              f"results/trees/{tag}.json")
    if a.save_model:
        fm = d["fmap"]
        torch.save(dict(model=model, fmap=[f.state() for f in fm] if isinstance(fm, list) else fm.state(),
                        idx=d["idx"], args=vars(a)),
                   f"results/trees/{tag}.pt")
