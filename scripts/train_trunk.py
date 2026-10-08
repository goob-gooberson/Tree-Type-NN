"""Train a SmallCNN trunk from scratch (then frozen and used as the tree's feature extractor).

    python scripts/train_trunk.py --dataset cifar10 --epochs 15
    python scripts/train_trunk.py --dataset catsdogs --epochs 15
    python scripts/train_trunk.py --dataset cifar10 --split half      # trunk sees only half the data
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from treenn import data as D
from treenn.trunks import SmallCNN, train_cnn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--split", default="all", help="all | half (first half of a fixed permutation)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    ap.add_argument("--out", default="results/trunks")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (an NVIDIA GPU, e.g. on Google Colab)")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    Xtr, ytr = D.load(a.dataset, "train")
    Xte, yte = D.load(a.dataset, "test")
    if a.split == "half":
        perm = np.random.default_rng(123).permutation(len(Xtr))
        keep = np.sort(perm[: len(Xtr) // 2])
        Xtr, ytr = Xtr[keep], ytr[keep]
    model = SmallCNN(D.num_classes(a.dataset), a.width, Xtr.shape[1])
    tag = f"{a.dataset}_cnn{a.width}_{a.split}"
    logf = open(os.path.join(a.out, tag + ".log"), "w")

    def log(r):
        print(r, flush=True)
        logf.write(json.dumps(r) + "\n")
        logf.flush()

    hist = train_cnn(model, Xtr, ytr, Xte, yte, epochs=a.epochs, bs=a.bs, lr=a.lr, log=log, device=a.device)
    torch.save(model.state_dict(), os.path.join(a.out, tag + ".pt"))
    json.dump(dict(args=vars(a), history=hist), open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
