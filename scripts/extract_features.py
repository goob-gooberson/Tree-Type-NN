"""Run a frozen trunk over a dataset and cache the pooled features.

    python scripts/extract_features.py --dataset cifar10 --trunk cnn
    python scripts/extract_features.py --dataset cifar10 --trunk resnet --res 128
    python scripts/extract_features.py --dataset cifar10 --trunk pixels
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from treenn import data as D
from treenn.trunks import SmallCNN, ResNet18Trunk, PixelTrunk, extract

RESNET_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "data", "resnet18_a1_0-d63eafa0.pth")


def build_trunk(dataset, trunk, res=128, split="all", width=32):
    if trunk == "cnn":
        X, _ = D.load(dataset, "test")
        m = SmallCNN(D.num_classes(dataset), width, X.shape[1])
        m.load_state_dict(torch.load(f"results/trunks/{dataset}_cnn{width}_{split}.pt"))
        return m.eval()
    if trunk == "resnet":
        return ResNet18Trunk(RESNET_WEIGHTS, res)
    if trunk == "pixels":
        return PixelTrunk(16)
    raise ValueError(trunk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--trunk", default="cnn")
    ap.add_argument("--split", default="all")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (an NVIDIA GPU, e.g. on Google Colab)")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    os.makedirs("data/feats", exist_ok=True)
    trunk = build_trunk(a.dataset, a.trunk, a.res, a.split)
    name = a.trunk if a.trunk != "cnn" or a.split == "all" else f"cnn_{a.split}"
    out = {}
    for split in ["train", "test"]:
        X, y = D.load(a.dataset, split)
        t0 = time.time()
        out["F" + split] = extract(trunk, X, log=print, device=a.device)
        out["y" + split] = y
        print(split, out["F" + split].shape, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(f"data/feats/{a.dataset}_{name}.npz", **out)
