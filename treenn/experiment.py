"""Shared helpers for the experiment scripts."""
import json
import os

import numpy as np
import torch

from . import data as D
from .trunks import FeatureMap
from .tree import BinaryTree, OvRForest, CorrectorTree, GatedTree, GrowConfig, summarise

FEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "feats")


def load_feats(dataset, feats, pca=128, n_train=0, noise=0.0, seed=0, separate=False, subset=None):
    """Load cached trunk features (comma-separated list => concatenation of trunks).

    Returns dict with torch tensors Xtr, ytr (possibly noisy), ytr_clean, Xte, yte, fmap, noisy mask
    and the subset indices used.
    """
    Ftr, Fte = [], []
    for f in feats.split(","):
        d = np.load(os.path.join(FEAT_DIR, f"{dataset}_{f}.npz"))
        Ftr.append(d["Ftrain"])
        Fte.append(d["Ftest"])
        ytr, yte = d["ytrain"], d["ytest"]
    Ftr, Fte = np.concatenate(Ftr, 1), np.concatenate(Fte, 1)
    idx = np.arange(len(Ftr))
    if subset in ("half_a", "half_b"):  # the two halves used by train_trunk.py --split half
        perm = np.random.default_rng(123).permutation(len(Ftr))
        half = perm[: len(Ftr) // 2] if subset == "half_a" else perm[len(Ftr) // 2:]
        idx = np.sort(half)
    if n_train and n_train < len(idx):
        idx = np.sort(np.random.default_rng(seed).permutation(idx)[:n_train])
    Ftr, ytr = Ftr[idx], ytr[idx]
    k = D.num_classes(dataset)
    ytr_noisy, flipped = D.label_noise(ytr, noise, k, seed=seed + 1)
    if separate:  # one FeatureMap (standardise + PCA) per trunk, concatenated -> column blocks
        sizes = [np.load(os.path.join(FEAT_DIR, f"{dataset}_{f}.npz"))["Ftest"].shape[1]
                 for f in feats.split(",")]
        cuts = np.cumsum([0] + sizes)
        fmaps = [FeatureMap(Ftr[:, cuts[i]:cuts[i + 1]], pca) for i in range(len(sizes))]
        Xtr = torch.cat([fm(Ftr[:, cuts[i]:cuts[i + 1]]) for i, fm in enumerate(fmaps)], 1)
        Xte = torch.cat([fm(Fte[:, cuts[i]:cuts[i + 1]]) for i, fm in enumerate(fmaps)], 1)
        dims = np.cumsum([0] + [fm.dim for fm in fmaps])
        blocks = [(int(dims[i]), int(dims[i + 1])) for i in range(len(fmaps))]
        return dict(Xtr=Xtr, ytr=torch.as_tensor(ytr_noisy), ytr_clean=torch.as_tensor(ytr), Xte=Xte,
                    yte=torch.as_tensor(yte), fmap=fmaps, blocks=blocks, flipped=flipped, idx=idx, k=k)
    fmap = FeatureMap(Ftr, pca)
    return dict(Xtr=fmap(Ftr), ytr=torch.as_tensor(ytr_noisy), ytr_clean=torch.as_tensor(ytr),
                Xte=fmap(Fte), yte=torch.as_tensor(yte), fmap=fmap, flipped=flipped, idx=idx, k=k)


def build(model, k, cfg):
    if model == "binary":
        assert k == 2
        return BinaryTree(cfg)
    if model == "ovr":
        return OvRForest(k, cfg)
    if model == "corrector":
        return CorrectorTree(k, cfg)
    if model == "gated":
        return GatedTree(k, cfg)
    raise ValueError(model)


def fit_and_summarise(model, d):
    model.fit(d["Xtr"], d["ytr"])
    s = summarise(model, d["Xtr"], d["ytr"], d["Xte"], d["yte"])
    if d["flipped"].any():
        p = model.predict(d["Xtr"]).numpy()
        f = d["flipped"]
        s["train_acc_on_noisy_labels"] = float((p[f] == d["ytr"].numpy()[f]).mean())
        s["train_acc_vs_clean_labels"] = float((p == d["ytr_clean"].numpy()).mean())
    return s


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
