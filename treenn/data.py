"""Dataset loading, normalisation, label noise and train-time augmentation."""
import os

import numpy as np
import torch

DATA_DIR = os.environ.get("TREENN_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
CD_CLASSES = ["cat", "dog"]

DATASETS = {
    "cifar10": dict(file="cifar10", classes=CIFAR_CLASSES),
    "catsdogs": dict(file="catsdogs64", classes=CD_CLASSES),
}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load(name, split):
    """Return uint8 images (N,H,W,3) and int64 labels (N,)."""
    f = os.path.join(DATA_DIR, f"{DATASETS[name]['file']}_{split}.npz")
    d = np.load(f)
    return d["X"], d["y"]


def num_classes(name):
    return len(DATASETS[name]["classes"])


def class_names(name):
    return DATASETS[name]["classes"]


def to_float(x_uint8):
    """uint8 NHWC numpy/tensor -> normalised float NCHW tensor."""
    x = torch.as_tensor(x_uint8).permute(0, 3, 1, 2).float().div_(255.0)
    return (x - MEAN) / STD


def denorm(x):
    """Inverse of to_float for visualisation: NCHW float -> NHWC in [0,1]."""
    x = x * STD + MEAN
    return x.clamp(0, 1).permute(0, 2, 3, 1)


def augment(x, pad=4):
    """Random crop with zero padding + random horizontal flip, on a float NCHW batch."""
    n, c, h, w = x.shape
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    i = torch.randint(0, 2 * pad + 1, (n,))
    j = torch.randint(0, 2 * pad + 1, (n,))
    rows = (i.view(n, 1) + torch.arange(h).view(1, h))  # n,h
    cols = (j.view(n, 1) + torch.arange(w).view(1, w))  # n,w
    out = xp[torch.arange(n).view(n, 1, 1), :, rows.view(n, h, 1), cols.view(n, 1, w)]  # n,h,w,c
    out = out.permute(0, 3, 1, 2)
    flip = torch.rand(n) < 0.5
    out[flip] = out[flip].flip(3)
    return out.contiguous()


def label_noise(y, rate, k, seed=0):
    """Symmetric label noise: a fraction `rate` of labels is replaced by a different random class.

    Returns the noisy labels and a boolean mask of corrupted samples.
    """
    if rate <= 0:
        return y.copy(), np.zeros(len(y), dtype=bool)
    rng = np.random.default_rng(seed)
    y2 = y.copy()
    flip = rng.random(len(y)) < rate
    shift = rng.integers(1, k, size=flip.sum())
    y2[flip] = (y[flip] + shift) % k
    return y2, flip
