"""Convert the raw image folders into compact uint8 numpy archives.

CIFAR-10   : folder layout  <root>/{train,test}/<class>/*.jpg  (32x32)
Cats vs Dogs: folder layout <root>/data/{train,test}/{cats,dogs}/*.jpg (arbitrary size)

Cats vs Dogs images are resized (shorter side -> S) and centre-cropped to SxS.

Usage:
    python scripts/prepare_data.py --cifar_root data/cifar_git --cd_root data/catsdogs_git \
        --out data --cd_size 64
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


def _load_cifar(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _load_cd(args):
    path, size = args
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = size / min(w, h)
            im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
            w, h = im.size
            l, t = (w - size) // 2, (h - size) // 2
            return np.asarray(im.crop((l, t, l + size, t + size)), dtype=np.uint8)
    except Exception as e:  # a few Kaggle files are corrupt
        print("skip", path, e)
        return None


def prepare_cifar(root, out):
    for split in ["train", "test"]:
        paths, labels = [], []
        for k, c in enumerate(CIFAR_CLASSES):
            d = os.path.join(root, split, c)
            for f in sorted(os.listdir(d)):
                paths.append(os.path.join(d, f))
                labels.append(k)
        with ProcessPoolExecutor() as ex:
            imgs = list(ex.map(_load_cifar, paths, chunksize=512))
        X = np.stack(imgs)
        y = np.array(labels, dtype=np.int64)
        np.savez(os.path.join(out, f"cifar10_{split}.npz"), X=X, y=y)
        print("cifar10", split, X.shape, np.bincount(y))


def prepare_catsdogs(root, out, size):
    for split in ["train", "test"]:
        paths, labels = [], []
        for k, c in enumerate(["cats", "dogs"]):
            d = os.path.join(root, "data", split, c)
            for f in sorted(os.listdir(d)):
                paths.append(os.path.join(d, f))
                labels.append(k)
        with ProcessPoolExecutor() as ex:
            imgs = list(ex.map(_load_cd, [(p, size) for p in paths], chunksize=128))
        keep = [i for i, im in enumerate(imgs) if im is not None]
        X = np.stack([imgs[i] for i in keep])
        y = np.array([labels[i] for i in keep], dtype=np.int64)
        np.savez(os.path.join(out, f"catsdogs{size}_{split}.npz"), X=X, y=y)
        print("catsdogs", split, X.shape, np.bincount(y))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar_root", default="data/cifar_git")
    ap.add_argument("--cd_root", default="data/catsdogs_git")
    ap.add_argument("--out", default="data")
    ap.add_argument("--cd_size", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prepare_cifar(a.cifar_root, a.out)
    prepare_catsdogs(a.cd_root, a.out, a.cd_size)
