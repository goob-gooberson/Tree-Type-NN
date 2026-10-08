"""Download everything the experiments need into data/ (works on Windows, macOS and Linux).

    python scripts/download_data.py

* CIFAR-10 as image folders      -> data/cifar_git      (git clone, ~260 MB, 60 000 files)
* Kaggle Cats vs Dogs            -> data/catsdogs_git   (git clone, ~800 MB, 25 000 files)
* ImageNet ResNet-18 weights     -> data/resnet18_a1_0-d63eafa0.pth (47 MB)

Anything already present is skipped, so the script can simply be run again after an interruption
(delete a half-cloned folder first). Needs `git` on the PATH.
"""
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "data")

REPOS = {
    "cifar_git": "https://github.com/YoongiKim/CIFAR-10-images.git",
    "catsdogs_git": "https://github.com/laxmimerit/dog-cat-full-dataset.git",
}
WEIGHTS = ("resnet18_a1_0-d63eafa0.pth",
           "https://github.com/huggingface/pytorch-image-models/releases/download/"
           "v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth")


def clone(name, url):
    dst = os.path.join(DATA, name)
    if os.path.isdir(dst) and os.listdir(dst):
        print(f"[skip] {name} already exists")
        return
    if shutil.which("git") is None:
        sys.exit("git was not found. Install it from https://git-scm.com/download/win and open a new terminal.")
    print(f"[clone] {url} -> data/{name}  (this can take several minutes)", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", url, dst], check=True)


def download(fname, url):
    dst = os.path.join(DATA, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 40_000_000:
        print(f"[skip] {fname} already exists")
        return
    print(f"[download] {fname} (47 MB)", flush=True)
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    for name, url in REPOS.items():
        clone(name, url)
    download(*WEIGHTS)
    print("done: data/ now holds", sorted(os.listdir(DATA)))
