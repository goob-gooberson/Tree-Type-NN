"""Visualise and quantify receptive fields of tree nodes.

    python scripts/receptive_fields.py --tree results/trees/cifar10_resnet_corrector_h32_n0.0.pt
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from treenn import data as D
from treenn import receptive as RF
from treenn.experiment import load_feats
from treenn.node import MLPNode
from treenn.tree import CorrectorTree, BinaryTree, OvRForest
from extract_features import build_trunk


class ClassLogit(torch.nn.Module):
    def __init__(self, net, k):
        super().__init__()
        self.net, self.k = net, k

    def forward(self, x):
        return self.net(x)[:, self.k]


def analyse(name, net, trunk, fmap, imgs_tr, Xtr, pos_idx, data_idx, Xte, n_grad=32):
    """Receptive-field statistics of one node network."""
    pos_idx = pos_idx[:n_grad]
    g, z = RF.input_gradients(trunk, fmap, net, imgs_tr[pos_idx.numpy()])
    sal = RF.saliency_maps(g)
    with torch.no_grad():
        ztr = net(Xtr[data_idx])
        zte = net(Xte)
    top = data_idx[torch.topk(ztr, min(6, len(ztr))).indices]
    g1, _ = RF.input_gradients(trunk, fmap, net, imgs_tr[top[:1].numpy()])
    return dict(
        top_sal=RF.saliency_maps(g1)[0],
        name=name,
        template=g.mean(0),
        saliency=sal.mean(0),
        top=top,
        area50=float(RF.area50(sal).nanmean()),
        frac_zero_grad=float(RF.area50(sal).isnan().float().mean()),
        area50_meanmap=float(RF.area50(sal.mean(0, keepdim=True))[0]),
        centre=float(RF.centre_mass(sal).nanmean()),
        fire_train=float((ztr >= 0).float().mean()),
        fire_test=float((zte >= 0).float().mean()),
        n_pos=int(len(pos_idx)),
        grad_input_cos=float(torch.nn.functional.cosine_similarity(
            g.flatten(1), (D.to_float(imgs_tr[pos_idx.numpy()])).flatten(1)).mean()),
    )


def plot_panel(rows, imgs_tr, path, title=None):
    n = len(rows)
    fig, axes = plt.subplots(n, 8, figsize=(10, 1.3 * n + 0.35))
    if n == 1:
        axes = axes[None]
    for r, row in enumerate(rows):
        for j in range(6):
            ax = axes[r, j]
            ax.axis("off")
            if j < len(row["top"]):
                ax.imshow(imgs_tr[int(row["top"][j])])
        axes[r, 0].text(-0.08, 0.5, row["name"].replace(" (", "\n("), transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
        axes[r, 6].imshow(imgs_tr[int(row["top"][0])])
        sal1 = row["top_sal"].numpy()
        sal1 = np.clip(sal1 / (np.percentile(sal1, 99) + 1e-12), 0, 1)
        axes[r, 6].imshow(sal1, cmap="inferno", alpha=0.7,
                          extent=(-0.5, imgs_tr.shape[2] - 0.5, imgs_tr.shape[1] - 0.5, -0.5))
        axes[r, 6].axis("off")
        axes[r, 7].imshow(row["saliency"].numpy(), cmap="inferno")
        axes[r, 7].axis("off")
    axes[0, 2].set_title("top-activating training images", fontsize=10, x=1.1)
    axes[0, 6].set_title("|grad| top-1", fontsize=10)
    axes[0, 7].set_title("mean |grad|", fontsize=10)
    fig.subplots_adjust(left=0.13, right=0.995, top=1 - 0.3 / (1.3 * n + 0.35), bottom=0.005,
                        wspace=0.05, hspace=0.06)
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--max_nodes", type=int, default=24)
    ap.add_argument("--per_depth", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = all cores)")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    ck = torch.load(a.tree, weights_only=False)
    args, model = ck["args"], ck["model"]
    tag = os.path.basename(a.tree).replace(".pt", "")
    d = load_feats(args["dataset"], args["feats"], args["pca"], args["n_train"], args["noise"], args["seed"],
                   subset=args.get("subset"))
    fmap = d["fmap"]
    trunk = build_trunk(args["dataset"], args["feats"].replace("cnn_half", "cnn"),
                        split="half" if "half" in args["feats"] else "all")
    for p in trunk.parameters():
        p.requires_grad_(False)
    imgs_tr = D.load(args["dataset"], "train")[0][d["idx"]]
    Xtr, Xte, ytr = d["Xtr"], d["Xte"], d["ytr"]
    classes = D.class_names(args["dataset"])
    rows_root, rows_nodes = [], []

    if isinstance(model, CorrectorTree):
        with torch.no_grad():
            pred = model.root(Xtr).argmax(1)
        for k in range(d["k"]):
            pos = torch.where((pred == ytr) & (ytr == k))[0]
            pos = pos[torch.randperm(len(pos), generator=torch.Generator().manual_seed(0))]
            rows_root.append(analyse(f"root:{classes[k]}", ClassLogit(model.root, k), trunk, fmap, imgs_tr,
                                     Xtr, pos, torch.where(ytr == k)[0], Xte))
            rows_root[-1].update(depth=0, kind="root")
    nodes = [n for n in model.nodes() if isinstance(n.net, MLPNode) or True]
    # choose nodes: all shallow nodes then a spread of deeper ones
    # stratify by depth: the largest `per_depth` nodes at every depth
    nodes.sort(key=lambda n: (n.depth, -int(n.ypos.sum())))
    chosen, count = [], {}
    for n in nodes:
        if count.get(n.depth, 0) < a.per_depth:
            chosen.append(n)
            count[n.depth] = count.get(n.depth, 0) + 1
    chosen = chosen[: a.max_nodes]
    for n in chosen:
        idx = n.idx.long()
        pos = idx[n.ypos]
        r = analyse(f"{n.path} ({n.net.kind}, {int(n.ypos.sum())}+)", n.net, trunk, fmap, imgs_tr, Xtr, pos,
                    idx, Xte)
        r.update(depth=n.depth, kind=n.net.kind, path=n.path)
        rows_nodes.append(r)
        print(r["name"], {k: r[k] for k in ["area50", "centre", "fire_train", "fire_test", "grad_input_cos"]},
              flush=True)

    os.makedirs(a.out, exist_ok=True)
    if rows_root:
        plot_panel(rows_root, imgs_tr, f"{a.out}/rf_{tag}_root.png", "Root (K-way) node: per-class receptive fields")
    for i in range(0, len(rows_nodes), 8):
        plot_panel(rows_nodes[i:i + 8], imgs_tr, f"{a.out}/rf_{tag}_nodes{i // 8}.png",
                   "Binary nodes: top-activating images, saliency of the top image, mean saliency over the node's positives")
    keep = ["name", "depth", "kind", "area50", "area50_meanmap", "centre", "fire_train", "fire_test", "n_pos",
            "grad_input_cos", "frac_zero_grad"]
    stats = [{k: r.get(k) for k in keep} for r in rows_root + rows_nodes]
    json.dump(stats, open(f"results/rf_{tag}.json", "w"), indent=1)
    torch.save(dict(rows=[{k: v for k, v in r.items()} for r in rows_root + rows_nodes]),
               f"results/rf_{tag}_maps.pt")
