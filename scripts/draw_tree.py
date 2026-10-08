"""Draw a grown tree (from a results/trees/*.json file) in the style of Fig. 4 of the note.

    python scripts/draw_tree.py results/trees/catsdogs_cnn_binary_h8_n0.0.json figures/tree_cd.pdf

Uses Graphviz (`dot`) when it is installed, otherwise a pure-matplotlib layout (same content), so
nothing extra has to be installed on Windows. `--matplotlib` forces the matplotlib version.
"""
import json
import shutil
import subprocess
import sys

COL = {"mlp": "#cfe2f3", "linear": "#d9ead3", "iso": "#f4cccc", "mlp-k": "#fff2cc", "proto": "#f4cccc"}


def _structure(nodes):
    """Return (label, colour) per path and (child, parent, edge label) triples."""
    paths = {n["path"] for n in nodes}
    labels, edges = {}, []
    for n in nodes:
        npos = f"+{n['npos']}" if n["npos"] >= 0 else f"err {n['err']}"
        lab = f"{n['path']}\n{n['kind']}{'' if n['kind'] in ('iso',) else ' h=' + str(n['h'])}\nn={n['n']} {npos}"
        if n["kind"] == "iso":
            lab += f"\n{n['h']} caps"
        labels[n["path"]] = (lab, COL.get(n["kind"], "#eeeeee"))
    for n in nodes:
        p = n["path"]
        if "." in p:
            parent, side = p.rsplit(".", 1)
        elif p.startswith("D"):
            parent, side = "R", p
        else:
            continue
        if parent not in paths:
            continue
        w = ""
        par = next(x for x in nodes if x["path"] == parent)
        if side == "A" and "wA" in par:
            w = f"w_A={par['wA']:.2f}"
        if side == "B" and "wB" in par:
            w = f"w_B={par['wB']:.2f}"
        if side.startswith("D") and "w_conn" in n:
            w = f"w={n['w_conn']:.1f}"
        edges.append((p, parent, w))
    return labels, edges


def draw_graphviz(labels, edges, dst):
    lines = ["digraph T {", "rankdir=BT;",
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9];',
             'edge [fontname="Helvetica", fontsize=8];']
    for p, (lab, col) in labels.items():
        lines.append(f'"{p}" [label="{lab.replace(chr(10), chr(92) + "n")}", fillcolor="{col}"];')
    for child, parent, w in edges:
        lines.append(f'"{child}" -> "{parent}" [label="{w}"];')
    lines.append("}")
    fmt = dst.rsplit(".", 1)[1]
    subprocess.run(["dot", f"-T{fmt}", "-o", dst], input="\n".join(lines).encode(), check=True)


def draw_matplotlib(labels, edges, dst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    children = {}
    for child, parent, _ in edges:
        children.setdefault(parent, []).append(child)
    has_parent = {c for c, _, _ in edges}
    roots = [p for p in labels if p not in has_parent]
    x, depth = {}, {}
    nxt = [0]

    def place(p, d):  # leaves get consecutive x; a parent sits above the middle of its children
        depth[p] = d
        kids = sorted(children.get(p, []))
        if not kids:
            x[p] = nxt[0]
            nxt[0] += 1
        else:
            for k in kids:
                place(k, d + 1)
            x[p] = (x[kids[0]] + x[kids[-1]]) / 2

    for r in roots:
        place(r, 0)
    width, height = max(nxt[0], 1), max(depth.values()) + 1
    fig, ax = plt.subplots(figsize=(max(6, 1.25 * width), 1.3 * height + 0.4))
    for child, parent, w in edges:
        ax.annotate("", xy=(x[parent], -depth[parent] - 0.22), xytext=(x[child], -depth[child] + 0.22),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.8))
        if w:
            ax.text((x[child] + x[parent]) / 2 + 0.05, -(depth[child] + depth[parent]) / 2, w, fontsize=6.5,
                    ha="left", va="center", color="#333333")
    for p, (lab, col) in labels.items():
        ax.text(x[p], -depth[p], lab, ha="center", va="center", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="#333333", lw=0.6))
    ax.set_xlim(-0.7, width - 0.3)
    ax.set_ylim(-height + 0.45, 0.45)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(dst, bbox_inches="tight")
    plt.close(fig)


def main(src, dst, max_nodes=80, force_mpl=False):
    nodes = json.load(open(src))["nodes"][:max_nodes]
    labels, edges = _structure(nodes)
    if shutil.which("dot") and not force_mpl:
        draw_graphviz(labels, edges, dst)
    else:
        draw_matplotlib(labels, edges, dst)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--matplotlib"]
    main(args[0], args[1], int(args[2]) if len(args) > 2 else 80, force_mpl="--matplotlib" in sys.argv)
