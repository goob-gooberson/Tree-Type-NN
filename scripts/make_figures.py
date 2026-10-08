"""Build every report figure and LaTeX table from results/.

    python scripts/make_figures.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FIG = "figures"
TAB = "report/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def tree_json(tag):
    p = f"results/trees/{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


# --------------------------------------------------------------------------------------
def fig_double_descent():
    runs = [("results/dd/cifar10_resnet_n4000_noise0.2_s0.jsonl", "20% label noise"),
            ("results/dd/cifar10_resnet_n4000_noise0.0_s0.jsonl", "clean labels")]
    runs = [(p, t) for p, t in runs if os.path.exists(p)]
    if not runs:
        return
    fig, axes = plt.subplots(2, len(runs), figsize=(3.6 * len(runs), 5.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1]), squeeze=False)
    for j, (p, title) in enumerate(runs):
        R = jl(p)
        mlp = sorted([r for r in R if r["model"] == "mlp"], key=lambda r: r["H"])
        tr = sorted([r for r in R if r["model"] == "tree"], key=lambda r: r["H"])
        ax = axes[0, j]
        ax.plot([r["H"] for r in mlp], [100 * r["test_err"] for r in mlp], "-o", color=BLUE, label="single MLP (= root)")
        ax.plot([r["H"] for r in tr], [100 * r["test_err"] for r in tr], "-s", color=ORANGE, label="tree (100% train acc.)")
        thr = next((r["H"] for r in mlp if r["train_err"] == 0), None)
        if thr:
            ax.axvline(thr, color=INK2, lw=1, ls="--")
            ax.text(thr * 0.9, ax.get_ylim()[1] * 0.98, "MLP interpolation\nthreshold", va="top", ha="right",
                    fontsize=7.5, color=INK2)
        ax.set_xscale("log", base=2)
        ax.set_title(f"CIFAR-10, n=4000, {title}", fontsize=9)
        if j == 0:
            ax.set_ylabel("test error (%)")
            ax.legend(fontsize=8, loc="center right", bbox_to_anchor=(1.0, 0.66))
        ax2 = axes[1, j]
        ax2.plot([r["H"] for r in mlp], [100 * r["train_err"] for r in mlp], "-o", color=BLUE)
        ax2.plot([r["H"] for r in tr], [100 * r["train_err"] for r in tr], "-s", color=ORANGE)
        ax2.set_xlabel("node width H (hidden units of root / of every node)")
        if j == 0:
            ax2.set_ylabel("train error (%)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/double_descent.pdf")
    plt.close(fig)

    # test error vs number of parameters
    fig, axes = plt.subplots(1, len(runs), figsize=(3.6 * len(runs), 2.9), squeeze=False)
    for j, (p, title) in enumerate(runs):
        R = jl(p)
        mlp = sorted([r for r in R if r["model"] == "mlp"], key=lambda r: r["params"])
        tr = sorted([r for r in R if r["model"] == "tree" and r["n_nodes"] > 1], key=lambda r: r["H"])
        ax = axes[0, j]
        ax.plot([r["params"] for r in mlp], [100 * r["test_err"] for r in mlp], "-o", color=BLUE, label="single MLP")
        ax.scatter([r["params"] for r in tr], [100 * r["test_err"] for r in tr], marker="s", color=ORANGE, zorder=3,
                   label="tree (root width H < threshold)")
        for r in tr:
            if r["H"] <= 2 or r is tr[-1]:
                ax.annotate(f"H={r['H']}", (r["params"], 100 * r["test_err"]), fontsize=7, color=INK2,
                            xytext=(4, 2), textcoords="offset points")
        ax.axvline(4000 * 10, color=INK2, lw=1, ls=":")
        ax.text(4000 * 10 * 1.1, ax.get_ylim()[0] + 2, "n·K", fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlabel("number of parameters")
        ax.set_title(title, fontsize=9)
        if j == 0:
            ax.set_ylabel("test error (%)")
            ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/double_descent_params.pdf")
    plt.close(fig)




# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------
def _row(label, tag, extra=""):
    J = tree_json(tag)
    if J is None:
        return f"{label} & \\multicolumn{{8}}{{c}}{{(missing: {tag})}}\\\\"
    s = J["summary"]
    pct = lambda v: f"{100 * v:.1f}"
    return (f"{label}{extra} & {pct(s['root_train_acc'])} & {pct(s['root_test_acc'])} & {pct(s['train_acc'])} & "
            f"\\textbf{{{pct(s['test_acc'])}}} & {s['n_nodes']} ({s['n_iso']}) & {s['params'] / 1000:.1f}k & "
            f"{s['max_depth']} & {s['test_changed_fixed']}/{s['test_changed_broken']}\\\\")


HEAD = ("\\begin{tabular}{@{}lrrrrrrrr@{}}\\toprule\n"
        " & \\multicolumn{2}{c}{root acc. (\\%)} & \\multicolumn{2}{c}{tree acc. (\\%)} & nodes & & & test fixed/\\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "configuration & train & test & train & test & (isol.) & params & depth & broken\\\\\\midrule\n")
FOOT = "\\bottomrule\\end{tabular}\n"


def tab_main():
    cifar = [("pixels", "cifar10_pixels_corrector_h8_n0.0"),
             ("scratch CNN", "cifar10_cnn_corrector_h8_n0.0"),
             ("ResNet-18 (ImageNet)", "cifar10_resnet_corrector_h8_n0.0"),
             ("CNN + ResNet, per-node trunk selection", "cifar10_cnn+resnet_sel_corrector_h8_n0.0")]
    cd = [("pixels", "catsdogs_pixels_binary_h8_n0.0"),
          ("scratch CNN", "catsdogs_cnn_binary_h8_n0.0"),
          ("ResNet-18 (ImageNet)", "catsdogs_resnet_binary_h8_n0.0"),
          ("CNN + ResNet, per-node trunk selection", "catsdogs_cnn+resnet_sel_binary_h8_n0.0"),
          ("scratch CNN, single neurons ($h=0$, the note)", "catsdogs_cnn_binary_h0_n0.0"),
          ("scratch CNN, $h=8$, LP (6)--(9) + $\\ell_1$", "catsdogs_cnn_binary_h8_lp")]
    out = HEAD + "\\multicolumn{9}{@{}l}{\\emph{CIFAR-10 -- corrector tree, root $128\\to64\\to10$, binary nodes $h=8$}}\\\\\n"
    out += "\n".join(_row(l, t) for l, t in cifar) + "\n\\midrule\n"
    out += "\\multicolumn{9}{@{}l}{\\emph{Cats vs Dogs -- the note's binary tree, every node $128\\to h\\to1$}}\\\\\n"
    out += "\n".join(_row(l, t) for l, t in cd) + "\n" + FOOT
    open(f"{TAB}/main.tex", "w").write(out)


def tab_width():
    rows = [(f"$h={h}$", f"cifar10_resnet_corrector_h{h}_n0.0") for h in [0, 2, 8, 32, 128]]
    rows[0] = ("$h=0$ (linear $\\to$ all isolation)", "cifar10_resnet_corrector_h0_v2")
    rows.append(("isolation nodes only", "cifar10_resnet_corrector_iso_n0.0"))
    out = HEAD + "\n".join(_row(l, t) for l, t in rows) + "\n" + FOOT
    open(f"{TAB}/width.tex", "w").write(out)


def tab_arch():
    rows = [("one-vs-rest forest", "cifar10_resnet_ovr_h8_n0.0"),
            ("corrector tree", "cifar10_resnet_corrector_h8_n0.0"),
            ("gated tree", "cifar10_resnet_gated_h8_n0.0")]
    out = HEAD + "\n".join(_row(l, t) for l, t in rows) + "\n" + FOOT
    open(f"{TAB}/arch.tex", "w").write(out)


def tab_half():
    rows = [("tree on $S_A$ (trunk's own training half)", "cifar10_cnnhalf_half_a_corrector_h8"),
            ("tree on $S_B$ (half never seen by the trunk)", "cifar10_cnnhalf_half_b_corrector_h8")]
    out = HEAD + "\n".join(_row(l, t) for l, t in rows) + "\n" + FOOT
    open(f"{TAB}/half.tex", "w").write(out)



# --------------------------------------------------------------------------------------
def fig_rf_depth():
    trunks = [("pixels", "pixels", BLUE, "o"), ("scratch CNN", "cnn", ORANGE, "s"), ("ResNet-18", "resnet", AQUA, "^")]
    dsets = [("CIFAR-10", "cifar10_{}_corrector_h8_n0.0"), ("Cats vs Dogs", "catsdogs_{}_binary_h8_n0.0")]
    V = json.load(open("results/veto.json")) if os.path.exists("results/veto.json") else {}
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.7))
    for di, (dname, pat) in enumerate(dsets):
        ax, ax2 = axes[di], axes[2 + di]
        for lab, t, col, mk in trunks:
            tag = pat.format(t)
            p = f"results/rf_{tag}.json"
            if os.path.exists(p):
                R = [r for r in json.load(open(p)) if r["area50"] == r["area50"]]
                depths = sorted({r["depth"] for r in R})
                ax.plot(depths, [np.mean([r["area50"] for r in R if r["depth"] == dd]) for dd in depths],
                        "-" + mk, color=col, label=lab)
            if tag in V:
                rows = [r for r in V[tag] if r["npos"] > 0]
                depths = sorted({r["depth"] for r in rows if r["depth"] > 0})
                depths = [dd for dd in depths if sum(r["npos"] for r in rows if r["depth"] == dd) >= 10]
                ys = [100 * sum(r["silent_pos"] * r["npos"] for r in rows if r["depth"] == dd) /
                      sum(r["npos"] for r in rows if r["depth"] == dd) for dd in depths]
                ax2.plot(depths, ys, "-" + mk, color=col, label=lab)
        ax.axhline(0.5, color=INK2, lw=1, ls=":")
        ax.set_ylim(0, 0.55)
        ax.set_title(f"{dname}: area holding 50% of saliency", fontsize=8.5)
        ax2.set_title(f"{dname}: positives fired by bias alone (%)", fontsize=8.5)
        ax2.set_ylim(0, 60)
        for a_ in (ax, ax2):
            a_.set_xlabel("node depth (0 = root)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIG}/rf_depth.pdf")
    plt.close(fig)


def fig_prune():
    sets = [("CIFAR-10, ResNet, $h$=8", "cifar10_resnet_corrector_h8_n0.0", BLUE, "o"),
            ("CIFAR-10, CNN, $h$=8", "cifar10_cnn_corrector_h8_n0.0", ORANGE, "s"),
            ("Cats vs Dogs, CNN, $h$=8", "catsdogs_cnn_binary_h8_n0.0", AQUA, "^")]
    sets = [x for x in sets if os.path.exists(f"results/minimize_{x[1]}.json")]
    if not sets:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for lab, tag, col, mk in sets:
        C = json.load(open(f"results/minimize_{tag}.json"))["subtree_curve"]
        base = C[-1]["test"]
        ax.plot([100 * c["train"] for c in C], [100 * (c["test"] - base) for c in C], "-" + mk, color=col,
                label=lab, markersize=3)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("training accuracy (%) as subtrees are added back")
    ax.set_ylabel("test acc. change vs. root (pp)")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/prune_curve.pdf")
    plt.close(fig)


def fig_noise():
    p = "results/analysis_cifar10_resnet_corrector_h8_n0.2.json"
    if not os.path.exists(p):
        return
    A = json.load(open(p))
    rows = [r for r in A["nodes"] if r["noisy_frac_pos"] is not None]
    depths = sorted({r["depth"] for r in rows})
    frac, cnt = [], []
    for dd in depths:
        rs = [r for r in rows if r["depth"] == dd]
        tot = sum(r["npos"] for r in rs)
        cnt.append(tot)
        frac.append(sum(r["noisy_frac_pos"] * r["npos"] for r in rs) / max(tot, 1))
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    xs = ["all train"] + ["root errors"] + [f"depth {d}\n(n={c})" for d, c in zip(depths, cnt)]
    ys = [A["base_noise"], A["noisy_frac_root_errors"]] + frac
    ax.bar(range(len(xs)), [100 * y for y in ys], color=[INK2, ORANGE] + [BLUE] * len(depths), width=0.6)
    for i, y in enumerate(ys):
        ax.text(i, 100 * y + 1, f"{100 * y:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(xs)), xs, rotation=0, fontsize=7.5)
    ax.set_ylabel("corrupted labels among\nsamples a node must fire on (%)")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(f"{FIG}/noise_depth.pdf")
    plt.close(fig)


def fig_trees():
    import subprocess
    for tag, out in [("catsdogs_cnn_binary_h8_n0.0", "tree_cd_cnn"), ("cifar10_resnet_corrector_h8_n0.0", "tree_cifar_resnet"),
                     ("catsdogs_cnn_binary_h0_n0.0", "tree_cd_cnn_h0")]:
        if os.path.exists(f"results/trees/{tag}.json"):
            subprocess.run([sys.executable, "scripts/draw_tree.py", f"results/trees/{tag}.json", f"{FIG}/{out}.pdf", "60"],
                           check=True)



def _n(v):
    return f"{v:,}".replace(",", "\\,")


def tab_prune():
    rows = [("CIFAR-10, ResNet, corrector $h$=8", "cifar10_resnet_corrector_h8_n0.0"),
            ("CIFAR-10, CNN, corrector $h$=8", "cifar10_cnn_corrector_h8_n0.0"),
            ("CIFAR-10, ResNet, corrector $h$=32", "cifar10_resnet_corrector_h32_n0.0"),
            ("Cats vs Dogs, CNN, binary $h$=8", "catsdogs_cnn_binary_h8_n0.0"),
            ("Cats vs Dogs, ResNet, binary $h$=8", "catsdogs_resnet_binary_h8_n0.0")]
    out = ("\\begin{tabular}{@{}lrrrrrr@{}}\\toprule\n"
           "tree & non-zero weights & after pruning & reduction & train acc. & test acc. before & after\\\\\\midrule\n")
    for lab, tag in rows:
        p = f"results/minimize_{tag}.json"
        if not os.path.exists(p) or "prune" not in json.load(open(p)):
            continue
        P = json.load(open(p))["prune"]
        out += (f"{lab} & {_n(P['nnz_before'])} & {_n(P['nnz_after'])} & {100 * (1 - P['nnz_after'] / P['nnz_before']):.0f}\\% & "
                f"{100 * P['train_acc']:.1f}\\% & {100 * P['test_acc_before']:.2f}\\% & {100 * P['test_acc']:.2f}\\%\\\\\n")
    out += "\\bottomrule\\end{tabular}\n"
    open(f"{TAB}/prune.tex", "w").write(out)


def tab_smallest():
    """Smallest network that interpolates, from the double-descent sweeps."""
    out = ("\\begin{tabular}{@{}llrr@{}}\\toprule\n"
           "labels & interpolating model & parameters & test error\\\\\\midrule\n")
    for p, lab in [("results/dd/cifar10_resnet_n4000_noise0.0_s0.jsonl", "clean"),
                   ("results/dd/cifar10_resnet_n4000_noise0.2_s0.jsonl", "20\\% noisy")]:
        R = jl(p)
        if not R:
            continue
        interp = [r for r in R if r["train_err"] == 0]
        mlps = [r for r in interp if r["model"] == "mlp"]
        trees_ = [r for r in interp if r["model"] == "tree" and r.get("n_nodes", 1) > 1]
        if not mlps or not trees_:
            print(f"  tab_smallest: {p} has no interpolating MLP or tree yet - skipped")
            continue
        best = min(R, key=lambda r: r["test_err"])
        mlp0 = min(mlps, key=lambda r: r["params"])
        tr0 = min(trees_, key=lambda r: r["params"])
        bestint = min(interp, key=lambda r: r["test_err"])
        for name, r in [(f"smallest single MLP ($H$={mlp0['H']})", mlp0),
                        (f"smallest tree ($H$={tr0['H']}; {tr0['n_nodes']} nodes)", tr0),
                        (f"best interpolating model ({bestint['model'].upper()}, $H$={bestint['H']})", bestint)]:
            out += f"{lab} & {name} & {_n(r['params'])} & {100 * r['test_err']:.1f}\\%\\\\\n"
            lab = ""
        out += "\\midrule\n"
    out = out[: -len("\\midrule\n")] + "\\bottomrule\\end{tabular}\n"
    open(f"{TAB}/smallest.tex", "w").write(out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for name, fn in list(globals().items()):
        if (name.startswith("fig_") or name.startswith("tab_")) and ("all" in which or name in which):
            try:
                fn()
                print("made", name)
            except Exception as e:  # one missing result file should not stop the other figures
                print(f"SKIPPED {name}: {type(e).__name__}: {e}  (did you run every step?)")
