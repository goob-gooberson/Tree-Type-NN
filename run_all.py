"""Run every experiment behind the report, in order (Windows, macOS and Linux).

    python run_all.py                    # everything (several hours on a laptop CPU)
    python run_all.py trees dd           # only some stages, in the order given
    python run_all.py --list             # show the stages and their commands
    python run_all.py trunks --device cuda   # train/extract trunks on an NVIDIA GPU (e.g. Google Colab)

Stages: data, quick, trunks, trees, dd, analysis, report. Each command's output streams to the
terminal; the run stops at the first command that fails, so you can fix it and rerun that stage.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

PY = sys.executable
T = "results/trees"


def stages(device):
    dev = [] if device == "cpu" else ["--device", device]
    tree = [PY, "scripts/run_tree.py", "--save_model"]
    S = {}
    S["data"] = [
        [PY, "scripts/download_data.py"],
        [PY, "scripts/prepare_data.py", "--cd_size", "64"],
    ]
    S["quick"] = [
        [PY, "scripts/extract_features.py", "--dataset", "catsdogs", "--trunk", "pixels"],
        [PY, "scripts/run_tree.py", "--dataset", "catsdogs", "--feats", "pixels", "--model", "binary",
         "--h", "8", "--n_train", "2000", "--tag", "quick_test", "--verbose"],
    ]
    S["trunks"] = [
        [PY, "scripts/extract_features.py", "--dataset", "cifar10", "--trunk", "pixels"],
        [PY, "scripts/extract_features.py", "--dataset", "catsdogs", "--trunk", "pixels"],
        [PY, "scripts/extract_features.py", "--dataset", "cifar10", "--trunk", "resnet", "--res", "128"] + dev,
        [PY, "scripts/extract_features.py", "--dataset", "catsdogs", "--trunk", "resnet", "--res", "128"] + dev,
        [PY, "scripts/train_trunk.py", "--dataset", "cifar10", "--epochs", "15"] + dev,
        [PY, "scripts/train_trunk.py", "--dataset", "catsdogs", "--epochs", "15"] + dev,
        [PY, "scripts/train_trunk.py", "--dataset", "cifar10", "--epochs", "15", "--split", "half"] + dev,
        [PY, "scripts/extract_features.py", "--dataset", "cifar10", "--trunk", "cnn"] + dev,
        [PY, "scripts/extract_features.py", "--dataset", "catsdogs", "--trunk", "cnn"] + dev,
        [PY, "scripts/extract_features.py", "--dataset", "cifar10", "--trunk", "cnn", "--split", "half"] + dev,
    ]
    c10 = ["--dataset", "cifar10", "--h_root", "64"]
    S["trees"] = (
        [tree + ["--dataset", "catsdogs", "--feats", f, "--model", "binary", "--h", "8"] for f in ("cnn", "resnet", "pixels")]
        + [tree + ["--dataset", "catsdogs", "--feats", "cnn", "--model", "binary", "--h", "0"],
           tree + ["--dataset", "catsdogs", "--feats", "cnn", "--model", "binary", "--h", "8", "--resolve", "lp",
                   "--lp_l1", "0.001", "--tag", "catsdogs_cnn_binary_h8_lp"],
           tree + ["--dataset", "catsdogs", "--feats", "cnn,resnet", "--select_trunk", "--model", "binary", "--h", "8"]]
        + [tree + c10 + ["--feats", "resnet", "--model", "corrector", "--h", h] for h in ("2", "8", "32", "128")]
        + [tree + c10 + ["--feats", "resnet", "--model", "corrector", "--h", "0", "--tag", "cifar10_resnet_corrector_h0_v2"],
           tree + c10 + ["--feats", "resnet", "--model", "corrector", "--h", "32", "--iso_below", "1000000",
                         "--tag", "cifar10_resnet_corrector_iso_n0.0"],
           tree + c10 + ["--feats", "resnet", "--model", "gated", "--h", "8"],
           tree + c10 + ["--feats", "resnet", "--model", "ovr", "--h", "8"],
           tree + c10 + ["--feats", "cnn", "--model", "corrector", "--h", "8"],
           tree + c10 + ["--feats", "cnn,resnet", "--select_trunk", "--model", "corrector", "--h", "8"],
           tree + c10 + ["--feats", "resnet", "--model", "corrector", "--h", "8", "--n_train", "10000", "--noise", "0.2"],
           tree + c10 + ["--feats", "cnn_half", "--subset", "half_a", "--model", "corrector", "--h", "8",
                         "--tag", "cifar10_cnnhalf_half_a_corrector_h8"],
           tree + c10 + ["--feats", "cnn_half", "--subset", "half_b", "--model", "corrector", "--h", "8",
                         "--tag", "cifar10_cnnhalf_half_b_corrector_h8"],
           tree + c10 + ["--feats", "pixels", "--model", "corrector", "--h", "8"]]
    )
    dd = [PY, "scripts/run_double_descent.py", "--dataset", "cifar10", "--feats", "resnet", "--n_train", "4000",
          "--pca", "64"]
    S["dd"] = [dd + ["--noise", "0.2"],
               dd + ["--noise", "0.0", "--widths", "1,2,4,8,16,32,64,128,256,512,1024,2048"]]
    rf_trees = ["cifar10_resnet_corrector_h8_n0.0", "cifar10_cnn_corrector_h8_n0.0", "cifar10_pixels_corrector_h8_n0.0",
                "catsdogs_cnn_binary_h8_n0.0", "catsdogs_resnet_binary_h8_n0.0", "catsdogs_pixels_binary_h8_n0.0"]
    min_trees = ["cifar10_resnet_corrector_h8_n0.0", "cifar10_cnn_corrector_h8_n0.0", "catsdogs_cnn_binary_h8_n0.0",
                 "catsdogs_resnet_binary_h8_n0.0", "cifar10_resnet_corrector_h32_n0.0"]
    S["analysis"] = (
        [[PY, "scripts/receptive_fields.py", "--tree", f"{T}/{t}.pt", "--max_nodes", "24", "--per_depth", "4"]
         for t in rf_trees]
        + [[PY, "scripts/run_minimize.py", "--tree", f"{T}/{t}.pt"] for t in min_trees]
        + [[PY, "scripts/veto_analysis.py"] + [f"{T}/{t}.pt" for t in rf_trees]]
        + [["@analyse_all"]]  # analyse_tree.py on every saved tree (expanded at run time)
    )
    S["report"] = [[PY, "scripts/make_figures.py"], [PY, "scripts/make_overleaf_zip.py"]]
    return S


def expand(cmd):
    if cmd == ["@analyse_all"]:
        return [[PY, "scripts/analyse_tree.py", "--tree", f] for f in sorted(glob.glob(f"{T}/*.pt"))]
    return [cmd]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    S = stages(a.device)
    order = ["data", "quick", "trunks", "trees", "dd", "analysis", "report"]
    chosen = a.stages or [s for s in order if s != "quick"]
    bad = [s for s in chosen if s not in S]
    if bad:
        sys.exit(f"unknown stage(s) {bad}; choose from {order}")
    if a.list or a.dry_run:
        for s in chosen:
            print(f"== {s}")
            for c in S[s]:
                line = " ".join(c).replace(PY, "python")
                print("   " + ("python scripts/analyse_tree.py --tree <every results/trees/*.pt>"
                               if line == "@analyse_all" else line))
        sys.exit(0)
    t_all = time.time()
    for s in chosen:
        print(f"\n===== stage {s} =====", flush=True)
        for c0 in S[s]:
            for c in expand(c0):
                print(f"\n$ {' '.join(c).replace(PY, 'python')}", flush=True)
                t0 = time.time()
                r = subprocess.run(c)
                if r.returncode != 0:
                    sys.exit(f"FAILED (exit {r.returncode}) after {time.time() - t0:.0f}s: {' '.join(c)}")
                print(f"[ok, {time.time() - t0:.0f}s]", flush=True)
    if "report" in chosen and shutil.which("pdflatex"):
        os.chdir("report")
        for _ in range(3):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], stdout=subprocess.DEVNULL)
        print("built report/report.pdf")
    print(f"\nall done in {(time.time() - t_all) / 60:.0f} min")
