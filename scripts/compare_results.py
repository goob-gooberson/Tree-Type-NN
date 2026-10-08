"""Put your numbers next to a reference run, so you can see what changed and fix the report text.

    python scripts/compare_results.py results_reference results

Prints, for every tree both folders contain: train / test accuracy of the root and of the tree,
number of nodes, and the test fixed/broken counts; then the key double-descent and pruning numbers.
Small differences (a few tenths of a point, a few nodes) are normal: they come from different CPUs,
thread counts and library versions.
"""
import glob
import json
import os
import sys


def trees(folder):
    out = {}
    for f in glob.glob(os.path.join(folder, "trees", "*.json")):
        out[os.path.basename(f)[:-5]] = json.load(open(f))["summary"]
    return out


def dd(folder):
    out = {}
    for f in glob.glob(os.path.join(folder, "dd", "*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        out[os.path.basename(f)[:-6]] = {(r["model"], r["H"]): r for r in rows}
    return out


def pct(v):
    return f"{100 * v:5.1f}" if v is not None else "    -"


if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else "results_reference"
    new = sys.argv[2] if len(sys.argv) > 2 else "results"
    A, B = trees(ref), trees(new)
    print(f"Trees  (columns: reference -> yours;  {ref} -> {new})\n")
    print(f"{'tree':44s} {'root test':>13s} {'tree train':>13s} {'tree test':>13s} {'nodes':>11s} {'fixed/broken':>21s}")
    for k in sorted(set(A) | set(B)):
        a, b = A.get(k), B.get(k)
        if a is None or b is None:
            print(f"{k:44s}  only in {'reference' if b is None else 'yours'}")
            continue
        print(f"{k:44s} {pct(a['root_test_acc'])}->{pct(b['root_test_acc'])} "
              f"{pct(a['train_acc'])}->{pct(b['train_acc'])} {pct(a['test_acc'])}->{pct(b['test_acc'])} "
              f"{a['n_nodes']:4d}->{b['n_nodes']:<4d}  "
              f"{a['test_changed_fixed']:4d}/{a['test_changed_broken']:<4d}->{b['test_changed_fixed']:4d}/{b['test_changed_broken']:<4d}")
    DA, DB = dd(ref), dd(new)
    for k in sorted(set(DA) & set(DB)):
        print(f"\nDouble descent {k}: test error % of the single MLP by width H (reference -> yours)")
        Hs = sorted(H for (m, H) in DA[k] if m == "mlp" and ("mlp", H) in DB[k])
        print("  " + "  ".join(f"H={H}: {pct(DA[k][('mlp', H)]['test_err'])}->{pct(DB[k][('mlp', H)]['test_err'])}"
                             for H in Hs))
        for name, D in (("reference", DA[k]), ("yours", DB[k])):
            mlp = [D[("mlp", H)] for H in Hs]
            thr = next((r["H"] for r in mlp if r["train_err"] == 0), None)
            best = min(mlp, key=lambda r: r["test_err"])
            print(f"  {name:9s}: first width with 0% train error H={thr}; best test error "
                  f"{pct(best['test_err'])}% at H={best['H']}")
    for f in sorted(glob.glob(os.path.join(new, "minimize_*.json"))):
        g = os.path.join(ref, os.path.basename(f))
        if not os.path.exists(g):
            continue
        a, b = json.load(open(g)).get("prune"), json.load(open(f)).get("prune")
        if a and b:
            print(f"\nPruning {os.path.basename(f)[9:-5]}: weights {a['nnz_before']}->{a['nnz_after']} (reference) | "
                  f"{b['nnz_before']}->{b['nnz_after']} (yours); test {pct(a['test_acc'])} | {pct(b['test_acc'])}")
