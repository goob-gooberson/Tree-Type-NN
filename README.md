# Tree-type networks of small neural networks (CIFAR-10, Cats vs Dogs)

Code for the graded assignment *"Tree type network revisited"*. It grows the tree-type network of
the course note (Jayadeva, *A Tree Type Neural Network*) with **every node a small neural network**,
on top of frozen **feature trunks**, and studies interpolation (100% training accuracy), double
descent, receptive fields of nodes, and how small the network can be made.

The full write-up is `report/report.pdf`.

## 1. Idea in one picture

```
image ──► trunk (frozen) ──► features ──► FeatureMap (standardise + PCA) ──► tree of small MLPs
          pixels | SmallCNN | ResNet-18                                       root + correcting children
```

Binary node (Table I/II of the note, generalised to neural nodes). A node with logit `z(x)` splits
its samples into C1 (z<0,y=0), C2 (z≥0,y=1), C3 (missed, z<0,y=1), C4 (false alarm, z≥0,y=0).

* child **A** is trained to fire on C3 and stay silent on C1 ∪ C4 (C2 = don't care);
* child **B** is trained to fire on C4 and stay silent on C2 ∪ C3 (C1 = don't care);
* corrected net input: `z(x) + w_A·A(x) + w_B·B(x)` with `w_A ≥ 0`, `w_B ≤ 0` (eq. 9).

With perfect children the LP (6)–(9) over `(w_A, w_B)` has the closed form
`w_A = max_{C3}(δ − z_i)`, `w_B = −max_{C4}(δ + z_i)`; `--resolve lp` re-solves the node's whole
output layer plus `(w_A, w_B)` by linear programming exactly as in (6)–(9) (optionally with an L1 term
to minimise the number of weights). Children recurse until every training sample is correct.
A child that cannot make progress is replaced by an **isolation node** – a closed-form two-layer
threshold network (OR of spherical caps) – which guarantees termination because every point of an
L2-normalised data set is an extreme point and hence linearly separable from the rest.

Each node's own objective is the neural version of eq. (1)–(3): minimise Σ c_i·max(0, 1 − (2y_i−1) f(x_i)),
with class weights c_i because child problems are extremely imbalanced.

Multiclass trees (`treenn/tree.py`):

| model | structure | children per node |
|---|---|---|
| `ovr` | K binary trees (one-vs-rest) | 2 |
| `corrector` | K-way root + per-class *detector* D_k that fires on misclassified samples of class k and is silent on every other class (correct samples of class k are don't-care). Exactly Table II when K = 2. Logits `z_k + w_k·D_k(x)`. | ≤ K |
| `gated` | K-way node + error detector E (binary tree) + re-classifier R (gated tree trained on the node's errors). Output = R(x) if E fires else node(x). | 2 |
| `binary` | the note's binary tree (Cats vs Dogs) | 2 |

## Key results (details in `report/report.pdf`)

| setting | root test acc. | tree (100% train) test acc. | nodes |
|---|---|---|---|
| CIFAR-10, scratch-CNN trunk, corrector tree | 84.8% | 81.8% | 83 |
| CIFAR-10, CNN + ResNet, per-node trunk selection | 84.8% | 82.2% | 35 |
| CIFAR-10, ResNet trunk, isolation-node children | 74.5% | 74.6% | 11 |
| Cats vs Dogs, scratch-CNN trunk, the note's binary tree | 94.1% | 92.1% | 17 |
| Cats vs Dogs, ResNet trunk, binary tree | 93.4% | 93.2% | 5 |

* Every tree interpolates its training set; learned correctors break ~3 test predictions per fix,
  isolation nodes (local memorisation) are harmless but large.
* With 20% label noise the root MLP shows double descent (33% → 54% at the interpolation threshold
  H=64 → 39% at H=4096); trees interpolate at every width and land on the over-parameterised branch.
* Saliency extent is set by the trunk, not by depth; deep nodes memorise outliers and increasingly fire
  "by exclusion" (all hidden units silent, positive bias).
* Per-node trunk selection: 3.3x fewer parameters; decision-preserving pruning: −20–32% weights at
  unchanged training decisions.

## 2. Repository layout

```
treenn/
  data.py         dataset loading, normalisation, augmentation, symmetric label noise
  trunks.py       PixelTrunk, SmallCNN (trained from scratch), ResNet18Trunk (ImageNet, frozen), FeatureMap
  node.py         MLPNode / linear node, IsoNode (closed-form memoriser), node training (weighted hinge)
  tree.py         BinaryTree, OvRForest, CorrectorTree, GatedTree, LP re-solve, summaries
  receptive.py    input-gradient receptive fields, saliency, localisation statistics
  minimize.py     decision-preserving pruning, cost-complexity subtree pruning
  experiment.py   shared loading/fitting helpers
scripts/
  prepare_data.py        image folders -> .npz
  train_trunk.py         train the SmallCNN trunk
  extract_features.py    cache trunk features
  run_tree.py            grow a tree and report train/test accuracy, size, test-time effect of children
  run_double_descent.py  model-wise double-descent sweep (MLP vs tree)
  receptive_fields.py    receptive-field figures and statistics for tree nodes
  run_minimize.py        size minimisation experiments
  analyse_tree.py        per-node analysis (label noise, test-time effect of children)
  veto_analysis.py       how nodes recognise positives (bias-only firing)
  draw_tree.py           drawing of a grown tree in the style of Fig. 4 (Graphviz if installed, else matplotlib)
  make_figures.py        all report figures/tables from results/
  download_data.py       one-command download of both datasets and the ResNet weights
  compare_results.py     your results next to a reference run
  make_overleaf_zip.py   the report bundled for Overleaf
results/                 JSON results of every run (trees, sweeps, pruning, receptive-field stats)
figures/                 figures used in the report
report/                  LaTeX source and PDF
run_all.py               runs every experiment in order (stages: data, quick, trunks, trees, dd, analysis, report)
```

## 3. Reproducing

Works on Windows, macOS and Linux with Python 3.10–3.13 and Git. Commands are the same in PowerShell
and bash; run them from the project folder.

```bash
python -m venv .venv            # then: .venv\Scripts\Activate.ps1 (Windows) or source .venv/bin/activate
pip install -r requirements.txt

python run_all.py --list        # every command behind the report, grouped in stages
python run_all.py data          # download CIFAR-10, Cats vs Dogs, ResNet-18 weights; build data/*.npz
python run_all.py quick         # 2-minute sanity check: one small tree
python run_all.py trunks        # pixel / ResNet features, train the CNN trunks (add --device cuda on a GPU)
python run_all.py trees         # the 20 trees of the report, each at 100% training accuracy
python run_all.py dd            # double-descent sweeps
python run_all.py analysis      # receptive fields, pruning, per-node analyses
python run_all.py report        # figures, LaTeX tables, overleaf.zip (and the PDF if pdflatex exists)

python scripts/compare_results.py results_reference results   # your numbers next to a reference run
```

Every stage can also be run command by command (`run_all.py --list` prints them). Total time on a
2-core CPU without a GPU: about 3.5 hours. Rename the shipped `results/` folder to `results_reference/`
before a fresh run so that every number is recomputed.
