#!/usr/bin/env python3
"""Generate notebooks/qubo_partition_colab.ipynb (nbformat v4).

Run:  python notebooks/build_colab_notebook.py
Keeps the notebook source in one readable place instead of hand-edited JSON.
"""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text: str) -> list:
    text = text.strip("\n")
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


cells = []

cells.append(md(r"""
# A Closed-Form QUBO for Graph Partitioning & Image Segmentation — Colab

Two superficially different problems — **minimum vertex cover** on a graph and
**foreground/background separation** in an image — written as a *single* QUBO
$H(x)=\sum_i a_i x_i+\sum_{i<j} b_{ij}x_ix_j$ and minimized with **one**
simulated-annealing sampler, then validated against an **exact reference**
(exhaustive search / maximum flow). Plus a bridge to graph-reconstruction
attacks (GraphMI).

This notebook installs everything, fetches the code, and runs every phase with
inline figures. **Runtime → Run all** works end to end. No quantum hardware
needed; runs on the free CPU Colab.
"""))

cells.append(md("## 1. Install dependencies"))
cells.append(code(r"""
!pip install -q dimod dwave-samplers dwave-networkx networkx scipy matplotlib scikit-image pillow
print("deps installed")
"""))

cells.append(md(r"""
## 2. Get the code

Pick **one** of three ways to bring the project into Colab:

* **A — GitHub (recommended):** set `REPO_URL` below to your repo and run.
* **B — Upload a zip:** leave `REPO_URL=""`; you'll be prompted to upload
  `qubo-partition.zip` (make it locally with `bash scripts/make_colab_zip.sh`).
* **C — Google Drive:** mount Drive and set `PROJECT_DIR` to the folder.

The cell auto-detects if the code is already present (e.g. when run locally).
"""))
cells.append(code(r"""
import os, sys, glob, zipfile

REPO_URL = ""                      # <-- set to "https://github.com/<you>/qubo-partition.git" for option A
PROJECT_DIR = None

def _has_pkg(d):
    return d and os.path.isdir(os.path.join(d, "src", "qubo_partition"))

# already present (local run, prior clone, or re-run)?
for cand in [os.getcwd(), "/content/qubo-partition"]:
    if _has_pkg(cand):
        PROJECT_DIR = cand
        break

if PROJECT_DIR is None and REPO_URL:                      # option A
    !git clone -q $REPO_URL /content/qubo-partition
    PROJECT_DIR = "/content/qubo-partition"

if PROJECT_DIR is None:                                    # option B (upload zip)
    from google.colab import files
    print("Upload qubo-partition.zip (made via scripts/make_colab_zip.sh) ...")
    up = files.upload()
    with zipfile.ZipFile(next(iter(up))) as z:
        z.extractall("/content/_proj")
    hits = glob.glob("/content/_proj/**/src/qubo_partition", recursive=True)
    PROJECT_DIR = os.path.dirname(os.path.dirname(hits[0]))

assert _has_pkg(PROJECT_DIR), f"could not locate src/qubo_partition under {PROJECT_DIR}"
os.chdir(PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
print("PROJECT_DIR =", PROJECT_DIR)
"""))

cells.append(md("## 3. Imports & sanity check"))
cells.append(code(r"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dimod

import qubo_partition
from qubo_partition.data.graphs import benchmark_graphs, erdos_renyi_graph, size_sweep
from qubo_partition.data.images import make_blob_image, make_two_region_image
from qubo_partition.evaluation.runner import run_vertex_cover, run_segmentation
from qubo_partition.qubo.vertex_cover import cover_from_sample
from qubo_partition.solvers.exact_vc import exact_min_vertex_cover

print("qubo_partition", qubo_partition.__version__, "| dimod", dimod.__version__)
"""))

cells.append(md(r"""
## Phase 1 — Minimum vertex cover (QUBO vs. exhaustive search)

$H(x)=\sum_{i\in V} x_i + P\sum_{(u,v)\in E}(1-x_u)(1-x_v)$, with $P>1$ so the
global minimizer is guaranteed a *minimum* vertex cover.
"""))
cells.append(code(r"""
def draw_cover(g, chosen, optimal=None, title="", seed=0):
    pos = nx.spring_layout(g, seed=seed)
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    colors = ["#d62728" if n in chosen else "#dddddd" for n in g.nodes()]
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#999")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, edgecolors="#333", node_size=420)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8)
    if optimal is not None:
        nx.draw_networkx_nodes(g, pos, nodelist=list(optimal), ax=ax, node_color="none",
                               edgecolors="#1f77b4", node_size=560, linewidths=2)
    ax.set_title(title); ax.axis("off"); plt.show()

g = nx.petersen_graph()
rec = run_vertex_cover(g, name="petersen", penalty=2.0, num_reads=200, num_sweeps=1000)
opt_cover, opt_size = exact_min_vertex_cover(g)
print(f"annealed |cover|={rec.annealed_cover_size} (exact {opt_size}), valid={rec.is_valid_cover}, "
      f"best_gap={rec.gap.best_gap:.3f}, success_rate={rec.gap.success_rate:.2f}")
draw_cover(g, cover_from_sample(rec.annealed_sample), opt_cover,
           title="Petersen: annealed cover (red), one exact optimum (blue outline)")
"""))

cells.append(md("### Benchmark suite + optimality gap vs. instance size"))
cells.append(code(r"""
rows = []
for name, gg in benchmark_graphs().items():
    r = run_vertex_cover(gg, name=name, penalty=2.0, num_reads=150, num_sweeps=1000)
    rows.append((name, r.n_nodes, r.n_edges, r.annealed_cover_size, r.optimal_cover_size,
                 r.is_valid_cover, round(r.gap.best_gap, 3), round(r.gap.success_rate, 2)))
print(f"{'graph':16s} n   m   |C| opt valid bestgap succ")
for x in rows:
    print(f"{x[0]:16s} {x[1]:<3d} {x[2]:<3d} {x[3]:<3d} {x[4]:<3d} {str(x[5]):5s} {x[6]:<7} {x[7]}")
valid = sum(r[5] for r in rows); opt = sum(1 for r in rows if r[6] <= 1e-6)
print(f"\n{valid}/{len(rows)} valid covers, {opt}/{len(rows)} reached the exact optimum")
"""))
cells.append(code(r"""
from collections import defaultdict
by_size = defaultdict(list)
for name, gg in size_sweep():
    r = run_vertex_cover(gg, name=name, num_reads=150, num_sweeps=1000)
    by_size[gg.number_of_nodes()].append(r.gap)
sizes = sorted(by_size)
best = [np.mean([s.best_gap for s in by_size[n]]) for n in sizes]
mean = [np.mean([s.mean_gap for s in by_size[n]]) for n in sizes]
std  = [np.mean([s.std_gap  for s in by_size[n]]) for n in sizes]
plt.figure(figsize=(5.2, 3.6))
plt.plot(sizes, best, "o-", label="best gap")
plt.plot(sizes, mean, "s--", label="mean gap")
plt.fill_between(sizes, np.array(mean)-std, np.array(mean)+std, alpha=0.2)
plt.axhline(0, color="k", lw=.8); plt.xlabel("nodes"); plt.ylabel("energy gap")
plt.title("Vertex cover: optimality gap vs. size"); plt.legend(); plt.show()
"""))

cells.append(md(r"""
## Phase 2 — Seeded segmentation (Boykov–Jolly QUBO vs. maximum flow)

$E(x)=\sum_i D_i(x_i)+\sum_{(i,j)} w_{ij}(x_i-x_j)^2$ — submodular, so its global
optimum is an exact maximum-flow min-cut.
"""))
cells.append(code(r"""
def show_seg(seeded, rec, truth_label="ground truth"):
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    ax[0].imshow(seeded.image, cmap="gray", vmin=0, vmax=1)
    fr, fc = np.where(seeded.fg_seeds); br, bc = np.where(seeded.bg_seeds)
    ax[0].scatter(fc, fr, c="#2ca02c", s=20, edgecolors="k", linewidths=.4, label="fg")
    ax[0].scatter(bc, br, c="#d62728", s=20, marker="s", edgecolors="k", linewidths=.4, label="bg")
    ax[0].legend(fontsize=7); ax[0].set_title("image + seeds")
    ax[1].imshow(rec.annealed_labels, cmap="gray", vmin=0, vmax=1); ax[1].set_title("annealed")
    ax[2].imshow(rec.optimal_labels, cmap="gray", vmin=0, vmax=1); ax[2].set_title("max-flow optimum")
    ax[3].imshow(seeded.truth.astype(float), cmap="gray", vmin=0, vmax=1); ax[3].set_title(truth_label)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    plt.suptitle(f"gap={rec.gap.best_gap:.2f}  IoU(opt)={rec.iou_optimal:.2f}"); plt.show()

seeded = make_blob_image(size=16, seed=0)
rec = run_segmentation(seeded, lambda_smooth=1.0, num_reads=200, num_sweeps=2000)
print(f"E_anneal={rec.annealed_energy:.2f}  E_opt={rec.optimal_energy:.2f}  "
      f"best_gap={rec.gap.best_gap:.3f}  IoU={rec.iou_optimal:.2f}")
show_seg(seeded, rec)
"""))

cells.append(md(r"""
## Phase 2 (real data) — 32×32 Weizmann horses

Real photographs with real binary ground-truth masks. Downloaded straight into
Colab. Improved method: **histogram data term** + **8-connectivity**.
"""))
cells.append(code(r"""
import os
if not os.path.isdir("datasets/weizmann_horse_32/images"):
    !git clone -q --depth 1 https://github.com/nkg114mc/dvn-horse.git /tmp/dvn-horse
    !mkdir -p /tmp/h && unzip -q -o /tmp/dvn-horse/pics32.zip -d /tmp/h
    !mkdir -p datasets/weizmann_horse_32 && cp -r /tmp/h/pics32/. datasets/weizmann_horse_32/
print("images:", len(os.listdir("datasets/weizmann_horse_32/images")))
"""))
cells.append(code(r"""
from qubo_partition.data.real import load_weizmann_horses
horses = load_weizmann_horses("datasets/weizmann_horse_32", limit=6, n_seeds=5, seed=0)
for s in horses:
    r = run_segmentation(s, lambda_smooth=2.0, data_model="histogram", connectivity=8,
                         num_reads=80, num_sweeps=1500)
    print(f"{s.name}: gap={r.gap.best_gap:.2f} IoU(opt)={r.iou_optimal:.2f} acc={r.pixel_acc_annealed:.2f}")
    show_seg(s, r)
"""))

cells.append(md(r"""
## Phase 2 (high-res) — clean scikit-image photos (Q-Seg-style look)

Real photographs (cameraman, coins, cell, ...) + crisp shapes, resized large for
clarity. Max-flow stays exact at any size. *On Colab's 2 vCPU we use 96×96 and a
small subset for speed; bump `SIZE`/`PHOTOS`/reads for the full 128×128 run.*
"""))
cells.append(code(r"""
from qubo_partition.data.real import load_skimage_demo
SIZE = 96                                   # try 128 for sharper figures (slower on Colab)
PHOTOS = ["camera", "coins", "cell"]        # add "clock","astronaut","coffee" for more
imgs = load_skimage_demo(size=SIZE, photos=PHOTOS, n_seeds=6, seed=0)
for s in imgs:
    is_gt = bool(getattr(s, "truth_is_gt", False))
    r = run_segmentation(s, lambda_smooth=3.0, data_model="histogram", n_bins=24,
                         connectivity=8, num_reads=30, num_sweeps=2000)
    print(f"{s.name}: gap={r.gap.best_gap:.2f} IoU(opt)={r.iou_optimal:.2f} "
          f"[{'GT' if is_gt else 'Otsu'}]")
    show_seg(s, r, truth_label=("ground truth" if is_gt else "Otsu reference"))
"""))

cells.append(md(r"""
## Phase 3 — The bridge: graph reconstruction as the *same* QUBO move

The feature-smoothness objective an adversary minimizes to recover a hidden
graph (GraphMI) mirrors the segmentation smoothness term. We hide a graph,
release smoothed node features, and reconstruct the edges by minimizing a QUBO —
validated against the analytic optimum and the hidden graph.
"""))
cells.append(code(r"""
from qubo_partition.bridge.reconstruction import run_reconstruction_demo

def draw_reconstruction(rec, seed=1):
    g_true = nx.Graph(); g_true.add_nodes_from(range(rec.n_nodes)); g_true.add_edges_from(rec.true_edges)
    pos = nx.spring_layout(g_true, seed=seed)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2))
    for a, edges, t in [(ax[0], rec.true_edges, "hidden graph"),
                        (ax[1], rec.annealed_edges, f"reconstructed (F1={rec.f1:.2f})")]:
        nx.draw_networkx_nodes(g_true, pos, ax=a, node_color="#dddddd", edgecolors="#333", node_size=300)
        nx.draw_networkx_labels(g_true, pos, ax=a, font_size=8)
        nx.draw_networkx_edges(nx.Graph(list(edges)), pos, ax=a, edge_color="#1f77b4", width=2)
        a.set_title(t); a.axis("off")
    plt.show()

for name, gg in {"cycle_8": nx.cycle_graph(8), "petersen": nx.petersen_graph(),
                 "er_n8": erdos_renyi_graph(8, 0.35, seed=1)}.items():
    rec = run_reconstruction_demo(gg, name=name, num_reads=300, num_sweeps=3000, seed=0)
    print(f"{name}: gap={rec.gap:.3f}  F1={rec.f1:.2f}  precision/recall={rec.precision:.2f}/{rec.recall:.2f}")
    draw_reconstruction(rec)
"""))

cells.append(md("## (optional) Run the test suite"))
cells.append(code(r"""
!cd "$PROJECT_DIR" && PYTHONPATH=src python -m pytest -q
"""))

cells.append(md(r"""
## Notes & scaling

* **Validation is the point.** Gaps are reported against an *exact* optimum
  (exhaustive search / maximum flow); on the larger images the annealer's best
  read may not hit the optimum, but its mask is visually identical and the gap
  is small relative to the total energy.
* **Scale up** by raising `SIZE` (→128), adding `PHOTOS`, and increasing
  `num_reads`/`num_sweeps`. Colab's 2 vCPU is far slower than a workstation;
  expect ~1 min per 128×128 image.
* **Quantum stretch goal.** The identical QUBO (`qubo.to_bqm()`) can be sent to a
  D-Wave `DWaveSampler` + `EmbeddingComposite` with Leap credentials — no change
  to the formulation.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = Path(__file__).with_name("qubo_partition_colab.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(cells)} cells)")
