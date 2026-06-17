#!/usr/bin/env python3
"""Benchmark a portfolio of classical solvers (SA / Tabu / Greedy) on the identical QUBO.

Mirrors the Q-Seg comparison-to-a-classical-solver methodology: every solver
minimizes the same QUBO, and is scored by its optimality gap to the exact
reference (exhaustive search for vertex cover, maximum flow for segmentation)
and its wall-clock time.

Run:  python experiments/solver_portfolio.py --num-reads 100
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from qubo_partition.data.graphs import benchmark_graphs
from qubo_partition.data.images import make_blob_image, make_two_region_image
from qubo_partition.io_utils import write_csv, write_latex_table
from qubo_partition.qubo.segmentation import segmentation_qubo
from qubo_partition.qubo.vertex_cover import vertex_cover_qubo
from qubo_partition.solvers.exact_vc import exact_min_vertex_cover_energy
from qubo_partition.solvers.maxflow import min_cut_segmentation
from qubo_partition.solvers.portfolio import run_portfolio

RESULTS = Path("results")


def _aggregate(per_instance: list[dict]) -> list[dict]:
    by_solver: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in per_instance:
        s = by_solver[r["solver"]]
        s["gap"].append(r["gap"])
        s["time"].append(r["time_s"])
        s["success"].append(float(r["success"]))
    rows = []
    for solver, d in by_solver.items():
        rows.append(
            {
                "solver": solver,
                "mean_gap": round(float(np.mean(d["gap"])), 4),
                "max_gap": round(float(np.max(d["gap"])), 4),
                "mean_time_s": round(float(np.mean(d["time"])), 4),
                "success_rate": round(float(np.mean(d["success"])), 3),
            }
        )
    return rows


def vertex_cover_portfolio(args) -> list[dict]:
    print("\n== Vertex cover: solver portfolio vs. exhaustive-search optimum ==")
    rows = []
    graphs = benchmark_graphs()
    for name, g in graphs.items():
        qubo = vertex_cover_qubo(g, penalty=2.0)
        _, opt = exact_min_vertex_cover_energy(g, penalty=2.0)
        for res in run_portfolio(
            qubo, opt, num_reads=args.num_reads, num_sweeps=args.num_sweeps, seed=args.seed
        ):
            rows.append(
                {
                    "instance": name,
                    "solver": res.name,
                    "gap": res.gap,
                    "time_s": res.time_s,
                    "success": res.success,
                }
            )
    return rows


def segmentation_portfolio(args) -> list[dict]:
    print("\n== Segmentation: solver portfolio vs. maximum-flow optimum ==")
    images = [
        make_blob_image(size=16, seed=0),
        make_two_region_image(size=16, seed=1),
        make_blob_image(size=16, noise=0.15, seed=2),
    ]
    rows = []
    for seeded in images:
        qubo, model = segmentation_qubo(
            seeded.image,
            seeded.fg_seeds,
            seeded.bg_seeds,
            lambda_smooth=1.0,
            data_model="histogram",
        )
        _, opt = min_cut_segmentation(model)
        for res in run_portfolio(
            qubo, opt, num_reads=args.num_reads, num_sweeps=args.num_sweeps, seed=args.seed
        ):
            rows.append(
                {
                    "instance": seeded.name,
                    "solver": res.name,
                    "gap": res.gap,
                    "time_s": res.time_s,
                    "success": res.success,
                }
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-reads", type=int, default=100)
    ap.add_argument("--num-sweeps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)

    vc = vertex_cover_portfolio(args)
    seg = segmentation_portfolio(args)

    vc_agg = _aggregate(vc)
    seg_agg = _aggregate(seg)
    for r in vc_agg:
        r["problem"] = "vertex cover"
    for r in seg_agg:
        r["problem"] = "segmentation"
    agg = vc_agg + seg_agg

    print("\nproblem        solver       mean_gap  max_gap  mean_time(s)  success")
    for r in agg:
        print(
            f"  {r['problem']:13s} {r['solver']:11s} {r['mean_gap']:8.4f} "
            f"{r['max_gap']:8.3f} {r['mean_time_s']:11.4f}  {r['success_rate']:.2f}"
        )

    write_csv(vc + seg, RESULTS / "solver_portfolio_per_instance.csv")
    write_csv(agg, RESULTS / "solver_portfolio.csv")
    write_latex_table(
        agg,
        columns=["problem", "solver", "mean_gap", "max_gap", "mean_time_s", "success_rate"],
        path=RESULTS / "tables" / "solver_portfolio.tex",
        caption="Classical solver portfolio on the identical QUBO: optimality gap "
        "(vs. the exact reference) and wall-clock time. SA = simulated "
        "annealing, Greedy = steepest descent.",
        label="tab:portfolio",
    )
    print("\nWrote results/solver_portfolio.csv and results/tables/solver_portfolio.tex")


if __name__ == "__main__":
    main()
