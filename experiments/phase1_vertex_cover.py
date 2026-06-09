#!/usr/bin/env python3
"""Phase one experiments: minimum vertex cover.

Run:  python experiments/phase1_vertex_cover.py --num-reads 200 --num-sweeps 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qubo_partition import viz
from qubo_partition.data.graphs import benchmark_graphs, erdos_renyi_graph, size_sweep
from qubo_partition.evaluation.runner import run_vertex_cover
from qubo_partition.io_utils import write_csv, write_json, write_latex_table
from qubo_partition.qubo.vertex_cover import cover_from_sample
from qubo_partition.solvers.exact_vc import exact_min_vertex_cover

RESULTS = Path("results")
FIGS = RESULTS / "figures"


def benchmark_suite(args) -> list[dict]:
    print("\n== Benchmark suite: anneal vs. exhaustive search ==")
    rows = []
    graphs = benchmark_graphs()
    for name, g in graphs.items():
        rec = run_vertex_cover(
            g,
            name=name,
            penalty=args.penalty,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
        )
        rows.append(rec.as_row())
        print(
            f"  {name:16s} n={rec.n_nodes:2d} m={rec.n_edges:3d} "
            f"|cover|={rec.annealed_cover_size:2d} (opt {rec.optimal_cover_size:2d}) "
            f"valid={rec.is_valid_cover!s:5s} best_gap={rec.gap.best_gap:.3f} "
            f"success={rec.gap.success_rate:.2f}"
        )
        if name in ("petersen", "er_n15_p0.3_s2"):
            opt_cover, _ = exact_min_vertex_cover(g)
            viz.plot_vertex_cover(
                g,
                cover_from_sample(rec.annealed_sample),
                optimal=opt_cover,
                title=f"{name}: annealed cover in red, one exact optimum outlined "
                f"(size {rec.annealed_cover_size}={rec.optimal_cover_size}; "
                "minima may differ)",
                path=str(FIGS / f"vc_{name}.png"),
            )
    write_csv(rows, RESULTS / "phase1_benchmarks.csv")
    write_latex_table(
        rows,
        columns=[
            "name",
            "n_nodes",
            "n_edges",
            "annealed_cover_size",
            "optimal_cover_size",
            "is_valid_cover",
            "best_gap",
            "mean_gap",
            "success_rate",
        ],
        path=RESULTS / "tables" / "phase1_benchmarks.tex",
        caption="Minimum vertex cover: simulated annealing vs. exhaustive search "
        "(200 reads per instance).",
        label="tab:vc-bench",
    )
    return rows


def gap_vs_size(args) -> list[dict]:
    print("\n== Optimality gap vs. instance size ==")
    instances = size_sweep()
    by_size: dict[int, list] = {}
    rows = []
    for name, g in instances:
        rec = run_vertex_cover(
            g,
            name=name,
            penalty=args.penalty,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
        )
        rows.append(rec.as_row())
        by_size.setdefault(g.number_of_nodes(), []).append(rec.gap)

    sizes = sorted(by_size)
    best = [np.mean([s.best_gap for s in by_size[n]]) for n in sizes]
    mean = [np.mean([s.mean_gap for s in by_size[n]]) for n in sizes]
    std = [np.mean([s.std_gap for s in by_size[n]]) for n in sizes]
    for n, b, m in zip(sizes, best, mean):
        print(f"  n={n:2d}: avg best gap={b:.3f}  avg mean gap={m:.3f}")

    viz.plot_gap_vs_size(
        sizes,
        best,
        mean,
        std,
        title="Vertex cover: optimality gap vs. instance size",
        path=str(FIGS / "vc_gap_vs_size.png"),
    )
    write_csv(rows, RESULTS / "phase1_gap_vs_size.csv")
    return rows


def penalty_sweep(args) -> list[dict]:
    print("\n== Penalty-weight sweep (P > 1 is the threshold) ==")
    g = erdos_renyi_graph(14, 0.3, seed=2)
    opt_cover, opt_size = exact_min_vertex_cover(g)
    rows = []
    penalties = [1.05, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]
    succ = []
    for P in penalties:
        rec = run_vertex_cover(
            g,
            name=f"P={P}",
            penalty=P,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
        )
        rows.append(
            {
                "penalty": P,
                **rec.gap.as_dict(),
                "valid": rec.is_valid_cover,
                "cover_size": rec.annealed_cover_size,
                "opt_size": opt_size,
            }
        )
        succ.append(rec.gap.success_rate)
        print(
            f"  P={P:5.2f}: valid={rec.is_valid_cover!s:5s} "
            f"|cover|={rec.annealed_cover_size} (opt {opt_size}) "
            f"success={rec.gap.success_rate:.2f}"
        )
    viz.plot_success_vs_param(
        penalties,
        succ,
        xlabel="penalty weight P",
        title="Vertex cover: success rate vs. penalty weight",
        path=str(FIGS / "vc_penalty_sweep.png"),
    )
    write_csv(rows, RESULTS / "phase1_penalty_sweep.csv")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-reads", type=int, default=200)
    ap.add_argument("--num-sweeps", type=int, default=1000)
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)

    bench = benchmark_suite(args)
    gap_vs_size(args)
    penalty_sweep(args)

    n_valid = sum(r["is_valid_cover"] for r in bench)
    n_opt = sum(1 for r in bench if r["best_gap"] <= 1e-6)
    write_json(
        {
            "benchmarks": len(bench),
            "valid_covers": n_valid,
            "reached_optimum": n_opt,
            "config": vars(args),
        },
        RESULTS / "phase1_summary.json",
    )
    print(
        f"\nPhase 1 done: {n_valid}/{len(bench)} valid covers, "
        f"{n_opt}/{len(bench)} reached the exact optimum."
    )


if __name__ == "__main__":
    main()
