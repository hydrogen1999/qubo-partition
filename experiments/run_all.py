#!/usr/bin/env python3
"""Run all three phases end-to-end and print a one-line summary of each.

Run:  python experiments/run_all.py            # full settings
      python experiments/run_all.py --quick    # fast smoke run
"""

from __future__ import annotations

import argparse
import sys
import time

import phase1_vertex_cover as p1
import phase2_segmentation as p2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer reads/sweeps")
    args = ap.parse_args()

    reads = 40 if args.quick else 200
    sweeps = 500 if args.quick else 2000
    size = 16

    t0 = time.time()

    p1_args = argparse.Namespace(num_reads=reads, num_sweeps=sweeps, penalty=2.0, seed=0)
    p1.FIGS.mkdir(parents=True, exist_ok=True)
    (p1.RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    p1.benchmark_suite(p1_args)
    p1.gap_vs_size(p1_args)
    p1.penalty_sweep(p1_args)

    p2_args = argparse.Namespace(
        size=size,
        num_reads=reads,
        num_sweeps=sweeps,
        lam=1.0,
        data_weight=1.0,
        seed=0,
        seed_trials=3,
    )
    p2.benchmark_suite(p2_args)
    p2.lambda_study(p2_args)
    p2.seed_sensitivity(p2_args)

    p3_args = argparse.Namespace(num_reads=reads, num_sweeps=sweeps, penalty=5.0, seed=0)
    sys.argv = ["phase3"]  # p3.main parses argv; call its body directly instead
    _run_phase3(p3_args)

    print(f"\nAll phases complete in {time.time() - t0:.1f}s. See results/.")


def _run_phase3(args):
    from pathlib import Path

    import networkx as nx

    from qubo_partition.bridge.reconstruction import run_reconstruction_demo
    from qubo_partition.data.graphs import erdos_renyi_graph
    from qubo_partition.io_utils import write_csv, write_json, write_latex_table

    print("\n== GraphMI bridge ==")
    instances = {
        "cycle_8": nx.cycle_graph(8),
        "path_9": nx.path_graph(9),
        "er_n8_s1": erdos_renyi_graph(8, 0.35, seed=1),
        "er_n10_s2": erdos_renyi_graph(10, 0.3, seed=2),
        "petersen": nx.petersen_graph(),
    }
    rows = []
    for name, g in instances.items():
        rec = run_reconstruction_demo(
            g,
            name=name,
            penalty=args.penalty,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            seed=args.seed,
        )
        rows.append(rec.as_row())
        print(f"  {name:10s} gap={rec.gap:.3f} F1={rec.f1:.2f}")
    write_csv(rows, Path("results") / "phase3_reconstruction.csv")
    write_latex_table(
        rows,
        columns=["name", "n_nodes", "n_true_edges", "gap", "precision", "recall", "f1", "edge_iou"],
        path=Path("results") / "tables" / "phase3_reconstruction.tex",
        caption="Graph reconstruction as a QUBO.",
        label="tab:graphmi",
    )
    mean_f1 = sum(r["f1"] for r in rows) / len(rows)
    write_json(
        {"instances": len(rows), "mean_f1": mean_f1}, Path("results") / "phase3_summary.json"
    )


if __name__ == "__main__":
    main()
