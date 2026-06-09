#!/usr/bin/env python3
"""Phase three: the bridge to graph-reconstruction attacks (GraphMI).

Run:  python experiments/phase3_graphmi_bridge.py --num-reads 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx

from qubo_partition.bridge.reconstruction import run_reconstruction_demo
from qubo_partition.data.graphs import erdos_renyi_graph
from qubo_partition.io_utils import write_csv, write_json, write_latex_table

RESULTS = Path("results")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-reads", type=int, default=200)
    ap.add_argument("--num-sweeps", type=int, default=2000)
    ap.add_argument("--penalty", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("\n== GraphMI bridge: reconstruct a hidden graph by minimizing a QUBO ==")
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
        print(
            f"  {name:10s} n={rec.n_nodes:2d} |E|={rec.n_true_edges:2d} "
            f"gap={rec.gap:.3f} F1={rec.f1:.2f} "
            f"P/R={rec.precision:.2f}/{rec.recall:.2f} edge_IoU={rec.edge_iou:.2f}"
        )

    write_csv(rows, RESULTS / "phase3_reconstruction.csv")
    write_latex_table(
        rows,
        columns=[
            "name",
            "n_nodes",
            "n_true_edges",
            "annealed_energy",
            "optimal_energy",
            "gap",
            "precision",
            "recall",
            "f1",
            "edge_iou",
        ],
        path=RESULTS / "tables" / "phase3_reconstruction.tex",
        caption="Graph reconstruction as a QUBO: annealed recovery vs. exact "
        "optimum and vs. the hidden graph.",
        label="tab:graphmi",
    )
    mean_f1 = sum(r["f1"] for r in rows) / len(rows)
    write_json(
        {"instances": len(rows), "mean_f1": mean_f1, "config": vars(args)},
        RESULTS / "phase3_summary.json",
    )
    print(f"\nPhase 3 done: mean edge-F1 = {mean_f1:.2f} across {len(rows)} hidden graphs.")


if __name__ == "__main__":
    main()
