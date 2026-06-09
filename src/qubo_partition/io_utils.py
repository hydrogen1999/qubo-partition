"""Tiny dependency-light helpers for writing result tables (CSV/JSON/LaTeX)."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path


def write_csv(rows: list[dict], path: str | Path) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


def _default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=repr)
    return str(o)


def write_latex_table(
    rows: list[dict],
    columns: Sequence[str],
    path: str | Path,
    caption: str = "",
    label: str = "",
    float_fmt: str = "{:.4g}",
) -> None:
    """Write a booktabs-style LaTeX table for the paper."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        if isinstance(v, bool):
            return r"\checkmark" if v else r"$\times$"
        return str(v).replace("_", r"\_")

    align = "l" + "r" * (len(columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}" if caption else r"\caption{}",
        rf"\label{{{label}}}" if label else "",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(c.replace("_", r"\_") for c in columns) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(fmt(r.get(c, "")) for c in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(line for line in lines if line != ""))
