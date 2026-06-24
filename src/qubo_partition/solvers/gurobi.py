"""Exact QUBO solver using Gurobi (optional dependency).

Gurobi is not installed by default and requires a license; it is imported lazily
so the rest of the package works without it. Install with ``pip install '.[gurobi]'``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from qubo_partition.qubo.base import QUBO


@dataclass
class GurobiResult:
    sample: dict
    energy: float
    time_s: float


def gurobi_available() -> bool:
    """True if gurobipy can be imported (does not verify the license)."""
    try:
        import gurobipy  # noqa: F401

        return True
    except Exception:
        return False


def gurobi_solve(qubo: QUBO) -> GurobiResult:
    """Solve a QUBO exactly with Gurobi (requires gurobipy and a valid license)."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "gurobi_solve requires the optional 'gurobipy' package and a Gurobi "
            "license. Install with: pip install '.[gurobi]'."
        ) from e

    model = gp.Model("qubo")
    model.Params.OutputFlag = 0

    # index-based names avoid Gurobi issues with tuple/space-containing keys
    order = list(qubo.variables)
    idx = {v: i for i, v in enumerate(order)}
    gvars = {v: model.addVar(vtype=GRB.BINARY, name=f"x{idx[v]}") for v in order}

    obj = gp.LinExpr()
    for v, bias in qubo.linear.items():
        obj += bias * gvars[v]
    for (u, v), bias in qubo.quadratic.items():
        obj += bias * gvars[u] * gvars[v]
    obj += qubo.offset

    model.setObjective(obj, GRB.MINIMIZE)

    t0 = time.perf_counter()
    model.optimize()
    dt = time.perf_counter() - t0

    sample = {v: int(round(gvars[v].X)) for v in order}
    return GurobiResult(sample=sample, energy=float(model.ObjVal), time_s=dt)
