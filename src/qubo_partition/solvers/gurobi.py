"""Exact QUBO solver using Gurobi."""

from __future__ import annotations

import time
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from qubo_partition.qubo.base import QUBO


@dataclass
class GurobiResult:
    sample: dict
    energy: float
    time_s: float


def gurobi_solve(qubo: QUBO) -> GurobiResult:
    """Solve a QUBO exactly with Gurobi."""

    model = gp.Model("qubo")
    model.Params.OutputFlag = 0

    vars_ = {
        v: model.addVar(vtype=GRB.BINARY, name=str(v))
        for v in qubo.variables
    }

    obj = gp.LinExpr()

    for v, bias in qubo.linear.items():
        obj += bias * vars_[v]

    for (u, v), bias in qubo.quadratic.items():
        obj += bias * vars_[u] * vars_[v]

    obj += qubo.offset

    model.setObjective(obj, GRB.MINIMIZE)

    t0 = time.perf_counter()
    model.optimize()
    dt = time.perf_counter() - t0

    sample = {
        v: int(round(vars_[v].X))
        for v in qubo.variables
    }

    return GurobiResult(
        sample=sample,
        energy=float(model.ObjVal),
        time_s=dt,
    )