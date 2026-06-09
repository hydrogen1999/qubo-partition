"""Shared closed-form QUBO container with energy eval, brute-force, and dimod interop."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import product

import numpy as np

Variable = Hashable
Sample = Mapping[Variable, int]


@dataclass
class QUBO:
    """A quadratic unconstrained binary optimization objective."""

    linear: dict[Variable, float] = field(default_factory=dict)
    quadratic: dict[tuple[Variable, Variable], float] = field(default_factory=dict)
    offset: float = 0.0

    def add_variable(self, v: Variable, bias: float = 0.0) -> None:
        """Register variable ``v`` (so it survives even with zero bias) and add ``bias``."""
        self.linear[v] = self.linear.get(v, 0.0) + float(bias)

    def add_linear(self, v: Variable, bias: float) -> None:
        self.linear[v] = self.linear.get(v, 0.0) + float(bias)

    def add_quadratic(self, u: Variable, v: Variable, bias: float) -> None:
        if u == v:
            # x_i^2 == x_i for binary variables, so a self-pair is a linear term.
            self.add_linear(u, bias)
            return
        key = self._edge(u, v)
        self.quadratic[key] = self.quadratic.get(key, 0.0) + float(bias)
        self.linear.setdefault(u, 0.0)
        self.linear.setdefault(v, 0.0)

    def add_offset(self, c: float) -> None:
        self.offset += float(c)

    @staticmethod
    def _edge(u: Variable, v: Variable) -> tuple[Variable, Variable]:
        """Canonical unordered key for a pair, robust to mixed/unorderable types."""
        try:
            return (u, v) if u <= v else (v, u)  # type: ignore[operator]
        except TypeError:
            return (u, v) if repr(u) <= repr(v) else (v, u)

    @property
    def variables(self) -> tuple[Variable, ...]:
        return tuple(self.linear.keys())

    @property
    def num_variables(self) -> int:
        return len(self.linear)

    def energy(self, sample: Sample) -> float:
        """Evaluate ``H(x)`` for a full assignment ``sample : var -> {0, 1}``."""
        e = self.offset
        for v, a in self.linear.items():
            xi = sample[v]
            if xi:
                e += a
        for (u, v), b in self.quadratic.items():
            if sample[u] and sample[v]:
                e += b
        return e

    def brute_force(self) -> tuple[dict[Variable, int], float]:
        """Exhaustively minimize over all ``2**n`` assignments (toy instances only)."""
        vars_ = list(self.linear.keys())
        n = len(vars_)
        if n > 24:
            raise ValueError(
                f"brute_force() refuses {n} variables (> 24); use a dedicated exact solver."
            )
        best_sample: dict[Variable, int] = {}
        best_energy = np.inf
        for bits in product((0, 1), repeat=n):
            sample = dict(zip(vars_, bits))
            e = self.energy(sample)
            if e < best_energy:
                best_energy, best_sample = e, sample
        return best_sample, float(best_energy)

    def to_bqm(self):
        """Return an equivalent ``dimod.BinaryQuadraticModel`` (BINARY vartype)."""
        import dimod

        bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
        for v, a in self.linear.items():
            bqm.add_variable(v, a)
        for (u, v), b in self.quadratic.items():
            bqm.add_interaction(u, v, b)
        bqm.offset = self.offset
        return bqm

    @classmethod
    def from_terms(
        cls,
        linear: Mapping[Variable, float] | None = None,
        quadratic: Mapping[tuple[Variable, Variable], float] | None = None,
        offset: float = 0.0,
    ) -> QUBO:
        q = cls(offset=float(offset))
        for v, a in (linear or {}).items():
            q.add_linear(v, a)
        for (u, v), b in (quadratic or {}).items():
            q.add_quadratic(u, v, b)
        return q

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"QUBO(n_vars={self.num_variables}, n_quad={len(self.quadratic)}, "
            f"offset={self.offset:g})"
        )


def assignment_to_array(sample: Sample, order: Iterable[Variable]) -> np.ndarray:
    """Project a ``var -> bit`` map onto a 0/1 numpy array in a fixed order."""
    return np.array([int(sample[v]) for v in order], dtype=np.int8)
