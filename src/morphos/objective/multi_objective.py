"""Multi-objective optimization: scalarised weighted-sum figure of merit over
several physics objectives, plus a Pareto archive of non-dominated designs.

Each component objective maps its own oracle's :class:`PhysicsResult` to a figure
of merit (higher is better, the engine convention). The combined scalar figure of
merit the optimizer maximises is the weighted sum ``weights . values``, and its
gradient is the matching weighted sum of the per-objective gradients. As the run
explores different weightings, every non-dominated objective vector is kept in a
Pareto archive so the trade-off front can be reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import List, Sequence, Tuple

import numpy as np

from morphos.field import Field
from morphos.objective.objective import Objective
from morphos.physics.oracle import PhysicsOracle


@dataclass
class ObjectiveVector:
    """A vector of figures of merit and the scalarised value/gradient."""

    values: np.ndarray              # (n_objectives,) per-objective FOM
    gradient: np.ndarray            # (n_objectives, *field_shape) per-objective grad
    weights: np.ndarray             # (n_objectives,) scalarisation weights
    scalar_fom: float               # weights . values
    scalar_grad: np.ndarray         # (*field_shape) weighted-sum gradient


class MultiObjective:
    """Scalarise several objectives into one figure of merit with Pareto archiving.

    Parameters
    ----------
    objectives:
        One :class:`Objective` per figure of merit.
    oracles:
        One :class:`PhysicsOracle` per objective (the same oracle may appear more
        than once; each objective is evaluated on its oracle's result).
    weights:
        Initial scalarisation weights; normalised to sum to 1.
    """

    def __init__(
        self,
        objectives: Sequence[Objective],
        oracles: Sequence[PhysicsOracle],
        weights: Sequence[float],
    ) -> None:
        if not (len(objectives) == len(oracles) == len(weights)):
            raise ValueError("objectives, oracles and weights must have equal length")
        if len(objectives) < 1:
            raise ValueError("need at least one objective")
        self.objectives = list(objectives)
        self.oracles = list(oracles)
        self.weights = self._normalise(weights)
        self.pareto_archive: List[Tuple[np.ndarray, np.ndarray]] = []

    @staticmethod
    def _normalise(weights) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        s = w.sum()
        if s <= 0:
            raise ValueError("weights must sum to a positive value")
        return w / s

    def update_weights(self, new_weights) -> None:
        self.weights = self._normalise(new_weights)

    def evaluate_design(self, field: Field) -> ObjectiveVector:
        """Run every oracle on ``field``, evaluate every objective, and combine
        into the scalarised figure of merit and gradient. Records the resulting
        objective vector in the Pareto archive."""
        values = []
        grads = []
        for objective, oracle in zip(self.objectives, self.oracles):
            ov = objective.evaluate(oracle.solve(field))
            if ov.gradient is None:
                raise ValueError(
                    "MultiObjective requires gradient-providing objectives/oracles"
                )
            values.append(float(ov.fom))
            grads.append(np.asarray(ov.gradient))
        values = np.asarray(values)
        grads = np.stack(grads, axis=0)
        scalar_fom = float(self.weights @ values)
        scalar_grad = np.tensordot(self.weights, grads, axes=(0, 0))

        self.update_pareto_archive(values)
        return ObjectiveVector(
            values=values,
            gradient=grads,
            weights=self.weights.copy(),
            scalar_fom=scalar_fom,
            scalar_grad=scalar_grad,
        )

    @staticmethod
    def is_dominated(fom_a: np.ndarray, fom_b: np.ndarray) -> bool:
        """True if ``fom_a`` is dominated by ``fom_b`` (b is >= a in every
        component and strictly greater in at least one). Higher is better."""
        a, b = np.asarray(fom_a), np.asarray(fom_b)
        return bool(np.all(b >= a) and np.any(b > a))

    def update_pareto_archive(self, fom_vector: np.ndarray) -> None:
        """Add ``fom_vector`` if it is non-dominated, pruning any archived
        solution it dominates."""
        fom = np.asarray(fom_vector, dtype=float)
        # Drop archived points dominated by the newcomer.
        self.pareto_archive = [
            (w, f) for (w, f) in self.pareto_archive if not self.is_dominated(f, fom)
        ]
        # Add the newcomer unless an existing point dominates it (or it duplicates).
        dominated = any(self.is_dominated(fom, f) for (w, f) in self.pareto_archive)
        duplicate = any(np.allclose(fom, f) for (w, f) in self.pareto_archive)
        if not dominated and not duplicate:
            self.pareto_archive.append((self.weights.copy(), fom.copy()))
