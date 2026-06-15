"""Engine: orchestration from DesignSpec to DesignResult.

The engine runs the optimizer against the oracle and objective under the
manufacturability constraint, then assembles the result, including the margin to
the physical limit when the objective exposes one.
"""

from __future__ import annotations

from morphos.spec import DesignSpec, DesignResult


class Engine:
    def run(self, spec: DesignSpec) -> DesignResult:
        opt = spec.optimizer.run(
            spec.initial, spec.oracle, spec.objective, spec.constraint
        )

        bound = spec.objective.bound()
        margin = None if bound is None else bound.margin(opt.fom)
        fraction = None if bound is None else bound.attained_fraction(opt.fom)

        manuf = {} if spec.constraint is None else spec.constraint.report(opt.field)

        return DesignResult(
            field=opt.field,
            figure_of_merit=opt.fom,
            bound=bound,
            margin=margin,
            attained_fraction=fraction,
            history=opt.history,
            iterations=opt.iterations,
            converged=opt.converged,
            used_finite_differences=opt.used_finite_differences,
            manufacturability=manuf,
        )
