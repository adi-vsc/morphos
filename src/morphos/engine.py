"""Engine: orchestration from DesignSpec to DesignResult.

The engine runs the optimizer against the oracle and objective under the
manufacturability constraint, then assembles the result, including the margin to
the physical limit when the objective exposes one.
"""

from __future__ import annotations

from typing import List

from morphos.spec import CoupledSpec, DesignSpec, DesignResult, ParametricSpec


def _result_from_opt(opt, objective, constraint) -> DesignResult:
    """Assemble a DesignResult from an optimizer result, including the margin to
    the physical limit when the objective exposes one. Shared by Engine and
    CoupledEngine."""
    bound = objective.bound()
    margin = None if bound is None else bound.margin(opt.fom)
    fraction = None if bound is None else bound.attained_fraction(opt.fom)
    manuf = {} if constraint is None else constraint.report(opt.field)
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


class Engine:
    def run(
        self,
        spec,
        export_dir=None,
        print_params=None,
        iso_value: float = 0.5,
        checkpoint_dir=None,
        checkpoint_every=None,
        resume_from=None,
        on_iteration=None,
    ) -> DesignResult:
        """Run ``spec`` through its optimizer.

        ``checkpoint_dir``/``checkpoint_every``/``resume_from``, when given,
        are set onto ``spec.optimizer`` before it runs (overriding whatever
        the optimizer was constructed with), threading the checkpoint/resume
        contract that :class:`~morphos.optimize.topopt.TopologyOptimizer` and
        :class:`~morphos.optimize.parametric.ParametricOptimizer` both expose
        through to the engine entry point, so callers do not need to reach
        into the optimizer object directly. Optimizers that already have
        these attributes set (constructed with them directly) are left
        unchanged when the corresponding ``Engine.run`` argument is omitted.
        """
        from pathlib import Path

        if checkpoint_dir is not None and hasattr(spec.optimizer, "checkpoint_dir"):
            spec.optimizer.checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_every is not None and hasattr(spec.optimizer, "checkpoint_every"):
            spec.optimizer.checkpoint_every = int(checkpoint_every)
        if resume_from is not None and hasattr(spec.optimizer, "resume_from"):
            spec.optimizer.resume_from = Path(resume_from)

        if isinstance(spec, ParametricSpec):
            opt = spec.optimizer.run(
                spec.initial_params,
                spec.build,
                spec.oracle,
                spec.objective,
                spec.constraint,
                on_iteration=on_iteration,
            )
        else:
            opt = spec.optimizer.run(
                spec.initial,
                spec.oracle,
                spec.objective,
                spec.constraint,
                on_iteration=on_iteration,
            )

        bound = spec.objective.bound()
        margin = None if bound is None else bound.margin(opt.fom)
        fraction = None if bound is None else bound.attained_fraction(opt.fom)

        manuf = {} if spec.constraint is None else spec.constraint.report(opt.field)

        result = DesignResult(
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

        # Optional geometry export: materialise the design to STL + voxel sidecar
        # and populate the result's export state (requires a 3D density field).
        if export_dir is not None:
            from morphos.manufacturing.export import PrintParams, export_bundle

            if print_params is None:
                print_params = PrintParams(
                    material="SS316L", layer_thickness_mm=0.04, laser_power_W=200.0,
                    scan_speed_mm_s=800.0, hatch_spacing_mm=0.1,
                )
            export_bundle(result, print_params, export_dir, iso_value=iso_value)

        return result


class CoupledEngine:
    """Orchestrates a multi-stage coupled optimization, staggered or monolithic.

    Staggered (``spec.coupling_mode == "staggered"``, the default): for each of
    ``n_outer`` outer iterations, run every stage's optimizer to convergence on
    the shared design Field, in order; after each stage re-solve its oracle once
    to recover the ``aux`` quantities and forward any ``passthrough`` attributes
    into the next stage's oracle. Returns one ``DesignResult`` per stage from the
    final outer iteration.

    Monolithic (``spec.coupling_mode == "monolithic"``): each stage's oracle is
    expected to already solve its physics jointly in one shot (e.g.
    :class:`morphos.physics.coupled.MonolithicCoupledOracle`), so there is no
    outer fixed-point loop or passthrough step to run -- every stage's optimizer
    runs exactly once, in order, on the shared field. This is the dispatch point
    that keeps the staggered path's code and behaviour completely unchanged
    while giving callers a one-shot alternative for strongly coupled physics.
    """

    def run(self, spec: CoupledSpec) -> List[DesignResult]:
        if not spec.stages:
            raise ValueError("CoupledSpec has no stages")
        if spec.coupling_mode == "monolithic":
            return self._run_monolithic(spec)
        return self._run_staggered(spec)

    def _run_staggered(self, spec: CoupledSpec) -> List[DesignResult]:
        field = spec.initial_field
        stage_results: List[DesignResult] = []

        for _ in range(spec.n_outer):
            stage_aux: dict = {}
            stage_results = []
            for i, (oracle, objective, constraint) in enumerate(spec.stages):
                # Forward passthrough quantities from the previous stage's solve.
                if (i - 1) in spec.passthrough and (i - 1) in stage_aux:
                    for attr in spec.passthrough[i - 1]:
                        if attr in stage_aux[i - 1]:
                            setattr(oracle, attr, stage_aux[i - 1][attr])
                            # Invalidate any per-spacing cache so the oracle
                            # rebuilds with the freshly forwarded field.
                            if hasattr(oracle, "_cache_h"):
                                oracle._cache_h = None

                opt = spec.optimizer.run(field, oracle, objective, constraint)
                field = opt.field
                stage_aux[i] = oracle.solve(field).aux
                stage_results.append(_result_from_opt(opt, objective, constraint))

        return stage_results

    def _run_monolithic(self, spec: CoupledSpec) -> List[DesignResult]:
        field = spec.initial_field
        stage_results: List[DesignResult] = []
        for oracle, objective, constraint in spec.stages:
            opt = spec.optimizer.run(field, oracle, objective, constraint)
            field = opt.field
            stage_results.append(_result_from_opt(opt, objective, constraint))
        return stage_results
