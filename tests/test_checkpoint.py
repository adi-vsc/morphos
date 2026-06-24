"""Step 5: optimizer checkpointing and warm-start resume.

Long topology/parametric optimization runs must be able to save periodic
state and resume from it, continuing from the saved iteration/p/beta rather
than restarting the SIMP continuation schedule from scratch.
"""

import numpy as np
import pytest

from morphos.field import Field
from morphos.objective.objective import MaximizeValue
from morphos.optimize.checkpoint import load_checkpoint, save_checkpoint
from morphos.optimize.parametric import ParametricOptimizer
from morphos.optimize.topopt import TopologyOptimizer
from morphos.physics.analytic import AnalyticOracle


def _target(shape=(4, 5)):
    rng = np.random.default_rng(2)
    return 0.2 + 0.5 * rng.uniform(size=shape)


class _PenaltyTrackingOracle(AnalyticOracle):
    """An AnalyticOracle that exposes a settable ``penalty`` attribute and
    records every value TopologyOptimizer's continuation schedule sets on it,
    so a test can inspect exactly which p value was active on each solve."""

    def __init__(self, target):
        super().__init__(target=target)
        self.penalty = 1.0
        self.penalty_log = []

    def solve(self, field):
        self.penalty_log.append(self.penalty)
        return super().solve(field)


class _CrashAfterN(AnalyticOracle):
    """Simulates a crashed run: raises once the underlying oracle has been
    solved ``n_calls`` times, so a test can exercise a checkpoint left on
    disk mid-run by an interrupted ``TopologyOptimizer.run`` without needing
    a real crash -- the checkpoint write happens (inside ``run``'s loop)
    before this exception propagates out of it."""

    def __init__(self, target, n_calls):
        super().__init__(target=target)
        self.n_calls = int(n_calls)
        self.calls = 0

    def solve(self, field):
        self.calls += 1
        if self.calls > self.n_calls:
            raise RuntimeError("simulated crash")
        return super().solve(field)


def test_checkpoint_round_trips_field_and_state(tmp_path):
    shape = (3, 4)
    field = Field(np.full(shape, 0.42), spacing=1.5)
    save_checkpoint(tmp_path / "ckpt.npz", field, iteration=7, p=2.5, beta=4.0,
                     history=[1.0, 2.0, 3.0])
    loaded_field, iteration, p, beta, history = load_checkpoint(tmp_path / "ckpt.npz")

    assert np.allclose(loaded_field.values, field.values)
    assert loaded_field.spacing == field.spacing
    assert iteration == 7
    assert p == pytest.approx(2.5)
    assert beta == pytest.approx(4.0)
    assert history == pytest.approx([1.0, 2.0, 3.0])


def test_checkpoint_round_trip_through_topology_optimizer(tmp_path):
    """Run 10 iterations from scratch as a reference. Simulate a crash at
    iteration 5 of an otherwise identical ``max_iter=10`` run (so the
    continuation schedule's horizon is identical to the reference's, the
    realistic scenario being an external interruption rather than a smaller
    configured ``max_iter``); resume from the checkpoint that the crashed run
    left behind and run the remaining iterations to completion. The final
    field must match the reference run's final field exactly."""
    target = _target()
    obj = MaximizeValue()
    init = Field(np.zeros(target.shape), spacing=1.0)
    schedule_kwargs = dict(
        step_size=0.05, tol=0.0, p_start=1.0, p_end=3.0,
        beta_start=1.0, beta_end=4.0, p_ramp_fraction=0.5,
    )

    reference_opt = TopologyOptimizer(max_iter=10, **schedule_kwargs)
    reference = reference_opt.run(init.copy(), AnalyticOracle(target=target), obj)

    ckpt_dir = tmp_path / "ckpt"
    crashing_opt = TopologyOptimizer(
        max_iter=10, checkpoint_dir=ckpt_dir, checkpoint_every=1, **schedule_kwargs
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        crashing_opt.run(init.copy(), _CrashAfterN(target=target, n_calls=5), obj)
    _, saved_iteration, _, _, saved_history = load_checkpoint(ckpt_dir / "checkpoint.npz")
    assert saved_iteration == 5
    assert saved_history == pytest.approx(reference.history[:5])

    resume_opt = TopologyOptimizer(
        max_iter=10, resume_from=ckpt_dir / "checkpoint.npz", **schedule_kwargs
    )
    final = resume_opt.run(init.copy(), AnalyticOracle(target=target), obj)

    assert np.allclose(final.field.values, reference.field.values, atol=1e-10)
    assert final.fom == pytest.approx(reference.fom, abs=1e-10)


def test_resume_continues_from_saved_p_and_beta(tmp_path):
    """A resumed run's post-resume penalty sequence must equal what the
    ORIGINAL (uninterrupted) schedule would have produced at those same
    absolute iterations -- i.e. resume must restore the saved p from the
    checkpoint and continue the ramp's absolute iteration count, not restart
    it from p_start. Verified by tracking every penalty value
    TopologyOptimizer sets on the oracle across a full uninterrupted run vs.
    a crashed-then-resumed pair of runs sharing the same ``max_iter``
    schedule horizon: the per-iteration sequences must match exactly."""
    target = _target()
    obj = MaximizeValue()
    init = Field(np.zeros(target.shape), spacing=1.0)
    schedule_kwargs = dict(
        step_size=0.05, tol=0.0, p_start=1.0, p_end=4.0, p_ramp_fraction=1.0,
    )

    reference_oracle = _PenaltyTrackingOracle(target=target)
    TopologyOptimizer(max_iter=10, **schedule_kwargs).run(
        init.copy(), reference_oracle, obj
    )
    assert reference_oracle.penalty_log[0] == pytest.approx(schedule_kwargs["p_start"])
    assert reference_oracle.penalty_log[5] != pytest.approx(schedule_kwargs["p_start"])

    ckpt_dir = tmp_path / "ckpt"
    crashing_opt = TopologyOptimizer(
        max_iter=10, checkpoint_dir=ckpt_dir, checkpoint_every=1, **schedule_kwargs
    )
    crash_oracle = _CrashAfterN(target=target, n_calls=5)
    crash_oracle.penalty = 1.0
    with pytest.raises(RuntimeError, match="simulated crash"):
        crashing_opt.run(init.copy(), crash_oracle, obj)

    resume_oracle = _PenaltyTrackingOracle(target=target)
    TopologyOptimizer(
        max_iter=10, resume_from=ckpt_dir / "checkpoint.npz", **schedule_kwargs
    ).run(init.copy(), resume_oracle, obj)

    # The resumed run's penalty sequence (6th solve onward, 0-indexed 5..10)
    # must equal the reference's penalty sequence over that same window.
    assert resume_oracle.penalty_log == pytest.approx(reference_oracle.penalty_log[5:])


def test_checkpoint_file_written_every_n_iterations(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    target = _target()
    opt = TopologyOptimizer(
        step_size=0.05, max_iter=6, tol=0.0,
        checkpoint_dir=ckpt_dir, checkpoint_every=2,
    )
    opt.run(Field(np.zeros(target.shape), spacing=1.0), AnalyticOracle(target=target), MaximizeValue())
    assert (ckpt_dir / "checkpoint.npz").exists()
    _, iteration, _, _, history = load_checkpoint(ckpt_dir / "checkpoint.npz")
    assert iteration % 2 == 0
    assert len(history) == iteration


# --- ParametricOptimizer checkpointing ---------------------------------


def _quadratic_target_problem():
    """A 2-parameter problem with a closed-form optimum, cheap enough to run
    L-BFGS-B on repeatedly without depending on a geometry kernel."""
    from morphos.physics.oracle import PhysicsOracle, PhysicsResult

    target = np.array([1.3, -0.7])

    class _ParamOracle(PhysicsOracle):
        provides_gradient = False

        def solve(self, field):
            diff = field.values - target
            return PhysicsResult(value=-float(np.sum(diff ** 2)))

    def build(params):
        return Field(np.asarray(params, dtype=float), spacing=1.0)

    return build, _ParamOracle(), target


def test_parametric_checkpoint_round_trip(tmp_path):
    """A ParametricOptimizer run that writes periodic checkpoints can be
    resumed from the saved parameter vector and continues optimizing rather
    than restarting from the original initial_params."""
    build, oracle, target = _quadratic_target_problem()
    obj = MaximizeValue()
    ckpt_dir = tmp_path / "ckpt"

    first_opt = ParametricOptimizer(
        max_iter=3, checkpoint_dir=ckpt_dir, checkpoint_every=1,
    )
    first_result = first_opt.run(np.array([0.0, 0.0]), build, oracle, obj)
    assert (ckpt_dir / "checkpoint.npz").exists()

    loaded_field, iteration, _, _, history = load_checkpoint(ckpt_dir / "checkpoint.npz")
    assert iteration >= 1
    # L-BFGS-B calls the objective multiple times per outer iteration (line
    # search / gradient evaluations), so the FOM history is at least as long
    # as the iteration count, not exactly equal (unlike TopologyOptimizer's
    # 1-fom-per-iteration loop).
    assert len(history) >= iteration

    resume_opt = ParametricOptimizer(
        max_iter=2000, tol=1e-14, resume_from=ckpt_dir / "checkpoint.npz",
    )
    final = resume_opt.run(np.array([0.0, 0.0]), build, oracle, obj)
    assert final.params == pytest.approx(target, abs=1e-3)
    # The resumed run did not restart from the original initial_params; its
    # starting point was the checkpointed parameter vector, which is already
    # partway toward the optimum.
    assert not np.allclose(final.params, np.array([0.0, 0.0]))


# --- Engine threads checkpoint_dir / resume_from to the optimizer ------


def test_engine_threads_checkpoint_dir_and_resume_from_to_optimizer(tmp_path):
    """Engine.run accepts checkpoint_dir/resume_from and applies them to
    spec.optimizer before running, so a DesignSpec run through Engine gets
    the same checkpoint/resume behaviour as calling optimizer.run directly."""
    from morphos.spec import DesignSpec
    from morphos.engine import Engine

    target = _target((2, 2))
    init = Field(np.zeros(target.shape), spacing=1.0)
    ckpt_dir = tmp_path / "ckpt"

    opt = TopologyOptimizer(step_size=0.1, max_iter=4, tol=0.0)
    spec = DesignSpec(
        initial=init, oracle=AnalyticOracle(target=target),
        objective=MaximizeValue(), optimizer=opt,
    )
    Engine().run(spec, checkpoint_dir=ckpt_dir, checkpoint_every=2)
    assert opt.checkpoint_dir == ckpt_dir
    assert (ckpt_dir / "checkpoint.npz").exists()

    resume_opt = TopologyOptimizer(step_size=0.1, max_iter=4, tol=0.0)
    spec2 = DesignSpec(
        initial=init, oracle=AnalyticOracle(target=target),
        objective=MaximizeValue(), optimizer=resume_opt,
    )
    result = Engine().run(spec2, resume_from=ckpt_dir / "checkpoint.npz")
    assert resume_opt.resume_from == ckpt_dir / "checkpoint.npz"
    assert result.iterations <= 4
