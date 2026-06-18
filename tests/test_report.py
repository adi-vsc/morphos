import numpy as np
import pytest

from morphos.field import Field
from morphos.engine import Engine
from morphos.intent import CantileverIntent, ChannelIntent
from morphos.report import PerformanceReport, build_report


def test_elasticity_report_has_engineering_quantities():
    intent = CantileverIntent(span=10, height=5, load=-1.0, volume_fraction=0.4, max_iter=10)
    spec = intent.build()
    res = Engine().run(spec)
    report = build_report(res, spec.oracle)
    assert isinstance(report, PerformanceReport)
    assert report.quantities["compliance"] > 0.0
    assert report.quantities["max_displacement"] > 0.0
    assert 0.0 < report.mass_fraction <= 1.0
    assert report.figure_of_merit == pytest.approx(res.figure_of_merit)


def test_darcy_report_has_engineering_quantities():
    intent = ChannelIntent(span=10, height=5, source=1.0, volume_fraction=0.4, max_iter=10)
    spec = intent.build()
    res = Engine().run(spec)
    report = build_report(res, spec.oracle)
    assert report.quantities["dissipation"] > 0.0
    assert report.quantities["peak_pressure"] > 0.0


def test_report_carries_margin_when_bound_is_known():
    intent = CantileverIntent(span=10, height=5, load=-1.0, volume_fraction=0.4, max_iter=10)
    spec = intent.build()
    res = Engine().run(spec)
    report = build_report(res, spec.oracle)
    assert report.margin == pytest.approx(res.margin)
    assert report.attained_fraction == res.attained_fraction
