"""Feedback ingestion and oracle calibration.

The loop the engine is missing: record what a physical test measured against what
an oracle predicted, then fit the oracle's calibration parameters to shrink the
gap. A :class:`FeedbackRecord` ties one measurement set to a design (by STL hash)
and auto-computes the relative discrepancy; an :class:`OracleCalibrator` fits
calibration parameters by least squares (Levenberg-Marquardt) over a collection
of records, given a forward model mapping parameters to predicted quantities.

Real test data is often not a single scalar bias: an oracle can be accurate in
one operating regime and systematically off in another (e.g. a heat oracle
that over-predicts at high temperature). :class:`PolynomialGainModel` fits a
polynomial gain in the record's ``operating_point`` scalars, and
:class:`OracleCalibrator` (when given a model with a ``.fit`` method) delegates
straight to it, returning a :class:`CalibrationResult` with the fitted
coefficients, before/after residuals, and an R^2 score -- structured enough to
attach to a :class:`~morphos.report.PerformanceReport`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as _dc_field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FeedbackRecord:
    """One physical test result linked to a design.

    ``discrepancy`` is filled automatically as ``(measured - predicted) / predicted``
    for every shared key. ``oracle_name``, ``operating_point``, and ``quantities``
    are optional structured fields (all default to empty) added for multi-record,
    multi-regime calibration: ``operating_point`` records the scalar conditions
    (e.g. temperature, pressure) at which the measurement was taken, and
    ``quantities`` pairs each measured quantity with its (predicted, measured)
    tuple for structured comparison rather than the single scalar discrepancy.
    """

    design_hash: str
    oracle_type: str
    predicted: Dict[str, float]
    measured: Dict[str, float]
    timestamp: str = ""
    notes: str = ""
    discrepancy: Dict[str, float] = _dc_field(default_factory=dict)
    oracle_name: str = ""
    operating_point: Dict[str, float] = _dc_field(default_factory=dict)
    quantities: Dict[str, Tuple[float, float]] = _dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.discrepancy:
            self.discrepancy = self._compute_discrepancy()

    def _compute_discrepancy(self) -> Dict[str, float]:
        out = {}
        for k, pred in self.predicted.items():
            if k in self.measured and abs(pred) > 1e-300:
                out[k] = (self.measured[k] - pred) / pred
        return out

    def to_json(self) -> str:
        return json.dumps(
            {
                "design_hash": self.design_hash,
                "oracle_type": self.oracle_type,
                "predicted": self.predicted,
                "measured": self.measured,
                "timestamp": self.timestamp,
                "notes": self.notes,
                "discrepancy": self.discrepancy,
                "oracle_name": self.oracle_name,
                "operating_point": self.operating_point,
                "quantities": {k: list(v) for k, v in self.quantities.items()},
            }
        )

    @classmethod
    def from_json(cls, s: str) -> "FeedbackRecord":
        d = json.loads(s)
        quantities = {k: tuple(v) for k, v in d.get("quantities", {}).items()}
        return cls(
            design_hash=d["design_hash"],
            oracle_type=d["oracle_type"],
            predicted=d["predicted"],
            measured=d["measured"],
            timestamp=d.get("timestamp", ""),
            notes=d.get("notes", ""),
            discrepancy=d.get("discrepancy", {}),
            oracle_name=d.get("oracle_name", ""),
            operating_point=d.get("operating_point", {}),
            quantities=quantities,
        )


def _default_gain_model(params: Dict[str, float], record: FeedbackRecord) -> Dict[str, float]:
    """Default forward model: a single multiplicative gain correcting a
    systematic bias, ``corrected = gain * predicted`` for every quantity."""
    gain = params.get("gain", 1.0)
    return {k: gain * v for k, v in record.predicted.items()}


@dataclass
class CalibrationResult:
    """Outcome of fitting a calibration model across a record collection.

    ``coeffs`` are the fitted model coefficients (interpretation is
    model-specific; for :class:`PolynomialGainModel` they are the polynomial
    coefficients, highest degree first, as returned by ``numpy.polyfit``).
    ``residuals`` is the per-(record, quantity) residual vector *after*
    fitting. ``r_squared`` is the coefficient of determination of the fit
    against the raw discrepancy it is correcting.
    """

    coeffs: List[float]
    residuals: List[float]
    r_squared: float


class PolynomialGainModel:
    """Fits a polynomial correction of the predicted value to the measured
    value, using a single scalar drawn from each record's ``operating_point``
    (or, if absent, the predicted value itself) as the independent variable.

    ``degree=1`` (the default) reduces to an affine fit ``measured = a*x + b``,
    equivalent in expressive power to the existing single-gain model but fit
    by polynomial least squares rather than nonlinear least squares. Higher
    degrees capture the structured, regime-dependent discrepancies (e.g.
    over-prediction at high temperature) that a single scalar gain cannot.
    """

    def __init__(self, degree: int = 1, operating_point_key: Optional[str] = None) -> None:
        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.degree = int(degree)
        self.operating_point_key = operating_point_key

    def _x_for(self, record: FeedbackRecord, quantity: str, predicted: float) -> float:
        if self.operating_point_key is not None:
            return float(record.operating_point[self.operating_point_key])
        if quantity in record.operating_point:
            return float(record.operating_point[quantity])
        if record.operating_point:
            return float(next(iter(record.operating_point.values())))
        return float(predicted)

    def _samples(self, records: Sequence[FeedbackRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten every record's quantities into parallel (x, predicted,
        measured) arrays, preferring the structured ``quantities`` dict and
        falling back to the plain ``predicted``/``measured`` dicts so the
        model also works on records built without the new fields."""
        xs, preds, meas = [], [], []
        for rec in records:
            pairs = rec.quantities if rec.quantities else {
                k: (v, rec.measured[k]) for k, v in rec.predicted.items() if k in rec.measured
            }
            for k, (p, m) in pairs.items():
                xs.append(self._x_for(rec, k, p))
                preds.append(float(p))
                meas.append(float(m))
        return np.asarray(xs, dtype=float), np.asarray(preds, dtype=float), np.asarray(meas, dtype=float)

    def fit(self, records: Sequence[FeedbackRecord]) -> CalibrationResult:
        """Fit ``measured ~ polynomial(x)`` by ordinary least squares and
        return a :class:`CalibrationResult` with coefficients, the fitted
        residuals, and R^2 against the measured values."""
        x, _pred, meas = self._samples(records)
        if x.size == 0:
            raise ValueError("no quantities to fit")
        degree = min(self.degree, max(0, x.size - 1))
        coeffs = np.polyfit(x, meas, degree)
        fitted = np.polyval(coeffs, x)
        residuals = fitted - meas
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((meas - meas.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-300 else 1.0
        return CalibrationResult(
            coeffs=coeffs.tolist(), residuals=residuals.tolist(), r_squared=float(r_squared),
        )


class OracleCalibrator:
    """Fit calibration parameters to reduce prediction error across records.

    Parameters
    ----------
    records:
        The :class:`FeedbackRecord` collection to fit against.
    calibration_params:
        Names of the scalar parameters to fit.
    model:
        Either ``model(params: dict, record) -> dict`` predicting each measured
        quantity under candidate parameters (the legacy nonlinear-least-squares
        path; defaults to a single multiplicative ``gain``, so
        ``calibration_params`` defaults to ``["gain"]``), or a model object
        exposing ``.fit(records) -> CalibrationResult`` such as
        :class:`PolynomialGainModel`, in which case :meth:`calibrate` delegates
        straight to it.
    x0:
        Initial parameter values (defaults to ones); unused for ``.fit``-style
        models.
    oracle_name:
        When given, restricts the fit to records whose ``oracle_name`` matches
        (records without ``oracle_name`` set are not filtered out, so legacy
        records remain usable).

    The residual for one (record, key) is the relative error
    ``(model - measured) / measured``; :meth:`calibrate` minimises the sum of
    their squares by Levenberg-Marquardt and returns the fitted parameters.
    """

    def __init__(
        self,
        records: Sequence[FeedbackRecord],
        calibration_params: Optional[Sequence[str]] = None,
        model: Optional[object] = None,
        x0: Optional[Sequence[float]] = None,
        oracle_name: Optional[str] = None,
    ) -> None:
        if not records:
            raise ValueError("OracleCalibrator needs at least one FeedbackRecord")
        records = list(records)
        if oracle_name is not None:
            records = [r for r in records if not r.oracle_name or r.oracle_name == oracle_name]
            if not records:
                raise ValueError(f"no records match oracle_name={oracle_name!r}")
        self.records = records
        self.model = model or _default_gain_model
        self.calibration_params = list(calibration_params) if calibration_params else ["gain"]
        if x0 is None:
            x0 = np.ones(len(self.calibration_params))
        self.x0 = np.asarray(x0, dtype=float)
        if not hasattr(self.model, "fit") and self.x0.shape != (len(self.calibration_params),):
            raise ValueError("x0 length must match calibration_params")

    def _params_dict(self, x: np.ndarray) -> Dict[str, float]:
        return {name: float(v) for name, v in zip(self.calibration_params, x)}

    def _residual_vector(self, x: np.ndarray) -> np.ndarray:
        params = self._params_dict(x)
        res = []
        for rec in self.records:
            pred = self.model(params, rec)
            for k, meas in rec.measured.items():
                if k in pred:
                    denom = meas if abs(meas) > 1e-300 else 1.0
                    res.append((pred[k] - meas) / denom)
        return np.asarray(res, dtype=float)

    def residuals(self) -> np.ndarray:
        """Current relative residual vector at the initial parameters."""
        return self._residual_vector(self.x0)

    def calibrate(self):
        """Fit the calibration parameters.

        When ``model`` exposes ``.fit`` (e.g. :class:`PolynomialGainModel`),
        delegates to it and returns its :class:`CalibrationResult` directly.
        Otherwise runs the legacy nonlinear-least-squares fit (Levenberg-
        Marquardt) and returns the fitted parameters as a dict.
        """
        if hasattr(self.model, "fit"):
            return self.model.fit(self.records)

        from scipy.optimize import least_squares

        n_res = self._residual_vector(self.x0).size
        if n_res == 0:
            raise ValueError("no overlapping predicted/measured quantities to fit")
        # 'lm' needs residuals >= parameters; fall back to 'trf' otherwise.
        method = "lm" if n_res >= self.x0.size else "trf"
        sol = least_squares(self._residual_vector, self.x0, method=method)
        return self._params_dict(sol.x)
