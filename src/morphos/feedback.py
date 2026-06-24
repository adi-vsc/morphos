"""Feedback ingestion and oracle calibration.

The loop the engine is missing: record what a physical test measured against what
an oracle predicted, then fit the oracle's calibration parameters to shrink the
gap. A :class:`FeedbackRecord` ties one measurement set to a design (by STL hash)
and auto-computes the relative discrepancy; an :class:`OracleCalibrator` fits
calibration parameters by least squares (Levenberg-Marquardt) over a collection
of records, given a forward model mapping parameters to predicted quantities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as _dc_field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class FeedbackRecord:
    """One physical test result linked to a design.

    ``discrepancy`` is filled automatically as ``(measured - predicted) / predicted``
    for every shared key.
    """

    design_hash: str
    oracle_type: str
    predicted: Dict[str, float]
    measured: Dict[str, float]
    timestamp: str = ""
    notes: str = ""
    discrepancy: Dict[str, float] = _dc_field(default_factory=dict)

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
            }
        )

    @classmethod
    def from_json(cls, s: str) -> "FeedbackRecord":
        d = json.loads(s)
        return cls(
            design_hash=d["design_hash"],
            oracle_type=d["oracle_type"],
            predicted=d["predicted"],
            measured=d["measured"],
            timestamp=d.get("timestamp", ""),
            notes=d.get("notes", ""),
            discrepancy=d.get("discrepancy", {}),
        )


def _default_gain_model(params: Dict[str, float], record: FeedbackRecord) -> Dict[str, float]:
    """Default forward model: a single multiplicative gain correcting a
    systematic bias, ``corrected = gain * predicted`` for every quantity."""
    gain = params.get("gain", 1.0)
    return {k: gain * v for k, v in record.predicted.items()}


class OracleCalibrator:
    """Fit calibration parameters to reduce prediction error across records.

    Parameters
    ----------
    records:
        The :class:`FeedbackRecord` collection to fit against.
    calibration_params:
        Names of the scalar parameters to fit.
    model:
        ``model(params: dict, record) -> dict`` predicting each measured quantity
        under candidate parameters. Defaults to a single multiplicative ``gain``
        (so ``calibration_params`` defaults to ``["gain"]``).
    x0:
        Initial parameter values (defaults to ones).

    The residual for one (record, key) is the relative error
    ``(model - measured) / measured``; :meth:`calibrate` minimises the sum of
    their squares by Levenberg-Marquardt and returns the fitted parameters.
    """

    def __init__(
        self,
        records: Sequence[FeedbackRecord],
        calibration_params: Optional[Sequence[str]] = None,
        model: Optional[Callable[[Dict[str, float], FeedbackRecord], Dict[str, float]]] = None,
        x0: Optional[Sequence[float]] = None,
    ) -> None:
        if not records:
            raise ValueError("OracleCalibrator needs at least one FeedbackRecord")
        self.records = list(records)
        self.model = model or _default_gain_model
        self.calibration_params = list(calibration_params) if calibration_params else ["gain"]
        if x0 is None:
            x0 = np.ones(len(self.calibration_params))
        self.x0 = np.asarray(x0, dtype=float)
        if self.x0.shape != (len(self.calibration_params),):
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

    def calibrate(self) -> Dict[str, float]:
        """Fit the calibration parameters and return them as a dict."""
        from scipy.optimize import least_squares

        n_res = self._residual_vector(self.x0).size
        if n_res == 0:
            raise ValueError("no overlapping predicted/measured quantities to fit")
        # 'lm' needs residuals >= parameters; fall back to 'trf' otherwise.
        method = "lm" if n_res >= self.x0.size else "trf"
        sol = least_squares(self._residual_vector, self.x0, method=method)
        return self._params_dict(sol.x)
