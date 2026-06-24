"""Tests for feedback ingestion and oracle calibration."""

import numpy as np
import pytest

from morphos.feedback import FeedbackRecord, OracleCalibrator


def _record(pred, meas, h="abc123"):
    return FeedbackRecord(
        design_hash=h, oracle_type="ElasticityOracle",
        predicted=pred, measured=meas,
    )


def test_feedback_record_round_trips_json():
    rec = _record({"compliance": 1.0, "max_disp": 0.2},
                  {"compliance": 1.1, "max_disp": 0.25})
    rec2 = FeedbackRecord.from_json(rec.to_json())
    assert rec2.design_hash == rec.design_hash
    assert rec2.predicted == rec.predicted
    assert rec2.measured == rec.measured
    assert rec2.discrepancy == rec.discrepancy


def test_discrepancy_is_relative_error():
    rec = _record({"q": 2.0}, {"q": 2.4})
    assert rec.discrepancy["q"] == pytest.approx((2.4 - 2.0) / 2.0)


def test_calibrator_reduces_residual_on_synthetic_data():
    # Synthetic measurements with a known multiplicative bias of 1.3 on the
    # predictions; the calibrator must recover the gain within 1%.
    rng = np.random.default_rng(0)
    true_gain = 1.3
    records = []
    for _ in range(6):
        base = float(rng.uniform(1.0, 5.0))
        records.append(_record({"q": base}, {"q": true_gain * base}))
    cal = OracleCalibrator(records)  # default single-gain model
    before = np.sqrt(np.mean(cal.residuals() ** 2))
    fitted = cal.calibrate()
    assert fitted["gain"] == pytest.approx(true_gain, rel=1e-2)
    # residual at the fitted gain is far smaller than at the initial gain=1.
    cal_fitted = OracleCalibrator(records, x0=[fitted["gain"]])
    after = np.sqrt(np.mean(cal_fitted.residuals() ** 2))
    assert after < 0.01 * before


def test_calibrator_with_custom_model_recovers_offset():
    # measured = predicted + offset; fit the additive offset parameter.
    true_offset = 0.7
    records = [_record({"q": float(b)}, {"q": float(b) + true_offset}) for b in (1, 2, 3, 4)]

    def model(params, rec):
        return {k: v + params["offset"] for k, v in rec.predicted.items()}

    cal = OracleCalibrator(records, calibration_params=["offset"], model=model, x0=[0.0])
    fitted = cal.calibrate()
    assert fitted["offset"] == pytest.approx(true_offset, abs=1e-6)


def test_calibrator_raises_on_empty_records():
    with pytest.raises(ValueError):
        OracleCalibrator([])
