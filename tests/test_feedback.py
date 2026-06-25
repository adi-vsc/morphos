"""Tests for feedback ingestion and oracle calibration."""

import numpy as np
import pytest

from morphos.feedback import (
    CalibrationResult,
    FeedbackRecord,
    OracleCalibrator,
    PolynomialGainModel,
)


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


def test_feedback_record_extended_fields_default_and_round_trip():
    # New structured fields default to empty so existing 4-arg construction
    # (design_hash/oracle_type/predicted/measured) keeps working unchanged.
    rec = _record({"q": 1.0}, {"q": 1.1})
    assert rec.oracle_name == ""
    assert rec.operating_point == {}
    assert rec.quantities == {}

    rec2 = FeedbackRecord(
        design_hash="abc123",
        oracle_type="ConjugateHeatOracle",
        predicted={"peak_temperature_K": 900.0},
        measured={"peak_temperature_K": 950.0},
        oracle_name="cht_3d",
        operating_point={"temperature_K": 900.0, "pressure_Pa": 1e6},
        quantities={"peak_temperature_K": (900.0, 950.0)},
    )
    s = rec2.to_json()
    rec3 = FeedbackRecord.from_json(s)
    assert rec3.oracle_name == "cht_3d"
    assert rec3.operating_point == {"temperature_K": 900.0, "pressure_Pa": 1e6}
    assert rec3.quantities == {"peak_temperature_K": (900.0, 950.0)}


def _nonlinear_records(n=10, seed=0):
    """Synthetic records where measured = predicted + 0.02 * predicted**2,
    a genuinely nonlinear discrepancy that a degree-1 (linear) gain model
    cannot fit as well as a higher-degree polynomial."""
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n):
        x = float(rng.uniform(1.0, 10.0))
        measured = x + 0.02 * x ** 2
        records.append(
            FeedbackRecord(
                design_hash=f"d{_}",
                oracle_type="ConjugateHeatOracle",
                predicted={"T": x},
                measured={"T": measured},
                oracle_name="cht_3d",
                operating_point={"T": x},
                quantities={"T": (x, measured)},
            )
        )
    return records


def test_polynomial_model_outperforms_linear_on_nonlinear_data():
    records = _nonlinear_records()
    linear = PolynomialGainModel(degree=1)
    cubic = PolynomialGainModel(degree=3)

    cal_linear = OracleCalibrator(records, model=linear, oracle_name="cht_3d")
    cal_cubic = OracleCalibrator(records, model=cubic, oracle_name="cht_3d")

    result_linear = cal_linear.calibrate()
    result_cubic = cal_cubic.calibrate()

    rms_linear = float(np.sqrt(np.mean(np.asarray(result_linear.residuals) ** 2)))
    rms_cubic = float(np.sqrt(np.mean(np.asarray(result_cubic.residuals) ** 2)))
    assert rms_cubic < rms_linear


def test_calibration_result_has_positive_r_squared_on_synthetic_data():
    records = _nonlinear_records()
    model = PolynomialGainModel(degree=2)
    cal = OracleCalibrator(records, model=model, oracle_name="cht_3d")
    result = cal.calibrate()
    assert isinstance(result, CalibrationResult)
    assert result.r_squared > 0.0
    assert result.r_squared <= 1.0 + 1e-9
