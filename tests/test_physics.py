"""Tests for src/tidetail/physics.py — surge prior helpers.

Every function is stateless and CPU-only; tests are deterministic.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tidetail.physics import (
    coastal_setup_from_stress,
    inverse_barometer,
    lag_kernel_calibration,
    pressure_anomaly_to_surge,
    surge_prior_kernel,
    tail_weight,
    wind_stress_proxy,
)

# ---------------------------------------------------------------------------
# 1. inverse_barometer
# ---------------------------------------------------------------------------

class TestInverseBarometer:
    def test_standard_atmosphere_zero(self):
        """p = p_ref → eta_IB = 0."""
        assert inverse_barometer(1013.25, 1013.25) == pytest.approx(0.0, abs=1e-6)

    def test_hurricane_pressure(self):
        """900 hPa relative to 1013.25 hPa → ~1.13 m IB surge.

        A Category 5 hurricane with 113 hPa pressure deficit produces
        a ~1.1 m equilibrium inverse-barometer component.  Dynamic
        amplification in real basins pushes this to the higher end of
        observed storm surges.
        """
        eta = inverse_barometer(900.0, 1013.25)
        assert eta > 0  # low pressure → higher sea
        # -(900 - 1013.25) * 100 Pa / (1025 * 9.81) ≈ 1.126 m
        assert eta == pytest.approx(113.25 * 100 / (1025 * 9.81), rel=1e-4)
        assert 1.0 < eta < 1.3  # physically plausible for major hurricane

    def test_high_pressure_negative(self):
        """1040 hPa → negative IB (sea level depressed ~0.27 m)."""
        eta = inverse_barometer(1040.0, 1013.25)
        assert eta < 0
        # -(1040 - 1013.25) * 100 / (1025 * 9.81) ≈ -0.266 m
        assert -0.30 < eta < -0.24

    def test_typical_daily_range(self):
        """Daily pressure variations (±10 hPa) → IB ±0.1 m range."""
        for p in (1003.25, 1008.25, 1018.25, 1023.25):
            eta = inverse_barometer(p, 1013.25)
            assert -0.15 < eta < 0.15, f"eta_IB={eta:.4f} m for p={p} hPa"

    # --- error paths ---
    def test_extreme_low_pressure_raises(self):
        with pytest.raises(ValueError, match="pressure_hpa"):
            inverse_barometer(800.0)

    def test_extreme_high_pressure_raises(self):
        with pytest.raises(ValueError, match="pressure_hpa"):
            inverse_barometer(1200.0)

    def test_zero_pref_raises(self):
        with pytest.raises(ValueError, match="pressure_ref_hpa"):
            inverse_barometer(1013.25, pressure_ref_hpa=0.0)

    def test_negative_pref_raises(self):
        with pytest.raises(ValueError, match="pressure_ref_hpa"):
            inverse_barometer(1013.25, pressure_ref_hpa=-1.0)

    def test_bad_rho_water_low(self):
        with pytest.raises(ValueError, match="rho_water"):
            inverse_barometer(1013.25, rho_water=800.0)

    def test_bad_rho_water_high(self):
        with pytest.raises(ValueError, match="rho_water"):
            inverse_barometer(1013.25, rho_water=1100.0)

    def test_bad_gravity(self):
        with pytest.raises(ValueError, match="g must be positive"):
            inverse_barometer(1013.25, g=0.0)


# ---------------------------------------------------------------------------
# 2. wind_stress_proxy
# ---------------------------------------------------------------------------

class TestWindStressProxy:
    def test_magnitude_only(self):
        """Strong eastward wind (20 m/s) — magnitude returned, no components."""
        result = wind_stress_proxy(u10_ms=20.0, v10_ms=0.0)
        assert "tau_magnitude" in result
        assert "tau_onshore" not in result  # no coastal_azimuth given
        assert "tau_alongshore" not in result
        assert result["tau_magnitude"] > 0
        # rho * Cd * |U|^2 = 1.225 * 0.0014 * 400
        assert result["tau_magnitude"] == pytest.approx(
            1.225 * 0.0014 * 400, rel=1e-3
        )

    def test_coastal_projection_perpendicular(self):
        """Eastward wind (u=+10) at coast facing south (az=180°).

        Coastal azimuth = 180° → direction from sea to land is south.
        Wind goes east → no component toward coast.
        """
        res = wind_stress_proxy(u10_ms=10.0, v10_ms=0.0, coastal_azimuth=180.0)
        assert "tau_onshore" in res
        assert "tau_alongshore" in res
        # Perpendicular: onshore component = 0
        assert res["tau_onshore"] == pytest.approx(0.0, abs=1e-9)

    def test_coastal_projection_directly_onshore(self):
        """Northward wind (v=+10) at coast facing north (az=0°).

        Coastal azimuth = 0° → land is north of sea.
        Wind goes north → directly onshore → tau_onshore = +tau_mag.
        """
        res = wind_stress_proxy(u10_ms=0.0, v10_ms=10.0, coastal_azimuth=0.0)
        tau_exp = 1.225 * 0.0014 * 10.0 ** 2
        assert res["tau_onshore"] == pytest.approx(tau_exp, rel=1e-6)
        assert res["tau_alongshore"] == pytest.approx(0.0, abs=1e-9)

    def test_wind_direction_cardinal_north(self):
        """u=0, v=10 → wind going north → az_wind = 0°."""
        res = wind_stress_proxy(u10_ms=0.0, v10_ms=10.0)
        assert res["wind_direction_deg"] == pytest.approx(0.0, abs=1e-6)

    def test_wind_direction_east(self):
        res = wind_stress_proxy(u10_ms=10.0, v10_ms=0.0)
        assert res["wind_direction_deg"] == pytest.approx(90.0, abs=1e-6)

    def test_wind_direction_south(self):
        res = wind_stress_proxy(u10_ms=0.0, v10_ms=-10.0)
        assert res["wind_direction_deg"] == pytest.approx(180.0, abs=1e-6)

    def test_wind_direction_west(self):
        res = wind_stress_proxy(u10_ms=-10.0, v10_ms=0.0)
        assert res["wind_direction_deg"] == pytest.approx(270.0, abs=1e-6)

    # --- error paths ---
    def test_zero_wind_raises(self):
        with pytest.raises(ValueError, match="wind_speed"):
            wind_stress_proxy(0.0, 0.0)

    def test_subthreshold_wind_raises(self):
        with pytest.raises(ValueError, match="wind_speed"):
            wind_stress_proxy(0.3, 0.4)  # |U| = 0.5

    def test_bad_azimuth_negative(self):
        with pytest.raises(ValueError, match="coastal_azimuth"):
            wind_stress_proxy(u10_ms=5.0, v10_ms=0.0, coastal_azimuth=-1.0)

    def test_bad_azimuth_over_360(self):
        with pytest.raises(ValueError, match="coastal_azimuth"):
            wind_stress_proxy(u10_ms=5.0, v10_ms=0.0, coastal_azimuth=360.0)

    def test_bad_rho_air(self):
        with pytest.raises(ValueError, match="rho_air"):
            wind_stress_proxy(u10_ms=5.0, v10_ms=0.0, rho_air=-1.0)

    def test_bad_cd(self):
        with pytest.raises(ValueError, match="cd"):
            wind_stress_proxy(u10_ms=5.0, v10_ms=0.0, cd=0.0)


# ---------------------------------------------------------------------------
# 3. coastal_setup_from_stress
# ---------------------------------------------------------------------------

class TestCoastalSetupFromStress:
    def test_typical_storm(self):
        """tau=0.3 Pa, 100 km basin, H=20 m → setup ≈ 0.15 m."""
        setup = coastal_setup_from_stress(
            tau_onshore_pa=0.3, water_depth_m=20.0, basin_width_m=100_000.0
        )
        expected = 0.3 * 100_000 / (1025 * 9.81 * 20)
        assert setup == pytest.approx(expected, rel=1e-3)
        assert 0.05 < setup < 0.5  # physically plausible

    def test_scaling(self):
        """Twice the stress → twice the setup."""
        s1 = coastal_setup_from_stress(0.2, 10.0, 50_000.0)
        s2 = coastal_setup_from_stress(0.4, 10.0, 50_000.0)
        assert s2 == pytest.approx(2 * s1, rel=1e-6)

    def test_scaling_depth_inverse(self):
        """Twice the depth → half the setup."""
        s1 = coastal_setup_from_stress(0.3, 10.0, 50_000.0)
        s2 = coastal_setup_from_stress(0.3, 20.0, 50_000.0)
        assert s1 / s2 == pytest.approx(2.0, rel=1e-6)

    # --- error paths ---
    def test_shallow_water_raises(self):
        with pytest.raises(ValueError, match="water_depth_m"):
            coastal_setup_from_stress(0.3, water_depth_m=0.1, basin_width_m=100_000.0)

    def test_zero_depth_raises(self):
        with pytest.raises(ValueError, match="water_depth_m"):
            coastal_setup_from_stress(0.3, 0.0, 100_000.0)

    def test_negative_stress_raises(self):
        with pytest.raises(ValueError, match="tau_onshore_pa"):
            coastal_setup_from_stress(-0.1, 10.0, 100_000.0)

    def test_zero_basin_width_raises(self):
        with pytest.raises(ValueError, match="basin_width_m"):
            coastal_setup_from_stress(0.3, 10.0, 0.0)


# ---------------------------------------------------------------------------
# 4. pressure_anomaly_to_surge
# ---------------------------------------------------------------------------

class TestPressureAnomaly:
    def test_zero_anomaly(self):
        assert pressure_anomaly_to_surge(1013.25) == pytest.approx(0.0, abs=1e-6)

    def test_negative_pressure_positive_anomaly(self):
        """950 hPa → -(950 - 1013.25) * 100 = +6325 Pa."""
        a = pressure_anomaly_to_surge(950.0)
        assert a == pytest.approx(6325.0, rel=1e-3)

    def test_high_pressure_negative_anomaly(self):
        a = pressure_anomaly_to_surge(1030.0)
        assert a == pytest.approx(-(1030 - 1013.25) * 100, rel=1e-3)

    def test_extreme_pressure_raises(self):
        with pytest.raises(ValueError, match="pressure_hpa"):
            pressure_anomaly_to_surge(800.0)


# ---------------------------------------------------------------------------
# 5. surge_prior_kernel
# ---------------------------------------------------------------------------

class TestSurgePriorKernel:
    def test_default_is_placeholder(self):
        k = surge_prior_kernel()
        assert k["calibrated"] is False
        assert "wind_lag_hours" in k
        assert "pressure_lag_hours" in k
        assert len(k["wind_lag_hours"]) > 0
        assert 0 in k["wind_lag_hours"]

    def test_custom_lags(self):
        k = surge_prior_kernel(
            lags_hours=24,
            wind_lag=[0, 6, 12, 24],
            pressure_lag=[0, 12],
        )
        assert k["wind_lag_hours"] == [0, 6, 12, 24]
        assert k["pressure_lag_hours"] == [0, 12]
        assert k["n_wind_lags"] == 4
        assert k["n_pressure_lags"] == 2


# ---------------------------------------------------------------------------
# 6. tail_weight
# ---------------------------------------------------------------------------

class TestTailWeight:
    def test_uniform_one_below_q95(self):
        vals = np.array([0.1, 0.2, 0.3])
        w = tail_weight(vals, train_q95=0.5, train_q99=1.0)
        np.testing.assert_array_equal(w, np.ones(3))

    def test_two_above_q99(self):
        vals = np.array([0.1, 1.5, 0.2])
        w = tail_weight(vals, train_q95=0.5, train_q99=1.0)
        assert w[1] == pytest.approx(2.0)
        assert w[0] == pytest.approx(1.0)
        assert w[2] == pytest.approx(1.0)

    def test_linear_ramp_between(self):
        vals = np.array([0.5, 0.75, 1.0])
        w = tail_weight(vals, train_q95=0.5, train_q99=1.0)
        assert w[0] == pytest.approx(0.0, abs=1e-9)  # at q95 → (0.5-0.5)/0.5 = 0
        assert w[1] == pytest.approx(0.5, abs=1e-9)
        assert w[2] == pytest.approx(1.0, abs=1e-9)  # at q99 → 1.0 (but not > q99)

    def test_nan_gets_weight_one(self):
        vals = np.array([0.1, np.nan, 0.8, 2.0])
        w = tail_weight(vals, train_q95=0.5, train_q99=1.0)
        assert math.isnan(w[1]) is False  # NaN replaced by 1
        assert w[1] == pytest.approx(1.0)

    # --- error paths ---
    def test_q95_ge_q99_raises(self):
        with pytest.raises(ValueError, match="q95"):
            tail_weight(np.array([0.1, 0.2]), train_q95=1.0, train_q99=1.0)

    def test_q95_greater_q99_raises(self):
        with pytest.raises(ValueError, match="q95"):
            tail_weight(np.array([0.1, 0.2]), train_q95=1.5, train_q99=0.5)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            tail_weight(np.array([0.1]), train_q95=-0.1, train_q99=0.5)


# ---------------------------------------------------------------------------
# 7. lag_kernel_calibration
# ---------------------------------------------------------------------------

class TestLagKernelCalibration:
    @pytest.fixture()
    def aligned_arrays(self):
        """Generate synthetic data where surge = 2*wind(t) + 0.5*IB(t-3).

        Uses only lags [0, 3] so least squares has no ambiguity —
        coefficients should be recoverable up to noise.
        """
        rng = np.random.default_rng(42)
        N = 500
        wind = rng.uniform(2.0, 15.0, size=N)
        # Pressure range chosen so IB values are small but non-negligible
        pressure = rng.uniform(990.0, 1020.0, size=N)
        # Build IB values once (in metres)
        ib_surge = np.array([
            inverse_barometer(float(p)) for p in pressure
        ])
        surge = np.full(N, 0.0)
        # Only fill from max_lag onward
        surge[3:] = (
            2.0 * wind[3:]
            + 0.5 * ib_surge[:N - 3]
            + rng.normal(0, 0.01, size=N - 3)
        )
        return surge, wind, pressure

    def test_recovers_coefficients(self, aligned_arrays):
        surge, wind, pressure = aligned_arrays
        result = lag_kernel_calibration(
            surge, wind, pressure, lags=[0, 3]
        )
        idx_0 = result["lags"].index(0)
        idx_3 = result["lags"].index(3)
        # Wind coefficient at lag 0 should be ~2.0
        assert result["w_wind"][idx_0] == pytest.approx(2.0, rel=0.2)
        # Pressure (IB) coefficient at lag 3 should be ~0.5
        assert result["w_pres"][idx_3] == pytest.approx(0.5, rel=0.3)
        # R² should be very high (low noise, correct lags)
        assert result["r_squared"] > 0.98

    def test_wrong_lag_near_zero(self, aligned_arrays):
        """Coefficients at lags that didn't generate surge should be ~0."""
        surge, wind, pressure = aligned_arrays
        result = lag_kernel_calibration(
            surge, wind, pressure, lags=[0, 6]
        )
        idx_0 = result["lags"].index(0)
        idx_6 = result["lags"].index(6)
        assert result["w_wind"][idx_0] == pytest.approx(2.0, rel=0.2)
        # Pressure at lag 6 should be near 0 (we used lag 3)
        assert abs(result["w_pres"][idx_6]) < 0.3

    def test_returned_shape(self, aligned_arrays):
        surge, wind, pressure = aligned_arrays
        lags = [0, 3, 6, 12, 24]
        r = lag_kernel_calibration(surge, wind, pressure, lags=lags)
        assert len(r["w_wind"]) == len(lags)
        assert len(r["w_pres"]) == len(lags)
        assert isinstance(r["intercept"], float)
        assert r["n_samples"] == len(surge) - max(lags)

    def test_short_arrays_raise(self):
        N = 5
        surge = np.zeros(N)
        wind = np.zeros(N)
        pressure = np.full(N, 1013.25)
        with pytest.raises(ValueError, match="max_lag"):
            lag_kernel_calibration(surge, wind, pressure, lags=[0, 24])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            lag_kernel_calibration(
                np.zeros(100), np.zeros(90), np.zeros(100), lags=[0]
            )

    def test_nan_raises(self):
        surge = np.zeros(100); surge[10] = np.nan
        wind = np.zeros(100)
        pressure = np.full(100, 1013.25)
        with pytest.raises(ValueError, match="NaN"):
            lag_kernel_calibration(surge, wind, pressure, lags=[0])
