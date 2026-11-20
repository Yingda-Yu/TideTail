"""Physically grounded surge priors for TideTail.

This module implements simple, CPU-only, unit-testable surge-related
quantities drawn from coastal physical oceanography.  Every function is
stateless, top-level, and has no side effects so that it can be exercised
directly in pytest.

DATUM AND UNITS
---------------
Consistent with FORECAST_TASK_CONTRACT.md §2 and total_water_eval.py, all
water-level quantities are in **metres above MLLW** unless explicitly
stated otherwise.

CORE PHYSICAL ASSUMPTIONS
-------------------------
1. **Inverse barometer effect**:  A 1 hPa atmospheric pressure anomaly
   relative to the regional mean sea-level pressure produces a surge
   of roughly 1 cm.  We use the hydrostatic balance
       eta_IB = -(p - p_ref) / (rho_water * g)
   with rho_water = 1025 kg/m³ and g = 9.81 m/s².  This is the classic
   equilibrium approximation (Proudman & Doodson 1924) and holds well
   for basins wider than the barotropic Rossby radius (~200 km).
   Limitations: ignores dynamic adjustments, phase lags, and density
   stratification.

2. **Wind stress proxy**:  Surface wind stress is estimated from the
   10-metre wind using the bulk drag formula
       tau = rho_air * C_D * |U10| * U10
   with C_D ≈ 1.4e-3 (Large & Pond 1981, standard for moderate winds).
   Below 1 m/s the drag law is unreliable (regime change), so we raise.

3. **Coastal setup**:  A steady onshore wind stress drives a coastal
   surge via the along-basin momentum balance.  The simplified form we
   use here is
       setup = tau_onshore * basin_width / (rho_water * g * H)
   where H is mean water depth.  This is NOT a full shallow-water
   solution — it neglects Coriolis, advection, nonlinearity, and
   time-dependence.  It is an order-of-magnitude proxy, useful for
   feature engineering or prior construction.

4. **Pressure anomaly**:  A trivial helper returning -(p - p_ref) in Pa.

5. **Lag kernel**:  A placeholder lag-structure recorder.  Actual
   calibration against ERA5 forcing is intentionally deferred until
   data-access issues are resolved — see docs/status/BLOCKERS.md.

6. **Tail weights**:  Per-sample weights that up-weight extreme-surge
   values during model training.  Piecewise linear between q95 and q99.

7. **Lag kernel calibration**:  A one-shot linear fit of
       surge ≈ sum over lags: w_wind[lag]*U10[lag] + w_pres[lag]*IB[lag]
   Uses simple least squares on a single training split.  This is
   single-gauge, no spatial forcing, no phase info, and should be
   treated as a diagnostic baseline only.

REFERENCES
----------
Proudman, J., & Doodson, A. T. (1924). The principal constituent of
   the tides of the North Sea. *Philosophical Transactions of the Royal
   Society of London* A, 222, 223-255.
Large, W. G., & Pond, S. (1981). Open ocean momentum flux measurements
   in moderate to strong winds. *Journal of Physical Oceanography*,
   11(3), 324-336.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Physical defaults
RHO_WATER_DEFAULT = 1025.0   # kg/m³, mean ocean surface water
G_DEFAULT = 9.81             # m/s², standard gravity
RHO_AIR_DEFAULT = 1.225      # kg/m³, 15 °C at sea level
CD_DEFAULT = 0.0014          # Large & Pond 1981, moderate winds (~1-20 m/s)
PRESSURE_REF_DEFAULT = 1013.25  # hPa, standard atmosphere

# Sanity bounds
_PRESSURE_MIN = 900.0   # hPa — lowest observed: ~870 hPa (Typhoon Tip)
_PRESSURE_MAX = 1150.0  # hPa — highest observed: ~1085 hPa (Siberian winter)
_RHO_WATER_MIN = 950.0  # kg/m³ — warm fresh water
_RHO_WATER_MAX = 1050.0  # kg/m³ — cold hypersaline

# Wind speed regime where Large & Pond drag law is unreliable
_WIND_MIN_MS = 1.0  # m/s


# ---------------------------------------------------------------------------
# 1. Inverse barometer
# ---------------------------------------------------------------------------

def inverse_barometer(
    pressure_hpa: float,
    pressure_ref_hpa: float = PRESSURE_REF_DEFAULT,
    rho_water: float = RHO_WATER_DEFAULT,
    g: float = G_DEFAULT,
) -> float:
    r"""Inverse-barometer surge eta_IB (metres above MLLW).

    Uses the equilibrium hydrostatic approximation:

        eta_IB = -(p - p_ref) / (rho_water * g)

    A 1 hPa low relative to p_ref lifts the sea by ~1 cm.

    Args:
        pressure_hpa: Instantaneous atmospheric pressure (hPa).
        pressure_ref_hpa: Regional mean / reference pressure (hPa).
            Must be > 0.
        rho_water: Seawater density (kg/m³).  Expected 950–1050.
        g: Gravitational acceleration (m/s²).  Must be > 0.

    Returns:
        eta_IB in metres (positive = pressure deficit → higher sea level).

    Raises:
        ValueError: If pressure is outside 900–1150 hPa, p_ref ≤ 0,
            rho_water outside 950–1050, or g ≤ 0.

    Note:
        This is the equilibrium approximation.  Real basins have dynamic
        adjustments (phase lags, resonant amplification) that the formula
        ignores.  For hurricanes, the IB typically contributes 0.3–0.6 m
        of the total surge; the rest comes from wind stress.
    """
    if not (_PRESSURE_MIN <= pressure_hpa <= _PRESSURE_MAX):
        raise ValueError(
            f"pressure_hpa={pressure_hpa} outside physical range "
            f"[{_PRESSURE_MIN}, {_PRESSURE_MAX}] hPa"
        )
    if pressure_ref_hpa <= 0:
        raise ValueError(f"pressure_ref_hpa must be positive, got {pressure_ref_hpa}")
    if not (_RHO_WATER_MIN <= rho_water <= _RHO_WATER_MAX):
        raise ValueError(
            f"rho_water={rho_water} outside reasonable range "
            f"[{_RHO_WATER_MIN}, {_RHO_WATER_MAX}] kg/m³"
        )
    if g <= 0:
        raise ValueError(f"g must be positive, got {g}")

    # hPa → Pa: 1 hPa = 100 Pa.  Hydrostatic balance uses Pascals.
    return -(pressure_hpa - pressure_ref_hpa) * 100.0 / (rho_water * g)


# ---------------------------------------------------------------------------
# 2. Wind stress proxy
# ---------------------------------------------------------------------------

def wind_stress_proxy(
    u10_ms: float,
    v10_ms: float,
    coastal_azimuth: float | None = None,
    rho_air: float = RHO_AIR_DEFAULT,
    cd: float = CD_DEFAULT,
) -> dict:
    r"""Wind stress magnitude and coastal-projected components.

    Uses the bulk drag formula (Large & Pond 1981):

        |tau| = rho_air * C_D * |U10|²

    If *coastal_azimuth* is supplied (0° = north, 90° = east,
    direction **FROM** sea **TO** land), the stress vector is projected
    onto the coast frame using dot products:

        onshore_unit = (sin(coastal_az), cos(coastal_az))
        tau_vec      = |tau| * (u10, v10) / |U10|
        tau_onshore  = tau_vec · onshore_unit
        tau_alongshore = tau_vec · (-cos(coastal_az), sin(coastal_az))

    Convention:
        "onshore" means wind pushing **toward** the coast.

    Args:
        u10_ms: 10-metre zonal wind component (m/s, positive = eastward).
        v10_ms: 10-metre meridional wind component (m/s, positive = northward).
        coastal_azimuth: Direction FROM sea TO land, in degrees
            (0° = north, 90° = east, etc.).  Must be in [0, 360).
            ``None`` → no projection, return only magnitude.
        rho_air: Air density (kg/m³).  Must be > 0.
        cd: Drag coefficient (dimensionless).  Must be > 0.

    Returns:
        dict with keys:
            "tau_magnitude"   — scalar stress magnitude in Pa (N/m²).
            "tau_onshore"     — stress component toward land (Pa),
                                 only present if coastal_azimuth given.
            "tau_alongshore"  — stress component parallel to coast (Pa),
                                 only present if coastal_azimuth given.
            "wind_speed_ms"   — |U10| for reference.
            "wind_direction_deg" — meteorological direction wind goes
                                    TO (0° = north), NaN for |U|=0.

    Raises:
        ValueError: If |U10| < 1 m/s (drag law unreliable),
            coastal_azimuth out of [0, 360), or rho_air/cd non-positive.

    Note:
        The Large & Pond C_D = 1.4e-3 is tuned for 1–20 m/s winds.
        For hurricane-force winds (> 20 m/s) C_D actually *decreases*
        with wind speed (Donelan et al. 2004) — this function does not
        handle that regime.
    """
    if rho_air <= 0:
        raise ValueError(f"rho_air must be positive, got {rho_air}")
    if cd <= 0:
        raise ValueError(f"cd must be positive, got {cd}")
    if coastal_azimuth is not None and not (0.0 <= coastal_azimuth < 360.0):
        raise ValueError(
            f"coastal_azimuth must be in [0, 360), got {coastal_azimuth}"
        )

    wind_speed = math.sqrt(u10_ms ** 2 + v10_ms ** 2)
    if wind_speed < _WIND_MIN_MS:
        raise ValueError(
            f"wind_speed={wind_speed:.3f} m/s below reliability threshold "
            f"{_WIND_MIN_MS} m/s — Large & Pond drag law not valid"
        )

    tau_mag = rho_air * cd * wind_speed ** 2

    # Wind direction (meteorological: where wind is GOING TO)
    if wind_speed > 0:
        # atan2(x_component, y_component) for standard math convention
        az_wind = math.degrees(math.atan2(u10_ms, v10_ms))  # 0° = north, 90° = east
        if az_wind < 0:
            az_wind += 360.0
    else:
        az_wind = float("nan")

    result = {
        "tau_magnitude": tau_mag,
        "wind_speed_ms": wind_speed,
        "wind_direction_deg": az_wind,
    }

    if coastal_azimuth is not None:
        # Project wind stress vector onto coast frame using dot products.
        # Stress vector: tau_vec = tau_mag * (u_hat, v_hat)  [N/m²]
        # where u_hat = u10/|U|, v_hat = v10/|U|.
        # Onshore unit vector (FROM sea TO land, 0° = north):
        #   (sin(coastal_rad), cos(coastal_rad))  — 0°→(0,1), 90°→(1,0).
        coastal_rad = math.radians(coastal_azimuth)
        u_hat = u10_ms / wind_speed
        v_hat = v10_ms / wind_speed
        tx = tau_mag * u_hat
        ty = tau_mag * v_hat
        # Component pushing toward land (positive = onshore)
        result["tau_onshore"] = tx * math.sin(coastal_rad) + ty * math.cos(coastal_rad)
        # Component along shore (right-hand rule from onshore direction)
        result["tau_alongshore"] = -tx * math.cos(coastal_rad) + ty * math.sin(coastal_rad)

    return result


# ---------------------------------------------------------------------------
# 3. Coastal setup from stress
# ---------------------------------------------------------------------------

def coastal_setup_from_stress(
    tau_onshore_pa: float,
    water_depth_m: float,
    basin_width_m: float,
    rho_water: float = RHO_WATER_DEFAULT,
    g: float = G_DEFAULT,
) -> float:
    r"""Simplified coastal surge setup from steady onshore wind stress.

    Uses a one-dimensional momentum balance:

        d(eta)/dx ≈ tau_onshore / (rho_water * g * H)

    Integrated over a basin of width *L*:

        setup = tau_onshore * L / (rho_water * g * H)

    Args:
        tau_onshore_pa: Onshore wind stress component (Pa = N/m²).
            Positive = pushing toward the coast → positive setup.
        water_depth_m: Mean basin depth (m).  Must be ≥ 0.5 m.
        basin_width_m: Distance from coast to the offshore boundary (m).
            Must be > 0.
        rho_water: Seawater density (kg/m³).
        g: Gravitational acceleration (m/s²).

    Returns:
        eta_setup in metres (positive = sea level raised near coast).

    Raises:
        ValueError: If water_depth_m < 0.5 m, tau_onshore_pa < 0,
            basin_width_m ≤ 0, rho_water non-positive, or g non-positive.

    Note:
        This is an order-of-magnitude proxy, NOT a full shallow-water
        solution.  It neglects Coriolis force, bottom friction, advection,
        nonlinearity, time-dependence, and the 2D geometry of real
        basins.  Typical storm surges of 0.5–2 m require onshore stresses
        of ~0.1–0.5 Pa acting over basins of 100–500 km at depths of
        10–100 m.
    """
    if water_depth_m < 0.5:
        raise ValueError(
            f"water_depth_m={water_depth_m} < 0.5 m — shallow-water PDE "
            f"breaks down; use at your own risk"
        )
    if tau_onshore_pa < 0:
        raise ValueError(
            f"tau_onshore_pa must be non-negative (must push toward coast), "
            f"got {tau_onshore_pa}"
        )
    if basin_width_m <= 0:
        raise ValueError(f"basin_width_m must be positive, got {basin_width_m}")
    if rho_water <= 0:
        raise ValueError(f"rho_water must be positive, got {rho_water}")
    if g <= 0:
        raise ValueError(f"g must be positive, got {g}")

    return tau_onshore_pa * basin_width_m / (rho_water * g * water_depth_m)


# ---------------------------------------------------------------------------
# 4. Pressure anomaly to surge (Pa)
# ---------------------------------------------------------------------------

def pressure_anomaly_to_surge(
    pressure_hpa: float,
    pressure_ref_hpa: float = PRESSURE_REF_DEFAULT,
) -> float:
    """Pressure anomaly (Pa) from pressure relative to reference.

    Simple helper: -(p - p_ref).  Returned in Pascals (NOT metres).

    Args:
        pressure_hpa: Instantaneous pressure (hPa).
        pressure_ref_hpa: Regional reference pressure (hPa).

    Returns:
        Anomaly in Pa (i.e., hPa-to-Pa conversion with sign flipped so
        a low-pressure system → positive anomaly).

    Raises:
        ValueError: If pressure is outside [900, 1150] hPa.
    """
    if not (_PRESSURE_MIN <= pressure_hpa <= _PRESSURE_MAX):
        raise ValueError(
            f"pressure_hpa={pressure_hpa} outside physical range "
            f"[{_PRESSURE_MIN}, {_PRESSURE_MAX}] hPa"
        )

    # hPa → Pa: multiply by 100.  Anomaly = -(p - p_ref).
    return -(pressure_hpa - pressure_ref_hpa) * 100.0


# ---------------------------------------------------------------------------
# 5. Surge prior kernel (placeholder)
# ---------------------------------------------------------------------------

def surge_prior_kernel(
    lags_hours: int = 72,
    wind_lag: list[int] | None = None,
    pressure_lag: list[int] | None = None,
) -> dict:
    """Record lag structure for later calibration.

    This is a **placeholder**.  Actual calibration against ERA5 wind and
    pressure forcing is deferred until data access is resolved.  The
    returned dict simply documents the intended lag structure.

    Args:
        lags_hours: Total horizon of lags considered (hours).
        wind_lag: List of lag hours where wind contributes
            (e.g., ``[0, 3, 6, 12, 24]``).  Defaults to a coarse grid
            up to *lags_hours*.
        pressure_lag: List of lag hours where pressure contributes.
            Defaults to the same as *wind_lag*.

    Returns:
        dict with metadata only — no calibrated coefficients yet.
    """
    if wind_lag is None:
        wind_lag = list(range(0, lags_hours + 1, max(1, lags_hours // 12)))
    if pressure_lag is None:
        pressure_lag = list(wind_lag)

    # Sanity — must be lists of non-negative integers, sorted
    wind_lag = sorted({max(0, int(x)) for x in wind_lag})
    pressure_lag = sorted({max(0, int(x)) for x in pressure_lag})

    return {
        "calibrated": False,
        "lags_hours": lags_hours,
        "wind_lag_hours": wind_lag,
        "pressure_lag_hours": pressure_lag,
        "n_wind_lags": len(wind_lag),
        "n_pressure_lags": len(pressure_lag),
        "notes": (
            "Placeholder only — coefficients must be calibrated against "
            "ERA5 forcing once data access is resolved.  "
            "See docs/status/BLOCKERS.md."
        ),
    }


# ---------------------------------------------------------------------------
# 6. Tail weight
# ---------------------------------------------------------------------------

def tail_weight(
    metric_values: np.ndarray,
    train_q95: float,
    train_q99: float,
) -> np.ndarray:
    """Per-sample weights for tail-aware training.

    Piecewise definition (per sample value *v*):

        weight = 1,                         v < q95
        weight = (v - q95) / (q99 - q95),  q95 ≤ v ≤ q99
        weight = 2,                         v > q99

    NaN values receive weight = 1 (not weighted).

    Args:
        metric_values: 1-D or N-D array of values (e.g., residuals,
            total-water levels).
        train_q95: Training-set 95th percentile threshold.
        train_q99: Training-set 99th percentile threshold.

    Returns:
        Array of weights with the same shape as *metric_values*, dtype
        np.float64.

    Raises:
        ValueError: If q95 >= q99 or either threshold is negative.
    """
    if train_q95 >= train_q99:
        raise ValueError(
            f"q95 ({train_q95}) must be strictly less than q99 ({train_q99})"
        )
    if train_q95 < 0 or train_q99 < 0:
        raise ValueError(
            f"thresholds must be non-negative, got q95={train_q95}, q99={train_q99}"
        )

    arr = np.asarray(metric_values, dtype=np.float64)
    weights = np.ones_like(arr)

    nan_mask = np.isnan(arr)
    between = ~nan_mask & (arr >= train_q95) & (arr <= train_q99)
    above = ~nan_mask & (arr > train_q99)

    weights[between] = (arr[between] - train_q95) / (train_q99 - train_q95)
    weights[above] = 2.0
    # NaN stays at 1.0 from the ones_like initialization

    return weights


# ---------------------------------------------------------------------------
# 7. Lag kernel calibration (simple linear regression)
# ---------------------------------------------------------------------------

def lag_kernel_calibration(
    obs_surge: np.ndarray,
    obs_wind: np.ndarray,
    obs_pressure: np.ndarray,
    lags: list[int] | None = None,
) -> dict:
    r"""Fit surge ≈ wind + IB-pressure using simple linear regression over lags.

    For each sample *t* and each lag *l* in *lags*, the features are
        U10(t - l),   IB(t - l) = -(p(t - l) - p_ref) / (rho_water * g)
    and the target is surge(t).  All lagged columns are stacked and
    fit with least squares via ``scipy.stats.linregress`` (univariate
    regression on the full feature matrix — actually we use
    ``np.linalg.lstsq`` for the multivariate case).

    Inputs must be **time-aligned numpy arrays** of the same length.
    The caller is responsible for slicing to a training subset only —
    this function has no split awareness.

    Args:
        obs_surge: (N,) array of observed surge (metres above MLLW).
        obs_wind: (N,) array of 10 m wind speed magnitude (m/s) or
            any scalar wind proxy — the function does NOT decompose.
        obs_pressure: (N,) array of atmospheric pressure (hPa).
        lags: List of lag hours to include.  Default ``[0, 3, 6, 12, 24]``.

    Returns:
        dict with keys:
            "w_wind"       — (n_lags,) coefficients for lagged wind.
            "w_pres"       — (n_lags,) coefficients for lagged IB.
            "intercept"    — scalar intercept.
            "r_squared"    — coefficient of determination on the train fit.
            "train_rmse"   — RMSE (metres) on the train fit.
            "n_samples"    — number of samples used (after lag trimming).
            "lags"         — the lag list used.

    Raises:
        ValueError: If input arrays have different lengths, are too short
            for the requested lags, or contain NaNs.

    Note:
        This is a **diagnostic baseline only**.  Limitations:
          * Single gauge, no spatial forcing.
          * Uses wind *speed* (scalar) — not vector; no coastal projection.
          * IB uses the default reference pressure (1013.25 hPa).
          * Static lags, no phase info.
          * Linear model — misses nonlinear surge mechanisms.
          * No train/val split enforced here; caller must slice correctly.
    """
    if lags is None:
        lags = [0, 3, 6, 12, 24]

    obs_surge = np.asarray(obs_surge, dtype=np.float64)
    obs_wind = np.asarray(obs_wind, dtype=np.float64)
    obs_pressure = np.asarray(obs_pressure, dtype=np.float64)

    N = len(obs_surge)
    if len(obs_wind) != N or len(obs_pressure) != N:
        raise ValueError(
            f"all inputs must have same length (got surge={N}, "
            f"wind={len(obs_wind)}, pressure={len(obs_pressure)})"
        )

    max_lag = max(lags)
    if N <= max_lag:
        raise ValueError(
            f"N={N} too short for max_lag={max_lag} — need N > max_lag"
        )

    if np.any(np.isnan(obs_surge)) or np.any(np.isnan(obs_wind)) or np.any(np.isnan(obs_pressure)):
        raise ValueError("NaN detected in input arrays — caller must pre-process")

    n_lags = len(lags)
    n_feat = 2 * n_lags  # wind lags + pressure (IB) lags
    # Use time-aligned samples where all lags are available:
    samples = np.arange(max_lag, N)
    n_samples = len(samples)

    # Build IB surrogate for each time step
    ib_surge = np.array([
        inverse_barometer(float(p)) for p in obs_pressure
    ])

    # Build feature matrix X (n_samples × n_feat)
    X = np.zeros((n_samples, n_feat), dtype=np.float64)
    for j, lag in enumerate(lags):
        src_idx = samples - lag  # t - lag
        X[:, j] = obs_wind[src_idx]           # wind lag
        X[:, n_lags + j] = ib_surge[src_idx]  # IB lag

    y = obs_surge[samples]

    # Least squares with intercept
    X_aug = np.column_stack([X, np.ones(n_samples)])
    coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

    # Extract results
    w_total = coef[:n_feat]
    intercept = float(coef[-1])
    w_wind = w_total[:n_lags]
    w_pres = w_total[n_lags:]

    # Fit quality
    y_pred = X_aug @ coef
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    return {
        "w_wind": w_wind,
        "w_pres": w_pres,
        "intercept": intercept,
        "r_squared": r_sq,
        "train_rmse": rmse,
        "n_samples": n_samples,
        "lags": list(lags),
    }
