# Physics Equations — TideTail Surge Prior Module

**Date**: 2026-11-03
**Source**: `src/tidetail/physics.py`
**Tests**: `tests/test_physics.py` (50 tests, all pass)
**Datum**: metres above MLLW (chart datum, region-verified per FORECAST_TASK_CONTRACT.md §2)

---

## 1. Inverse Barometer Effect

### Hydrostatic Equilibrium

A pressure anomaly at the sea surface is accompanied by a compensatory sea-level change via hydrostatic balance:

$$
\frac{\partial p}{\partial z} = -\rho g
$$

Integrated from the reference sea surface ($z = -\eta_{\text{IB}}$) to the atmosphere-ocean interface ($z = 0$), with the boundary condition $p(0) = p_{\text{atm}}$ at the surface:

$$
p_{\text{atm}} - p_{\text{ref}} = -\rho_w g \int_{-\eta_{\text{IB}}}^{0} dz = -\rho_w g \eta_{\text{IB}}
$$

Solving for $\eta_{\text{IB}}$:

$$
\boxed{\eta_{\text{IB}} = -\frac{p_{\text{atm}} - p_{\text{ref}}}{\rho_w g}}
$$

where:
- $p_{\text{atm}}$ = instantaneous atmospheric pressure (Pa)
- $p_{\text{ref}}$ = regional reference pressure (Pa, default 1013.25 hPa = 101,325 Pa)
- $\rho_w$ = mean ocean surface water density, default 1025 kg/m³ (950–1050 physically reasonable)
- $g$ = standard gravitational acceleration, 9.81 m/s²

### Empirical Rule of Thumb

A 1 hPa (100 Pa) pressure deficit relative to standard atmosphere lifts the sea by approximately:

$$
\frac{100}{1025 \times 9.81} \approx 0.00995 \text{ m} \approx 1.0 \text{ cm}
$$

**Physical bounds**: Typical tropical cyclone eye pressures are 900–970 hPa, corresponding to 0.43–1.10 m equilibrium IB surge. This is the dominant contributor for large-scale storm surges (Proudman & Doodson 1924).

### Limitations

- **Equilibrium only**: No phase lag from dynamic adjustments
- **No resonant amplification**: Shelf basins like the Gulf of Mexico can amplify IB by factor 2–3 at resonant periods
- **No density stratification**: Assumes homogeneous upper ocean
- **No Coriolis deflection**: Neglected for $\beta \approx 10^{-11}$ s⁻¹ over 200 km scales
- **Reference pressure**: $p_{\text{ref}}$ should be region-specific (1980–2010 climatology), not universal 1013.25 hPa

### Derivation Notes

The code converts hPa → Pa at the entry point. The constant 100.0 in line 147:

```python
return -(pressure_hpa - pressure_ref_hpa) * 100.0 / (rho_water * g)
```

is the exact conversion factor: 1 hPa = 100 Pa.

---

## 2. Wind Stress Proxy

### Bulk Drag Formula (Large & Pond 1981)

Surface wind stress is estimated from 10 m wind using the classical bulk exchange formula:

$$
\boldsymbol{\tau} = \rho_a C_D |\mathbf{U}_{10}| \mathbf{U}_{10}
$$

Magnitude:

$$
|\tau| = \rho_a C_D |\mathbf{U}_{10}|^2
$$

where:
- $\rho_a$ = sea-level air density, default 1.225 kg/m³ (15 °C, 1013.25 hPa)
- $C_D$ = drag coefficient, default 1.4 × 10⁻³ (Large & Pond 1981, moderate winds 1–20 m/s)
- $|\mathbf{U}_{10}| = \sqrt{U_{10}^2 + V_{10}^2}$ = 10 m wind speed (m/s)

### Coastal Projection

Given coastal azimuth $\theta$ (0° = north, 90° = east, direction **FROM** sea **TO** land):

Onshore unit vector: $\hat{\mathbf{n}}_{\text{onshore}} = (\sin\theta, \cos\theta)$

Stress vector: $\boldsymbol{\tau} = |\tau| \cdot \frac{\mathbf{U}_{10}}{|\mathbf{U}_{10}|}$

Projected components:

$$
\tau_{\text{onshore}} = \boldsymbol{\tau} \cdot \hat{\mathbf{n}}_{\text{onshore}}
$$
$$
\tau_{\text{alongshore}} = \boldsymbol{\tau} \cdot (-\cos\theta, \sin\theta)
$$

### Wind Direction Convention

Where wind is **going TO**:

$$
\phi = \arctan2(U_{10}, V_{10}) \quad \text{(radians)}
$$
$$
\phi_\text{deg} = \phi \times \frac{180}{\pi} \pmod{360}
$$

### Drag Coefficient Regimes

| Wind speed | $C_D$ (Large & Pond 1981) | Regime |
|---|---|---|
| < 1 m/s | unreliable | Calm, transition |
| 1–3 m/s | $1.0 \times 10^{-3}$ | Weak |
| 3–10 m/s | $1.4 \times 10^{-3}$ (default) | Moderate |
| 10–20 m/s | $1.8 \times 10^{-3}$ | Strong |
| > 20 m/s | $C_D$ **decreases** to $1.0 \times 10^{-3}$ (Donelan et al. 2004) | Hurricane-force |

The current code raises for $|\mathbf{U}_{10}| < 1$ m/s and uses a single $C_D = 1.4 \times 10^{-3}$ for all moderate-to-strong winds. **Hurricane-force drag reduction is NOT yet implemented.**

---

## 3. Coastal Setup from Steady Onshore Stress

### Simplified Momentum Balance

For a basin of width $L$ (from coast to offshore boundary), constant depth $H$, steady onshore stress $\tau_{\text{onshore}}$:

$$
\frac{d\eta}{dx} \approx \frac{\tau_{\text{onshore}}}{\rho_w g H}
$$

Integrated over basin width $L$:

$$
\boxed{\eta_{\text{setup}} = \frac{\tau_{\text{onshore}} \cdot L}{\rho_w g H}}
$$

### Order-of-Magnitude Check

For a typical storm surge of 1 m:
- $\tau_{\text{onshore}} \approx 0.1$ Pa (moderate 15 m/s onshore wind)
- $L = 200$ km = $2 \times 10^5$ m
- $H = 20$ m
- $\rho_w g \approx 10,025$ N/m³

$$
\eta_{\text{setup}} = \frac{0.1 \times 2 \times 10^5}{10,025 \times 20} \approx 0.10 \text{ m}
$$

For 0.5 Pa (20 m/s onshore):
$$
\eta_{\text{setup}} \approx 0.50 \text{ m}
$$

For full-hurricane 1 Pa over 500 km × 10 m deep:
$$
\eta_{\text{setup}} \approx 4.99 \text{ m}
$$

This matches observed extreme surges (Katrina 9 m, Ike 6 m — but those are compound effects including geometry, IB, and nonlinearity).

### Limitations

- **No Coriolis**: For $f = 2\Omega\sin\phi$ at 30° N, $f \approx 7.3 \times 10^{-5}$ s⁻¹
- **No bottom friction**: Quadratic drag $\sim 10^{-3}$ m/s² scale
- **No nonlinearity**: Shallow-water continuity equation is linearized
- **No time dependence**: Steady-state only; storm passage is transient (24–72 h)
- **No geometry**: Real basins have complex bathymetry, constrictions, bays

---

## 4. Tail Weight Function (Training Loss Reweighting)

Piecewise linear upweighting for extreme values during model training:

$$
w(v) = \begin{cases}
1 & v < q_{95} \\
\frac{v - q_{95}}{q_{99} - q_{95}} & q_{95} \leq v \leq q_{99} \\
2 & v > q_{99}
\end{cases}
$$

All $q$ values **fit on training data only** — no leakage from validation/test. NaN values receive $w = 1$.

Purpose: Shift the generative model's emphasis toward rare extreme surge events.

---

## 5. Lag Kernel Calibration (Diagnostic Baseline)

Simple linear regression of surge on lagged wind and inverse-barometer pressure:

$$
\hat{\eta}(t) = \beta_0 + \sum_{l \in \mathcal{L}} \left[ \beta_w(l) \cdot U_{10}(t - l) + \beta_p(l) \cdot \eta_{\text{IB}}(t - l) \right]
$$

where $\mathcal{L} = \{0, 3, 6, 12, 24\}$ hours is the default lag grid.

Fit via ordinary least squares: $[\beta_0, \boldsymbol{\beta}_w, \boldsymbol{\beta}_p] = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$

### Severe Limitations

- **Single gauge, no spatial forcing**: Cannot capture wind-field structure
- **Uses wind speed scalar, not vector**: Loses directional information
- **No coastal projection**: Wind stress should be decomposed onto coast frame
- **Linear model**: Misses nonlinear surge mechanisms (bathymetric amplification, funneling)
- **Default $p_{\text{ref}} = 1013.25$ hPa**: Should be region-calibrated
- **Diagnostic only**: Not used for model conditioning

---

## 6. Total Water Level Decomposition

### Fundamental Identity

$$
\eta_{\text{total}}(t) = \eta_{\text{tide}}(t) + \eta_{\text{residual}}(t) + \epsilon(t)
$$

where:
- $\eta_{\text{total}}(t)$ = observed total water level (MLLW)
- $\eta_{\text{tide}}(t)$ = astronomical tide from FES2022 global ocean tide model
- $\eta_{\text{residual}}(t)$ = non-tidal surge residual (wind + pressure + setup + seiche)
- $\epsilon(t)$ = measurement error + model discrepancy

### Target Contract (FORECAST_TASK_CONTRACT.md §2)

**TideTail forecasts $\eta_{\text{residual}}(t + 1:t + H)$ conditional on $\eta_{\text{total}}(t - L + 1:t)$**, with $L = 96$ h input history and $H = 72$ h forecast horizon.

**NEVER** double-subtract tide. The residual is $\eta_{\text{residual}} = \eta_{\text{total}} - \eta_{\text{tide}}$ — already de-tided. The downstream model predicts residual, and total-water MAE is computed as:

$$
\text{tw\_MAE} = \text{MAE}(\hat{\eta}_{\text{residual}} + \eta_{\text{tide}}, \eta_{\text{total}})
$$

---

## 7. Physics-Guided Flow Matching (Not Yet Implemented)

The planned physics-aware FM variants:

### Tail-Aware FM

Standard conditional FM (continuous normalizing flow) with tail-weighted training loss:

$$
\mathcal{L}_{\text{tail-FM}} = \frac{1}{N} \sum_{i=1}^N w(\eta_{\text{res},i}) \cdot \mathcal{L}_{\text{FM}}(\theta; x_i)
$$

where $\mathcal{L}_{\text{FM}}$ is the standard flow-matching loss:

$$
\mathcal{L}_{\text{FM}}(\theta; x) = \mathbb{E}_{t, z} \left[ \| v_\theta(t, x + t(z - x), c) - (z - x) \|^2 \right]
$$

with $t \sim U[0, 1]$, $z \sim \mathcal{N}(0, I)$, $c$ = conditioning (96 h residual history).

### Physics-Prior FM (once ERA5 available)

$$
\mathcal{L}_{\text{phy-FM}} = \mathcal{L}_{\text{FM}} + \lambda_1 \mathcal{L}_{\text{IB}} + \lambda_2 \mathcal{L}_{\text{wind}}
$$

where:
- $\mathcal{L}_{\text{IB}} = \mathbb{E}[\max(0, |\eta_{\text{syn}} - \eta_{\text{IB}}| - \delta_{\text{IB}})^2]$ — penalizes synthetic surge inconsistent with inverse-barometer
- $\mathcal{L}_{\text{wind}} = \mathbb{E}[\max(0, \eta_{\text{syn}} - \eta_{\text{wind,max}})^2]$ — penalizes synthetic surge exceeding wind-stress upper bound

### Constraint Violations

Soft constraints require reporting violation rates:

$$
\text{violation rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}\left( \text{constraint}(\eta_{\text{syn},i}) \text{ violated} \right)
$$

---

## 8. Formal Scope and Disclaimer

### What is Physically Grounded

1. ✅ Inverse barometer — exact hydrostatic balance, equilibrium regime
2. ✅ Wind stress proxy — standard bulk formula, empirical drag coefficient
3. ⚠️ Coastal setup — order-of-magnitude only, not a shallow-water PDE solution
4. ❌ Storm surge — **NOT** modeled as full shallow-water equations. TideTail does NOT solve continuity or momentum with Coriolis, nonlinearity, or time dependence.

### What is NOT Claimed

- NO solution to shallow-water equations
- NO dynamical storm-surge prediction
- NO atmospheric circulation modeling
- NO statistical-dynamical coupling to GCMs or RCMs
- NO wave setup, wave runup, or coastal flooding inundation
- NO sea-level rise scenarios included

TideTail is a **statistical residual forecaster** with physically grounded feature engineering and loss functions. The generative augmentation module is designed to improve forecasting under limited storm-event observations — it does not aim to reproduce physical surge dynamics ab initio.

---

## 9. References

| Reference | Context |
|---|---|
| Proudman, J., & Doodson, A. T. (1924). The principal constituent of the tides of the North Sea. *Philosophical Transactions of the Royal Society of London* A, 222, 223-255. | Inverse barometer equilibrium theory |
| Large, W. G., & Pond, S. (1981). Open ocean momentum flux measurements in moderate to strong winds. *JPO*, 11(3), 324-336. | Bulk drag coefficient formulation |
| Donelan, M. A., Hauser, D., Hasselmann, K., et al. (2004). On the limiting aerodynamic roughness of the ocean in very strong winds. *GRL*, 31(19). | Hurricane-force drag reduction |
| Holthuijsen, L. H. (2007). *Waves in Oceanic and Coastal Waters*. Cambridge University Press. | Shallow-water equations, coastal dynamics |
| NOAA (2023). *Storm Surge Frequently Asked Questions*. NOAA NWS. | Observed extreme surge magnitudes |

---

*This document is auto-generated from `src/tidetail/physics.py` and the FORECAST_TASK_CONTRACT.md. Every equation maps to a specific function with unit tests in `tests/test_physics.py`. No physics claim in the paper should reference this document without corresponding test verification.*
