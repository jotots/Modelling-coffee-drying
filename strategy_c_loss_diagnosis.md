# Strategy C loss and optimization diagnosis

> **Model-scope notice (2026-11-18):** This report documents the pre-direct-field
> Strategy C implementation and its immutable historical runs. Its
> Bernstein/analytic-mean constraint analysis, raw-loss comparisons, gradient
> measurements, and bounded replay are historical evidence, not measurements
> of the new `strategy-c-local-fields-v1` model. The current model uses
> quadrature-integrated direct fields and a frozen-scaled warmup/ramp objective;
> see the session development report at
> `C:\Users\pushe\.copilot\session-state\e940b76e-fec8-4e6a-95fb-d507313e71dd\files\direct-fields\development-report.md`.

## Executive findings

1. **The initial-temperature residual had a real target bug.** The model
   hard-enforces the initial product temperature, for which the signed
   temperature ratio is one. The former residual used that ratio directly,
   adding a constant unit mean-square error with zero trainable gradient. It
   now subtracts one. This corrects the objective value but does not change
   optimizer gradients or, by itself, cure fit quality or drying speed.
2. **The moisture initial-condition loss was roundoff.** The hard-constrained
   moisture ratio starts at one; the historical artifact's IC loss was about
   1e-32.
3. **Temperature remains trainable after the hard initial condition.** The
   initial-condition term itself has zero gradient, but the temperature field
   varies with depth at positive times. Heat/mass PDE and boundary coupling
   provide trainable gradients; the first-Adam-step temperature-network
   parameter change was nonzero.
4. **Historical and current-default runs are distinct.** The immutable
   historical artifact initialized reference diffusivity at 1e-10 m2/s; the
   current PhysicsConfig default is 1e-7 m2/s. The historical data-loss rise
   is not a reproduced current-default full-run result.
5. **Current-default training has an early surface-temperature validity    failure** A bounded, deterministic Adam-only replay reached nine updates.
   The objective's post-update validation after update 9 rejected a predicted
   surface temperature below the saturation-pressure model's 273.15 K
   minimum. This corresponds to the next pre-update objective evaluation in
   the ordinary training loop. No clamp, extrapolation, continued training,
   L-BFGS, artifact export, or held-out scoring was used.
6. **A diffusivity-unit error is not supported.** The old Arrhenius expression
   was already dimensionally valid in absolute Kelvin. The code now makes its
   equivalent dimensionless groups explicit while retaining SI inputs and
   outputs. The historical fitted value near 1e-10 is not a recovered value
   from the current-default run.

## Source traces

These are repository-relative links to the current worktree. The exact
implementation links for every objective component are also listed in the
loss ledger below.

| Topic | Source |
|---|---|
| Current SI constants, default diffusivity initialization and scales | [`src/strategy_c/physics.py` lines 43-161](src/strategy_c/physics.py#L43-L161) |
| Hard moisture field, temperature ansatz and network initialization | [`src/strategy_c/physics.py` lines 248-390](src/strategy_c/physics.py#L248-L390) |
| Dimensionless diffusivity groups and Kelvin Arrhenius equivalence | [`src/strategy_c/physics.py` lines 413-455](src/strategy_c/physics.py#L413-L455) |
| Heat/mass transfer coefficients and coupled surface flux | [`src/strategy_c/physics.py` lines 687-805](src/strategy_c/physics.py#L687-L805) |
| PDE, IC, base and surface residual definitions | [`src/strategy_c/physics.py` lines 880-1059](src/strategy_c/physics.py#L880-L1059) |
| Residual-to-mean-square mapping | [`src/strategy_c/physics.py` lines 1062-1075](src/strategy_c/physics.py#L1062-L1075) |
| Measured moisture targets and normalization | [`src/strategy_c/data.py` lines 151-278](src/strategy_c/data.py#L151-L278) |
| Training/sampling defaults and loss weights | [`src/strategy_c/training.py` lines 51-149 and 253-523](src/strategy_c/training.py#L51-L149) |
| Equal-curve data loss and combined objective | [`src/strategy_c/training.py` lines 565-618](src/strategy_c/training.py#L565-L618) |
| Adam snapshots are recorded before each update | [`src/strategy_c/training.py` lines 801-817](src/strategy_c/training.py#L801-L817) |
| L-BFGS configuration and closure calls | [`src/strategy_c/training.py` lines 621-681 and 819-839](src/strategy_c/training.py#L621-L681) |
| Temperature IC and diffusivity-equivalence regressions | [`tests/test_strategy_c_physics.py` lines 675-760](tests/test_strategy_c_physics.py#L675-L760) |

## Variables, conventions and diffusivity

Readable variable legend:

- `t`: elapsed time in seconds; `z`: depth into the bed in metres.
- `tau = t / 14400`; the modeled duration `t_s` is 14,400 seconds.
- `xi = z / 0.01`; the bed depth `L` is 0.01 metres.
- `X`: local moisture content; `X0`: that curve's initial moisture content.
- `u = X / X0`: local dimensionless moisture ratio.
- `B(tau, curve)`: predicted depth-mean moisture ratio.
- `T`: absolute product temperature in kelvin; `T0 = 298.15 K`;
  `T_air = 343.15 K`.
- `theta = (T - T_air) / (T0 - T_air)`. This is a signed temperature
  coordinate: it is one initially and zero at air temperature. Its
  denominator is `T0 - T_air = -45 K`; it is not clipped.
- `Tstar = T / Tref`, where `Tref` is the diffusivity reference temperature.
  `Tstar` belongs only to the Arrhenius law and is not `theta`.

The model imposes `u(tau=0, xi)=1` and `T(tau=0, xi)=T0`. It predicts the
temperature through `T = T0 * exp(tau * g)`, where `g` depends on time,
depth and curve conditions. Thus an exact initial temperature does not
constrain away depth variation in the positive-time temperature field.
Neural condition features are standardized from training curves; physical
constitutive laws and fluxes receive the original SI-valued conditions.

The diffusivity law, written without specialized notation, is:

```text
d_ratio = particle_diameter / reference_particle_diameter
v_ratio = air_speed / reference_air_speed
Tstar = absolute_temperature_K / reference_temperature_K
activation_energy_J_per_mol = 2044 K * gas_constant_J_per_mol_K
                            = 16994.76 J/mol   (fixed, Barnard et al. 2024 Table 1)
Ea_star = activation_energy_J_per_mol
          / (gas_constant_J_per_mol_K * reference_temperature_K)
        = 2044 / reference_temperature_K

D(T, d, v) = D_ref * d_ratio^alpha * v_ratio^beta
             * exp(Ea_star * (1 - 1/Tstar))
```

`D` and `D_ref` are in m2/s; `Ea_star` and the ratios are dimensionless.
The exponent equals the usual absolute-Kelvin Arrhenius exponent
`(Ea/R) * (1/Tref - 1/T)`. It is not a reciprocal of a standardized or
Celsius temperature. The moisture PDE coefficient is `D_ref * t_s / L^2`;
surface fluxes use the physical SI diffusivity.

At `D_ref = 1e-10 m2/s`, the reference moisture Fourier coefficient is 0.0144
and the diffusion time `L^2/D` is 1e6 seconds. At `D_ref = 1e-7 m2/s`, these
are 14.4 and 1e3 seconds. This thousand-fold initialization difference
explains a much larger initial PDE residual, but does not identify which
diffusivity represents the spent-coffee-grounds bed.

## Complete loss ledger and scaling

Every residual is evaluated on its own fixed sample set. `mean_square(r)`
means the arithmetic mean of the squared entries of residual `r`. The terms
are summed; they are not divided by one shared point count.

```text
DATA
For each training curve, compare its predicted depth-mean B(tau, curve)
against its available normalized measured bulk moisture y.
L_data = mean over training curves of
         (mean over that curve's observations of (B - y)^2)
Curves receive equal top-level weight, not weight proportional to row count.
```

Source: [`src/strategy_c/training.py` lines 572-599](src/strategy_c/training.py#L572-L599); target derivation: [`src/strategy_c/data.py` lines 243-278](src/strategy_c/data.py#L243-L278).

```text
INTERIOR MASS PDE
R_mass = du/dtau
          - (t_s/L^2) * d/dxi( D(T,d,v) * du/dxi )
L_mass = mean_square(R_mass)

INTERIOR HEAT PDE (Incropera, Fourier time)
X_local    = X0 * u
alpha      = k_p / (rho_d * (cp_s + X_local * cp_w))
Fo_s       = alpha * t_s / L^2            (local heat_fourier_scale)
Fo_heat    = alpha * t / L^2 = Fo_s * tau
R_heat = (1 / Fo_s) * dtheta/dtau - d2theta/dxi2
       = dtheta/dFo_heat - d2theta/dxi2
L_heat = mean_square(R_heat)
```

The mass residual retains the full variable-coefficient divergence. Heat uses
the wet-bed volumetric capacity `rho_d * (cp_s + X0*u*cp_w)` (dry solids plus
local liquid water, `cp_w = 4180 J/(kg K)`), evaluated at each collocation
point; the network still receives `tau`, `xi` and returns `theta`. The sampled
64 interior points are **per training curve**.

Source: [`src/strategy_c/physics.py` lines 880-895 and 941-958](src/strategy_c/physics.py#L941-L958); MSE mapping: [`src/strategy_c/physics.py` lines 1062-1075](src/strategy_c/physics.py#L1062-L1075).

```text
INITIAL CONDITIONS
R_initial_moisture = u(tau=0, xi) - 1
L_initial_moisture = mean_square(R_initial_moisture)

R_initial_temperature = theta(tau=0, xi) - 1
L_initial_temperature = mean_square(R_initial_temperature)
```

Each IC term samples 16 points **per training curve**. The former
temperature residual used `theta` without subtracting one, producing exactly
one unit of MSE despite zero trainable gradient. The corrected residual is
zero under the hard initial ansatz. The moisture IC was already correct;
its measured loss was floating-point roundoff.

Source: [`src/strategy_c/physics.py` lines 929-938 and 995-1003](src/strategy_c/physics.py#L929-L938); loss mapping: [`src/strategy_c/physics.py` lines 1018-1021](src/strategy_c/physics.py#L1018-L1021).

```text
BASE BOUNDARY
R_base_moisture = du/dxi at xi=0
L_base_moisture = mean_square(R_base_moisture)

R_base_temperature = dtheta/dxi at xi=0
L_base_temperature = mean_square(R_base_temperature)
```

The base is impermeable and insulated. Each term samples 16 points **per
training curve**.

Source: [`src/strategy_c/physics.py` lines 940-944 and 1003-1006](src/strategy_c/physics.py#L940-L944); loss mapping: [`src/strategy_c/physics.py` lines 1019-1022](src/strategy_c/physics.py#L1019-L1022).

```text
EXPOSED SURFACE
X_surface = X0 * u_surface
water_activity (Barnard et al. 2024 Table 1; fitted to potato, assumed for
spent coffee grounds; not clipped, so -0.0008 at X = 0):
    X < 0.22:         1232.3 X^4 - 646.14 X^3 + 98.541 X^2 - 0.3406 X - 0.0008
    0.22 <= X <= 0.6: -0.5129 X^2 + 0.6398 X + 0.5995
    X > 0.6:          0.0013 X^3 - 0.0197 X^2 + 0.1164 X + 0.7448
    (with X = X_surface)
J = h_m * M_water/R_gas
    * (water_activity * Psat(T_surface)/T_surface
       - RH * Psat(T_air)/T_air)

h_v = 1000 * (h_fg + dh_w)   in J/kg, evaluated at every surface point
    h_fg = -2.394 * (T_surface - 273.15) + 2502.1                kJ/kg
    dh_w = 1420.95                                    X <= 0.02  kJ/kg
         = -8.2068e6 X^4 + 4.0003e6 X^3 - 6.1614e5 X^2
           + 2.3682e4 X + 1.1631e3                    0.02 < X <= 0.2
         = -22.583 X + 130.18                         X > 0.2
q_convection = h_T * (T_air - T_surface)
q_required = q_convection - h_v * J

R_surface_mass = (-rho_d * X0 * D * du/dxi / L - J) / 0.01
L_surface_mass = mean_square(R_surface_mass)

Bi_heat = h_T * L / k_p
E_evap  = h_v * J * L / (k_p * (T_air - T0))
R_surface_heat = dtheta/dxi + Bi_heat * theta_surface - E_evap   at xi = 1
L_surface_heat = mean_square(R_surface_heat)
```

Positive `J` denotes evaporation. The mass-flux scale is 0.01 kg/(m2 s).
`R_surface_heat` is exactly `(k_p * (T0 - T_air) * dtheta/dxi / L - q_required)`
divided by the conductive scale `k_p * (T0 - T_air) / L`; the former fixed
1000 W/m2 heat-flux scale was removed. It is a pure flux balance with no
accumulation term (a surface has no volume). With `T_air > T0`, `E_evap > 0`
when water evaporates. Each surface term samples 16 positive-time
points **per training curve**; the exactly incompatible initial Robin corner
is excluded.

Source: [`src/strategy_c/physics.py` lines 709-789](src/strategy_c/physics.py#L709-L789); residuals: [`src/strategy_c/physics.py` lines 1011-1043](src/strategy_c/physics.py#L1011-L1043); losses: [`src/strategy_c/physics.py` lines 1062-1075](src/strategy_c/physics.py#L1062-L1075).

```text
COMPLETE WEIGHTED OBJECTIVE
L_total = L_data
          + w_physics * (L_mass + L_heat)
          + w_initial * (L_initial_moisture + L_initial_temperature)
          + w_boundary * (L_base_moisture + L_base_temperature
                          + L_surface_mass + L_surface_heat)

Current defaults: w_physics = w_initial = w_boundary = 1.
```

Source: [`src/strategy_c/training.py` lines 92-104](src/strategy_c/training.py#L92-L104); complete objective: [`src/strategy_c/training.py` lines 600-618](src/strategy_c/training.py#L600-L618).

Adam now uses two parameter groups: the three learned diffusivity-law parameters (log D_ref, diameter and speed exponents; Ea is a fixed buffer) at `physics_parameter_learning_rate = 0.05`, all network weights at `adam_learning_rate = 0.001` ([`src/strategy_c/training.py` lines 700-749](src/strategy_c/training.py#L700-L749)); L-BFGS is unchanged.

Training uses 16 complete curves and 372 available targets in the locked
seed-42 split; eight other curves are held out. There are **no direct
temperature targets** in `L_data`. The hard temperature IC has no trainable
gradient because the ansatz fixes the initial value, but it does not freeze
the spatially varying positive-time model. Temperature is trained through the
heat and mass PDEs and boundary coupling.

## Evidence and interpretation

### Historical immutable artifact (not current defaults)

The immutable run is
[`artifacts/strategy_c_inverse/20261118T141959.814677Z`](artifacts/strategy_c_inverse/20261118T141959.814677Z).
Its manifest records schema `strategy-c-hybrid-ca-v1`, seed 42, 16 training
and eight held-out curves, 372 training targets, 64/16/16/16 collocation
points **per training curve**, 50 Adam epochs at 0.001 and 20 L-BFGS
iterations. It records an initial reference diffusivity of 1e-10 m2/s, while
the current default is 1e-7 m2/s. This old artifact was not overwritten and
has no source revision fingerprint.

Selected pre-update Adam snapshots:

| Adam epoch | Data MSE | Mass PDE MSE | Heat PDE MSE | Moisture IC MSE | Temperature IC MSE | Total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0075779 | 1.66148 | 6.20e-7 | 3.24e-32 | 1.0 | 2.69072 |
| 2 | 0.0075604 | 1.63351 | 0.004291 | 4.84e-32 | 1.0 | 2.65526 |
| 10 | 0.0095936 | 1.34445 | 0.000637 | 3.31e-32 | 1.0 | 2.35819 |
| 25 | 0.0737741 | 0.38473 | 8.59e-5 | 4.97e-32 | 1.0 | 1.46206 |
| 50 | 0.2232070 | 0.015400 | 4.02e-5 | 6.91e-32 | 1.0 | 1.24190 |

Data MSE bottoms at 0.0075604 (epoch 2) and is 29.5 times larger by epoch
50, while mass-PDE MSE improves about 108-fold. The total falls as the mass
residual falls; the former temperature-IC error added a constant one. First
and last recorded L-BFGS-closure data MSEs are 0.22317 and 0.18837. These are
historical logs, not a completed current-default trajectory.

The old artifact's final fitted parameters are reference diffusivity
1.14505e-10 m2/s, activation energy 29.434 kJ/mol, alpha -0.05255 and beta
-0.04054. Its held-out normalized-moisture scores are equal-curve MAE 0.32330
and RMSE 0.42393 over eight curves/188 targets (pooled MAE 0.32287, RMSE
0.42426). They are not scores from current defaults and do not validate
internal temperature.

Independent artifact QA found initial-state maximum absolute errors no
larger than 2e-15, a finite nonzero temperature time derivative (2.8945 at
one off-grid probe), and nonzero finite gradients for all four diffusivity
parameters. The saved field's theta range was [0.91765, 1.0]. The old
unit temperature-IC loss therefore did not freeze the temperature network.

### Current-default first step and bounded replay

The first-step results below use the local workbook parser, locked seed-42
split, float64/CUDA, the default 1e-7 m2/s reference diffusivity, and 64/16/
16/16 sampled points **per training curve**. Adam's pre-update loss values
match the reported first-step probe:

| Quantity | Before update 1 | After update 1 |
|---|---:|---:|
| Data MSE | 0.00757789 | 0.00755699 |
| Mass PDE MSE | 97.10837 | 48.06688 |
| Heat PDE MSE | 6.20e-7 | 0.00390656 |
| Moisture IC MSE | 3.24e-32 | 3.66e-32 |
| Corrected temperature IC MSE | 0 | 0 |
| Total | 97.14105 | 48.09846 |
| Surface temperature range | 298.1500-298.1850 K | 295.7863-298.1479 K |

At initialization, the raw mass-PDE gradient norms were 1,135.13 for the
moisture network, 2,248.40 for the temperature network, and 187.17/0.1271/
10.68/8.87 for log reference diffusivity/log activation energy/alpha/beta.
The data-gradient norm for the moisture network was 0.006382; its data-vs-
mass gradient cosine was +0.0961. The raw first-step temperature-network
parameter change was 0.07590. These norms do not establish Adam-update
causality or that a gradient-scale imbalance alone caused all poor curves.

One diagnostic replay was performed, at most 50 Adam updates and no L-BFGS.
It used the same parser definitions from notebook cell 3, seed-42 16/8 split,
current defaults, float64/CUDA, and fixed 64 interior plus 16 initial/base/
surface points per training curve. The split used 372 train targets; held-out
curve IDs were recorded for provenance but held-out observations were not
used by fitting, diagnostics, or optimization. For every successful step the
helper recorded actual pre-update and post-update objective values, parameters,
surface temperature range, exact IC errors, and the representative training
curve's depth-temperature range and dT/dxi at tau=0.25, 0.50 and 0.75.

The replay completed nine Adam updates. Post-update objective validation
after update 9 rejected a surface temperature of 270.83759 K, below the
supported 273.15-373.15 K interval; the sampled surface range was
270.83759-298.12841 K. The last fully valid objective was pre-update 9
(identical to post-update 8): data MSE 0.00767037, mass MSE 6.91171, heat
MSE 0.506061 and total 7.56722. The error is re-raised after writing evidence;
there is no clamping, catch-and-continue, extrapolation or fabricated
successful result. No L-BFGS or held-out score was produced.

All loss values in this subsection (first-step table and bounded replay) were measured under the previous heat formulation (time-scaled heat PDE, surface heat residual divided by 1000 W/m2, dry-solid heat capacity) and are not comparable with current loss values.

Initial raw gradient norms by parameter block were saved for data, mass-PDE,
heat-PDE and surface-heat losses. Per-loss block gradients were not separately
recomputed at every step; the Adam update is not inferred from those initial
raw norms. The replay reached only pre-update epochs 1-9, so epochs 10, 25 and
50 were not reachable. The full machine-readable trace records all successful
pre/post evaluations and the failure:

- Helper: `C:\Users\pushe\.copilot\session-state\e940b76e-fec8-4e6a-95fb-d507313e71dd\files\strategy_c_current_default_Adam_diagnostic.py`
- JSON: `C:\Users\pushe\.copilot\session-state\e940b76e-fec8-4e6a-95fb-d507313e71dd\files\strategy_c_current_default_Adam_diagnostic.json`
- CSV: `C:\Users\pushe\.copilot\session-state\e940b76e-fec8-4e6a-95fb-d507313e71dd\files\strategy_c_current_default_Adam_diagnostic.csv`
- Traceback: `C:\Users\pushe\.copilot\session-state\e940b76e-fec8-4e6a-95fb-d507313e71dd\files\strategy_c_current_default_Adam_traceback.txt`

The JSON records source and workbook SHA-256 fingerprints, commit/branch and
initial dirty status, Python/library/device versions, complete PhysicsConfig/
TrainingConfig/SamplingConfig values, train/held-out IDs, observation and
collocation counts, all captured milestone states, initial raw gradient
norms, exact failure step/stage and the range-validation traceback.

### Degree-eight bulk-curve restriction versus training observations

The depth-mean moisture family is a positive mixture
`B(tau) = sum from k=0 to 8 of w_k * (1-tau)^k`, with nonnegative weights
that sum to one. On `0 <= tau <= 1`, its maximum possible drying slope is
bounded by `abs(dB/dtau) <= sum(k*w_k) <= 8`.

Using only available adjacent scheduled train observations, one of 354
intervals exceeds this hard bound: training curve `v0.48_d236_r1`, from 120
to 180 seconds, falls from normalized moisture 0.973929 to 0.927385. Its
observed secant drying rate is 11.170538 per tau. Over 60 seconds, the
degree-eight family's maximum possible drop is `8 * 60/14400 = 0.033333`;
the observed drop is 0.046544. This is an isolated measured incompatibility,
not evidence that the restriction explains the overall poor fit. Measurement
variability is plausible, and no held-out observations or optimization were
used for this comparison.

Source: [`src/strategy_c/physics.py` degree-eight positive bulk mixture](src/strategy_c/physics.py#L248-L335); train-target construction: [`src/strategy_c/data.py` lines 151-278](src/strategy_c/data.py#L151-L278); locked split and train-only batch: [`src/strategy_c/training.py` lines 253-523](src/strategy_c/training.py#L253-L523).

## Causal limits and next diagnosis

**Established:** the former temperature IC loss was a constant one; moisture
IC error was roundoff; historical data MSE rose while mass-PDE MSE fell in
the distinct 1e-10-initialization artifact; current-default initial mass
residual and gradients are large; temperature parameters update; the
degree-eight model cannot exactly match the one measured train-only segment
above its slope bound; and the current-default replay fails surface
saturation-range validation after Adam update 9.

**Not established:** that this loss tradeoff caused all poor fits; that a
higher reference diffusivity improves predictions; that the isolated
degree-eight incompatibility limits the overall fit; that internal
temperature is experimentally accurate; or that one diffusivity is
identifiable from bulk-moisture curves. Temperature/boundary assumptions,
measurement variation, geometry, property values and the soft physics/data
balance remain possible confounders. No new convergence or recovered-
diffusivity claim is made.

The appropriate next step is a separately authorized diagnosis of the
range-validation failure and intended diffusivity initialization, with
explicit acceptance criteria before any optimizer, scaling, architecture or
temperature-stability change. Preserve a new immutable run and the locked
held-out split for later evaluation; do not silently alter configuration or
use held-out curves to choose changes.

## Earlier implementation verification and scope

The original Strategy C fix changed only the initial-temperature residual and
made the equivalent SI diffusivity law explicit; focused regression tests
covered both. Its development record reports 76 passing Strategy C tests and
Ruff passing on changed Python files. Repository-wide mypy remained blocked
by missing pandas stubs and existing module/test errors. This follow-up adds
no production Python, tests, optimizer settings, architecture, scaling,
notebook outputs or repository artifacts; it changes this report only. The
diagnostic script and outputs are in session files, not the repository.
