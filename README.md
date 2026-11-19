# Modelling-coffee-drying
Modelling drying process of coffee with PINNs for y chemical engineering thesis

## Illustrative drying curve

Run `python notebooks\Drying_curve_generator.py` to generate the Portuguese
PNG and PDF figures in `src\drying_curve`. The numbers are arbitrary: the
figure illustrates an idealized drying process, not measured coffee data.
Equilibrium moisture is approached asymptotically. A constant-rate period
is not necessarily present in real coffee drying.

## Strategy C result displays

`notebooks/strategy_c.ipynb` uses two six-input tanh MLPs to predict direct
local moisture and temperature log fields. Exponential readouts enforce exact
initial fields and positivity without imposing bulk monotonicity or convexity.
Bulk moisture is integrated from the same local field used by the PDE residuals
with differentiable Gauss-Legendre quadrature.

The notebook displays internal fields and a final loss figure, returns it as
`strategy_c_run["loss_figure"]`, and exports `loss_history.png` with
`loss_history.csv`. Adam's x-axis is epochs (pre-update snapshots); L-BFGS has
a separate closure-evaluation axis, including line-search trials, not epochs.
Adam defaults to data-only warmup (fraction 0.2), then a linear physics ramp
(fraction 0.4) and at least one full-objective epoch. Six physics/boundary
residual RMS scales are frozen after warmup from training collocation only.
Explicit raw, scaled, and weighted per-residual history fields reconstruct the
objective; exact initial-condition residuals are diagnostics, not objective
terms. Data loss remains the unchanged equal-curve MSE, with no data-relative
physics discount or alternating training. The grouped contributions are
dimensionless squared losses, not independent fits or errors in K².

The current seeded split preserves the original **16 complete training
curves** and divides the eight previously held-out curves into **4 validation
and 4 final-test curves**. The two evaluation partitions are disjoint, each
contains two curves per air speed and one per diameter, and never divide
timepoints from a curve. Only training features fit the feature transform.
Available training moisture targets fit the model; valid noisy mass increases
are retained. There are **no temperature observations**: temperature is latent
and constrained by coupled physics.

New histories include `validation_data`: the equal-curve average of per-curve
MSE in depth-averaged X/X0, unweighted and dimensionless. It uses training
quadrature, no gradients, and is excluded from the objective. Final-test
targets are not read during Adam or L-BFGS and are scored only during final
reporting. No early stopping or model-selection rule is added. The historical
`20261118T222054.508698Z` run used 16 training curves and 8 held-out test
curves with no separate validation set; its `held_out_data` history remains
clearly labeled as legacy test monitoring and is never relabeled validation.
Final metrics can differ due to finer evaluation quadrature or a different
final accepted optimizer state.

The temperature field title and profile labels are simplified, but its
normalization remains θ=(T−T_air)/(T_initial−T_air), with signed scale −45 K
by default. Overshoots are not clipped.
Existing run artifacts are not rewritten. The stale cached notebook run from
the former eight-curve holdout was cleared rather than presented as a
16/4/4 result. Existing CSVs can be passed to
`strategy_c.loss_visualization.plot_loss_history`, with the original
physics/initial/boundary weights; legacy `held_out_data` remains labeled as
test monitoring, and missing validation history is labeled unavailable rather
than reconstructed from the final checkpoint. Rerun to collect current history.

## Notebook code style

Lint `notebooks/data.ipynb` from the repository root with the following
PowerShell command (validated with Flake8 7.1.1). It extracts the seven code
cells into a temporary Python file outside the repository, prints a cell/line
map (cell numbers are one-based), and runs Flake8's default rules, including
the 79-character limit. It does not execute notebook code or training.
The temporary file is removed after the check.

```powershell
@'
import json
from pathlib import Path
import subprocess
import sys
import tempfile

notebook = json.loads(
    Path("notebooks/data.ipynb").read_text(encoding="utf-8")
)
parts = []
line = 1
for index, cell in enumerate(notebook["cells"], start=1):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"]).rstrip("\n")
    print(f"Cell {index}: lines {line}-{line + len(source.splitlines()) - 1}")
    parts.append(source)
    line += source.count("\n") + 3

with tempfile.TemporaryDirectory(prefix="coffee-notebook-lint-") as directory:
    extracted = Path(directory) / "data.py"
    extracted.write_text("\n\n\n".join(parts) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable, "-m", "flake8", str(extracted),
            "--statistics", "--count",
        ],
        check=False,
    )
sys.exit(result.returncode)
'@ | python -
```

For bulk notebook-aware formatting, use
`python -m ruff format notebooks/data.ipynb --line-length 79` (Ruff 0.12.0).
Then run the Flake8 command above: the formatter can leave long strings or
comments and inserts slice whitespace that Flake8 reports as E203. Wrap text
without changing string values and remove that whitespace manually rather than
ignoring rules. Keep notebook outputs and execution counts when editing source.
