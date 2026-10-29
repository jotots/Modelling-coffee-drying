# Modelling-coffee-drying
Modelling drying process of coffee with PINNs for y chemical engineering thesis

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
