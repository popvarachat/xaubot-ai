"""Run the GOLDmicro baseline without importing the upstream src package initializer.

The upstream ``src/__init__.py`` eagerly imports the full trading/ML stack
(polars, HMM, etc.).  The GOLDmicro baseline only needs the isolated broker,
risk, cost and replay modules, so this bootstrap supplies a lightweight
package shell and then executes ``run_goldmicro_baseline.py``.

This is read-only with respect to MT5: it does not place, modify or close
orders.
"""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
RUNNER = REPO_ROOT / "scripts" / "run_goldmicro_baseline.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid executing src/__init__.py, which pulls in the complete upstream ML
# dependency tree.  Submodules such as src.broker_profile and
# src.goldmicro_risk remain importable through this package path.
if "src" not in sys.modules:
    src_package = types.ModuleType("src")
    src_package.__path__ = [str(SRC_DIR)]
    src_package.__package__ = "src"
    sys.modules["src"] = src_package

runpy.run_path(str(RUNNER), run_name="__main__")
