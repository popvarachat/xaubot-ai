"""Dependency preflight for the isolated GOLDmicro research environment.

Run this with the dedicated Python 3.11 virtual environment before any challenger
training.  It performs imports only; it does not connect to MT5, train models,
place orders, or mutate model state.
"""
from __future__ import annotations

import importlib
import sys


MODULES = (
    "polars",
    "pyarrow",
    "MetaTrader5",
    "numpy",
    "xgboost",
    "sklearn",
    "hmmlearn",
    "joblib",
    "loguru",
    "dotenv",
    "pandas",
    "openpyxl",
)


def main() -> int:
    failed: list[tuple[str, str]] = []
    print("=== GOLDmicro dependency preflight ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version   : {sys.version}")

    for name in MODULES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "ok")
            print(f"PASS {name}: {version}")
        except Exception as exc:  # noqa: BLE001 - preflight must report every import failure
            failed.append((name, repr(exc)))
            print(f"FAIL {name}: {exc}")

    if failed:
        print("\nPreflight FAILED:")
        for name, detail in failed:
            print(f"- {name}: {detail}")
        return 2

    print("\nPreflight PASSED. Research dependencies are importable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
