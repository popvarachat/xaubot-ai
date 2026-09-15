"""Causal research entrypoint for the GOLDmicro 24 x N trainer.

The underlying candidate trainer is research-only, but its historical SMC import
uses the shared analyzer whose order-block chart annotations are retroactive.  We
replace only that research module's analyzer reference with the prefix-causal
wrapper before importing and invoking the batch runner.

This does not modify src/smc_polars.py, main_live.py, active models, or MT5 order
execution behavior.
"""
from __future__ import annotations

import src.goldmicro_candidate_trainer as candidate_trainer
from src.goldmicro_causal_smc import GoldmicroCausalSMCAnalyzer

# train_candidate() resolves SMCAnalyzer from its module globals at call time.
# Patch the research module only inside this Python process.
candidate_trainer.SMCAnalyzer = GoldmicroCausalSMCAnalyzer

from scripts.train_goldmicro_challenger_batch import main  # noqa: E402


if __name__ == "__main__":
    main()
