"""GULFOILLUB factor regression on IIM Ahmedabad's Indian factors (stockstalker-front-door Day 5).

``factors.screen_check`` answers "does Stock Stalker's screen survive factor
adjustment?" against Ken French's USD factors and a 3-ticker RELIANCE/
TATACHEM/CROMPTON universe. GULFOILLUB, the front-door project's actual
subject, is in neither that universe nor that currency - Day 4 already
loaded the Indian-market substitute (``factors.iima``) for exactly this
reason. This module is the regression that substitute was for: GULFOILLUB's
own monthly returns against IIMA's SMB/HML/WML/MF/RF, long enough
(2014-08 to 2025-12, 137 months - the inner join of the ~12-year OHLCV
fixture against IIMA's factor history, which stops at 2025-12) to not share
Day 5-7's 22-month, "underpowered" caveat on the RELIANCE screen check.

Three specifications, all built on the one generic multi-factor OLS path
(``factors.multifactor.run_factor_model``) rather than ``factors.capm``'s
own dataclass, so even the single-factor case returns a ``loadings`` dict:

- ``capm``   - IIMA's MF alone (MF is already a market *excess* return, the
  same role Ken French's Mkt-RF plays, so this is CAPM's single-factor
  spec, not a literal call into ``factors.capm.run_capm``, which hardcodes
  the Ken French column name ``mkt_rf`` and would not find it here)
- ``ff3``    - MF, SMB, HML, the three factors IIMA actually publishes
- ``ff3wml`` - the above plus WML (momentum)

IIMA has no RMW/CMA, so there is no FF5+Mom equivalent here - only Ken
French's own factors reach that model (``factors.screen_check``).

``--diagnostics`` reuses Day 4's ``factors.diagnostics.run_diagnostics``
unchanged (it is already generic over factor columns), the same
Newey-West/Ljung-Box/Breusch-Pagan block ``regress.py``/``screen_check.py``
already print.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from factors.diagnostics import run_diagnostics
from factors.iima import load_iima_factors
from factors.multifactor import FactorResult, run_factor_model
from factors.screen_check import load_monthly_returns

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OHLCV_PATH = ROOT / "fixtures" / "stockstalker" / "ohlcv" / "GULFOILLUB_NS.csv"

MODEL_FACTOR_COLUMNS: dict[str, tuple[str, ...]] = {
    "capm": ("mf",),
    "ff3": ("mf", "smb", "hml"),
    "ff3wml": ("mf", "smb", "hml", "wml"),
}
MODEL_LABELS = {
    "capm": "CAPM (IIMA MF only)",
    "ff3": "FF3 (IIMA MF/SMB/HML)",
    "ff3wml": "FF3+WML (IIMA MF/SMB/HML/WML)",
}

# The contract ships the richest spec. CAPM/FF3 can regress fine but are not
# written as the spine contract - one ticker gets one factor fixture, and
# the whole point of Day 5 over Day 4 is the WML loading.
CONTRACT_MODEL = "ff3wml"


def load_gulfoillub_returns(path: str | Path = DEFAULT_OHLCV_PATH) -> pd.Series:
    """GULFOILLUB's own monthly returns, via the same loader ``screen_check`` uses."""
    return load_monthly_returns(path)


def evaluate(
    model: str,
    returns: pd.Series | None = None,
    factors_table: pd.DataFrame | None = None,
) -> tuple[FactorResult, pd.DataFrame]:
    """Run one of ``MODEL_FACTOR_COLUMNS``'s specs; defaults load the real fixtures.

    ``returns``/``factors_table`` are overridable so tests can hand in a
    synthetic frame without touching disk.
    """
    if model not in MODEL_FACTOR_COLUMNS:
        raise ValueError(f"unknown model '{model}'; choose one of {sorted(MODEL_FACTOR_COLUMNS)}")
    returns = load_gulfoillub_returns() if returns is None else returns
    factors_table = load_iima_factors() if factors_table is None else factors_table
    result = run_factor_model(returns, factors_table, MODEL_FACTOR_COLUMNS[model])
    return result, factors_table


def to_contract(result: FactorResult) -> dict[str, Any]:
    """The ``factor`` block this module publishes to the spine (v0.2 shape).

    Mirrors ``screen_check.to_contract``: the market factor is written
    under ``mkt`` regardless of the source library's own column name (Ken
    French's ``mkt_rf`` there, IIMA's ``mf`` here), every other loading
    keeps its own name. ``gates.factor.load_contract`` (STOCKSTALKER) only
    checks that the four top-level keys are present, not which keys
    ``loadings`` itself carries - confirmed by reading that gate before
    relying on it - so ``wml``, a key no Ken French contract has ever
    shipped, passes through unchanged rather than needing a gate change.
    """
    loadings = {("mkt" if c == "mf" else c): result.loadings[c] for c in result.factor_columns}
    return {
        "factor": {
            "alpha_annual": result.alpha_annual,
            "alpha_t": result.alpha_t,
            "significant": result.significant,
            "loadings": loadings,
        }
    }


def _print_result(model: str, result: FactorResult) -> None:
    label = MODEL_LABELS[model]
    print(f"{label}: GULFOILLUB vs IIMA factors")
    print(f"  months regressed  : {result.n_obs}")
    print(f"  alpha (monthly)   : {result.alpha_monthly:+.4%}")
    print(f"  alpha (annualized): {result.alpha_annual:+.4%}")
    print(f"  alpha t-stat      : {result.alpha_t:+.2f} (OLS, not Newey-West - see --diagnostics)")
    loadings = ", ".join(
        f"{c}={result.loadings[c]:+.3f} (t={result.loadings_t[c]:+.2f})" for c in result.factor_columns
    )
    print(f"  loadings          : {loadings}")
    print(f"  R-squared         : {result.r_squared:.3f}")
    print(f"  significant (5%)  : {result.significant}")


def _print_diagnostics(diag) -> None:
    print("  --- diagnostics ---")
    print(f"  alpha t-stat (NW) : {diag.alpha_t_nw:+.2f} (lag={diag.nw_lag}, vs {diag.alpha_t_ols:+.2f} plain OLS)")
    print(f"  significant (NW)  : {diag.significant_nw}")
    loadings_nw = ", ".join(f"{c}={t:+.2f}" for c, t in diag.loadings_t_nw.items())
    print(f"  loadings t (NW)   : {loadings_nw} (vs plain-OLS t above)")
    print(
        f"  residual autocorr : {diag.autocorrelated} "
        f"(Ljung-Box lag={diag.ljung_box_lag}, stat={diag.ljung_box_stat:.2f}, p={diag.ljung_box_pvalue:.3f})"
    )
    print(
        f"  heteroskedastic   : {diag.heteroskedastic} "
        f"(Breusch-Pagan stat={diag.breusch_pagan_stat:.2f}, p={diag.breusch_pagan_pvalue:.3f})"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", default="ff3wml", choices=sorted(MODEL_FACTOR_COLUMNS), help="which IIMA spec to run"
    )
    parser.add_argument(
        "--diagnostics", action="store_true", help="add the Newey-West/Ljung-Box/Breusch-Pagan block"
    )
    parser.add_argument(
        "--contract",
        metavar="PATH",
        default=None,
        help=f"write the v0.2 factor contract (requires --model {CONTRACT_MODEL}) as JSON to this "
        "path, for the spine to read as a file - never as a Python import",
    )
    args = parser.parse_args(argv)

    if args.contract and args.model != CONTRACT_MODEL:
        parser.error(f"--contract requires --model {CONTRACT_MODEL} (the shipped spine spec)")

    try:
        result, factors_table = evaluate(args.model)
    except ValueError as exc:
        # D15: an insufficient-data regression (a too-thin inner join) is a
        # normal, expected outcome -- print the one-line reason and stop,
        # never a full traceback, and never write a fabricated --contract.
        raise SystemExit(str(exc)) from exc
    _print_result(args.model, result)

    if args.diagnostics:
        diag = run_diagnostics(load_gulfoillub_returns(), factors_table, MODEL_FACTOR_COLUMNS[args.model])
        _print_diagnostics(diag)

    if args.contract:
        contract_path = Path(args.contract)
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(to_contract(result), indent=2) + "\n")
        print(f"\nwrote GULFOILLUB factor contract to {contract_path}")


if __name__ == "__main__":
    main()
