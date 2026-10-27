"""CLI: alpha decay report across CAPM -> FF3 -> FF5+momentum (Day 3-4).

The point of adding factors is watching what happens to alpha as each one is
added - a shrinking alpha means the earlier model was mistaking a factor
exposure for skill. Runs all three models against all 10 Ken French
industry portfolios and prints one row per industry per model, then writes
the same table to ``outputs/alpha_decay.csv``.

Day 4 adds two things to the Day 3 table:

- Newey-West-corrected alpha t-stats and autocorrelation/heteroskedasticity
  flags alongside the plain-OLS ones (``factors.diagnostics``), so "is this
  alpha significant" has both an optimistic and an honest answer in the
  same row instead of a separate report.
- ``--since`` restricts every model - including CAPM and FF3, which
  otherwise run on the full 1926-2026 history - to the same start month.
  Day 3 found FF5+Mom's alpha moving in the *opposite* direction from FF3's
  for some industries and flagged a confound: FF5+Mom only has data from
  1963-07, so that comparison mixed "more factors" with "a shorter, more
  recent sample". Running ``--since 1963-07`` puts CAPM and FF3 on FF5+Mom's
  own window, isolating the factor-count effect from the window effect.

Usage:
    python -m factors.decay_report
    python -m factors.decay_report --since 1963-07
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

from factors.capm import run_capm
from factors.diagnostics import run_diagnostics
from factors.industry import INDUSTRY_COLUMNS, load_monthly_value_weighted
from factors.kenfrench import load_monthly_factors
from factors.multifactor import (
    FF3_COLUMNS,
    FF5_MOM_COLUMNS,
    alpha_change_attribution,
    load_ff5_mom_factors,
    run_factor_model,
)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "alpha_decay.csv"


def _restrict(frame: pd.DataFrame | pd.Series, since: str | None) -> pd.DataFrame | pd.Series:
    if since is None:
        return frame
    cutoff = pd.Period(since, freq="M")
    return frame.loc[frame.index >= cutoff]


def _rows(since: str | None = None) -> list[dict[str, object]]:
    ff3_factors = _restrict(load_monthly_factors(), since)
    ff5mom_factors = _restrict(load_ff5_mom_factors(), since)
    value_weighted = _restrict(load_monthly_value_weighted(), since)

    rows: list[dict[str, object]] = []
    for industry in INDUSTRY_COLUMNS:
        portfolio = value_weighted[industry]

        capm = run_capm(portfolio, ff3_factors)
        ff3 = run_factor_model(portfolio, ff3_factors, FF3_COLUMNS)
        ff5mom = run_factor_model(portfolio, ff5mom_factors, FF5_MOM_COLUMNS)

        capm_diag = run_diagnostics(portfolio, ff3_factors, ("mkt_rf",))
        ff3_diag = run_diagnostics(portfolio, ff3_factors, FF3_COLUMNS)
        ff5mom_diag = run_diagnostics(portfolio, ff5mom_factors, FF5_MOM_COLUMNS)

        for model, result, diag in (
            ("CAPM", capm, capm_diag),
            ("FF3", ff3, ff3_diag),
            ("FF5+Mom", ff5mom, ff5mom_diag),
        ):
            rows.append(
                {
                    "industry": industry,
                    "model": model,
                    "n_obs": result.n_obs,
                    "alpha_annual": result.alpha_annual,
                    "alpha_t": result.alpha_t,
                    "significant": result.significant,
                    "alpha_t_nw": diag.alpha_t_nw,
                    "significant_nw": diag.significant_nw,
                    "autocorrelated": diag.autocorrelated,
                    "heteroskedastic": diag.heteroskedastic,
                    "r_squared": result.r_squared,
                }
            )
    return rows



def decay_tally(rows: list[dict[str, object]]) -> dict[str, object]:
    """P127 (CRITIQUES.md): per industry, does |alpha| shrink or grow from CAPM to FF5+Mom?

    The README's question is about all ten industries, not HiTec alone.
    ``matched_window`` is False when the two models ran on different
    months (the default run: 1,201 vs 757), where the comparison mixes in
    the window confound; use ``--since 1963-07`` for the honest tally.
    """
    by = {(r["industry"], r["model"]): r for r in rows}
    industries = list(dict.fromkeys(r["industry"] for r in rows))
    out: dict[str, object] = {"shrinks": [], "grows": [], "unchanged": [], "matched_window": True}
    for industry in industries:
        capm, ff5 = by[(industry, "CAPM")], by[(industry, "FF5+Mom")]
        if capm["n_obs"] != ff5["n_obs"]:
            out["matched_window"] = False
        before, after = abs(capm["alpha_annual"]), abs(ff5["alpha_annual"])
        key = "shrinks" if after < before else "grows" if after > before else "unchanged"
        out[key].append(industry)
    return out


def _print_attribution(industry: str, since: str | None) -> None:
    portfolio = _restrict(load_monthly_value_weighted(), since)[industry]
    r = alpha_change_attribution(
        portfolio,
        _restrict(load_monthly_factors(), since), FF3_COLUMNS,
        _restrict(load_ff5_mom_factors(), since), FF5_MOM_COLUMNS,
    )
    print(
        f"\n{industry} alpha-change attribution, FF3 -> FF5+Mom (n = {r['n_obs'][0]} / {r['n_obs'][1]} months; "
        "loading x mean premium, monthly x 12, each model's own factor series):"
    )
    print(f"{'factor':<8}{'FF3 beta':>10}{'FF5+Mom beta':>14}{'pp/yr':>9}")
    for factor, value in r["contributions"].items():
        print(
            f"{factor:<8}{r['base_loadings'].get(factor, 0.0):>+10.3f}"
            f"{r['rich_loadings'].get(factor, 0.0):>+14.3f}{value * 1200:>+9.2f}"
        )
    print(f"{'residual':<32}{r['residual'] * 1200:>+9.2f}")
    print(f"{'total alpha change':<32}{r['alpha_change'] * 1200:>+9.2f}")

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        default=None,
        metavar="YYYY-MM",
        help=(
            "restrict every model to months >= this one, e.g. 1963-07 (FF5+Mom's own "
            "start) to isolate factor-count effects from Day 3's sample-window confound"
        ),
    )
    parser.add_argument(
        "--attribution",
        default=None,
        metavar="INDUSTRY",
        help="also print, for this industry, which factors move alpha from FF3 to FF5+Mom (P128)",
    )
    args = parser.parse_args(argv)
    rows = _rows(args.since)

    header = (
        f"{'industry':<7}{'model':<10}{'n':>6}{'alpha (ann.)':>14}"
        f"{'t (OLS)':>10}{'t (NW)':>10}{'sig':>6}{'sig_nw':>8}{'autocorr':>10}{'het':>6}{'R2':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['industry']:<7}{r['model']:<10}{r['n_obs']:>6}"
            f"{r['alpha_annual']:>+14.4%}{r['alpha_t']:>+10.2f}{r['alpha_t_nw']:>+10.2f}"
            f"{str(r['significant']):>6}{str(r['significant_nw']):>8}"
            f"{str(r['autocorrelated']):>10}{str(r['heteroskedastic']):>6}{r['r_squared']:>7.3f}"
        )

    tally = decay_tally(rows)
    total = len(tally["shrinks"]) + len(tally["grows"]) + len(tally["unchanged"])
    print(
        f"\n|alpha| shrinks CAPM -> FF5+Mom in {len(tally['shrinks'])} of {total} industries "
        f"({', '.join(tally['shrinks'])}); grows in {len(tally['grows'])} ({', '.join(tally['grows'])})"
    )
    if not tally["matched_window"]:
        print("(CAPM and FF5+Mom ran on different months here; rerun with --since 1963-07 for the matched-window tally)")

    if args.attribution:
        _print_attribution(args.attribution, args.since)

    out_path = OUTPUT_PATH if args.since is None else OUTPUT_PATH.with_name(
        f"alpha_decay_since_{args.since.replace('-', '')}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
