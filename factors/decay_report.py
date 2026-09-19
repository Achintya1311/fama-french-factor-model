"""CLI: alpha decay report across CAPM -> FF3 -> FF5+momentum (Day 3).

The point of adding factors is watching what happens to alpha as each one is
added - a shrinking alpha means the earlier model was mistaking a factor
exposure for skill. Runs all three models against all 10 Ken French
industry portfolios and prints one row per industry per model, then writes
the same table to ``outputs/alpha_decay.csv``.

Usage:
    python -m factors.decay_report
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from factors.capm import run_capm
from factors.industry import INDUSTRY_COLUMNS, load_monthly_value_weighted
from factors.kenfrench import load_monthly_factors
from factors.multifactor import FF3_COLUMNS, FF5_MOM_COLUMNS, load_ff5_mom_factors, run_factor_model

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "alpha_decay.csv"


def _rows() -> list[dict[str, object]]:
    ff3_factors = load_monthly_factors()
    ff5mom_factors = load_ff5_mom_factors()
    value_weighted = load_monthly_value_weighted()

    rows: list[dict[str, object]] = []
    for industry in INDUSTRY_COLUMNS:
        portfolio = value_weighted[industry]

        capm = run_capm(portfolio, ff3_factors)
        ff3 = run_factor_model(portfolio, ff3_factors, FF3_COLUMNS)
        ff5mom = run_factor_model(portfolio, ff5mom_factors, FF5_MOM_COLUMNS)

        for model, result in (("CAPM", capm), ("FF3", ff3), ("FF5+Mom", ff5mom)):
            rows.append(
                {
                    "industry": industry,
                    "model": model,
                    "n_obs": result.n_obs,
                    "alpha_annual": result.alpha_annual,
                    "alpha_t": result.alpha_t,
                    "significant": result.significant,
                    "r_squared": result.r_squared,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> None:
    del argv
    rows = _rows()

    header = f"{'industry':<7}{'model':<10}{'n':>6}{'alpha (ann.)':>14}{'alpha t':>10}{'sig':>6}{'R2':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['industry']:<7}{r['model']:<10}{r['n_obs']:>6}"
            f"{r['alpha_annual']:>+14.4%}{r['alpha_t']:>+10.2f}"
            f"{str(r['significant']):>6}{r['r_squared']:>7.3f}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main(sys.argv[1:])
