"""CLI: rolling factor-loading charts with confidence bands (Day 6).

Day 4's ``rolling_report.py`` computed a single-factor (CAPM) rolling beta
and explicitly punted charting to Day 6 ("Charting this with confidence
bands is Day 6's job, not today's"). This module charts FF3 (default) or
FF5+Mom's rolling loadings for one industry, one PNG with a subplot per
factor, each with a 95% Wald confidence band
(``factors.diagnostics.rolling_loadings`` / ``confidence_band``).

Usage:
    python -m factors.rolling_chart --portfolio hitec
    python -m factors.rolling_chart --portfolio hitec --model ff5mom
    python -m factors.rolling_chart --portfolio hitec --window 36
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factors.diagnostics import confidence_band, rolling_loadings
from factors.industry import INDUSTRY_COLUMNS, load_monthly_value_weighted
from factors.kenfrench import load_monthly_factors
from factors.multifactor import FF3_COLUMNS, FF5_MOM_COLUMNS, load_ff5_mom_factors

MODEL_FACTOR_COLUMNS = {"ff3": FF3_COLUMNS, "ff5mom": FF5_MOM_COLUMNS}
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _resolve_industry(name: str) -> str:
    by_lower = {col.lower(): col for col in INDUSTRY_COLUMNS}
    key = name.strip().lower()
    if key not in by_lower:
        choices = ", ".join(INDUSTRY_COLUMNS)
        raise SystemExit(f"unknown --portfolio '{name}'; choose one of: {choices}")
    return by_lower[key]


def _load_factors(model: str):
    return load_ff5_mom_factors() if model == "ff5mom" else load_monthly_factors()


def render_chart(industry: str, model: str, window: int, output_dir: Path) -> tuple[Path, Path]:
    """Compute rolling loadings + bands for one industry/model and write PNG+CSV.

    Returns (png_path, csv_path). Raises ``SystemExit`` if ``window`` months
    don't fit inside the available overlap, same guard ``rolling_report.py``
    uses for the single-factor case.
    """
    factor_columns = MODEL_FACTOR_COLUMNS[model]
    factors = _load_factors(model)
    portfolio = load_monthly_value_weighted()[industry]

    result = rolling_loadings(portfolio, factors, factor_columns, window=window)
    if result.params.empty:
        raise SystemExit(f"no window of {window} months fits inside the available history")
    lower, upper = confidence_band(result.params, result.se)
    months = result.params.index.to_timestamp()

    fig, axes = plt.subplots(
        len(factor_columns), 1, figsize=(9, 2.6 * len(factor_columns)), sharex=True
    )
    axes = [axes] if len(factor_columns) == 1 else list(axes)
    for ax, col in zip(axes, factor_columns):
        ax.plot(months, result.params[col], color="#1f5fa8", linewidth=1.2, label=f"{col} loading")
        ax.fill_between(
            months, lower[col], upper[col], color="#1f5fa8", alpha=0.2, label="95% CI (plain OLS)"
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(col)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("month")
    fig.suptitle(f"{industry}: rolling {window}-month {model.upper()} loadings")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"rolling_loadings_{industry.lower()}_{model}.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    csv_path = output_dir / f"rolling_loadings_{industry.lower()}_{model}.csv"
    table = result.params.add_suffix("_loading").join(result.se.add_suffix("_se"))
    table.to_csv(csv_path)

    print(f"{industry}: {len(result.params)} rolling {model.upper()} windows ({window}mo)")
    for col in factor_columns:
        series = result.params[col]
        print(f"  {col:<8} min={series.min():+.3f} current={series.iloc[-1]:+.3f} max={series.max():+.3f}")
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    return png_path, csv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio", required=True, help="Ken French industry, e.g. hitec")
    parser.add_argument("--model", choices=("ff3", "ff5mom"), default="ff3")
    parser.add_argument(
        "--window", type=int, default=60, help="trailing months per loading estimate (default 60)"
    )
    args = parser.parse_args(argv)

    industry = _resolve_industry(args.portfolio)
    render_chart(industry, args.model, args.window, OUTPUT_DIR)


if __name__ == "__main__":
    main(sys.argv[1:])
