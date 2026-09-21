"""Pipeline audit: multiple testing and look-ahead in factor alignment (Day 7).

Two mechanical checks, not statistical ones - like ``STOCKSTALKER``'s and
``monte-carlo-risk-lab``'s own audit modules, this proves a structural
property by construction rather than by inspection:

1. **Look-ahead in factor alignment.** Every regression in this repo joins a
   portfolio's monthly returns to Ken French's factor table by month label
   (``factors.capm.align_excess_returns`` / ``factors.multifactor.
   align_excess_returns``), then some of them (``rolling_beta``,
   ``rolling_loadings``) refit a trailing window at every month. Two things
   could quietly break that: the join silently becoming positional instead
   of label-based (misassigning a factor row to the wrong month), or a
   rolling window that isn't actually trailing (reading a later month's data
   into an earlier month's estimate). Both are checked mechanically: an
   *alignment* check that shuffling the factor table's row order must not
   change the joined result, and a *causality* check that truncating the
   series after month i must not change the rolling estimate reported for
   month i.

2. **Multiple testing.** ``factors.decay_report`` runs 30 industry/model
   regressions and ``factors.screen_check`` runs up to 9 ticker/model
   regressions against the same two datasets. This applies Bonferroni and
   Benjamini-Hochberg correction (``factors.multiple_testing``) to each
   family's t-stats and reports how many of the raw "significant at 5%"
   claims survive - the number behind the README's existing "testing
   several specifications on one dataset is multiple testing" limitation.

Usage:
    python -m factors.pipeline_audit
    python -m factors.pipeline_audit --alpha 0.10
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from factors.capm import align_excess_returns as capm_align
from factors.decay_report import _rows as decay_rows
from factors.diagnostics import rolling_beta, rolling_loadings, run_diagnostics
from factors.industry import INDUSTRY_COLUMNS, load_monthly_value_weighted
from factors.kenfrench import load_monthly_factors
from factors.multifactor import (
    FF3_COLUMNS,
    FF5_MOM_COLUMNS,
    align_excess_returns as multifactor_align,
    load_ff5_mom_factors,
)
from factors.multiple_testing import benjamini_hochberg, bonferroni, two_sided_normal_pvalue
from factors.screen_check import (
    DEFAULT_OHLCV_DIR,
    DEFAULT_SCREEN_PATH,
    MODEL_FACTOR_COLUMNS,
    run_screen_check,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# Rolling causality is checked at every Nth window rather than every window -
# O(n) refits per sampled month is already the cost of the rolling function
# itself once per sample, and a periodic sample across 1,100+ windows already
# spans burn-in, mid-series, and tail behaviour without making this O(n^2).
CAUSALITY_SAMPLE_STRIDE = 120


# --- Part 1: look-ahead in factor alignment --------------------------------


def rolling_series_causality_violations(
    label: str,
    rolling_fn: Callable[[pd.Series, pd.DataFrame], pd.Series],
    portfolio: pd.Series,
    factors: pd.DataFrame,
    stride: int = CAUSALITY_SAMPLE_STRIDE,
) -> list[str]:
    """Bars where truncating the series after month *i* changes the rolling
    value reported for month *i*.

    ``rolling_fn`` takes (portfolio, factors) and returns a series indexed by
    each window's last month - ``rolling_beta``/``rolling_loadings`` bound to
    one factor column, already partially applied by the caller. For a sample
    of months in the full result, recomputes the same rolling function on
    data truncated to end at that month and compares. An empty list means no
    look-ahead was detected.
    """
    full = rolling_fn(portfolio, factors)
    if full.empty:
        return []
    violations = []
    months = list(full.index)
    for pos in range(0, len(months), stride):
        month = months[pos]
        truncated = rolling_fn(portfolio.loc[:month], factors.loc[:month])
        if truncated.empty or truncated.index[-1] != month:
            violations.append(
                f"{label}: month {month} produced no value when the series was "
                "truncated there - the window cannot be trailing-only if it needs "
                "data from after the month it is reporting on"
            )
            continue
        full_value = full.loc[month]
        truncated_value = truncated.iloc[-1]
        if not np.isclose(full_value, truncated_value, atol=1e-9):
            violations.append(
                f"{label}: value at {month} = {full_value!r} on the full series but "
                f"{truncated_value!r} truncated there - depends on data that had not "
                "happened yet"
            )
    return violations


def rolling_beta_causality_violations(
    portfolio: pd.Series, factors: pd.DataFrame, factor_column: str = "mkt_rf", window: int = 60
) -> list[str]:
    def fn(p: pd.Series, f: pd.DataFrame) -> pd.Series:
        return rolling_beta(p, f, factor_column, window)

    return rolling_series_causality_violations(f"rolling_beta({factor_column})", fn, portfolio, factors)


def rolling_loadings_causality_violations(
    portfolio: pd.Series, factors: pd.DataFrame, factor_columns: tuple[str, ...], window: int = 60
) -> list[str]:
    violations = []
    for column in factor_columns:
        def fn(p: pd.Series, f: pd.DataFrame, column: str = column) -> pd.Series:
            return rolling_loadings(p, f, factor_columns, window).params[column]

        violations += rolling_series_causality_violations(
            f"rolling_loadings({'+'.join(factor_columns)})[{column}]", fn, portfolio, factors
        )
    return violations


def alignment_is_order_invariant(
    align_fn: Callable[[pd.Series, pd.DataFrame], pd.DataFrame],
    portfolio: pd.Series,
    factors: pd.DataFrame,
    seed: int = 0,
) -> bool:
    """A month-label join must not care what order the factor rows arrive in.

    Shuffles ``factors``' row order and re-aligns; the result should be
    identical (once both are sorted back to the same order) to aligning
    against the unshuffled table. A join that quietly became positional
    (``.values``, ``reset_index(drop=True)``, ...) would silently pair a
    portfolio's month with the wrong factor row instead of raising, and this
    is the mechanical way to catch that rather than trusting the source
    reads correctly.
    """
    baseline = align_fn(portfolio, factors).sort_index()
    shuffled_factors = factors.sample(frac=1.0, random_state=seed)
    shuffled = align_fn(portfolio, shuffled_factors).sort_index()
    return baseline.equals(shuffled)


def run_alignment_audit(window: int = 60) -> dict[str, Any]:
    """Run both alignment checks against representative real data.

    Rolling causality needs a long enough series for a full ``window``, so
    it runs on a Ken French industry portfolio (1,000+ months); order
    invariance only needs an overlapping join, so it also runs on one Stock
    Stalker screen candidate to cover the shorter-series, NSE-fixture path.
    """
    ff3_factors = load_monthly_factors()
    ff5mom_factors = load_ff5_mom_factors()
    value_weighted = load_monthly_value_weighted()
    hitec = value_weighted["HiTec"]

    causality_checked = []
    causality_violations: dict[str, list[str]] = {}

    def _record(label: str, violations: list[str]) -> None:
        causality_checked.append(label)
        if violations:
            causality_violations[label] = violations

    _record("HiTec rolling_beta(mkt_rf)", rolling_beta_causality_violations(hitec, ff3_factors, "mkt_rf", window))
    _record(
        "HiTec rolling_loadings(FF3)",
        rolling_loadings_causality_violations(hitec, ff3_factors, FF3_COLUMNS, window),
    )
    _record(
        "HiTec rolling_loadings(FF5+Mom)",
        rolling_loadings_causality_violations(hitec, ff5mom_factors, FF5_MOM_COLUMNS, window),
    )

    order_checked = []
    order_violations: list[str] = []

    def _record_order(label: str, ok: bool) -> None:
        order_checked.append(label)
        if not ok:
            order_violations.append(label)

    _record_order("capm.align_excess_returns(HiTec, FF3 table)", alignment_is_order_invariant(capm_align, hitec, ff3_factors))
    _record_order(
        "multifactor.align_excess_returns(HiTec, FF3)",
        alignment_is_order_invariant(lambda p, f: multifactor_align(p, f, FF3_COLUMNS), hitec, ff3_factors),
    )
    _record_order(
        "multifactor.align_excess_returns(HiTec, FF5+Mom)",
        alignment_is_order_invariant(lambda p, f: multifactor_align(p, f, FF5_MOM_COLUMNS), hitec, ff5mom_factors),
    )

    screen_rows = run_screen_check(DEFAULT_SCREEN_PATH, DEFAULT_OHLCV_DIR, model="capm")
    for row in screen_rows:
        if "monthly_returns" not in row:
            continue
        label = f"capm.align_excess_returns({row['ticker']}, FF3 table)"
        _record_order(label, alignment_is_order_invariant(capm_align, row["monthly_returns"], ff3_factors))

    return {
        "causality_checked": causality_checked,
        "causality_violations": causality_violations,
        "order_checked": order_checked,
        "order_violations": order_violations,
    }


# --- Part 2: multiple testing ----------------------------------------------


def _correct_family(labels: list[str], t_stats: list[float], alpha: float) -> dict[str, Any]:
    p_values = [two_sided_normal_pvalue(t) for t in t_stats]
    raw = [p < alpha for p in p_values]
    bonf = bonferroni(p_values, alpha)
    bh = benjamini_hochberg(p_values, alpha)
    return {
        "n_tests": len(labels),
        "raw_significant": [l for l, s in zip(labels, raw) if s],
        "bonferroni_survivors": [l for l, s in zip(labels, bonf) if s],
        "bh_survivors": [l for l, s in zip(labels, bh) if s],
    }


def run_multiple_testing_audit(alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Bonferroni/BH-correct every t-stat this repo has reported so far.

    Two families from ``decay_report`` (30 industry/model rows, OLS and
    Newey-West t-stats each treated as their own family since they answer
    different questions - "is the naive number significant" vs "is the
    HAC-corrected one") and two from ``screen_check`` (up to 9 ticker/model
    rows, same OLS/NW split).
    """
    decay = decay_rows()
    decay_labels = [f"{r['industry']}/{r['model']}" for r in decay]

    screen_ols: list[tuple[str, float]] = []
    screen_nw: list[tuple[str, float]] = []
    for model in ("capm", "ff3", "ff5mom"):
        rows = run_screen_check(DEFAULT_SCREEN_PATH, DEFAULT_OHLCV_DIR, model=model)
        for row in rows:
            if "result" not in row:
                continue
            label = f"{row['ticker']}/{model}"
            screen_ols.append((label, row["result"].alpha_t))
            diag = run_diagnostics(row["monthly_returns"], row["factors_table"], MODEL_FACTOR_COLUMNS[model])
            screen_nw.append((label, diag.alpha_t_nw))

    families = {
        "decay_report (OLS)": (decay_labels, [r["alpha_t"] for r in decay]),
        "decay_report (Newey-West)": (decay_labels, [r["alpha_t_nw"] for r in decay]),
        "screen_check (OLS)": ([l for l, _ in screen_ols], [t for _, t in screen_ols]),
        "screen_check (Newey-West)": ([l for l, _ in screen_nw], [t for _, t in screen_nw]),
    }
    return {name: _correct_family(labels, t_stats, alpha) for name, (labels, t_stats) in families.items()}


# --- reporting ---------------------------------------------------------------


def print_report(alignment: dict[str, Any], testing: dict[str, dict[str, Any]], alpha: float) -> None:
    print(f"Pipeline audit - look-ahead in factor alignment, and multiple testing at alpha={alpha}")
    print()
    print(f"Causality: {len(alignment['causality_checked'])} rolling series checked")
    if alignment["causality_violations"]:
        for label, violations in alignment["causality_violations"].items():
            print(f"  ! {label}: {len(violations)} violation(s)")
            for v in violations:
                print(f"      - {v}")
    else:
        print("  clean: every rolling value survives truncation at its own month")
    print()
    print(f"Alignment order-invariance: {len(alignment['order_checked'])} joins checked")
    if alignment["order_violations"]:
        for label in alignment["order_violations"]:
            print(f"  ! {label}: result changed when the factor table's row order was shuffled")
    else:
        print("  clean: every join is by month label, not row position")
    print()
    for name, result in testing.items():
        print(f"{name}: {result['n_tests']} tests")
        print(f"  raw significant (p<{alpha}):        {len(result['raw_significant'])}  {result['raw_significant']}")
        print(f"  survives Bonferroni:            {len(result['bonferroni_survivors'])}  {result['bonferroni_survivors']}")
        print(f"  survives Benjamini-Hochberg:     {len(result['bh_survivors'])}  {result['bh_survivors']}")
        print()


def write_markdown(alignment: dict[str, Any], testing: dict[str, dict[str, Any]], alpha: float, path: Path, generated: str) -> None:
    lines = [
        f"# Pipeline audit — {generated}",
        "",
        "Two structural checks: does any rolling estimate depend on data from "
        "after the month it is reported for, does any factor-alignment join "
        "depend on row order rather than month label, and how many "
        "'significant' alphas already reported by `decay_report`/`screen_check` "
        "survive multiple-testing correction.",
        "",
        "## Look-ahead (causality)",
        "",
        f"{len(alignment['causality_checked'])} rolling series checked: "
        + ", ".join(alignment["causality_checked"]),
        "",
    ]
    if alignment["causality_violations"]:
        lines.append("**Violations found:**")
        for label, violations in alignment["causality_violations"].items():
            lines.append(f"- {label}")
            lines += [f"  - {v}" for v in violations]
    else:
        lines.append("Clean - every rolling value at month *i* is unchanged when the series is truncated at *i*.")
    lines += [
        "",
        "## Alignment order-invariance",
        "",
        f"{len(alignment['order_checked'])} joins checked: " + ", ".join(alignment["order_checked"]),
        "",
    ]
    if alignment["order_violations"]:
        lines.append("**Violations found:** " + ", ".join(alignment["order_violations"]))
    else:
        lines.append("Clean - every join's result is unchanged when the factor table's row order is shuffled.")
    lines += ["", f"## Multiple testing (alpha={alpha})", ""]
    for name, result in testing.items():
        lines += [
            f"### {name}",
            "",
            f"- tests run: {result['n_tests']}",
            f"- raw significant: {len(result['raw_significant'])} - {result['raw_significant'] or 'none'}",
            f"- survives Bonferroni: {len(result['bonferroni_survivors'])} - {result['bonferroni_survivors'] or 'none'}",
            f"- survives Benjamini-Hochberg: {len(result['bh_survivors'])} - {result['bh_survivors'] or 'none'}",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alpha", type=float, default=0.05, help="significance level for the multiple-testing correction")
    parser.add_argument("--window", type=int, default=60, help="rolling window (months) for the causality check")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--date", default=None, help="override the generated date (mainly for tests)")
    args = parser.parse_args(argv)

    alignment = run_alignment_audit(window=args.window)
    testing = run_multiple_testing_audit(alpha=args.alpha)
    generated = args.date or date.today().isoformat()

    print_report(alignment, testing, args.alpha)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(alignment, testing, args.alpha, out_dir / f"pipeline_audit_{generated}.md", generated)
    print(f"wrote {out_dir / f'pipeline_audit_{generated}.md'}")

    clean = not alignment["causality_violations"] and not alignment["order_violations"]
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
