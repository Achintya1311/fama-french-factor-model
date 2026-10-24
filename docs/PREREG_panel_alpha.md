# Pre-registration: is there alpha in the NSE F&O cross-section? (R17)

Committed before any panel data is fetched or any regression is run, per
`portfolio-fixes/CRITIQUES.md` row R17 and `PLAN.md`'s pre-registration
convention ("changing X because I expect Y" is stated up front, not after
seeing results). The single-stock result below (GULFOILLUB, Day 5) is
kept as the worked example motivating this day, not replaced by it.

## The question this answers

Day 5's headline finding -- GULFOILLUB's FF3+WML alpha is +10.64%/yr but
t=+1.01 (OLS)/+1.20 (NW), "not significant" -- is one stock on 137 months.
The same README (`README.md:244-246`) already draws the underpowered-vs-null
distinction explicitly for the 22-month RELIANCE screen check ("'insignificant'
here mostly means 'underpowered,' not 'the edge is beta'") but never draws it
for GULFOILLUB's own, larger-but-still-single-asset result. A single time
series, however long, can only ever answer "is this one asset's alpha
distinguishable from zero"; it cannot answer "is there alpha in the NSE
F&O cross-section generally," which needs a cross-section (many assets,
not many months of one asset) to have any statistical power at all.

This day re-asks the question with power: not "does GULFOILLUB have alpha"
but "across the NSE F&O universe, is there alpha, and how much of the
panel's returns is plain factor exposure rather than anything stock-specific."

## Universe

- **Source**: STOCKSTALKER's committed `fixtures/fo_mktlots.csv` (NSE's own
  F&O lot-size snapshot, `list_last_modified: 2026-09-29`), already the
  portfolio's standard "is this a genuine F&O underlying" source (hub
  CRITIQUES.md D2). A copy is committed into this repo's own
  `fixtures/stockstalker/` snapshot directory (the same file-contract
  convention Day 5 already used for STOCKSTALKER's screen/OHLCV fixtures --
  never a cross-repo import, never a sibling-checkout read at run time).
- **Filter**: the file lists 6 index underlyings (NIFTY, BANKNIFTY,
  FINNIFTY, MIDCPNIFTY, NIFTYNXT50, NIFTYFPI) before a sub-header row
  ("Derivatives on Individual Securities,Symbol,..."); every row after that
  sub-header is a single-stock underlying. Only stock rows are in scope --
  indices are dropped. This gives **213 stock symbols** (verified by
  parsing the committed file directly: `python3 -I -c` count = 213), each
  suffixed `.NS` for yfinance, well above the day's 150-name floor.
- **No vendor data beyond what the repo already commits as a fixture**:
  the universe *list* (tickers only, no prices) is NSE's own published
  lot-size table, already committed elsewhere in this portfolio; only the
  *prices* behind it are newly fetched, and only through a GitHub Actions
  workflow (`scripts/fetch_panel_prices.py` /
  `.github/workflows/fetch-panel-prices.yml`, modeled on
  `markowitz-portfolio-optimizer`'s `fetch_largecaps.py`/
  `fetch-largecaps.yml`), never from this interactive sandbox (LESSONS.md
  point 4: this sandbox's own IP is not reliable for free-tier market-data
  fetches; GitHub's hosted runners are).

## Factor model

- **FF3+WML on IIM Ahmedabad's committed Indian factors** (`factors/iima.py`:
  `mf`, `smb`, `hml`, `wml`, `rf`), the exact same spec `factors/gulfoillub.py`
  already runs as `"ff3wml"` (`factors.multifactor.run_factor_model`,
  factor columns `("mf", "smb", "hml", "wml")`) -- not Ken French's USD
  factors, for the same currency-mismatch reason Day 4 introduced IIMA.
- **Standard errors**: both plain OLS (`run_factor_model`) and Newey-West
  (`factors.diagnostics.run_diagnostics`, automatic lag rule, this repo's
  existing convention) are computed and reported per stock; the panel-level
  tests below use the OLS alpha and its OLS standard error, consistently
  with how `factors/multiple_testing.py` already treats every other t-stat
  in this repo (the plain-OLS `|t|>1.96` reading, not a second, NW-based
  definition of "significant"). The NW alpha t-stat is reported alongside
  for comparison, per stock, the same way Day 5 reports both for GULFOILLUB.
- **Per-stock minimum sample**: at least 60 overlapping months between the
  stock's own price history and IIMA's factor history (which runs
  1993-10 to 2025-12). This is well above `factors/multifactor.py`'s own
  D15 floor (3x the parameter count = 15 months for FF3+WML's 5 parameters)
  -- the day's own 60-month floor is a *power* requirement, not just the
  existing anti-degenerate-fit guard. A stock below 60 overlapping months
  is dropped from the panel and the drop is logged with its actual month
  count, never silently.

## Primary tests (pre-registered; pass/fail recorded as found)

1. **GRS joint test of zero alpha** (Gibbons, Ross & Shanken, 1989). Unlike
   the per-stock regressions, GRS requires every included asset's residuals
   to be measured over the *same* T months (it tests one joint null across
   a shared covariance structure, not T independent per-stock windows). The
   GRS panel is therefore restricted to a **common window**: the latest
   common-start month across all panel stocks with full coverage through
   IIMA's 2025-12 cutoff. Only stocks with no gap across that common window
   are included in the GRS panel; the rest stay in the per-stock/BH
   reporting (test 2 below, which has no common-window requirement) but
   are excluded from GRS, and the resulting GRS panel size N is reported
   honestly even if it falls under 150 -- the pre-registered universe floor
   (150) applies to the per-stock panel (test 2, R-squared, and the power
   statement), not to the common-window GRS subset, since GRS's own
   statistical requirement (a shared T) is structurally different from a
   per-stock test's. If the GRS panel is too small to compute (N >= K+2 is
   the bare minimum for `Sigma` to be invertible; a meaningfully powered
   test needs N well above that), that shortfall itself is reported as a
   finding, not patched by loosening the common-window requirement after
   the fact.
   - Statistic: `GRS = ((T - N - K) / N) * (alpha' @ Sigma^-1 @ alpha) / (1 + mu_f' @ Omega_f^-1 @ mu_f)`,
     where `alpha` is the length-N OLS-alpha vector, `Sigma` is the N x N
     residual covariance matrix (from the same common-window OLS fits),
     `mu_f`/`Omega_f` are the K-factor mean vector and covariance matrix
     over the same window. Under H0 (all alphas jointly zero), GRS is
     distributed `F(N, T-N-K)`.
   - Implementation: new code in `factors/panel.py` (no existing GRS
     implementation anywhere in this repo's own code, confirmed by
     `grep -rln "GRS\|Gibbons" --include=*.py .` excluding `.venv`,
     returning nothing), using `scipy.stats.f` for the p-value -- the exact
     F-distribution, not the normal approximation `multiple_testing.py`
     uses elsewhere, since GRS has a known small-sample exact distribution
     and there is no reason to approximate it away.
2. **Benjamini-Hochberg at 10% FDR over per-stock alpha p-values.** Every
   stock meeting the 60-month floor (not just the GRS common-window subset)
   gets a two-sided p-value from its own OLS alpha t-stat, via
   `factors.multiple_testing.two_sided_normal_pvalue` (the existing normal
   approximation, consistent with every other `significant` flag in this
   repo), then `factors.multiple_testing.benjamini_hochberg(p_values, alpha=0.10)`
   -- reusing the existing implementation exactly, at the 10% FDR level
   R17 specifies (not this repo's other, 5%-default usages). The result is
   a count of how many stocks' alphas survive FDR control, reported
   alongside the raw count that would be "significant" at 5% with no
   correction at all, the same before/after-correction framing
   `decay_report`/`screen_check` already use.

## Secondary reporting (not a pass/fail test, reported either way)

- **Market share of R-squared per stock**: the per-stock panel's
  `FactorResult.r_squared` distribution (mean, median, min/max) -- how much
  of each stock's return variance the four factors explain, versus the
  residual (stock-specific) share.
- **Power statement**: the smallest alpha detectable at 80% power, per
  stock, from the standard two-sided minimum-detectable-effect formula
  `MDE = (z_{1-alpha/2} + z_{power}) * se(alpha_hat)` (z_0.975=1.96,
  z_0.80=0.8416), using each stock's own OLS `se(alpha)` from its fitted
  regression (not a single assumed residual volatility) and reporting the
  panel's median/mean MDE, annualized the same way `alpha_annual` already
  is elsewhere in this repo. This replaces guessing a "typical" sample size
  and volatility with the panel's own, actually-observed distribution.

## Rule

Pass or fail on both primary tests is recorded as found, in the README,
the same run the result is computed -- no re-running with a different
window, factor set, or correction level after seeing whether the panel
finds alpha. If the Actions-fetched price data does not arrive by the end
of this routine's run, this document, the fetch script and its workflow
are committed as this day's real, honest progress, and `NEXT_STEPS.md`'s
Day 83 is marked blocked with that one-line reason -- never a fabricated
or partially-faked panel result.

## Scope limits (unchanged portfolio rules)

No per-stock recommendation, rating, target, or direction call is produced
or published anywhere by this day's work -- the panel and GRS/BH results
are aggregate statistics about the cross-section as a whole (how many
stocks, what fraction of variance, whether the joint null is rejected),
never a per-ticker "this stock has alpha" claim. Nothing here is a pick,
a signal, or a hit rate (`stockstalker-lab`'s exclusive domain, PLAN.md
rule 1) -- it is a methodology question about this repo's own factor
model, asked with enough statistical power to mean something.
