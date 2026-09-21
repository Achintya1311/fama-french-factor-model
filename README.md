# Fama-French factor model

Decomposes returns into market, size, value and momentum exposures to test whether an apparent edge is alpha or just beta wearing a disguise.

**Status:** Last checkpoint 2026-09-20 · Next: Day 6 - rolling factor-loading charts with confidence bands

## What this is

CAPM, then FF3, then FF5 plus momentum, reporting how alpha decays as factors are added. The decay is the result.

Pointed at the Stock Stalker screen output to answer one question directly: does that screen survive factor adjustment? If the answer is no, both READMEs say so.

## Correctness gate

Replicates a published FF3 result on a known portfolio within tolerance before being trusted on new data.

This is the test that decides whether the repo is finished. A result that has not passed it is a draft.

## Data sources

Every source is free. Nothing in this project requires a paid tier, a subscription, or a funded account.

- Ken French Data Library - factor returns and industry test portfolios, both cached as committed fixtures

`factors/kenfrench.py` loads `fixtures/ken_french/F-F_Research_Data_Factors.csv`
(refresh with `python scripts/fetch_ken_french.py`, network only, never on the
test/CLI path) and parses both the monthly and annual sections French ships
in one file. `factors/industry.py` loads
`fixtures/ken_french/10_Industry_Portfolios.csv` (refresh with
`python scripts/fetch_industry_portfolios.py`) the same way - these ten
industry portfolios are the test assets the regressions in
`factors/regress.py` run against, standing in for the yfinance test
portfolios sketched in the original plan (no key needed, so it's the
zero-spend, no-network path). See `fixtures/ken_french/README.md` for
provenance and format notes on both files.

- Stock Stalker's own screen output and OHLCV fixtures - a committed
  snapshot, not a live read of the sibling repo

`factors/screen_check.py` reads `fixtures/stockstalker/screen_2026-09-20.json`
(Stock Stalker's `schema/v1` screen contract) and the three OHLCV CSVs
behind it under `fixtures/stockstalker/ohlcv/`, both copied in as-is from
`STOCKSTALKER/outputs/` and `STOCKSTALKER/fixtures/ohlcv/`. Committing the
snapshot here (rather than reading the other checkout by relative path)
follows the same file-contract discipline the hub uses everywhere else:
this repo's tests never depend on `STOCKSTALKER` being checked out
alongside it, and never will unless the screen genuinely needs refreshing.

## How to run

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m factors.regress --portfolio hitec --model capm    # or ff3, ff5mom
python -m factors.regress --portfolio hitec --model ff5mom --diagnostics  # + Newey-West/autocorr/heteroskedasticity
python -m factors.decay_report                              # all 10 industries x all 3 models
python -m factors.decay_report --since 1963-07               # same, restricted to FF5+Mom's own window
python -m factors.rolling_report                             # 60-month trailing CAPM beta, all 10 industries
python -m factors.screen_check --model capm                  # Stock Stalker's screen vs Ken French factors
python -m factors.screen_check --model ff5mom --diagnostics  # same, FF5+Mom + Newey-West block
```

`--portfolio` takes one of the 10 Ken French industry portfolios (NoDur,
Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, Other), the CAPM/FF3/
FF5 test assets - not yet a path to an arbitrary returns CSV. `--model` is
`capm` (Mkt-RF), `ff3` (+ SMB, HML), or `ff5mom` (+ RMW, CMA, Mom).
`--diagnostics` adds a Day 4 block: Newey-West-corrected alpha t-stat,
Ljung-Box residual-autocorrelation test, Breusch-Pagan heteroskedasticity
test. `factors.decay_report` runs all three models against all ten
industries (plus the same diagnostics columns) and writes
`outputs/alpha_decay.csv`; `--since YYYY-MM` restricts every model to the
same start month, for isolating a factor-count effect from a sample-window
effect. `factors.rolling_report` writes a trailing-window CAPM beta series
per industry to `outputs/rolling_betas.csv` (`--window` to change the
default 60 months). `factors.screen_check` regresses each candidate in
Stock Stalker's committed screen snapshot (`--screen`, `--ohlcv-dir` to
point elsewhere) against Ken French's factors under `--model`
(capm/ff3/ff5mom), with the same `--diagnostics` block, and reports
whether any candidate's alpha survives adjustment. Runs entirely offline
against committed fixtures; nothing here needs network access or a key.

## Findings

**Day 2 - CAPM baseline.** Regressed each of the 10 industry value-weighted
portfolios on Mkt-RF, 1926-07 to 2026-07 (1,201 months). Results are in the
expected direction with no surprises: NoDur and Utils have the lowest betas
(0.74, 0.76 - defensive sectors), Durbl and HiTec the highest (1.28, 1.23 -
cyclical/growth), R² ranges 0.53-0.91. Two industries (NoDur, Hlth) show a
plain-OLS-significant positive alpha (t > 1.96) - consistent with the
well-documented value/quality tilt in those sectors, but that t-stat is not
yet Newey-West-corrected (Day 4), so "significant" here should be read as
optimistic, not final. The regression machinery itself was validated
against an exact known answer first: reconstructing the market portfolio as
`mkt_rf + rf` and regressing it on `mkt_rf` returns alpha=0, beta=1, R²=1
to 1e-9 - the only way that isn't a bug in the OLS wiring.

**Day 3 - FF3, then FF5+momentum; alpha decay is not what the plan expected.**
`factors/multifactor.py` generalizes the single-factor OLS in `capm.py` to
an arbitrary factor set, so FF3 (Mkt-RF, SMB, HML) and FF5+Mom (+ RMW, CMA,
Mom) share one regression path. Same correctness gate as Day 2, extended:
there's no internet access in this sandbox to a journal's published FF3
table to replicate against, so the check instead regresses each Fama-French
factor on the full set it belongs to - an exact analytical identity (its
own loading=1, every other loading and alpha=0, R²=1), not an estimate -
for SMB, HML (FF3), RMW, and Mom (FF5+Mom); see `tests/test_multifactor.py`
for why this stands in for "replicates a published result."

Running `factors/decay_report.py` across all 10 industries and all 3
models (`outputs/alpha_decay.csv`) does **not** show the monotonic
shrinkage the plan expected. NoDur's alpha does shrink and flip
insignificant as expected (CAPM +1.89% t=2.35 -> FF3 +1.77% t=2.20 ->
FF5+Mom -1.28% t=-1.32). But HiTec's does the opposite: CAPM +0.57%
(t=0.54, n.s.) -> FF3 +1.51% (t=1.55, n.s.) -> FF5+Mom **+5.44% (t=4.47,
significant)** - alpha *grows* and turns significant as more factors are
added, the reverse of "the edge was just beta in disguise." The honest
reason is a confound the plan didn't anticipate, not a discovery: FF5
data only starts 1963-07 (RMW/CMA need book equity and profitability data
CRSP/Compustat doesn't reach as far back as Mkt-RF/SMB/HML), so CAPM/FF3
run on 1,201 months (1926-2026) and FF5+Mom on 757 (1963-2026) - a
different, shorter, more tech-heavy sample window, not just a richer
factor set. Comparing alpha across models here conflates "more factors"
with "a different 63 years of history"; a same-window comparison (FF3
re-run restricted to 1963-2026) is the honest fix and is Day 4/5 scope,
not done yet. Reported as-is rather than reframed to fit the expected
story - see Limitations.

**Day 4 - diagnostics, and isolating Day 3's window confound.**
`factors/diagnostics.py` adds Newey-West (HAC) alpha t-stats, a Ljung-Box
test for residual autocorrelation, a Breusch-Pagan test for
heteroskedasticity, and a rolling CAPM beta, on top of the plain-OLS
regressions `capm.py`/`multifactor.py` already run - `factors/regress.py
--diagnostics` prints them for one portfolio, `factors/decay_report.py`
adds them as extra columns across all 30 industry/model combinations, and
`factors/rolling_report.py` reports a 60-month trailing beta per industry.

The Newey-West correction is not cosmetic: **NoDur's FF3 alpha flips from
plain-OLS-significant (t=2.20) to Newey-West-insignificant (t=1.92)** -
exactly the "read the plain t-stat as optimistic, not final" warning both
`capm.py`'s and `multifactor.py`'s own docstrings already carried. Across
all 30 industry x model rows, 16 (53%) show significant residual
autocorrelation at a 12-month Ljung-Box lag and 20 (67%) show
heteroskedasticity - both diagnostics that make a plain-OLS t-stat
unreliable are common in this dataset, not edge cases. Not one row gained
significance under the Newey-West correction (only lost it or stayed the
same), consistent with the standard result that positive serial
correlation inflates OLS t-stats rather than deflating them.

Rolling 60-month CAPM betas (`outputs/rolling_betas.csv`) show market
exposure is not a stable per-industry constant: Durbl's beta ranges from
0.75 to 2.06 across the sample, Telcm from 0.27 to 1.38. The most recent
window's beta lines up reasonably with Day 2's full-sample beta for most
industries (HiTec: 1.26 rolling vs 1.23 full-sample) - a sanity check that
the two methods agree on the same portfolio, not evidence that beta is
actually constant.

**The window-isolation re-run (`--since 1963-07`, matching FF5+Mom's own
start) answers Day 3's open question, and the answer is "both."**
Restricting CAPM and FF3 to the same 757-month window FF5+Mom already runs
on: HiTec's CAPM alpha stays small and insignificant (+0.26%, t=0.17), but
FF3's alpha - which was insignificant in the mismatched full-window
comparison (+1.51%, t=1.55) - becomes significant once window-matched
(+2.69%, t=2.17). So part of Day 3's "alpha grows as factors are added"
finding for HiTec was genuinely a window artifact: comparing FF3 on 1,201
months to FF5+Mom on 757 was misleading. But the growth doesn't stop
there - FF5+Mom's alpha (+5.44%, t=4.32) is still well above the
window-matched FF3 figure, on the *identical* 757 months. That residual
growth cannot be a window effect, since the window is now held fixed; it
has to be the RMW/CMA/Mom loadings themselves pulling alpha up. The honest
summary: Day 3's HiTec finding was partly a confound (window) and partly
real (factors) - neither the original "alpha shrinks with more factors"
story nor a claim that the whole Day 3 finding was an artifact is
correct on its own.

**Day 5 - pointed at Stock Stalker's own screen, and the honest answer is
"the test is too small to say."** `factors/screen_check.py` reads Stock
Stalker's `screen_2026-09-20.json` (3 NSE candidates: RELIANCE.NS,
TATACHEM.NS, CROMPTON.NS, ranked by technical score - a snapshot committed
here under `fixtures/stockstalker/` alongside the OHLCV fixtures behind it,
so this never depends on the two repos being checked out together) and
runs each candidate's monthly returns through CAPM, FF3, and FF5+Mom
against Ken French's factors. Every model on every candidate comes back
insignificant, plain-OLS and Newey-West alike (CAPM alpha, annualized:
RELIANCE -9.98% t=-0.64, TATACHEM -34.31% t=-1.53, CROMPTON -28.78%
t=-1.43; FF3 and FF5+Mom move the point estimates but not the
insignificance - none crosses |t|>1.96 under either standard error).
That is **not** the clean "the edge was beta, adjustment killed it" story
Day 3's traps anticipated: the honest limiter is that Stock Stalker's
OHLCV fixtures only span 2024-09 to 2026-09, and Ken French's own factor
fixture stops 2026-07, leaving 22 overlapping months to estimate up to
seven parameters (FF5+Mom). Twenty-two months is not enough data to reject
*or* confirm an alpha of any plausible size - "insignificant" here mostly
means "underpowered," not "the edge is beta." One directionally suggestive
but statistically meaningless pattern: the technical score's own ranking
(RELIANCE > TATACHEM > CROMPTON) matches the ordering of least-to-most
negative CAPM alpha, which is at least not *inconsistent* with the screen
carrying information, but with n=22 this is not evidence of anything - see
Limitations.

**Day 6 - rolling factor loadings finally get a confidence band, and the
bands say the three FF3 loadings aren't equally trustworthy.**
`factors/diagnostics.py` gains `rolling_loadings` (generalizing Day 4's
single-factor `rolling_beta` to an arbitrary factor set) and
`confidence_band` (a 95% Wald band, `loading +/- 1.96*se`); `factors/
rolling_chart.py` charts both for one industry at a time
(`python -m factors.rolling_chart --portfolio hitec`), writing a PNG (one
subplot per factor) and the underlying loadings+SE table to `outputs/`.
Run on HiTec's 60-month rolling FF3 loadings (1,142 windows, 1931-06 to
2026-07): mkt_rf's band **excludes zero in all 1,142 windows** (avg width
0.31) - unsurprising, since a near-zero market beta would be a strange
result for a tech-heavy industry portfolio to ever produce, but a real
confirmation the band isn't uselessly wide. hml's band excludes zero in
868/1,142 windows (76%, avg width 0.50) - the loading swings from about 0
to -1.39 over the sample (matching Day 4's rolling-beta-style read of
"exposure isn't a stable constant"), and most of that swing is outside
the band, not band noise. smb is the one that doesn't hold up as well:
its band excludes zero in only 351/1,142 windows (31%, avg width 0.51) -
so for roughly 7 windows in 10, HiTec's rolling SMB loading is not
statistically distinguishable from flat exposure, even though the point
estimate itself ranges from -0.33 to +0.60. Reading the point estimate's
full range as "SMB loading varies a lot" without the band would overstate
how much of the visible wiggle is real versus estimation noise -
precisely the failure mode a confidence band exists to catch.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
| 2026-09-20 | `1d1f8a1` | Day 5: does Stock Stalker's screen survive factor adjustment? factors/screen_check.py reads Stock Stalker's screen_2026-09-20.json contract (3 NSE candidates) plus the OHLCV fixtures behind it, both committed as a fixture snapshot, and regresses each candidate's monthly returns through the existing CAPM/FF3/FF5+Mom machinery. Every candidate is insignificant under every model, plain-OLS and Newey-West alike (CAPM alpha annualized: RELIANCE -9.98% t=-0.64, TATACHEM -34.31% t=-1.53, CROMPTON -28.78% t=-1.43) - but the honest read is underpowered, not disproven: only 22 months overlap between Stock Stalker's OHLCV window and Ken French's own factor fixture, not enough to estimate up to 7 parameters (FF5+Mom) with any confidence. Also newly flagged: Day 5 regresses INR closes directly against USD factors with no currency adjustment, on top of the already-noted US-factors-on-NSE-names approximation. Both limitations recorded in README rather than glossed over. 62/62 tests pass (pytest, 11 new); screen_check run by hand against capm/ff3/ff5mom, with and without --diagnostics. | Day 6 - rolling factor-loading charts with confidence bands |
| 2026-09-20 | `54a693b` | Day 4: diagnostics on top of Day 2-3's plain-OLS regressions - factors/diagnostics.py adds Newey-West (HAC) alpha t-stats, a Ljung-Box residual-autocorrelation test, a Breusch-Pagan heteroskedasticity test, and a 60-month rolling CAPM beta, wired into regress.py --diagnostics and decay_report.py's new columns; rolling_report.py reports the beta series for all 10 industries. Not cosmetic: NoDur's FF3 alpha flips from OLS-significant (t=2.20) to NW-insignificant (t=1.92), and 16/30 (autocorrelation) and 20/30 (heteroskedasticity) of all industry/model rows are flagged. Also closes the loop Day 3's checkpoint flagged: decay_report.py --since 1963-07 restricts every model to FF5+Mom's own window, isolating HiTec's alpha growth from the sample-window confound - window-matching turns FF3's alpha significant (t=1.55 -> t=2.17), so part of Day 3's finding was a window artifact, but FF5+Mom's alpha (t=4.32) still exceeds window-matched FF3 on the identical 757 months, so the rest is a genuine factor effect. Recorded as both, not smoothed either way. 51/51 tests pass (pytest, 7 new); regress.py --diagnostics, decay_report.py (default and --since), and rolling_report.py all run by hand against the full fixture set. | Day 5 - run it on Stock Stalker's screen output: does that edge survive factor adjustment? |
| 2026-09-19 | `68e65fd` | Day 3: FF3 and FF5+momentum regressions (factors/multifactor.py generalizes capm.py's single-factor OLS to an arbitrary factor set), plus the two new Ken French fixtures this needs (F-F_Research_Data_5_Factors_2x3.csv, F-F_Momentum_Factor.csv, fetched fresh - free, no login) and their loaders/fetch scripts. Correctness gate: no internet access here to a journal's published FF3 table, so 'replicates a known result' is instead an exact analytical identity (each Fama-French factor regressed on its own factor set returns alpha=0, own loading=1, all others=0, R2=1) verified for SMB/HML (FF3) and RMW/Mom (FF5+Mom). factors/decay_report.py runs CAPM/FF3/FF5+Mom across all 10 industries (outputs/alpha_decay.csv) and finds the alpha decay is NOT monotonic as the plan expected - HiTec's alpha grows from +0.57% (CAPM) to +5.44% and turns significant (FF5+Mom) instead of shrinking. Root cause is an honest confound, not a finding about the industry: FF5 data only starts 1963-07, so FF5+Mom runs on 757 months vs CAPM/FF3's 1,201 - different sample windows, not just more factors. Recorded as-is in the README rather than reframed to fit the expected story. 44/44 tests pass (pytest), both regress.py --model ff3/ff5mom and decay_report.py run by hand against all 10 industries. Found and fixed both this repo's and the hub's local main branches stuck in the same recurring stale-detached-HEAD issue before committing. | Day 4 - Newey-West standard errors, residual autocorrelation, heteroskedasticity tests, rolling betas; also re-run FF3 restricted to the 1963-2026 window to isolate factor effect from Day 3's sample-window confound |
| 2026-09-19 | `71a2758` | Day 2: CAPM baseline regression (factors/capm.py) - OLS of portfolio excess return on Mkt-RF, alpha/beta/t-stats/R-squared. Test portfolios are the Ken French 10 Industry Portfolios (new fixture, factors/industry.py, fetched via scripts/fetch_industry_portfolios.py), standing in for the yfinance portfolios sketched in the plan at zero cost. Regression machinery validated first against an exact known answer (market-on-itself: alpha=0, beta=1, R2=1 to 1e-9), then run on all 10 industries via 'python -m factors.regress --portfolio <name> --model capm'; results land where intuition says they should (NoDur/Utils low beta, Durbl/HiTec high beta). Two industries show a plain-OLS-significant alpha but that t-stat is not yet Newey-West-corrected (Day 4), so README records it as provisional. 21/21 tests pass (pytest), CLI run by hand against all 10 industries plus the bad-argument path. Found and fixed both this repo's and the hub's local main branches stuck in the same recurring stale-detached-HEAD issue before committing. | Day 3 - FF3, then FF5 plus momentum; report how alpha decays as factors are added |
| 2026-09-18 | `49966a5` | Day 1: Ken French loader (factors/kenfrench.py) parses the Data Library's real monthly+annual CSV off a committed fixture (fixtures/ken_french/, refreshed via scripts/fetch_ken_french.py). Cross-checking the two sections found a genuine quirk, not a bug: RF compounds monthly to annual within 0.03pp, but SMB/HML don't (up to 21pp off in 2020) since they're annually-reconstituted long/short portfolios, not one held-all-year position -- recorded in the README limitations. 9/9 tests pass (pytest), fetch script run by hand and reproduces the committed fixture byte-for-byte. | Day 2 - CAPM baseline regression: alpha, t-statistic, R-squared |
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- Ken French factors are constructed on US data. Applying them to NSE names is an approximation that needs stating every time.
- Overlapping windows inflate t-statistics. **`factors/capm.py` and `factors/multifactor.py` still report plain-OLS t-stats by design** - Day 4 added the Newey-West correction as a separate, additive module (`factors/diagnostics.py`, surfaced via `regress.py --diagnostics` and `decay_report.py`'s extra columns) rather than changing what those two dataclasses return, so any code that reads `CAPMResult.significant` or `FactorResult.significant` directly still gets the non-robust flag; only the diagnostics path gives the corrected one. One industry/model pair (NoDur, FF3) flips from significant to not once Newey-West is applied; see Findings for the count across all 30 rows.
- **The Ljung-Box and Breusch-Pagan tests use fixed defaults (a 12-month lag, a 5% threshold) that were not tuned per portfolio.** They're reasonable choices for monthly data, not a claim that 12 months is the right horizon for every industry's autocorrelation structure.
- **The Newey-West lag itself follows an automatic rule (Newey & West 1994's `floor(4*(n/100)**(2/9))`), not a cross-validated or per-series choice.** Standard practice, but still a formula substituting for judgment about how much serial dependence actually needs correcting.
- **Rolling betas/loadings use one fixed 60-month window by default** (`factors/rolling_report.py --window` / `factors/rolling_chart.py --window` to change it) - no sensitivity check across window lengths has been run.
- **Day 6's rolling confidence bands are plain-OLS (Wald), not Newey-West.** `statsmodels.regression.rolling.RollingOLS.fit` only accepts `cov_type` of `'nonrobust'`, `'HC0'`, or `'HCCM'` - it raises `ValueError` on `'HAC'` - so there is no drop-in fix analogous to Day 4's `diagnostics.py` for the rolling case. Overlapping monthly windows are serially correlated, which the full-sample Newey-West caveat already flags; that same inflation risk applies to every rolling band `rolling_chart.py` draws, uncorrected. The bands are still useful for telling "point estimate moves a lot but stays inside a wide band" from "point estimate moves outside its own band" (Day 6's HiTec SMB-vs-HML finding is exactly that contrast), just not for a precise coverage-rate claim.
- `factors/rolling_chart.py` charts one industry and one model (FF3 or FF5+Mom) per invocation - no batch mode across all 10 industries the way `rolling_report.py`/`decay_report.py` have.
- The 10 industry portfolios (`factors/industry.py`) are US-constructed test assets, not a Stock Stalker/NSE universe - useful for proving the regression machinery is correct, not for saying anything about NSE names yet.
- **Day 5's screen-check regression has only 22 overlapping months (2024-10 to 2026-07, bounded by Stock Stalker's OHLCV fixture window on one side and Ken French's factor fixture on the other) to estimate up to 7 parameters (FF5+Mom).** Every candidate came back insignificant under every model, but with this few observations "insignificant" mostly means the test lacks the power to detect anything short of an implausibly large alpha - it is not evidence the screen's edge is beta in disguise, just an absence of evidence either way. A real answer needs years more OHLCV history than this sandbox's fixtures carry.
- **Day 5 regresses INR-denominated NSE closing prices directly against USD-denominated Ken French factors, with no currency adjustment.** USD/INR moves are folded into the "raw" NSE return series, mislabeled as US-factor exposure or alpha. Combined with the point above (US factors are already an approximation for Indian names), this makes Day 5's alpha estimates directional at best, not a number to trade on.
- Stock Stalker's screen only ever ranks 3 tickers (its committed OHLCV fixture universe), so Day 5's "does the screen's ranking survive" question is answered on n=3 candidates - too few to say anything about the screening *methodology* in general, only about these three names in this window.
- Testing several specifications on one dataset is multiple testing. The alpha that survives all of them is the only one worth quoting.
- **Day 3's CAPM -> FF3 -> FF5+Mom comparison was confounded by sample window, not just factor count - Day 4's `--since 1963-07` re-run isolates the two, and the answer is "both mattered."** Window-matching FF3 to FF5+Mom's own 757-month range turns HiTec's FF3 alpha significant (t=1.55 -> t=2.17) that wasn't significant in the original mismatched comparison, so part of Day 3's finding was a window artifact. But FF5+Mom's alpha (t=4.32) still exceeds the window-matched FF3 figure on the *identical* 757 months, so the rest of the growth is a genuine RMW/CMA/Mom effect, not more window artifact. Only tested at this single cutoff (FF5+Mom's own start); other windows not explored.
- **Day 3's correctness gate does not replicate an actual published academic figure.** This sandbox has no internet access to a journal's FF3/FF5 replication table, so "replicates a known result" is instead each Fama-French factor's exact analytical self-loading (SMB/HML on FF3, RMW/Mom on FF5+Mom: alpha=0, own loading=1, all others=0, R²=1 to 1e-9) - a real identity check, not an estimate, but not the same evidentiary bar as matching a peer-reviewed number.
- **Compounding monthly SMB/HML to a year does not reproduce French's published annual figure**, and the gap is not small: measured across all 99 complete years in the fixture, the worst case (HML, 2020) is 21 percentage points off. RF (a real return) compounds to within 0.03pp, so this isn't a parser bug - it's the annually-reconstituted long/short portfolios' own arithmetic, most likely reflecting month-to-month changes in the underlying six size/book-to-market portfolios rather than one fixed portfolio held all year. `factors.kenfrench.annual_compounding_gaps` therefore checks RF tightly but only sanity-checks SMB/HML/Mkt-RF loosely (catches a wrong column or a forgotten /100, not fine-grained correctness). Anything downstream that needs annual factor returns should read the published annual section directly, not compound the monthly one.

## Where this sits

Part of a nine-repo research pipeline. Stock Stalker screens the NSE universe; this repo publishes a versioned artifact it reads back:

```json
{
  "factor": {
    "alpha_annual": 0.021,
    "alpha_t": 1.3,
    "significant": false
  }
}
```

Communication is by file contract, not imports, so either side can be refactored without breaking the other.

## Exam mapping

Series XV ch.12.5 (beta), ch.12.9 (risk-adjusted returns), ch.4.5 (quantitative research)

---

CLI only, by design. No dashboard, no server. Charts and documents are written to `outputs/`.
