# Fama-French factor model

Decomposes returns into market, size, value and momentum exposures to test whether an apparent edge is alpha or just beta wearing a disguise.

**Status:** Last checkpoint 2026-09-19 · Next: Day 4 - Newey-West standard errors, residual autocorrelation, heteroskedasticity tests, rolling betas; also re-run FF3 restricted to the 1963-2026 window to isolate factor effect from Day 3's sample-window confound

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

## How to run

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m factors.regress --portfolio hitec --model capm    # or ff3, ff5mom
python -m factors.decay_report                              # all 10 industries x all 3 models
```

`--portfolio` takes one of the 10 Ken French industry portfolios (NoDur,
Durbl, Manuf, Enrgy, HiTec, Telcm, Shops, Hlth, Utils, Other), the CAPM/FF3/
FF5 test assets - not yet a path to an arbitrary returns CSV. `--model` is
`capm` (Mkt-RF), `ff3` (+ SMB, HML), or `ff5mom` (+ RMW, CMA, Mom).
`factors.decay_report` runs all three models against all ten industries and
writes `outputs/alpha_decay.csv`. Runs entirely offline against committed
fixtures; nothing here needs network access or a key.

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

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
| 2026-09-19 | `68e65fd` | Day 3: FF3 and FF5+momentum regressions (factors/multifactor.py generalizes capm.py's single-factor OLS to an arbitrary factor set), plus the two new Ken French fixtures this needs (F-F_Research_Data_5_Factors_2x3.csv, F-F_Momentum_Factor.csv, fetched fresh - free, no login) and their loaders/fetch scripts. Correctness gate: no internet access here to a journal's published FF3 table, so 'replicates a known result' is instead an exact analytical identity (each Fama-French factor regressed on its own factor set returns alpha=0, own loading=1, all others=0, R2=1) verified for SMB/HML (FF3) and RMW/Mom (FF5+Mom). factors/decay_report.py runs CAPM/FF3/FF5+Mom across all 10 industries (outputs/alpha_decay.csv) and finds the alpha decay is NOT monotonic as the plan expected - HiTec's alpha grows from +0.57% (CAPM) to +5.44% and turns significant (FF5+Mom) instead of shrinking. Root cause is an honest confound, not a finding about the industry: FF5 data only starts 1963-07, so FF5+Mom runs on 757 months vs CAPM/FF3's 1,201 - different sample windows, not just more factors. Recorded as-is in the README rather than reframed to fit the expected story. 44/44 tests pass (pytest), both regress.py --model ff3/ff5mom and decay_report.py run by hand against all 10 industries. Found and fixed both this repo's and the hub's local main branches stuck in the same recurring stale-detached-HEAD issue before committing. | Day 4 - Newey-West standard errors, residual autocorrelation, heteroskedasticity tests, rolling betas; also re-run FF3 restricted to the 1963-2026 window to isolate factor effect from Day 3's sample-window confound |
| 2026-09-19 | `71a2758` | Day 2: CAPM baseline regression (factors/capm.py) - OLS of portfolio excess return on Mkt-RF, alpha/beta/t-stats/R-squared. Test portfolios are the Ken French 10 Industry Portfolios (new fixture, factors/industry.py, fetched via scripts/fetch_industry_portfolios.py), standing in for the yfinance portfolios sketched in the plan at zero cost. Regression machinery validated first against an exact known answer (market-on-itself: alpha=0, beta=1, R2=1 to 1e-9), then run on all 10 industries via 'python -m factors.regress --portfolio <name> --model capm'; results land where intuition says they should (NoDur/Utils low beta, Durbl/HiTec high beta). Two industries show a plain-OLS-significant alpha but that t-stat is not yet Newey-West-corrected (Day 4), so README records it as provisional. 21/21 tests pass (pytest), CLI run by hand against all 10 industries plus the bad-argument path. Found and fixed both this repo's and the hub's local main branches stuck in the same recurring stale-detached-HEAD issue before committing. | Day 3 - FF3, then FF5 plus momentum; report how alpha decays as factors are added |
| 2026-09-18 | `49966a5` | Day 1: Ken French loader (factors/kenfrench.py) parses the Data Library's real monthly+annual CSV off a committed fixture (fixtures/ken_french/, refreshed via scripts/fetch_ken_french.py). Cross-checking the two sections found a genuine quirk, not a bug: RF compounds monthly to annual within 0.03pp, but SMB/HML don't (up to 21pp off in 2020) since they're annually-reconstituted long/short portfolios, not one held-all-year position -- recorded in the README limitations. 9/9 tests pass (pytest), fetch script run by hand and reproduces the committed fixture byte-for-byte. | Day 2 - CAPM baseline regression: alpha, t-statistic, R-squared |
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- Ken French factors are constructed on US data. Applying them to NSE names is an approximation that needs stating every time.
- Overlapping windows inflate t-statistics, so Newey-West standard errors are used throughout. **Not true yet for Day 2's CAPM baseline** - `factors/capm.py` currently reports plain OLS t-stats, which is exactly the kind of inflated t-stat this bullet warns about. Two of the ten industries flag as significant under that non-robust test; treat that as provisional until Day 4 adds the Newey-West correction.
- The 10 industry portfolios (`factors/industry.py`) are US-constructed test assets, not a Stock Stalker/NSE universe - useful for proving the regression machinery is correct, not for saying anything about NSE names yet. That comes on Day 5, applied to Stock Stalker's own screen output, with the same US-factors-on-NSE-names caveat repeated there.
- Testing several specifications on one dataset is multiple testing. The alpha that survives all of them is the only one worth quoting.
- **The Day 3 CAPM -> FF3 -> FF5+Mom comparison is confounded by sample window, not just factor count.** FF5 data starts 1963-07 (RMW/CMA need data CRSP/Compustat doesn't have further back), so FF5+Mom regressions run on 757 months (1963-2026) against CAPM/FF3's 1,201 (1926-2026) - a shorter, more tech-heavy, post-1963 America. HiTec's alpha *growing* from +0.57% (CAPM) to +5.44% and turning significant (FF5+Mom) is reported honestly rather than smoothed into the "alpha shrinks as factors are added" story the plan expected; a same-window FF3 re-run restricted to 1963-2026 would isolate factor effect from window effect and hasn't been done yet.
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
