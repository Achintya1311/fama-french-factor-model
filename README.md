# Fama-French factor model

Educational research on historical data. Not investment advice. Not a SEBI-registered research analyst.

**Question:** as factors (market → size/value → momentum) are added one at a time, does an industry portfolio's alpha decay toward zero the way a pure-beta story predicts?
**Method:** CAPM, then FF3, then FF5+momentum on all 10 Ken French industry portfolios, with Newey-West SEs, residual-autocorrelation/heteroskedasticity diagnostics, and a rolling 60-month loading chart with a 95% confidence band.
**Finding (Day 3):** alpha decay is not monotonic — HiTec's alpha grows from +0.57% (CAPM) to +5.44% and turns significant (FF5+Mom), the opposite of the expected shrinkage, traced to an honest confound (FF5+Mom's shorter 757-month sample, not more factors).
**Chart:** `outputs/rolling_loadings_hitec_ff3.png` — HiTec's rolling FF3 loadings with a 95% Wald band, one subplot per factor.
**Reproduce:** `python -m factors.rolling_chart --portfolio hitec`

Decomposes returns into market, size, value and momentum exposures to test whether an apparent edge is alpha or just beta wearing a disguise.

**Status:** Complete (Sep 2026; Indian IIM-A factors added Oct 2026). Part of an eight-repo equity research pipeline (STOCKSTALKER plus seven research repos; the separate stockstalker-lab is not counted); [STOCKSTALKER](https://github.com/Achintya1311/STOCKSTALKER) is the front door.

*Educational research on historical data. Not investment advice. Not a SEBI-registered research analyst.*

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

- IIM Ahmedabad's Indian Fama-French + Momentum library - free, no login, Indian-market factors in INR

`factors/iima.py` loads `fixtures/iima/2025-12_FourFactors_and_Market_Returns_Monthly.csv`
(refresh with `python scripts/fetch_iima.py`) - SMB, HML, WML and MF (market
excess return), plus RF, built by Agarwalla/Jacob/Varma from CMIE Prowess
data on BSE/NSE stocks, 1993-10 on. Added for the stockstalker-front-door
project (Day 4): the Ken French factors above are USD portfolios, the
wrong benchmark for GULFOILLUB's INR prices, so the GULFOILLUB regression
(front-door Day 5, `factors/gulfoillub.py`) runs against this library
instead.

- GULFOILLUB's own OHLCV - a committed snapshot, not a live read of the sibling repo

`factors/gulfoillub.py` reads `fixtures/stockstalker/ohlcv/GULFOILLUB_NS.csv`
(front-door Day 5), copied in unmodified from STOCKSTALKER's own
`fixtures/ohlcv/GULFOILLUB_NS.csv` (2014-07-31 to 2026-10-09), the same
copy-in-as-a-fixture discipline the RELIANCE/TATACHEM/CROMPTON OHLCV above
already follows.

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
python -m factors.pipeline_audit                              # Day 7: look-ahead + multiple-testing audit
python -m factors.pipeline_audit --alpha 0.10                # same, at a looser significance level
python -m factors.gulfoillub --model ff3wml --diagnostics    # front-door Day 5: GULFOILLUB vs IIMA factors
python -m factors.gulfoillub --model ff3wml --contract fixtures/factor.json  # + writes the v0.2 contract
python -m factors.stock --ticker ANY --prices <ohlcv.csv> --model ff3wml --contract out.json  # Day 22: same regression, any ticker's own price CSV
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
`factors.pipeline_audit` (Day 7) checks two structural properties
mechanically rather than by inspection - that every rolling estimate is
trailing-only (truncating the series after month *i* never changes the
value reported for month *i*) and that every factor-alignment join is by
month label, not row order - then Bonferroni/BH-corrects every t-stat
`decay_report`/`screen_check` have reported so far and prints how many
survive. Exits 1 only on a look-ahead/alignment violation; a multiple-
testing result where nothing survives correction is an expected finding,
not a pipeline failure.
`factors.gulfoillub` (front-door Day 5) regresses GULFOILLUB's own monthly
returns against IIMA's factors instead of Ken French's - `--model`
(capm/ff3/ff3wml, default ff3wml) picks IIMA's MF alone, +SMB/HML, or
+WML; there is no ff5mom choice here, since IIMA has no RMW/CMA.
`--diagnostics` adds the same Newey-West/Ljung-Box/Breusch-Pagan block.
`--contract PATH` writes the v0.2 spine contract and requires
`--model ff3wml` - the only spec this module ships to STOCKSTALKER.

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

**stockstalker-front-door Day 4 (2026-10-12) - an Indian factor library, because GULFOILLUB needs one.** Every regression above runs USD-denominated Ken French factors against NSE-listed test assets - a mismatch the hub's `knowaboutit.md` flagged from the start but Days 2-7 never had to resolve, since the industry portfolios and even Stock Stalker's own screen check were about *whether the pipeline runs end to end*, not about getting the benchmark currency right. The front-door project's actual subject, GULFOILLUB, is the first case where that mismatch would land in a number someone might act on, so this day adds `factors/iima.py`, a loader for IIM Ahmedabad's free Indian Fama-French + Momentum library (SMB, HML, WML, MF, RF; CMIE Prowess data on BSE/NSE stocks, 1993-10 to 2025-12; already committed as `fixtures/iima/2025-12_FourFactors_and_Market_Returns_Monthly.csv` by an earlier session). Percent columns convert to decimal and month strings (`YYYY-MM`) convert to a `pandas.PeriodIndex`, the same convention `kenfrench.load_monthly_factors` uses, so the GULFOILLUB regression (front-door Day 5) can swap this loader in for that one without touching the regression code itself. NEXT_STEPS.md's plan described dropping "the NA rows before MF starts," expecting a leading block; the committed file does not match that shape exactly - 1993-10 (the first row) is missing MF/RF as expected, but 1994-11, well into the series, is separately missing SMB/HML/WML. Both rows are dropped (`dropna` across all five columns, not a positional slice), and the README here says so rather than leaving the original, slightly-off description uncorrected. MF is documented by IIMA as already a market *excess* return (built the same way Ken French's Mkt-RF is, under a different name), so `iima.py` does not subtract RF from it a second time. 8 new tests pass (97/97 total), including one pinning the first kept row (1993-11) and the last row (2025-12) against the committed CSV's own published values, and one asserting exactly 385 of the file's 387 data rows survive the NA drop.

**stockstalker-front-door Day 5 (2026-10-12) - GULFOILLUB's own factor regression, on 137 real months instead of RELIANCE's underpowered 22.** Day 4 loaded IIMA's Indian factors for exactly this: GULFOILLUB's monthly returns (`fixtures/stockstalker/ohlcv/GULFOILLUB_NS.csv`, copied in unmodified from STOCKSTALKER's own fixture, 2014-07-31 to 2026-10-09) inner-joined against IIMA's factor history (which stops at 2025-12) give 2014-08 to 2025-12, 137 months - six times the RELIANCE screen check's 22, and long enough that "underpowered" is no longer the limiter it was there. `factors/gulfoillub.py` is the new module: three specs sharing one generic OLS path (`factors.multifactor.run_factor_model`, not `factors.capm.run_capm`, which hardcodes Ken French's `mkt_rf` column name and would not find IIMA's `mf`) - CAPM (MF alone), FF3 (MF/SMB/HML), and FF3+WML (+ momentum). IIMA ships no RMW/CMA, so there is no FF5+Mom equivalent here, unlike the Ken French side of this repo. The honest result: alpha is **not** significant under any spec, plain-OLS or Newey-West, despite the much larger sample - FF3+WML's annualized alpha is +10.64% but t=+1.01 (OLS) / t=+1.20 (NW, lag=4); CAPM's is +8.37% (t=+0.86), FF3's +10.87% (t=+1.12). This is a genuinely different finding from the RELIANCE screen check's "underpowered, can't say" - here there is enough data to say, and the honest answer is that GULFOILLUB's real excess return over this window does not show a statistically distinguishable alpha once market exposure is priced in. The market loading is large and significant (`mf`=+0.797, t=+3.98), `smb`=+0.470 (t=+2.38, a real small-cap tilt) and `hml`=-0.237 (t=-1.16, not significant); `wml`=+0.011 (t=+0.06) is indistinguishable from zero - no detectable momentum tilt, the first time this repo's momentum loading has come back this flat for a real name. Diagnostics find neither residual autocorrelation (Ljung-Box p=0.344) nor heteroskedasticity (Breusch-Pagan p=0.665) on this spec, unlike several of the Ken French industry/model pairs Day 4 flagged. `to_contract()` writes `STOCKSTALKER/fixtures/factor/GULFOILLUB.json` in the existing v0.2 shape with the market loading renamed `mf` -> `mkt` (matching every other satellite's convention) and the extra `wml` key passed through unchanged - `gates.factor.load_contract` was read first and confirmed to check only the four top-level keys, not which keys `loadings` itself carries, so no gate change was needed. 13 new tests pass (110/110 total), including two exact-identity checks in the Day 2/3 style (regressing a synthetic market-plus-noise-free-factor combination recovers alpha=0 and the construction's own loadings to 1e-9) and a restriction test confirming `--contract` refuses CAPM/FF3 (only FF3+WML, the model actually shipped, is allowed to write the spine file); both `python -m factors.gulfoillub --model ff3wml --diagnostics` and `--contract` were run by hand against the real fixtures.

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

**Day 7 - pipeline audit: look-ahead is clean, multiple testing is not.**
`factors/pipeline_audit.py` runs two structural checks. Look-ahead:
truncate the aligned series after month *i* and confirm `rolling_beta`/
`rolling_loadings` still report the same value for month *i* - checked on
HiTec's rolling CAPM/FF3/FF5+Mom loadings, clean (0 violations). Alignment:
shuffle the factor table's row order and confirm `align_excess_returns`
still returns the identical joined frame - checked on both `capm.py`'s and
`multifactor.py`'s join, against both the Ken French industry data and
Stock Stalker's screen candidates, also clean. Both checks were verified
against a manufactured negative control before trusting the clean result on
real data (a synthetic rolling series shifted 6 months into the future, and
a synthetic positional join) - the audit does detect a real violation when
one exists (see `tests/test_pipeline_audit.py`), it isn't just passing
because nothing is checked.

Multiple testing is the actual finding. Applying Bonferroni and
Benjamini-Hochberg correction to every t-stat `decay_report`/`screen_check`
have reported so far: `decay_report`'s 30 industry/model rows go from 8
raw-significant (OLS, p<0.05) to **2 surviving Bonferroni** (HiTec/FF5+Mom,
Other/FF3) and 4 surviving Benjamini-Hochberg; the Newey-West reading moves
7 raw-significant to the same 2 Bonferroni survivors and 5 BH survivors.
`screen_check`'s 9 ticker/model rows go from 0 (OLS) or 1 (Newey-West,
TATACHEM.NS/capm) raw-significant to 0 surviving either correction - the
one Newey-West "significant" reading among Stock Stalker's three candidates
does not survive being one test among nine. HiTec/FF5+Mom - the industry
Day 3's checkpoint already flagged as the one case where alpha *grew* with
more factors instead of shrinking - is the only industry/model pair that
survives the strictest correction across every reading. That doesn't make
it real; with only two corrections and one dataset, it just means it's the
single candidate this repo's own multiple-testing check cannot rule out.

**Day 83 (R17) - re-asking the alpha question with power: a cross-section,
not one stock.** Day 5's GULFOILLUB alpha (+10.64%/yr, t=+1.01 OLS) is one
stock on 137 months - long enough to beat the 22-month RELIANCE screen
check's own "underpowered" caveat, but still a single time series, which
can only ever answer "is this one asset's alpha distinguishable from
zero," never "is there alpha in the NSE F&O cross-section." Pre-registered
first (`docs/PREREG_panel_alpha.md`, committed before any of this was
computed): FF3+WML on the 213 stock-only NSE F&O underlyings
(`factors/universe.py`, from STOCKSTALKER's committed `fo_mktlots.csv`
snapshot), prices fetched via a GitHub Actions workflow
(`.github/workflows/fetch-panel-prices.yml`, same pattern as
`markowitz-portfolio-optimizer`'s `fetch-largecaps.yml`) rather than this
sandbox's own blocked IP, with a 60-overlapping-month-per-stock floor.

189 of 213 stocks clear that floor (24 don't: 17 with fewer than 60
overlapping months, 7 below `multifactor.run_factor_model`'s own D15
anti-degenerate-fit floor entirely). The pre-registered **Benjamini-Hochberg
test, at the 10% FDR level, is the headline result: 0 of 189 stocks'
alphas survive FDR correction**, against 24/189 (12.7%) raw-"significant"
at the uncorrected 5% level - close to what a 5%-per-look false-positive
rate run across 189 independent-ish tests would produce by chance alone,
not evidence of a real cross-sectional alpha. The factors explain a real
but partial share of return variance (R-squared mean 0.331, median 0.326,
range 0.014-0.582) - roughly two-thirds of the typical panel stock's
monthly variance is stock-specific, not factor exposure. The pre-registered
power statement: the panel's own fitted standard errors imply a median
minimum detectable alpha, at 80% power, of **30.6%/yr annualized** (mean
36.1%, as low as 15.3% for the longest, least volatile histories and as
high as 181.6% for the shortest) - in the same range this README's own Day
5 finding already estimated for GULFOILLUB alone, now confirmed across the
whole panel rather than guessed from one stock's residual volatility.

**The pre-registered GRS joint test could not be computed, and that is
itself the finding, not a gap this day patches by dropping stocks after
seeing the result.** GRS needs every included asset's residuals measured
over one shared window (a common T), so the test is restricted to the
subset of the 189-stock panel with full coverage over a common window;
that window's start is forced to the *latest* of every included stock's
own first month, and with 213 F&O underlyings spanning listing dates from
2015 to well into the 2020s, the shared window shrinks to just 62 months
(2020-11 to 2025-12) across N=188 stocks - `Sigma`, GRS's N x N residual
covariance matrix, needs `T > N+K` (here `192`) to be invertible at all,
and `62 < 192` by a wide margin. This is a real, structural limit of
applying a classic GRS test to a *wide* (213-name), *heterogeneous-listing-date*
cross-section, not a bug: the same breadth that makes the Benjamini-Hochberg
test well-powered (189 roughly-independent stocks) is exactly what makes
GRS's shared-window requirement bite. `factors/panel.py`'s
`grs_common_window_panel` reports this honestly (`infeasible_reason`,
surfaced on the CLI) rather than silently narrowing the panel - the
pre-registration explicitly ruled out that patch.

Reproduce: `python -m factors.panel` (reads the committed
`fixtures/panel_ohlcv/` fixtures `fetch-panel-prices.yml` wrote). GULFOILLUB's
own single-stock regression (Day 5, `factors.gulfoillub`) is kept exactly as
it was - the worked example this cross-section test was pre-registered to
generalize, not a result this day replaces.

## Limitations and what would make me wrong

- Ken French factors are constructed on US data. Applying them to NSE names is an approximation that needs stating every time.
- Overlapping windows inflate t-statistics. **`factors/capm.py` and `factors/multifactor.py` still report plain-OLS t-stats by design, for both alpha and every loading** - Day 4 added the Newey-West correction as a separate, additive module (`factors/diagnostics.py`, surfaced via `regress.py --diagnostics` and `decay_report.py`'s extra columns) rather than changing what those two dataclasses return, so any code that reads `CAPMResult.significant`, `FactorResult.significant`, or any `*.loadings_t` value directly still gets the non-robust reading; only the diagnostics path (now printing `diag.loadings_t_nw` alongside `diag.alpha_t_nw` in every `--diagnostics` block, R11) gives the corrected one. One industry/model pair (NoDur, FF3) flips from significant to not once Newey-West is applied to alpha; see Findings for the count across all 30 rows. Re-checking GULFOILLUB's three headline loading claims (Day 5) under Newey-West moves the t-stats (`smb` 2.38->2.32, `hml` -1.16->-1.34, `wml` 0.06->0.05) but flips none of them - the reporting gap was real even though no currently-reported finding turned out wrong.
- **The Ljung-Box and Breusch-Pagan tests use fixed defaults (a 12-month lag, a 5% threshold) that were not tuned per portfolio.** They're reasonable choices for monthly data, not a claim that 12 months is the right horizon for every industry's autocorrelation structure.
- **The Newey-West lag itself follows an automatic rule (Newey & West 1994's `floor(4*(n/100)**(2/9))`), not a cross-validated or per-series choice.** Standard practice, but still a formula substituting for judgment about how much serial dependence actually needs correcting.
- **Rolling betas/loadings use one fixed 60-month window by default** (`factors/rolling_report.py --window` / `factors/rolling_chart.py --window` to change it) - no sensitivity check across window lengths has been run. Day 21 (D21) added a floor: `--window` below `3x` the model's own parameter count (incl. intercept) now raises a clear error instead of running - a window of 7 against FF5+Mom's 7 parameters used to run to completion with loadings swinging into the thousands and `inf` internal standard errors from `RollingOLS`, the rolling-window equivalent of the fabricated one-shot fit Day 15 already guards against. The default (60 months) is unaffected for any model offered here.
- **Day 6's rolling confidence bands are plain-OLS (Wald), not Newey-West.** `statsmodels.regression.rolling.RollingOLS.fit` only accepts `cov_type` of `'nonrobust'`, `'HC0'`, or `'HCCM'` - it raises `ValueError` on `'HAC'` - so there is no drop-in fix analogous to Day 4's `diagnostics.py` for the rolling case. Overlapping monthly windows are serially correlated, which the full-sample Newey-West caveat already flags; that same inflation risk applies to every rolling band `rolling_chart.py` draws, uncorrected. The bands are still useful for telling "point estimate moves a lot but stays inside a wide band" from "point estimate moves outside its own band" (Day 6's HiTec SMB-vs-HML finding is exactly that contrast), just not for a precise coverage-rate claim.
- `factors/rolling_chart.py` charts one industry and one model (FF3 or FF5+Mom) per invocation - no batch mode across all 10 industries the way `rolling_report.py`/`decay_report.py` have.
- The 10 industry portfolios (`factors/industry.py`) are US-constructed test assets, not a Stock Stalker/NSE universe - useful for proving the regression machinery is correct, not for saying anything about NSE names yet.
- **Day 5's screen-check regression has only 22 overlapping months (2024-10 to 2026-07, bounded by Stock Stalker's OHLCV fixture window on one side and Ken French's factor fixture on the other) to estimate up to 7 parameters (FF5+Mom).** Every candidate came back insignificant under every model, but with this few observations "insignificant" mostly means the test lacks the power to detect anything short of an implausibly large alpha - it is not evidence the screen's edge is beta in disguise, just an absence of evidence either way. A real answer needs years more OHLCV history than this sandbox's fixtures carry.
- **Day 5 regresses INR-denominated NSE closing prices directly against USD-denominated Ken French factors, with no currency adjustment.** USD/INR moves are folded into the "raw" NSE return series, mislabeled as US-factor exposure or alpha. Combined with the point above (US factors are already an approximation for Indian names), this makes Day 5's alpha estimates directional at best, not a number to trade on.
- Stock Stalker's screen only ever ranks 3 tickers (its committed OHLCV fixture universe), so Day 5's "does the screen's ranking survive" question is answered on n=3 candidates - too few to say anything about the screening *methodology* in general, only about these three names in this window.
- **Testing several specifications on one dataset is multiple testing - Day 7's `pipeline_audit.py` turns that into a number instead of a caveat.** Bonferroni/BH-correcting `decay_report`'s 30 industry/model rows shrinks 8 raw-significant (OLS, 5%) down to 2 surviving Bonferroni and 4 surviving Benjamini-Hochberg; `screen_check`'s 9 rows go from 1 raw-significant (Newey-West) to 0 surviving either correction. Read every "significant" alpha in this README's earlier Findings against that: most of them are one test among many and do not survive being treated that way.
- **`pipeline_audit.py`'s p-values use the normal approximation (`2*(1-Phi(|t|))`), not the exact t-distribution**, to stay consistent with every `significant`/`significant_nw` flag already computed elsewhere in this repo - all of them use the fixed `|t| > 1.96` cutoff regardless of degrees of freedom. This makes the audit's "raw significant" count agree exactly with what `decay_report`/`screen_check` already print, at the cost of being slightly liberal for the shorter-sample rows (screen_check's n≈22) where the exact t-distribution has fatter tails than the normal.
- **The look-ahead and alignment-order checks in `pipeline_audit.py` are structural, not statistical** - like the STOCKSTALKER/monte-carlo-risk-lab audits this one follows the same pattern of, a clean result proves no *look-ahead bug* was found in the checked functions, not that the regressions' findings are correct. Both checks were run against a manufactured negative control (a synthetic rolling series shifted 6 months into the future, a synthetic positional join) before being trusted on real data, so a clean result here is not merely "the check never fires."
- **Day 3's CAPM -> FF3 -> FF5+Mom comparison was confounded by sample window, not just factor count - Day 4's `--since 1963-07` re-run isolates the two, and the answer is "both mattered."** Window-matching FF3 to FF5+Mom's own 757-month range turns HiTec's FF3 alpha significant (t=1.55 -> t=2.17) that wasn't significant in the original mismatched comparison, so part of Day 3's finding was a window artifact. But FF5+Mom's alpha (t=4.32) still exceeds the window-matched FF3 figure on the *identical* 757 months, so the rest of the growth is a genuine RMW/CMA/Mom effect, not more window artifact. Only tested at this single cutoff (FF5+Mom's own start); other windows not explored.
- **Day 3's correctness gate does not replicate an actual published academic figure.** This sandbox has no internet access to a journal's FF3/FF5 replication table, so "replicates a known result" is instead each Fama-French factor's exact analytical self-loading (SMB/HML on FF3, RMW/Mom on FF5+Mom: alpha=0, own loading=1, all others=0, R²=1 to 1e-9) - a real identity check, not an estimate, but not the same evidentiary bar as matching a peer-reviewed number.
- **Compounding monthly SMB/HML to a year does not reproduce French's published annual figure**, and the gap is not small: measured across all 99 complete years in the fixture, the worst case (HML, 2020) is 21 percentage points off. RF (a real return) compounds to within 0.03pp, so this isn't a parser bug - it's the annually-reconstituted long/short portfolios' own arithmetic, most likely reflecting month-to-month changes in the underlying six size/book-to-market portfolios rather than one fixed portfolio held all year. `factors.kenfrench.annual_compounding_gaps` therefore checks RF tightly but only sanity-checks SMB/HML/Mkt-RF loosely (catches a wrong column or a forgotten /100, not fine-grained correctness). Anything downstream that needs annual factor returns should read the published annual section directly, not compound the monthly one.
- **`factors/iima.py` drops every row with any `NA` in it (2 of 387), not just a leading block.** NEXT_STEPS.md's plan described the gap as "the NA rows before MF starts," which held for 1993-10 but not for 1994-11, a separate mid-series gap in SMB/HML/WML alone. The loader's behavior (`dropna` across all five columns) is correct either way, but the original plan's description of *why* a row might be missing was incomplete, and this README corrects it rather than quietly matching the code to the wrong explanation.
- **IIMA's own documentation states MF is already a market excess return, not a raw market return** - this repo takes that at face value and does not independently verify it against a separate raw-market/RF reconstruction, the same trust-the-source-documentation posture `kenfrench.py` takes for Mkt-RF. If that documentation is wrong, every regression that uses `mf` instead of subtracting `rf` from a raw return would be mis-specified in a way this repo's own tests cannot catch, since they only check the loader against the file as published.
- **No cross-check between the IIMA and Ken French fixtures exists yet** (unlike `kenfrench.annual_compounding_gaps`, which cross-checks that library's own monthly/annual sections against each other) - `iima.py` has no annual section to check against, and nothing here compares IIMA's Indian market factor to Ken French's US one on the same months to sanity-check the two libraries' broad direction of agreement/disagreement.
- **This repo does not independently verify whether IIM Ahmedabad's Indian factor library (`factors/iima.py`, built from CMIE Prowess data) is survivorship-bias-free or point-in-time, the way Ken French's CRSP-based industry portfolios are documented to be** - the same take-the-source-documentation-at-face-value posture as the MF-is-already-excess-return bullet above. GULFOILLUB vs IIMA (Day 5) is this repo's only real, non-test-asset, INR-denominated finding, so its alpha and loadings inherit whatever survivorship or look-ahead bias, if any, IIMA's own construction carries; this repo's tests only check the loader against the file as published and cannot catch a bias in the source data itself.
- **IIMA has no RMW/CMA, so `factors/gulfoillub.py` has no FF5+Mom spec** - the richest model it can run is FF3+WML. GULFOILLUB's profitability/investment exposure is simply not measured here, unlike RELIANCE's Ken French-based screen check.
- **GULFOILLUB's FF3+WML regression finds no significant alpha (t=+1.01 OLS, +1.20 NW) on 137 real months, not 22 underpowered ones - but "not significant" is still not the same claim as "no edge exists."** A larger sample narrows the confidence interval around zero; it does not prove the true alpha is zero, only that this window's data cannot distinguish the point estimate (+10.64% annualized) from zero at the 5% level. `wml`'s loading (+0.011, t=+0.06) is close enough to exactly zero that "no detectable momentum tilt" is a safe reading; the alpha finding is weaker than that.
- **`factors/gulfoillub.py`'s `--contract` always writes the FF3+WML spec, never CAPM or FF3** - a deliberate restriction (the gate reads one fixture per ticker, and FF3+WML is the richest available), but it means a reader of the shipped contract never sees the CAPM/FF3 comparison this README's Findings section records; they would need to come back here or re-run the CLI.
- **This day changed only `fama-french-factor-model`.** `STOCKSTALKER/fixtures/factor/GULFOILLUB.json` was written by this repo's own `--contract` flag and copied over as a plain data file; no code in STOCKSTALKER changed, only its test suite (two tests updated to expect a real GULFOILLUB factor section instead of "not available", following the same pattern Day 3's risk contract used).
- **D15 fix (18 Oct 2026): `run_factor_model`/`run_capm` now refuse to fit below a 3x-the-parameter-count minimum sample.** Before this fix, a thin or newly-listed ticker's inner join against IIMA/Ken French factors could land at 4-6 months, which a 4- or 5-parameter OLS fits exactly or over-exactly - a rank-deficient or exactly-determined design matrix that silently returns a "result" (R-squared=1.0, t-stats collapsed to 0.0/0.0) rather than erroring, the one case this repo's "never a fabricated number" discipline had no guard for. Both functions now raise `ValueError` naming `n_obs` and the factor count when `n_obs < 3 * (len(factor_columns) + 1)`; `gulfoillub.py`'s and `stock.py`'s CLIs catch it and exit with that one-line reason instead of a traceback or a written `--contract`. The floor is deliberately loose enough to leave every case this repo already treats as merely "underpowered" untouched - Day 5-7's 22-month Ken French screen check passes FF5+Mom's 7-parameter floor (21 months) by a 1-month margin, and GULFOILLUB's 137 months are nowhere near either floor. The one real-world case this did change: STOCKSTALKER's TATAELXSI smoke test (14 months against FF3+WML's 15-month floor) now correctly reports "insufficient data" instead of the `alpha_t=-0.98` it previously fabricated from a near-rank-deficient fit - a stricter, more honest answer, not a regression.
- **Day 83's panel (R17) replays *today's* 213-name F&O lot-size list back to each stock's own full price history, with no point-in-time F&O-membership history available in this sandbox** - the same survivorship caveat D23 already states for `markowitz-portfolio-optimizer`'s large-cap universe. A stock that lost (or gained) F&O eligibility partway through its history is still included (or excluded) as if today's membership always held; the panel's Benjamini-Hochberg and R-squared findings inherit whatever bias, if any, that implies.
- **The GRS common-window subset (Day 83, R17) is not the same 189-stock panel the Benjamini-Hochberg test and R-squared figures use** - it is a 188-stock subset restricted to one shared 62-month window (2020-11 to 2025-12), and even that subset's GRS statistic could not be computed (`T=62 < N+K=192`; see Findings). A reader comparing "0/189 survive BH" against "GRS not computable" should not read the second as a weaker version of the first test on the same data - they are different sample windows answering structurally different questions (per-stock FDR control vs. one joint test requiring a shared T).

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
| 2026-10-18 | (pending) | D15 fix (NEXT_STEPS.md day 39): `factors/multifactor.run_factor_model` and `factors/capm.run_capm` now raise a clear `ValueError` naming `n_obs` and the factor count when `n_obs < 3 * (parameter count incl. intercept)` - the floor that stops a thin/newly-listed ticker's regression from silently returning a fabricated "result" (R-squared=1.0, t-stats collapsed to 0.0/0.0) off a rank-deficient or exactly-determined design matrix (reproduced first at n_obs=4/5/6 against the real IIMA fixture, per the critique). `gulfoillub.py`'s and `stock.py`'s CLIs catch it and exit with that one-line reason instead of a traceback, never reaching `--contract`. Failing tests added first (12 new: the degenerate n=4/5/6 cases, the exact floor boundary for both functions, and a regression check that the 22-month Ken French screen check - FF5+Mom's worst case, passing by a 1-month margin - and GULFOILLUB's 137 months are unaffected), then the fix; 134/134 tests pass. STOCKSTALKER's real sibling-checkout smoke test for TATAELXSI (14 months, below FF3+WML's 15-month floor) now correctly asserts "insufficient data" instead of the alpha it previously fabricated from a near-rank-deficient fit; that test file was updated to match (coordinated in the same run, see STOCKSTALKER's own log). | next critique run picks the next project (no day currently queued in NEXT_STEPS.md) |
| 2026-10-12 | (pending) | stockstalker-front-door Day 4: factors/iima.py loads IIM Ahmedabad's committed Indian four-factor CSV (SMB/HML/WML/MF/RF, percent -> decimal, month strings -> PeriodIndex), the free Indian benchmark GULFOILLUB's regression needs instead of USD Ken French factors. NEXT_STEPS.md's "drop NA rows before MF starts" undersold the file: 1993-10 is missing MF/RF as expected, but 1994-11 separately drops SMB/HML/WML - both rows (2 of 387) are dropped via dropna across all five columns, and the README corrects the original plan's description rather than leaving it. MF is not re-reduced by RF, since IIMA documents it as already an excess return. 8 new tests pass (97/97 total), including known-value pins on the first kept row (1993-11) and the last row (2025-12). No STOCKSTALKER changes this day - the GULFOILLUB regression and its factor contract land on Day 5. | front-door Day 5 - GULFOILLUB factor regression on IIMA factors (CAPM/FF3/FF3+WML, Newey-West), write STOCKSTALKER/fixtures/factor/GULFOILLUB.json |
| 2026-09-22 | `8f2114c` | Integration day: shipped Stock Stalker v0.2 (factor-adjusted ranking) using this repo's factor contract. factors/screen_check.py gained to_contract() plus --contract/--contract-ticker CLI flags, writing the v0.2 JSON block ({"factor": {"alpha_annual", "alpha_t", "significant", "loadings"}}, FF5+Mom's 6-factor loadings with mkt_rf renamed mkt) as a file, never a Python import -- only ff3/ff5mom can fill it, CAPM has no loadings dict so --contract with --model capm is rejected. On the spine side, STOCKSTALKER's new stockstalker.gates.factor reads that file and attaches it to the one screen candidate whose ticker matches; report.py's new --factor-contract/--factor-ticker flags apply the gate and add three columns (Alpha (ann.), Mkt loading, Sig.(5%)) to the screen table when populated. Unlike v0.7's GULFOILLUB non-match, this is a genuine one like v0.3's: fama-french-factor-model committed a snapshot of Stock Stalker's own screen and all three OHLCV fixtures, so RELIANCE.NS is a real shared ticker, not a synthetic stand-in. The honest result: all three candidates come back factor-insignificant under every model (CAPM/FF3/FF5+Mom, plain-OLS and Newey-West), but Day 5-7's own findings already flagged only 22 overlapping months between the two fixtures' windows as underpowered for up to 7 parameters -- read as 'underpowered,' not as proof the technical edge is beta in disguise, and recorded as such in both repos' READMEs. Verified honestly: 89/89 fama-french-factor-model tests pass (8 new), 186/186 STOCKSTALKER tests pass (0 skipped), and both CLIs were run by hand end to end including the real RELIANCE match. STOCKSTALKER bumped 0.7.1 -> 0.7.2, not 0.2.0 -- shipping landed out of pipeline order (behind v0.3 and v0.7, both still present), the same version-can't-move-backward deviation v0.3's own changelog already recorded. Found and fixed all three repos' (STOCKSTALKER, fama-french-factor-model, and the hub itself) local main branches stuck in the same recurring stale-detached-HEAD issue before committing -- fetching origin/main first confirmed none of the prior pushes had actually failed, the tracking refs were just stale locally. | markowitz-portfolio-optimizer Day 1 - returns and covariance estimation with estimation-error note |
| 2026-09-21 | `4ca30c8` | Day 7: ml-pipeline-audit pass - factors/pipeline_audit.py mechanically checks look-ahead in factor alignment (truncating the series after month i must not change a rolling_beta/rolling_loadings value already reported for month i; shuffling the factor table's row order must not change align_excess_returns's joined result) and applies Bonferroni/Benjamini-Hochberg correction to every t-stat decay_report/screen_check have reported so far. Both structural checks come back clean on real data, verified against manufactured negative controls (a rolling series shifted 6 months into the future, a positional instead of label-based join) so a clean result means something. Multiple testing is the real finding, turning an existing README caveat into a number: decay_report's 30 industry/model rows go from 8 raw-significant (OLS, 5%) to 2 surviving Bonferroni and 4 surviving Benjamini-Hochberg; screen_check's 9 rows go from 1 raw-significant (Newey-West) to 0 surviving either correction. factors/multiple_testing.py's Benjamini-Hochberg implementation is checked against the worked example from Benjamini & Hochberg's own 1995 paper, not just re-deriving the same formula the code uses. 97/97 tests pass (pytest, 16 new); pipeline_audit run by hand at the default and a loosened --alpha 0.10. fama-french-factor-model is now feature-complete at 7/7 days. | Integration day: ship Stock Stalker v0.2 (factor-adjusted ranking) using this repo's factor contract on the screen candidates |
| 2026-09-21 | `3bf7423` | Day 6: rolling factor-loading charts with confidence bands. factors/diagnostics.py gains rolling_loadings (generalizes Day 4's single-factor rolling_beta to an arbitrary factor set, FF3/FF5+Mom) and confidence_band (95% Wald band); factors/rolling_chart.py is the new CLI, writing one PNG (subplot per factor) plus a loadings+SE CSV per industry/model to outputs/. RollingOLS has no HAC option (statsmodels rejects cov_type='HAC'), so the bands are plain-OLS, not Newey-West - recorded as a limitation, not glossed over. Run on HiTec's rolling FF3 loadings (1,142 windows): mkt_rf and hml's visible swings are mostly real (zero excluded in 100%/76% of windows) while SMB's swing is mostly noise (zero excluded in only 31%) - the exact distinction a confidence band exists to catch. 65/65 tests pass (pytest, 6 new); rolling_chart run by hand for hitec under ff3 and ff5mom, plus the bad-portfolio and too-large-window error paths. | Day 7 - ml-pipeline-audit pass for multiple testing and look-ahead in factor alignment |
| 2026-09-20 | `1d1f8a1` | Day 5: does Stock Stalker's screen survive factor adjustment? factors/screen_check.py reads Stock Stalker's screen_2026-09-20.json contract (3 NSE candidates) plus the OHLCV fixtures behind it, both committed as a fixture snapshot, and regresses each candidate's monthly returns through the existing CAPM/FF3/FF5+Mom machinery. Every candidate is insignificant under every model, plain-OLS and Newey-West alike (CAPM alpha annualized: RELIANCE -9.98% t=-0.64, TATACHEM -34.31% t=-1.53, CROMPTON -28.78% t=-1.43) - but the honest read is underpowered, not disproven: only 22 months overlap between Stock Stalker's OHLCV window and Ken French's own factor fixture, not enough to estimate up to 7 parameters (FF5+Mom) with any confidence. Also newly flagged: Day 5 regresses INR closes directly against USD factors with no currency adjustment, on top of the already-noted US-factors-on-NSE-names approximation. Both limitations recorded in README rather than glossed over. 62/62 tests pass (pytest, 11 new); screen_check run by hand against capm/ff3/ff5mom, with and without --diagnostics. | Day 6 - rolling factor-loading charts with confidence bands |
| 2026-09-20 | `54a693b` | Day 4: diagnostics on top of Day 2-3's plain-OLS regressions - factors/diagnostics.py adds Newey-West (HAC) alpha t-stats, a Ljung-Box residual-autocorrelation test, a Breusch-Pagan heteroskedasticity test, and a 60-month rolling CAPM beta, wired into regress.py --diagnostics and decay_report.py's new columns; rolling_report.py reports the beta series for all 10 industries. Not cosmetic: NoDur's FF3 alpha flips from OLS-significant (t=2.20) to NW-insignificant (t=1.92), and 16/30 (autocorrelation) and 20/30 (heteroskedasticity) of all industry/model rows are flagged. Also closes the loop Day 3's checkpoint flagged: decay_report.py --since 1963-07 restricts every model to FF5+Mom's own window, isolating HiTec's alpha growth from the sample-window confound - window-matching turns FF3's alpha significant (t=1.55 -> t=2.17), so part of Day 3's finding was a window artifact, but FF5+Mom's alpha (t=4.32) still exceeds window-matched FF3 on the identical 757 months, so the rest is a genuine factor effect. Recorded as both, not smoothed either way. 51/51 tests pass (pytest, 7 new); regress.py --diagnostics, decay_report.py (default and --since), and rolling_report.py all run by hand against the full fixture set. | Day 5 - run it on Stock Stalker's screen output: does that edge survive factor adjustment? |
| 2026-09-19 | `68e65fd` | Day 3: FF3 and FF5+momentum regressions (factors/multifactor.py generalizes capm.py's single-factor OLS to an arbitrary factor set), plus the two new Ken French fixtures this needs (F-F_Research_Data_5_Factors_2x3.csv, F-F_Momentum_Factor.csv, fetched fresh - free, no login) and their loaders/fetch scripts. Correctness gate: no internet access here to a journal's published FF3 table, so 'replicates a known result' is instead an exact analytical identity (each Fama-French factor regressed on its own factor set returns alpha=0, own loading=1, all others=0, R2=1) verified for SMB/HML (FF3) and RMW/Mom (FF5+Mom). factors/decay_report.py runs CAPM/FF3/FF5+Mom across all 10 industries (outputs/alpha_decay.csv) and finds the alpha decay is NOT monotonic as the plan expected - HiTec's alpha grows from +0.57% (CAPM) to +5.44% and turns significant (FF5+Mom) instead of shrinking. Root cause is an honest confound, not a finding about the industry: FF5 data only starts 1963-07, so FF5+Mom runs on 757 months vs CAPM/FF3's 1,201 - different sample windows, not just more factors. Recorded as-is in the README rather than reframed to fit the expected story. 44/44 tests pass (pytest), both regress.py --model ff3/ff5mom and decay_report.py run by hand against all 10 industries. Found and fixed both this repo's and the hub's local main branches stuck in the same recurring stale-detached-HEAD issue before committing. | Day 4 - Newey-West standard errors, residual autocorrelation, heteroskedasticity tests, rolling betas; also re-run FF3 restricted to the 1963-2026 window to isolate factor effect from Day 3's sample-window confound |
| 2026-09-19 | `71a2758` | Day 2: CAPM baseline regression (factors/capm.py) - OLS of portfolio excess return on Mkt-RF, alpha/beta/t-stats/R-squared. Test portfolios are the Ken French 10 Industry Portfolios (new fixture, factors/industry.py, fetched via scripts/fetch_industry_portfolios.py), standing in for the yfinance portfolios sketched in the plan at zero cost. Regression machinery validated first against an exact known answer (market-on-itself: alpha=0, beta=1, R2=1 to 1e-9), then run on all 10 industries via 'python -m factors.regress --portfolio <name> --model capm'; results land where intuition says they should (NoDur/Utils low beta, Durbl/HiTec high beta). Two industries show a plain-OLS-significant alpha but that t-stat is not yet Newey-West-corrected (Day 4), so README records it as provisional. 21/21 tests pass (pytest), CLI run by hand against all 10 industries plus the bad-argument path. Found and fixed both this repo's and the hub's local main branches stuck in the same recurring stale-detached-HEAD issue before committing. | Day 3 - FF3, then FF5 plus momentum; report how alpha decays as factors are added |
| 2026-09-18 | `49966a5` | Day 1: Ken French loader (factors/kenfrench.py) parses the Data Library's real monthly+annual CSV off a committed fixture (fixtures/ken_french/, refreshed via scripts/fetch_ken_french.py). Cross-checking the two sections found a genuine quirk, not a bug: RF compounds monthly to annual within 0.03pp, but SMB/HML don't (up to 21pp off in 2020) since they're annually-reconstituted long/short portfolios, not one held-all-year position -- recorded in the README limitations. 9/9 tests pass (pytest), fetch script run by hand and reproduces the committed fixture byte-for-byte. | Day 2 - CAPM baseline regression: alpha, t-statistic, R-squared |
<!-- CHECKPOINTS:END -->

## Where this sits

Part of an eight-repo research pipeline. Stock Stalker screens the NSE universe; this repo publishes a versioned artifact it reads back, via `factors.screen_check --contract PATH --contract-ticker TICKER` (requires `--model ff3` or `--model ff5mom` — CAPM has no factor loadings to report):

```json
{
  "factor": {
    "alpha_annual": -0.0386,
    "alpha_t": -0.19,
    "significant": false,
    "loadings": {"mkt": 0.02, "smb": 0.10, "hml": -0.25, "rmw": 0.02, "cma": -0.76, "mom": -0.22}
  }
}
```

`loadings` carries one entry per factor the chosen model actually regresses on — six for
FF5+Mom (the model the integration day used), three for FF3 — under each factor's Ken
French column name, with `mkt_rf` shortened to `mkt`. NEXT_STEPS.md's original shape
sketched four loadings (`mkt`, `smb`, `hml`, `mom`) before this module was built; no
model here regresses on exactly that set (FF3 has three factors, FF5+Mom has six), so the
integration ships whatever the chosen model actually produced rather than force-fitting a
four-key example written before the code existed — recorded here rather than silently
matched.

Communication is by file contract, not imports, so either side can be refactored without breaking the other. Stock Stalker's `fixtures/factor/RELIANCE.json` is this exact output, generated against Stock Stalker's own screen and committed there — see its README and CHANGELOG for the integration.

The front-door project's subject gets its own contract the same way, via `factors.gulfoillub --model ff3wml --contract PATH` (no `--contract-ticker`; this module only ever regresses GULFOILLUB). Stock Stalker's `fixtures/factor/GULFOILLUB.json` is this exact output — the real FF3+WML regression against IIMA factors, not a stand-in:

```json
{
  "factor": {
    "alpha_annual": 0.1064,
    "alpha_t": 1.01,
    "significant": false,
    "loadings": {"mkt": 0.797, "smb": 0.470, "hml": -0.237, "wml": 0.011}
  }
}
```

`loadings` here carries `wml`, a key no Ken French-based contract (RELIANCE's above, or any FF3/FF5+Mom result) has ever shipped — `stockstalker.gates.factor.load_contract` only checks that the four top-level keys are present, not which keys `loadings` itself has, so this passed through on the first try, confirmed by reading that module before relying on it rather than assuming.

**Day 22 (C1): any-stock tear sheet, factor exposure.** `factors.gulfoillub` only ever regresses GULFOILLUB against its one committed fixture. `factors/stock.py` is the ticker-agnostic version of the same regression (`--ticker` for the label, `--prices` for any OHLCV CSV with `date`/`close` columns), sharing every underlying function with `gulfoillub.py` so the two necessarily agree bit-for-bit on the same input — `tests/test_stock.py` pins this with a regression test against GULFOILLUB's own contract. Stock Stalker's front door (`stockstalker.live_contracts.generate_factor_contract`) shells out to `python -m factors.stock` as a subprocess, the same pattern Day 21 already used for the risk contract, writing `outputs/contracts/<TICKER>/factor.json` for a ticker with no committed `fixtures/factor/<TICKER>.json` — a file contract, never a cross-repo import. `gulfoillub.py` and its own pinned GULFOILLUB/RELIANCE fixtures are untouched.

## Exam mapping

Series XV ch.12.5 (beta), ch.12.9 (risk-adjusted returns), ch.4.5 (quantitative research)

---

CLI only, by design. No dashboard, no server. Charts and documents are written to `outputs/`.
