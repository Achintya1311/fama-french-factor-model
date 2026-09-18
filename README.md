# Fama-French factor model

Decomposes returns into market, size, value and momentum exposures to test whether an apparent edge is alpha or just beta wearing a disguise.

**Status:** Last checkpoint 2026-09-18 · Next: Day 2 - CAPM baseline regression: alpha, t-statistic, R-squared

## What this is

CAPM, then FF3, then FF5 plus momentum, reporting how alpha decays as factors are added. The decay is the result.

Pointed at the Stock Stalker screen output to answer one question directly: does that screen survive factor adjustment? If the answer is no, both READMEs say so.

## Correctness gate

Replicates a published FF3 result on a known portfolio within tolerance before being trusted on new data.

This is the test that decides whether the repo is finished. A result that has not passed it is a draft.

## Data sources

Every source is free. Nothing in this project requires a paid tier, a subscription, or a funded account.

- Ken French Data Library - factor returns, cached as committed fixtures
- yfinance - test portfolio returns

`factors/kenfrench.py` loads `fixtures/ken_french/F-F_Research_Data_Factors.csv`
(refresh with `python scripts/fetch_ken_french.py`, network only, never on the
test/CLI path) and parses both the monthly and annual sections French ships
in one file. See `fixtures/ken_french/README.md` for provenance and format
notes.

## How to run

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m factors.regress --portfolio fixtures/sample_returns.csv --model ff5
```

Runs offline against committed fixtures by default. Live data needs a key in `.env` (see `.env.example`); the fixture path is the default so nothing blocks on network access.

## Findings

Nothing yet. This section fills in as the work lands, including the results that do not flatter the method.

## Checkpoint log

<!-- CHECKPOINTS:START -->
| Date | Commit | What changed | Next |
|------|--------|--------------|------|
| 2026-09-18 | `49966a5` | Day 1: Ken French loader (factors/kenfrench.py) parses the Data Library's real monthly+annual CSV off a committed fixture (fixtures/ken_french/, refreshed via scripts/fetch_ken_french.py). Cross-checking the two sections found a genuine quirk, not a bug: RF compounds monthly to annual within 0.03pp, but SMB/HML don't (up to 21pp off in 2020) since they're annually-reconstituted long/short portfolios, not one held-all-year position -- recorded in the README limitations. 9/9 tests pass (pytest), fetch script run by hand and reproduces the committed fixture byte-for-byte. | Day 2 - CAPM baseline regression: alpha, t-statistic, R-squared |
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- Ken French factors are constructed on US data. Applying them to NSE names is an approximation that needs stating every time.
- Overlapping windows inflate t-statistics, so Newey-West standard errors are used throughout.
- Testing several specifications on one dataset is multiple testing. The alpha that survives all of them is the only one worth quoting.
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
