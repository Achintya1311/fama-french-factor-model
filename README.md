# Fama-French factor model

Decomposes returns into market, size, value and momentum exposures to test whether an apparent edge is alpha or just beta wearing a disguise.

**Status:** Not started · Next: Day 1 - Ken French loader with cached fixtures

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
<!-- CHECKPOINTS:END -->

## Limitations and what would make me wrong

- Ken French factors are constructed on US data. Applying them to NSE names is an approximation that needs stating every time.
- Overlapping windows inflate t-statistics, so Newey-West standard errors are used throughout.
- Testing several specifications on one dataset is multiple testing. The alpha that survives all of them is the only one worth quoting.

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
