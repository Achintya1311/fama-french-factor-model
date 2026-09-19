# Ken French Data Library fixtures

`F-F_Research_Data_Factors.csv` is the unmodified CSV inside
`F-F_Research_Data_Factors_CSV.zip`, downloaded from the Ken French Data
Library on 2026-09-18:

```
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip
```

Refresh it with `python scripts/fetch_ken_french.py` (network required; not
part of any test or CLI runtime path — everything here reads this committed
copy).

Free for academic/research use per the Data Library's terms; no login or
API key needed, so this stays inside the zero-spend rule.

## Format, as shipped by French

- 4 lines of header prose, then a monthly section: `,Mkt-RF,SMB,HML,RF`
  followed by rows keyed `YYYYMM`, values in **percent**.
- A blank line, ` Annual Factors: January-December `, then the same header
  and rows keyed by 4-digit year.
- A trailing copyright line.

`factors/kenfrench.py` parses both sections directly out of this file; nothing
is hand-edited or reformatted before committing it.

## `10_Industry_Portfolios.csv`

Unmodified CSV inside `10_Industry_Portfolios_CSV.zip`, downloaded on
2026-09-19:

```
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Industry_Portfolios_CSV.zip
```

Refresh with `python scripts/fetch_industry_portfolios.py`. Same free,
no-login terms as the factors file above.

Ten value- and equal-weighted industry portfolios (NoDur, Durbl, Manuf,
Enrgy, HiTec, Telcm, Shops, Hlth, Utils, Other) - standard CAPM/FF3/FF5 test
assets, used here as the portfolios the regression in `factors/regress.py`
is run against. These are US industries, not an NSE universe; see the
project README's caveat on applying US-constructed portfolios to Stock
Stalker's screen output.

## Format, as shipped by French (industry file)

Seven sections, each starting with a blank line and a label, then the same
`,NoDur,Durbl,...,Other` header repeated: monthly value-weighted, monthly
equal-weighted, annual value-weighted, annual equal-weighted, number of
firms, average firm size, and two BE/ME sections, ending in a copyright
line. `factors/industry.py` currently parses only the first section
(monthly value-weighted returns, percent) - the one the regressions need.
The other six sections are left unparsed; a later day can extend the loader
if number-of-firms or BE/ME data becomes useful.

## `F-F_Research_Data_5_Factors_2x3.csv`

Unmodified CSV inside `F-F_Research_Data_5_Factors_2x3_CSV.zip`, downloaded
on 2026-09-19:

```
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip
```

Refresh with `python scripts/fetch_five_factors.py`. Same free, no-login
terms as the files above. Adds RMW (profitability) and CMA (investment) to
Mkt-RF/SMB/HML/RF - the two factors FF5 adds over FF3. Same two-section
(monthly + annual) layout as `F-F_Research_Data_Factors.csv`, six data
columns instead of four; `factors/five_factor.py` parses it the same way.

Note the shorter history: this file starts 1963-07, not 1926-07 like the
FF3 file - RMW and CMA need book equity and operating profitability data
that CRSP/Compustat coverage doesn't reach as far back for. Any comparison
across CAPM/FF3 (1926-) and FF5+Mom (1963-) results is therefore over
different sample windows, not just different factor sets - see the project
README's Day 3 findings for how that shows up in practice.

## `F-F_Momentum_Factor.csv`

Unmodified CSV inside `F-F_Momentum_Factor_CSV.zip`, downloaded on
2026-09-19:

```
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip
```

Refresh with `python scripts/fetch_momentum_factor.py`. Same free, no-login
terms. Mom ships as its own single-column download, not a seventh column on
the FF5 file, so `factors/momentum.py` is a separate loader;
`factors/multifactor.load_ff5_mom_factors()` inner-joins the two on month.
Same two-section layout, one data column (`,Mom`) instead of four or six -
note French spells the annual label across two lines here (`Annual
Factors:` then `January-December` on the next line) instead of one; the
loader doesn't care, since it matches on the `,Mom` header row itself, not
the label text above it.
