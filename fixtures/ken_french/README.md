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
