# IIM Ahmedabad Indian factor fixture

`2025-12_FourFactors_and_Market_Returns_Monthly.csv` is IIM Ahmedabad's Indian
Fama-French + Momentum monthly file (Agarwalla, Jacob and Varma; built from
CMIE Prowess data on BSE and NSE stocks), release prefix `2025-12`, from:

```
https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/DATA/2025-12_FourFactors_and_Market_Returns_Monthly.csv
```

**Provenance (D56).** The file arrived in commit `7998b18` on 2026-10-11. The
date it was downloaded was not recorded, and the site cannot be reached from
the sandbox these fixes run in, so the copy has not been re-checked against the
live file. It is pinned instead:

- sha256 `3d54c27b20a133fdd210d988ce21a4dc5b1f525df9ac19ccd437dee0a78cc8e9`
  (`tests/test_fixture_provenance.py`)

IIMA re-releases the series under a new `YYYY-MM` prefix. Refreshing means
changing `RELEASE` in `scripts/fetch_iima.py`, rerunning it (network required,
never on the test or CLI path), recording the download date here and updating
the pinned hash.

## Format, as shipped

`Date` (`YYYY-MM`), `SMB`, `HML`, `WML`, `MF` (market excess return), `RF`, all
in percent per month, 1993-10 to 2025-12 (387 rows). Two rows carry `NA`:
1993-10 (MF/RF) and 1994-11 (SMB/HML/WML); `factors/iima.py` drops both.
