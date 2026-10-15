# Data

Raw inputs live in `data/` (gitignored: the Datastream rows are licensed through
WRDS and are not redistributed). Each source, how it was obtained, and what it holds:

| file | source | content |
|---|---|---|
| `ncl_1983_2005.csv.gz`, `ncl_2006_2028.csv.gz` | LSEG Datastream futures via WRDS (`tr_ds_fut`, Future Contracts web query 11791858 and 11791860, 2026-10-15) | Daily settlement, volume, open interest, last trading and first notice dates for every NYMEX CL contract month NCL0183-NCL1228; 515,119 rows, 1983-03-30 to 2026-04-02 |
| `F1.csv`-`F4.csv` | EIA, Cushing OK WTI futures contracts 1-4 (RCLC1-4), daily | Reconciliation target, 2000-01 to 2024-04 |
| `gsci_crude.xls` | S&P Dow Jones Indices, S&P GSCI Crude Oil (index id 10003087), daily, export of 2026-10-15 | Reconciliation target, 2016-09 to 2026-10 |
| `cot_wti_disagg.csv` | CFTC Public Reporting Environment, disaggregated futures-only, NYMEX WTI (067651) | Weekly positions incl. spreading, 2006-06 to 2026-10 |
| `USCRUDE.csv`, `CUSHING.csv` | EIA weekly stocks (WCESTUS1, Cushing) | Mechanism test only |
| `aqr_tsmom_monthly.xlsx` | AQR Data Sets, "Time Series Momentum: Factors, Monthly" | TSMOM^CM for the novelty test |
| `CL_yahoo.csv` | Yahoo `CL=F` | Tie-break source only (identified the Datastream 2007 expiry error) |

To rebuild: run the two WRDS Future Contracts queries with mnemonics
NCL0183 ... NCL1205 and NCL0106 ... NCL1228, fields futcode, date_, volume,
settlement, openinterest, dsmnem, contrdate, lasttrddate, firstnoticedate, CSV, gzip.
