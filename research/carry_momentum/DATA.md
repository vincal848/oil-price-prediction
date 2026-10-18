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

Multi-market book (Part 6 of the README), same source, Future Contracts web queries
11802169-11802184 of 2026-10-18 (fields dsmnem, lasttrddate, date_, volume,
settlement, openinterest; mnemonics ROOT + MMYY for 2000-2030):

| file | Datastream root | market |
|---|---|---|
| `llc_2000_2026.csv.gz` | LLC | ICE Brent |
| `nng_2000_2026.csv.gz` | NNG | NYMEX Henry Hub natural gas |
| `nho_2000_2026.csv.gz` | NHO | NYMEX heating oil (ULSD) |
| `nhu_2000_2026.csv.gz`, `nrb_2000_2026.csv.gz` | NHU, NRB | NYMEX unleaded gasoline, then RBOB |
| `lle_2000_2026.csv.gz` | LLE | ICE low-sulphur gasoil |
| `nue_2000_2026.csv.gz` | NUE | Rotterdam 3.5% fuel oil barges swap |
| `nbg_2000_2026.csv.gz` | NBG | Mont Belvieu LDH propane (OPIS) swap |
| `nuj`, `nbq`, `nbe`, `ndh`, `nbn` files | NUJ, NBQ, NBE, NDH, NBN | Euro jet, physical ethane, ethane, butane and natural gasoline: pulled and left out for lack of trading |

The rows were de-duplicated (WRDS repeats some contract-days) and cut to contracts
within 15 months of delivery (16 for Brent) before transfer.

To rebuild: run the two WRDS Future Contracts queries with mnemonics
NCL0183 ... NCL1205 and NCL0106 ... NCL1228, fields futcode, date_, volume,
settlement, openinterest, dsmnem, contrdate, lasttrddate, firstnoticedate, CSV, gzip.
