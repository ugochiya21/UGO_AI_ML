"""Download intraday bars for every market and sanity-check them.

  python download_intraday.py                    # 5-minute, 2019-2025, report on 2019-2023
  python download_intraday.py --tf 15m

Downloads are kept under data/raw/, so re-running only fetches what's
missing. The report stops at --report-end so we don't look at the hidden
2024-2025 period.
"""
import argparse

import pandas as pd

from agent import instruments
from agent.data import intraday

SKIP = {instruments.EQUITIES}   # no free multi-year intraday data for stocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--report-end", default="2023-12-31")
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--tf", choices=["5m", "15m"], default="5m")
    args = ap.parse_args()
    syms = args.symbols or [s for s, i in instruments.UNIVERSE.items() if i.asset_class not in SKIP]

    print(f"{'market':12s} {'bars':>7s}  {'first':10s} {'last':10s} {'low':>11s} {'high':>11s}"
          f" {'spread':>8s} {'holes>1h':>8s}")
    for sym in syms:
        try:
            df = intraday.load(instruments.get(sym), args.start, args.end,
                               rule={"5m": "5min", "15m": "15min"}[args.tf])
        except Exception as e:  # noqa: BLE001 - report and carry on
            print(f"{sym:12s} FAILED: {e}")
            continue
        df = df.loc[:pd.Timestamp(args.report_end, tz="UTC") + pd.Timedelta(days=1)]
        if df.empty:
            print(f"{sym:12s} no bars in the report window")
            continue
        gap = df.index.to_series().diff()
        holes = ((gap > pd.Timedelta("1h")) & (gap < pd.Timedelta("36h"))).sum()
        spread = f"{df.spread.median():.5f}" if "spread" in df else "-"
        print(f"{sym:12s} {len(df):7d}  {df.index[0]:%Y-%m-%d} {df.index[-1]:%Y-%m-%d}"
              f" {df.low.min():11.4f} {df.high.max():11.4f} {spread:>8s} {holes:8d}", flush=True)


if __name__ == "__main__":
    main()
