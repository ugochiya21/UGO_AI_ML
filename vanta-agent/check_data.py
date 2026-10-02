"""Download (or refresh) price data and sanity-check it.

  python check_data.py                                  # download 2019-2025, report on 2019-2023
  python check_data.py --report-end 2025-12-31          # also report on the hidden period

Downloading the hidden period's prices is fine; only backtesting or tuning
on them uses it up. The report stops at --report-end so we don't look.
"""
import argparse

import pandas as pd

from agent import instruments
from agent.data import loaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--report-end", default="2023-12-31")
    ap.add_argument("--symbols", nargs="*", default=list(instruments.UNIVERSE))
    args = ap.parse_args()

    print(f"{'market':12s} {'bars':>6s}  {'first':10s} {'last':10s} {'low':>10s} {'high':>10s}"
          f" {'gaps>4d':>7s} {'jumps>8%':>8s}")
    for sym in args.symbols:
        inst = instruments.get(sym)
        try:
            df = loaders.load(inst, args.start, args.end)
        except Exception as e:  # noqa: BLE001 - report and carry on
            print(f"{sym:12s} FAILED: {e}")
            continue
        df = df.loc[:pd.Timestamp(args.report_end, tz="UTC") + pd.Timedelta(days=1)]
        if df.empty:
            print(f"{sym:12s} no bars in the report window")
            continue
        gaps = (df.index.to_series().diff() > pd.Timedelta(days=4)).sum()
        jumps = (df.close.pct_change().abs() > 0.08).sum()
        print(f"{sym:12s} {len(df):6d}  {df.index[0]:%Y-%m-%d} {df.index[-1]:%Y-%m-%d}"
              f" {df.low.min():10.4f} {df.high.max():10.4f} {gaps:7d} {jumps:8d}")


if __name__ == "__main__":
    main()
