"""Compare backtest runs side by side, focused on passing the challenge fast.

  python compare_runs.py runs3/*/          # one row per run folder

Columns: challenges finished, passed, failed, pass rate, median days to
pass, share of ALL finished challenges passed within 14 days, trades per
weekday, win rate, share of trades that hit TP, average R, net P&L.
"""
import sys
from pathlib import Path

import pandas as pd


def summarize(folder: Path) -> dict:
    ch = pd.read_csv(folder / "challenges.csv")
    tr = pd.read_csv(folder / "trades.csv", parse_dates=["opened_at", "closed_at"])
    done = ch[ch.status != "unfinished"]
    passed = done[done.status == "passed"]
    days = (pd.to_datetime(tr.opened_at).dt.normalize())
    weekdays = days[days.dt.weekday < 5]
    span = pd.bdate_range(weekdays.min(), weekdays.max()) if len(weekdays) else []
    return {
        "run": folder.name,
        "finished": len(done),
        "passed": len(passed),
        "failed": int((done.status == "failed").sum()),
        "pass%": round(100 * len(passed) / len(done)) if len(done) else 0,
        "med_days": round(passed.days.median(), 1) if len(passed) else None,
        "pass<=14d%": round(100 * (passed.days <= 14).sum() / len(done)) if len(done) else 0,
        "trades/day": round(len(weekdays) / max(len(span), 1), 1),
        "win%": round(100 * (tr.pnl > 0).mean()) if len(tr) else 0,
        "TP%": round(100 * (tr.reason == "target").mean()) if len(tr) else 0,
        "avg_R": round(tr.r_multiple.mean(), 3) if len(tr) else 0,
        "net_pnl": round(tr.pnl.sum()),
    }


def main():
    rows = [summarize(Path(p)) for p in sys.argv[1:] if (Path(p) / "challenges.csv").exists()]
    if not rows:
        raise SystemExit("no run folders with results given")
    print(pd.DataFrame(rows).set_index("run").to_string())


if __name__ == "__main__":
    main()
