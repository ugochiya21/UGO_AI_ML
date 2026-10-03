"""Plain-language performance and position-sizing report for one backtest.

  python performance_report.py results/

Shows: win rate, TP rate, results per market / strategy / month, how fast
challenges pass, and a snapshot of the moment the most trades were open
together - each trade's risk, share of the 1% budget, size and lots.
"""
import sys
from pathlib import Path

import pandas as pd

from agent.instruments import FOREX, get

BUDGET = 50.0   # 1% of $5,000


def lots(row) -> str:
    """Forex: standard lots (100,000 units of the base currency). For
    crosses without USD the USD value is used, so it is approximate (~)."""
    if get(row.symbol).asset_class != FOREX:
        return "-"
    if row.symbol.startswith("USD"):
        return f"{row.notional / 100_000:.2f}"
    if row.symbol.endswith("USD"):
        return f"{row.notional / row.entry / 100_000:.2f}"
    return f"~{row.notional / 100_000:.2f}"


def main(folder: str):
    d = Path(folder)
    t = pd.read_csv(d / "trades.csv", parse_dates=["opened_at", "closed_at"])
    ch = pd.read_csv(d / "challenges.csv")
    n = len(t)
    print("=" * 70)
    print("WIN PERFORMANCE")
    print("=" * 70)
    print(f"Trades: {n}   Won (closed in profit): {(t.pnl > 0).mean():.0%}   "
          f"Hit TP: {(t.reason == 'target').mean():.0%}   Hit SL: {(t.reason == 'stop').mean():.0%}")
    print(f"Closed early for news: {(t.reason == 'news').mean():.0%}   "
          f"for the weekend: {(t.reason == 'weekend').mean():.0%}")
    print(f"Average result per trade: {t.r_multiple.mean():+.2f}R   Net: ${t.pnl.sum():,.0f}")
    wd = t[t.opened_at.dt.weekday < 5].opened_at.dt.normalize()
    print(f"Trades per weekday: {len(wd) / max(len(pd.bdate_range(wd.min(), wd.max())), 1):.1f}")

    done = ch[ch.status != "unfinished"]
    passed = done[done.status == "passed"]
    print(f"\nChallenges: {len(done)} finished - {len(passed)} PASSED, "
          f"{(done.status == 'failed').sum()} failed")
    if len(passed):
        print(f"Days to pass: fastest {passed.days.min():.0f}, middle {passed.days.median():.0f}, "
              f"slowest {passed.days.max():.0f}.  Passed within 14 days: {(passed.days <= 14).sum()}")
    for r in done.itertuples():
        print(f"  #{r.number:<3d} {str(r.started)[:10]} -> {str(r.ended)[:10]}  {r.status:7s} "
              f"{r.days:6.1f} days  {r.trades:4d} trades  {r.reason}")

    print("\nBy strategy:")
    print(t.groupby("strategy").agg(trades=("pnl", "size"), won=("pnl", lambda x: f"{(x > 0).mean():.0%}"),
                                    hit_TP=("reason", lambda x: f"{(x == 'target').mean():.0%}"),
                                    avg_R=("r_multiple", "mean"), net=("pnl", "sum")).round(2).to_string())
    print("\nBy market:")
    print(t.groupby("symbol").agg(trades=("pnl", "size"), won=("pnl", lambda x: f"{(x > 0).mean():.0%}"),
                                  hit_TP=("reason", lambda x: f"{(x == 'target').mean():.0%}"),
                                  avg_R=("r_multiple", "mean"), net=("pnl", "sum"))
          .sort_values("net", ascending=False).round(2).to_string())
    print("\nBy month (net $):")
    m = t.groupby(t.closed_at.dt.to_period("M")).agg(trades=("pnl", "size"), won=("pnl", lambda x: f"{(x > 0).mean():.0%}"),
                                                     net=("pnl", "sum"))
    print(m.round(0).to_string())

    print("\n" + "=" * 70)
    print("POSITION SIZING - how the 1% ($50) is shared")
    print("=" * 70)
    print(f"Risk per trade: median ${t.risk_usd.median():.2f}, smallest ${t.risk_usd.min():.2f}, "
          f"largest ${t.risk_usd.max():.2f}")
    print(f"Combined risk of open trades at any entry: at most ${t.budget_used.max():.2f} "
          f"(limit ${BUDGET:.0f})")
    print(f"Most trades open at once: {t.opened_with.max()}")
    # Snapshot: the entry that had the most trades open together.
    k = t.sort_values(["opened_with", "budget_used"]).iloc[-1]
    when = k.opened_at
    snap = t[(t.opened_at <= when) & (t.closed_at > when)].copy()
    print(f"\nSnapshot at {when:%Y-%m-%d %H:%M} UTC - {len(snap)} trades open:")
    snap["side"] = snap.direction.map({1: "BUY", -1: "SELL"})
    snap["share_of_1%"] = (snap.risk_usd / BUDGET).map("{:.0%}".format)
    snap["stop_%"] = ((snap.entry - snap.stop).abs() / snap.entry * 100).round(3)
    snap["leverage"] = (snap.notional / 5000).round(2)
    snap["lots"] = snap.apply(lots, axis=1)
    snap["result"] = snap.reason + snap.pnl.map(" ${:+.2f}".format)
    cols = ["symbol", "side", "strategy", "entry", "stop", "take_profit", "stop_%",
            "risk_usd", "share_of_1%", "notional", "leverage", "lots", "result"]
    print(snap[cols].round({"risk_usd": 2, "notional": 0}).to_string(index=False))
    print(f"Total risk: ${snap.risk_usd.sum():.2f} = {snap.risk_usd.sum() / 5000:.2%} of the account")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
