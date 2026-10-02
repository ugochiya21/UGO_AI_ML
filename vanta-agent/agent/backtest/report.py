"""Turns a BacktestResult into numbers a trader cares about."""
import numpy as np
import pandas as pd

from agent.backtest.engine import BacktestResult


def trades_frame(res: BacktestResult) -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in res.trades])


def challenges_frame(res: BacktestResult) -> pd.DataFrame:
    rows = [{**c.__dict__, "days": round(c.days, 1)} for c in res.challenges]
    return pd.DataFrame(rows)


def _stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0}
    wins = df[df.pnl > 0]
    losses = df[df.pnl <= 0]
    pf = wins.pnl.sum() / -losses.pnl.sum() if losses.pnl.sum() < 0 else float("inf")
    return {
        "trades": len(df),
        "win_rate": round(len(wins) / len(df), 3),
        "avg_R": round(df.r_multiple.mean(), 3),
        "profit_factor": round(pf, 2),
        "net_pnl": round(df.pnl.sum(), 2),
    }


def summary(res: BacktestResult) -> str:
    t = trades_frame(res)
    c = challenges_frame(res)
    lines = ["=" * 64, "BACKTEST SUMMARY", "=" * 64]
    s = _stats(t)
    lines.append("All trades: " + ", ".join(f"{k}={v}" for k, v in s.items()))
    if not t.empty:
        lines.append("\nBy strategy:")
        for name, g in t.groupby("strategy"):
            lines.append(f"  {name:16s} " + ", ".join(f"{k}={v}" for k, v in _stats(g).items()))
        lines.append("\nBy market:")
        for name, g in t.groupby("symbol"):
            lines.append(f"  {name:12s} " + ", ".join(f"{k}={v}" for k, v in _stats(g).items()))
        lines.append("\nExit reasons: " + ", ".join(
            f"{k}={v}" for k, v in t.reason.value_counts().items()))
    if not c.empty:
        done = c[c.status.isin(["passed", "failed"])]
        passed = (c.status == "passed").sum()
        failed = (c.status == "failed").sum()
        lines += ["", "-" * 64, "BACK-TO-BACK $5,000 CHALLENGES (Vanta rules)", "-" * 64]
        lines.append(f"Finished: {len(done)}   PASSED: {passed}   FAILED: {failed}   "
                     f"unfinished at end: {(c.status == 'unfinished').sum()}")
        if len(done):
            lines.append(f"Pass rate: {passed / len(done):.0%}")
        if passed:
            lines.append(f"Days to pass: median {c[c.status == 'passed'].days.median():.0f}, "
                         f"max {c[c.status == 'passed'].days.max():.0f}")
        lines.append(f"Worst static drawdown seen: {c.worst_static_dd.max():.2%}   "
                     f"worst intraday: {c.worst_intraday_dd.max():.2%}   (Vanta limit 5.00%)")
        for _, r in c[c.status == "failed"].iterrows():
            lines.append(f"  FAILED #{r.number} {r.started:%Y-%m-%d} -> {r.ended:%Y-%m-%d}: {r.reason}")
    if res.rejections:
        lines.append("\nIdeas rejected (top 8): " + ", ".join(
            f"{k}={v}" for k, v in sorted(res.rejections.items(), key=lambda x: -x[1])[:8]))
    return "\n".join(lines)
