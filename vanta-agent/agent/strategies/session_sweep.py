"""Strategy 2 - London/New York liquidity sweep of the Asian range.

Idea: during the quiet Asian session (00:00-07:00 UTC) a range forms, and
stop orders pile up just beyond its high and low. When London or New York
opens, price often spikes through one side to trigger those stops, then
snaps back. We enter on the snap-back - but only in the direction of the
daily trend, so we are joining the bigger players, not fighting them.

Markets: forex, gold/silver, US indices (not crypto: no real sessions).

Score out of 10 (8+ needed):
  2  daily trend aligned
  2  4-hour structure agrees
  2  sweep beyond the Asian range and close back inside   (required)
  1  4-hour ADX > 20
  1  1-hour RSI not stretched in the trade direction
  1  Asian range of normal size (2-8 x 1-hour ATR)
  1  inside London/NY hours (07:00-15:00 UTC)               (required)
"""
import numpy as np
import pandas as pd

from agent.instruments import CRYPTO, EQUITIES, Instrument

NAME = "session_sweep"


def generate(f: pd.DataFrame, inst: Instrument, rr: float = 2.5) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["direction", "stop", "score", "take_profit", "strategy"])
    if inst.asset_class in (CRYPTO, EQUITIES):
        return empty

    day = f.index.floor("D")
    asia = f[f.index.hour < 7]
    rng = asia.groupby(asia.index.floor("D")).agg(a_high=("high", "max"), a_low=("low", "min"))
    a_high = pd.Series(day.map(rng["a_high"]), index=f.index)
    a_low = pd.Series(day.map(rng["a_low"]), index=f.index)
    window = (f.index.hour >= 7) & (f.index.hour < 15)
    size = (a_high - a_low) / f.atr
    size_ok = size.between(2, 8)

    out = []
    for d in (1, -1):
        if d == 1:
            swept = (f["low"] < a_low) & (f["close"] > a_low) & (f["close"] > f["open"])
            stop = f["low"] - 0.2 * f.atr
            rsi_ok = f.rsi < 65
        else:
            swept = (f["high"] > a_high) & (f["close"] < a_high) & (f["close"] < f["open"])
            stop = f["high"] + 0.2 * f.atr
            rsi_ok = f.rsi > 35
        dist = (f["close"] - stop) * d
        stop = np.where(dist < 0.7 * f.atr, f["close"] - d * 0.7 * f.atr, stop)

        score = (2 * (f.bias == d) + 2 * (f.h4_trend == d) + 2 * swept
                 + (f.h4_adx > 20) + rsi_ok + size_ok + window)
        ok = swept & window & (f.bias == d) & (f.atr_pct < 0.95)
        # First qualifying sweep of the day only.
        ok = ok & ~ok.groupby(day).cumsum().gt(1)
        sig = pd.DataFrame({"direction": d, "stop": stop, "score": score.astype(int)},
                           index=f.index)[ok]
        entry = f.loc[sig.index, "close"]
        sig["take_profit"] = entry + rr * (entry - sig["stop"])
        out.append(sig)
    res = pd.concat(out).sort_index()
    res["strategy"] = NAME
    return res
