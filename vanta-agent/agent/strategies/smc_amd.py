"""AMD + ICT/SMC (Oct 2026). Rules fixed before testing, 1:2.

  4H trend     SMC market structure: bullish after a 4H close above the last
               confirmed 4H swing high (break of structure), bearish after a
               4H close below the last confirmed swing low. Swings are 2-bar
               fractals, usable only once confirmed (2 bars later).
  15M AMD      accumulation: Asian range 00:00-06:00 UTC (indices: pre-market
               08:00-09:30 New York); manipulation window: London 08:00-11:00
               (indices: New York 09:30-11:00); the sweep must go AGAINST the
               4H trend (bullish trend -> sweep below the Asian low).
  15M entry    market-structure shift after the sweep: a 15M close above the
               highest high of the previous four 15M candles (below the
               lowest low for sells), back inside the range.
  Stop         beyond the sweep's extreme + 0.1 x 15M ATR (at least 0.5 x 15M
               ATR and the cost floor). TP 2R. One trade per market per day,
               closed by 16:00 New York.
Runs inside the 5-minute engine: a signal is placed on the 5M bar that
closes when its 15M candle closes (same price).
"""
import numpy as np
import pandas as pd

from agent.indicators import align, atr, resample
from agent.instruments import CRYPTO, INDICES, Instrument

M15 = pd.Timedelta("15min")


def _swings(h: pd.Series, l: pd.Series):
    """Confirmed 2-bar fractal swing levels, known at the close of bar i
    (the swing itself is bar i-2)."""
    sh = (h.shift(2) > h.shift(3)) & (h.shift(2) > h.shift(4)) & \
         (h.shift(2) > h.shift(1)) & (h.shift(2) > h)
    sl = (l.shift(2) < l.shift(3)) & (l.shift(2) < l.shift(4)) & \
         (l.shift(2) < l.shift(1)) & (l.shift(2) < l)
    return h.shift(2).where(sh).ffill(), l.shift(2).where(sl).ffill()


def trend_4h(df: pd.DataFrame) -> pd.Series:
    """+1 / -1 / 0 on 4H close times: direction of the last break of structure."""
    h4 = resample(df, "4h")
    last_hi, last_lo = _swings(h4["high"], h4["low"])
    c = h4["close"].to_numpy()
    hi, lo = last_hi.to_numpy(), last_lo.to_numpy()
    state, out = 0, np.zeros(len(c), dtype=int)
    for i in range(len(c)):
        if not np.isnan(hi[i]) and c[i] > hi[i]:
            state = 1
        elif not np.isnan(lo[i]) and c[i] < lo[i]:
            state = -1
        out[i] = state
    return pd.Series(out, index=h4.index)


def _clock(close_times: pd.DatetimeIndex, tz: str):
    t = close_times.tz_convert(tz)
    return pd.Series(t.date, index=close_times), pd.Series(t.hour * 60 + t.minute, index=close_times)


def generate(df: pd.DataFrame, inst: Instrument, rr: float = 2.0,
             min_stop_frac: float = 0.0) -> pd.DataFrame:
    if inst.asset_class == CRYPTO:
        return pd.DataFrame()
    m = resample(df, "15min")                       # indexed by 15M close time
    a15 = atr(m)
    trend = align(trend_4h(df), m.index - M15, M15)  # known at each 15M close
    trend.index = m.index
    if inst.asset_class == INDICES:
        day, mins = _clock(m.index, "America/New_York")
        in_acc = (mins > 8 * 60) & (mins <= 9 * 60 + 30)
        in_man = (mins > 9 * 60 + 30) & (mins <= 11 * 60)
    else:
        day, mu = _clock(m.index, "UTC")
        in_acc = (mu > 0) & (mu <= 6 * 60)
        _, ml = _clock(m.index, "Europe/London")
        in_man = (ml > 8 * 60) & (ml <= 11 * 60)
    hi = m["high"].where(in_acc).groupby(day).transform("max").where(in_man)
    lo = m["low"].where(in_acc).groupby(day).transform("min").where(in_man)
    run_hi = m["high"].where(in_man).groupby(day).cummax()
    run_lo = m["low"].where(in_man).groupby(day).cummin()
    c = m["close"]
    mss_up = c > m["high"].shift(1).rolling(4).max()
    mss_dn = c < m["low"].shift(1).rolling(4).min()
    buy = in_man & (trend == 1) & (run_lo < lo) & ~(run_hi > hi) & mss_up & (c > lo) & (c < hi)
    sell = in_man & (trend == -1) & (run_hi > hi) & ~(run_lo < lo) & mss_dn & (c < hi) & (c > lo)
    any_sig = buy | sell
    first = any_sig & (any_sig.astype(int).groupby(day).cumsum() == 1)
    ny_day, _ = _clock(m.index, "America/New_York")
    exit_at = pd.to_datetime(ny_day.astype(str) + " 16:00").dt.tz_localize(
        "America/New_York").dt.tz_convert("UTC").set_axis(m.index)
    out = []
    for d, want, extreme in ((1, buy, run_lo), (-1, sell, run_hi)):
        stop = extreme - d * 0.1 * a15
        floor = np.maximum(0.5 * a15, c * min_stop_frac)
        dist = (c - stop) * d
        stop = pd.Series(np.where(dist < floor, c - d * floor, stop), index=m.index)
        dist = (c - stop) * d
        ok = first & want & (dist > 0) & (dist <= 4 * a15)
        sig = pd.DataFrame({"direction": d, "stop": stop[ok], "score": 10,
                            "exit_at": exit_at[ok]})
        sig["take_profit"] = c[ok] + rr * (c[ok] - sig["stop"])
        sig["strategy"] = "smc_amd"
        out.append(sig)
    sig = pd.concat(out).sort_index()
    # Place on the 5M bar that closes at the 15M close (5M index = open time).
    sig.index = sig.index - pd.Timedelta("5min")
    return sig[sig.index.isin(df.index)]
