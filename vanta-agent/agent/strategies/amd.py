"""AMD - Accumulation, Manipulation, Distribution ("Power of 3"). Rules
fixed before testing (Oct 2026), 5-minute bars, 1:2.

  Accumulation   forex & metals: Asian range 00:00-06:00 UTC
                 indices: pre-market range 08:00-09:30 New York
  Manipulation   price trades beyond one side of that range during
                 forex & metals: 08:00-11:00 London time (London open)
                 indices: 09:30-10:30 New York (New York open)
  Entry          the first 5M close back INSIDE the range after the sweep:
                 sweep above -> sell, sweep below -> buy (distribution)
  Stop           beyond the sweep's extreme + 0.1 x 15M ATR (at least
                 0.5 x 15M ATR and the cost floor).  TP: 2R.
  One trade per market per day; close by 16:00 New York if still open.
"""
import numpy as np
import pandas as pd

from agent.indicators import align, atr, resample
from agent.instruments import CRYPTO, INDICES, Instrument

BAR = pd.Timedelta("5min")


def _clock(idx, tz):
    close = (idx + BAR).tz_convert(tz)
    return pd.Series(close.date, index=idx), pd.Series(close.hour * 60 + close.minute, index=idx)


def generate(df: pd.DataFrame, inst: Instrument, rr: float = 2.0,
             min_stop_frac: float = 0.0) -> pd.DataFrame:
    if inst.asset_class == CRYPTO:
        return pd.DataFrame()
    f = df[["open", "high", "low", "close"]].copy()
    a15 = align(atr(resample(df, "15min")), f.index, BAR)
    if inst.asset_class == INDICES:
        day, mins = _clock(f.index, "America/New_York")
        in_acc = (mins > 8 * 60) & (mins <= 9 * 60 + 30)
        in_man = (mins > 9 * 60 + 30) & (mins <= 10 * 60 + 30)
    else:
        day, mins_utc = _clock(f.index, "UTC")
        in_acc = (mins_utc > 0) & (mins_utc <= 6 * 60)
        _, mins_ldn = _clock(f.index, "Europe/London")
        in_man = (mins_ldn > 8 * 60) & (mins_ldn <= 11 * 60)
    hi = f["high"].where(in_acc).groupby(day).transform("max")
    lo = f["low"].where(in_acc).groupby(day).transform("min")
    # Range is known only after accumulation ends; use it in the manipulation window.
    hi, lo = hi.where(in_man), lo.where(in_man)
    c = f["close"]
    # Running extreme of the manipulation window so far (per day).
    run_hi = f["high"].where(in_man).groupby(day).cummax()
    run_lo = f["low"].where(in_man).groupby(day).cummin()
    swept_up = run_hi > hi
    swept_dn = run_lo < lo
    back_in = (c < hi) & (c > lo)
    sell = in_man & swept_up & back_in & ~swept_dn
    buy = in_man & swept_dn & back_in & ~swept_up
    any_sig = sell | buy
    first = any_sig & (any_sig.astype(int).groupby(day).cumsum() == 1)
    ny_day, _ = _clock(f.index, "America/New_York")
    exit_at = pd.to_datetime(ny_day.astype(str) + " 16:00").dt.tz_localize(
        "America/New_York").dt.tz_convert("UTC").set_axis(f.index)
    out = []
    for d, want, extreme in ((-1, sell, run_hi), (1, buy, run_lo)):
        stop = extreme - d * 0.1 * a15
        floor = np.maximum(0.5 * a15, c * min_stop_frac)
        dist = (c - stop) * d
        stop = np.where(dist < floor, c - d * floor, stop)
        dist = (c - stop) * d
        ok = first & want & (dist > 0) & (dist <= 4 * a15)
        sig = pd.DataFrame({"direction": d, "stop": stop, "score": 10, "exit_at": exit_at},
                           index=f.index)[ok]
        sig["take_profit"] = c[ok] + rr * (c[ok] - sig["stop"])
        sig["strategy"] = "amd"
        out.append(sig)
    return pd.concat(out).sort_index()
