"""Intraday, three timeframes: 1-hour trend, 15-minute setup, 5-minute entry.
The owner's style - intraday only, 1:2 on every trade, best setups only.

Runs on 5-minute bars. The 1-hour and 15-minute values on each 5-minute bar
are what a trader would have seen at that bar's close (finished higher-
timeframe bars only - no peeking).

1. mtf_pullback - buy the dip in an intraday up-trend (sell the rally in a
   down-trend).
     2  1-hour trend: EMA20 above EMA50 and price above EMA50
     1  1-hour ADX > 20 (the trend has strength)
     2  15-minute trend agrees: EMA20 above EMA50
     2  15-minute pullback into the EMA20 zone, structure intact (required)
     1  5-minute trigger: close above the previous high, bullish candle,
        above the 5-minute EMA20 (required)
     1  15-minute RSI healthy (45-70 long, 30-55 short)
     1  liquid hours for that market
   Stop: beyond the last hour's 5-minute low/high. TP: 2x the stop.

2. orb5 - London / New York opening range (first 30 minutes), entered on a
   5-minute close beyond it, only in the direction of the 1-hour AND
   15-minute trend, within 2 hours of the open. Stop: middle of the range.
     2 1-hour trend   1 1-hour ADX > 20   2 15-minute trend
     2 breakout (required)   1 range a sensible size   1 RSI not stretched
     1 first breakout of the session

Stops are never tighter than `min_stop_frac` (set so Vanta's spread +
slippage stay a small part of the risk) or 1 x 15-minute ATR.
"""
import numpy as np
import pandas as pd

from agent.indicators import adx, align, atr, ema, resample, rsi
from agent.instruments import COMMODITIES, CRYPTO, EQUITIES, FOREX, INDICES, Instrument

BAR = pd.Timedelta("5min")


def features(df: pd.DataFrame, inst: Instrument) -> pd.DataFrame:
    f = df[["open", "high", "low", "close"]].copy()

    h1 = resample(df, "1h")
    h1_ema20, h1_ema50 = ema(h1["close"], 20), ema(h1["close"], 50)
    f["h1_close"] = align(h1["close"], f.index, BAR)
    f["h1_ema20"] = align(h1_ema20, f.index, BAR)
    f["h1_ema50"] = align(h1_ema50, f.index, BAR)
    f["h1_adx"] = align(adx(h1), f.index, BAR)

    m15 = resample(df, "15min")
    f["m15_ema20"] = align(ema(m15["close"], 20), f.index, BAR)
    f["m15_ema50"] = align(ema(m15["close"], 50), f.index, BAR)
    f["m15_atr"] = align(atr(m15), f.index, BAR)
    f["m15_rsi"] = align(rsi(m15["close"]), f.index, BAR)
    f["m15_low3"] = align(m15["low"].rolling(3).min(), f.index, BAR)
    f["m15_high3"] = align(m15["high"].rolling(3).max(), f.index, BAR)
    f["m15_close"] = align(m15["close"], f.index, BAR)

    f["atr"] = atr(df)
    f["ema20"] = ema(df["close"], 20)

    up = (f.h1_ema20 > f.h1_ema50) & (f.h1_close > f.h1_ema50)
    dn = (f.h1_ema20 < f.h1_ema50) & (f.h1_close < f.h1_ema50)
    f["h1_trend"] = up.astype(int) - dn.astype(int)
    f["m15_trend"] = ((f.m15_ema20 > f.m15_ema50).astype(int)
                      - (f.m15_ema20 < f.m15_ema50).astype(int))

    h = f.index.hour + f.index.minute / 60
    if inst.asset_class == CRYPTO:
        f["liquid"] = True
    elif inst.asset_class in (INDICES, EQUITIES):
        f["liquid"] = (h >= 13.5) & (h < 20)
    elif inst.asset_class == COMMODITIES:
        f["liquid"] = (h >= 7) & (h < 20)
    else:
        f["liquid"] = (h >= 7) & (h < 17)
    return f


def _stop_floor(f, stop, d, min_stop_frac):
    """Widen a stop that is too tight (noise, or costs eating the risk)."""
    floor = np.maximum(f.m15_atr, f["close"] * min_stop_frac)
    dist = (f["close"] - stop) * d
    return np.where(dist < floor, f["close"] - d * floor, stop)


def _out(f, ok, d, stop, score, rr, name):
    sig = pd.DataFrame({"direction": d, "stop": stop, "score": score.astype(int)},
                       index=f.index)[ok]
    entry = f.loc[sig.index, "close"]
    sig["take_profit"] = entry + rr * (entry - sig["stop"])
    sig["strategy"] = name
    return sig


def mtf_pullback(f: pd.DataFrame, inst: Instrument, rr: float = 2.0,
                 min_stop_frac: float = 0.0) -> pd.DataFrame:
    out = []
    for d in (1, -1):
        if d == 1:
            pulled = (f.m15_low3 <= f.m15_ema20 + 0.25 * f.m15_atr) & (f.m15_close > f.m15_ema50)
            trigger = (f["close"] > f["high"].shift(1)) & (f["close"] > f["open"]) \
                & (f["close"] > f.ema20)
            extreme = f["low"].rolling(12).min()
            rsi_ok = f.m15_rsi.between(45, 70)
        else:
            pulled = (f.m15_high3 >= f.m15_ema20 - 0.25 * f.m15_atr) & (f.m15_close < f.m15_ema50)
            trigger = (f["close"] < f["low"].shift(1)) & (f["close"] < f["open"]) \
                & (f["close"] < f.ema20)
            extreme = f["high"].rolling(12).max()
            rsi_ok = f.m15_rsi.between(30, 55)
        stop = _stop_floor(f, extreme - d * 0.2 * f.atr, d, min_stop_frac)
        dist = (f["close"] - stop) * d
        sane = (dist > 0) & (dist <= 3 * f.m15_atr.clip(lower=f["close"] * min_stop_frac))
        score = (2 * (f.h1_trend == d) + (f.h1_adx > 20) + 2 * (f.m15_trend == d)
                 + 2 * pulled + trigger + rsi_ok + f.liquid.astype(bool))
        ok = pulled & trigger & sane & (f.h1_trend == d) & f.liquid.astype(bool)
        out.append(_out(f, ok, d, stop, score, rr, "mtf_pullback"))
    return pd.concat(out).sort_index()


SESSIONS = {
    "london": ("Europe/London", 8 * 60, {FOREX, COMMODITIES}),
    "newyork": ("America/New_York", 9 * 60 + 30, {FOREX, COMMODITIES, INDICES}),
}


def orb5(f: pd.DataFrame, inst: Instrument, rr: float = 2.0, min_stop_frac: float = 0.0,
         range_minutes: int = 30, window_minutes: int = 120) -> pd.DataFrame:
    if inst.asset_class == CRYPTO:
        return pd.DataFrame()
    out = []
    for tz, opens, classes in SESSIONS.values():
        if inst.asset_class not in classes:
            continue
        local = f.index.tz_convert(tz)
        mins = local.hour * 60 + local.minute - opens
        day = pd.Series(local.date, index=f.index)
        in_range = (mins >= 0) & (mins < range_minutes)
        after = (mins >= range_minutes) & (mins < range_minutes + window_minutes)
        hi = f["high"].where(in_range).groupby(day).transform("max").where(after)
        lo = f["low"].where(in_range).groupby(day).transform("min").where(after)
        width = hi - lo
        size_ok = (width >= 0.5 * f.m15_atr) & (width <= 3 * f.m15_atr)
        for d in (1, -1):
            brk = (f["close"] > hi) if d == 1 else (f["close"] < lo)
            first = brk & (brk.astype(int).groupby(day).cumsum() == 1)
            stop = _stop_floor(f, (hi + lo) / 2, d, min_stop_frac)
            dist = (f["close"] - stop) * d
            sane = (dist > 0) & (dist <= 3 * f.m15_atr.clip(lower=f["close"] * min_stop_frac))
            rsi_ok = f.m15_rsi.between(40, 75) if d == 1 else f.m15_rsi.between(25, 60)
            score = (2 * (f.h1_trend == d) + (f.h1_adx > 20) + 2 * (f.m15_trend == d)
                     + 2 * brk + size_ok + rsi_ok + first)
            ok = first & sane & size_ok & (f.h1_trend == d) & (f.m15_trend == d)
            out.append(_out(f, ok, d, stop, score, rr, "orb5"))
    return pd.concat(out).sort_index() if out else pd.DataFrame()


STRATEGIES = {"mtf_pullback": mtf_pullback, "orb5": orb5}
