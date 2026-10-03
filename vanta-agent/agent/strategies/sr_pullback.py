"""Trend following with support/resistance pullbacks (Oct 2026). Rules
fixed before testing, 1:2.

  1H trend      up: close > EMA50 and EMA20 > EMA50 (mirror for down)
  15M area      in an up-trend, a confirmed 15M swing high that price then
                closes above becomes SUPPORT (broken resistance). Valid for
                two days. Price must pull back into that zone (within 0.25 x
                15M ATR) during the last three 15M candles and hold it
                (no 15M close more than 0.25 ATR below it). Mirror for
                down-trends (broken support becomes resistance).
  5M entry      bullish 5M candle closing above the previous 5M high, with
                volatility expanding: its range >= 1.2 x the average of the
                last 20 candles. (Volume would be used where the data has it;
                forex/CFD data has no real volume.)
  Stop          beyond the zone: support - 0.5 x 15M ATR (at least 1 x 15M
                ATR and the cost floor). TP 2R.
"""
import numpy as np
import pandas as pd

from agent.indicators import align, atr, ema, resample
from agent.instruments import COMMODITIES, CRYPTO, EQUITIES, FOREX, INDICES, Instrument
from agent.strategies.smc_amd import _swings

BAR5, M15 = pd.Timedelta("5min"), pd.Timedelta("15min")
VALID = pd.Timedelta("2D")


def _broken_levels(m: pd.DataFrame, a15: pd.Series):
    """Latest broken swing high (now support) and broken swing low (now
    resistance) on 15M close times, with the time each was broken."""
    sw_hi, sw_lo = _swings(m["high"], m["low"])
    c = m["close"]
    up_break = c > sw_hi
    dn_break = c < sw_lo
    support = sw_hi.where(up_break).ffill()
    sup_time = pd.Series(m.index.where(up_break), index=m.index).ffill()
    resist = sw_lo.where(dn_break).ffill()
    res_time = pd.Series(m.index.where(dn_break), index=m.index).ffill()
    return support, sup_time, resist, res_time


def generate(df: pd.DataFrame, inst: Instrument, rr: float = 2.0,
             min_stop_frac: float = 0.0) -> pd.DataFrame:
    if inst.asset_class == EQUITIES:
        return pd.DataFrame()
    f = df[["open", "high", "low", "close"]].copy()
    h1 = resample(df, "1h")
    e20, e50 = ema(h1["close"], 20), ema(h1["close"], 50)
    trend = ((h1["close"] > e50) & (e20 > e50)).astype(int) - ((h1["close"] < e50) & (e20 < e50)).astype(int)
    f["trend"] = align(trend, f.index, BAR5)

    m = resample(df, "15min")
    a15 = atr(m)
    support, sup_t, resist, res_t = _broken_levels(m, a15)
    z = 0.25 * a15
    now = pd.Series(m.index, index=m.index)
    sup_ok = (now - sup_t) <= VALID
    res_ok = (now - res_t) <= VALID
    in_sup = sup_ok & (m["low"].rolling(3).min() <= support + z) & (m["close"].rolling(3).min() >= support - z)
    in_res = res_ok & (m["high"].rolling(3).max() >= resist - z) & (m["close"].rolling(3).max() <= resist + z)
    for name, s in {"a15": a15, "support": support, "resist": resist,
                    "in_sup": in_sup.astype(float), "in_res": in_res.astype(float)}.items():
        # 15M values (indexed by 15M close time) as known at each 5M close.
        f[name] = s.reindex(f.index + BAR5, method="ffill").to_numpy()

    rng = f["high"] - f["low"]
    vol_up = rng >= 1.2 * rng.shift(1).rolling(20).mean()
    if "volume" in df and df["volume"].notna().any():
        vol_up &= df["volume"] >= 1.5 * df["volume"].shift(1).rolling(20).mean()
    c = f["close"]
    h = f.index.hour + f.index.minute / 60
    if inst.asset_class == CRYPTO:
        liquid = pd.Series(True, index=f.index)
    else:
        lo_h, hi_h = {FOREX: (7, 17), INDICES: (13.5, 20), COMMODITIES: (7, 20)}[inst.asset_class]
        liquid = pd.Series((h >= lo_h) & (h < hi_h), index=f.index)
    out = []
    for d in (1, -1):
        if d == 1:
            zone_ok = f.in_sup == 1
            trig = (c > f["high"].shift(1)) & (c > f["open"])
            stop = f.support - 0.5 * f.a15
        else:
            zone_ok = f.in_res == 1
            trig = (c < f["low"].shift(1)) & (c < f["open"])
            stop = f.resist + 0.5 * f.a15
        floor = np.maximum(f.a15, c * min_stop_frac)
        dist = (c - stop) * d
        stop = pd.Series(np.where(dist < floor, c - d * floor, stop), index=f.index)
        dist = (c - stop) * d
        ok = (f.trend == d) & zone_ok & trig & vol_up & liquid & (dist > 0) & (dist <= 4 * f.a15)
        sig = pd.DataFrame({"direction": d, "stop": stop[ok], "score": 10})
        sig["take_profit"] = c[ok] + rr * (c[ok] - sig["stop"])
        sig["strategy"] = "sr_pullback"
        out.append(sig)
    return pd.concat(out).sort_index()
