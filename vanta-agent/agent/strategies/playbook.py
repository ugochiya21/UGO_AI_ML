"""The owner's playbook (Oct 2026): one approach per asset class, always
1H regime -> 15M setup confirmation -> 5M entry, ATR-based stops, ~1:2.

  Forex        trend + momentum, London / New York session breakouts
  Indices      trend + breakout + pullback
  Commodities  trend-following + volatility breakouts
  Crypto       momentum/trend + volatility (and volume, when the data has it)
  Equities     trend + momentum / opening-range breakout (no free multi-year
               intraday stock data, so not backtested)

1H regime (every market): price above EMA50, EMA20 above EMA50 and ADX > 20
for longs (mirror for shorts). No regime, no trade.

15M setup (must have been confirmed within the last hour):
  forex        a 15M close beyond the session's first 30 minutes (London
               08:00 London time, New York 08:00 New York time), within 3h,
               with 15M RSI > 55 (< 45 for shorts) = momentum
  indices      a 15M close beyond the prior 20 x 15M high (low)
  commodities  a 15M volatility-expansion bar: range >= 1.5 x 15M ATR, closing
               in the top (bottom) 30% of its range and beyond the prior
               20-bar high (low)
  crypto       a 15M close beyond the prior 20-bar high (low) while 15M ATR is
               above its 100-bar median (volatility rising); if volume is
               available, 15M volume >= 1.5 x its 20-bar average

5M entry:
  forex, commodities, crypto  momentum: 5M close above the previous 5M high
                              and above the 5M EMA20 (mirror for shorts)
  indices                     pullback: a 5M low touched the 5M EMA20 in the
                              last 3 bars, then a 5M close above the
                              previous 5M high

Stop: entry -/+ 1.5 x 15M ATR (never tighter than the cost floor). TP: 2R.
Score out of 10: regime 3 + setup 3 + entry 2 (all required = 8) + 1 if
1H ADX > 25 + 1 if 15M RSI is not stretched (< 75 long, > 25 short).
Every value is what a trader would have seen at that 5M bar's close.
"""
import numpy as np
import pandas as pd

from agent.indicators import adx, align, atr, ema, resample, rsi
from agent.instruments import COMMODITIES, CRYPTO, EQUITIES, FOREX, INDICES, Instrument

BAR = pd.Timedelta("5min")
SETUP_VALID = pd.Timedelta("60min")
ATR_STOP = 1.5


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = resample(df, rule)
    if "volume" in df:
        vol = df["volume"].resample(rule, label="right", closed="left").sum()
        out["volume"] = vol.reindex(out.index)
    return out


def features(df: pd.DataFrame, inst: Instrument) -> pd.DataFrame:
    f = df[["open", "high", "low", "close"]].copy()

    h1 = resample(df, "1h")
    e20, e50 = ema(h1["close"], 20), ema(h1["close"], 50)
    h1_adx = adx(h1)
    up = (h1["close"] > e50) & (e20 > e50) & (h1_adx > 20)
    dn = (h1["close"] < e50) & (e20 < e50) & (h1_adx > 20)
    f["regime"] = align(up.astype(int) - dn.astype(int), f.index, BAR)
    f["h1_adx"] = align(h1_adx, f.index, BAR)

    m15 = _resample(df, "15min")
    a15 = atr(m15)
    f["m15_atr"] = align(a15, f.index, BAR)
    f["m15_rsi"] = align(rsi(m15["close"]), f.index, BAR)
    f["ema20"] = ema(df["close"], 20)
    f.attrs["m15"] = m15
    f.attrs["m15_atr"] = a15
    return f


def _m15_setups(f: pd.DataFrame, inst: Instrument) -> dict[int, pd.Series]:
    """{direction: boolean Series on 15M close times} - the setup confirmation."""
    m15, a15 = f.attrs["m15"], f.attrs["m15_atr"]
    c, h, l = m15["close"], m15["high"], m15["low"]
    prior_hi = h.shift(1).rolling(20).max()
    prior_lo = l.shift(1).rolling(20).min()
    r = rsi(c)
    out = {}
    if inst.asset_class == FOREX:
        # Session range: first 30 minutes after each open, known once closed.
        brk = {1: pd.Series(False, index=m15.index), -1: pd.Series(False, index=m15.index)}
        opened = m15.index - pd.Timedelta("15min")          # 15M bar open times
        for tz, hhmm in (("Europe/London", 8 * 60), ("America/New_York", 8 * 60)):
            local = opened.tz_convert(tz)
            mins = local.hour * 60 + local.minute - hhmm
            day = pd.Series(local.date, index=m15.index)
            in_rng = (mins >= 0) & (mins < 30)
            after = (mins >= 30) & (mins < 30 + 180)
            hi = h.where(in_rng).groupby(day).transform("max").where(after)
            lo = l.where(in_rng).groupby(day).transform("min").where(after)
            brk[1] |= (c > hi) & (r > 55)
            brk[-1] |= (c < lo) & (r < 45)
        out = brk
    elif inst.asset_class == INDICES:
        out = {1: c > prior_hi, -1: c < prior_lo}
    elif inst.asset_class == COMMODITIES:
        rng = (h - l)
        pos = (c - l) / rng.replace(0, np.nan)
        big = rng >= 1.5 * a15
        out = {1: big & (pos >= 0.7) & (c > prior_hi), -1: big & (pos <= 0.3) & (c < prior_lo)}
    elif inst.asset_class == CRYPTO:
        vol_up = a15 > a15.rolling(100).median()
        if "volume" in m15 and m15["volume"].notna().any():
            vol_up &= m15["volume"] >= 1.5 * m15["volume"].shift(1).rolling(20).mean()
        out = {1: vol_up & (c > prior_hi), -1: vol_up & (c < prior_lo)}
    return {d: s.fillna(False).astype(bool) for d, s in out.items()}


def _recent(setup: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """True on 5M bars whose close falls within SETUP_VALID after a 15M setup
    bar closed (setup is indexed by 15M close time)."""
    when = setup.index[setup.to_numpy(bool)].as_unit("ns").asi8
    closes = (idx + BAR).as_unit("ns").asi8
    pos = np.searchsorted(when, closes, side="right") - 1
    last = np.where(pos >= 0, when[np.clip(pos, 0, None)] if len(when) else 0, 0)
    ok = (pos >= 0) & (closes - last <= SETUP_VALID.value)
    return pd.Series(ok, index=idx)


def generate(f: pd.DataFrame, inst: Instrument, rr: float = 2.0,
             min_stop_frac: float = 0.0) -> pd.DataFrame:
    if inst.asset_class == EQUITIES:
        return pd.DataFrame()
    setups = _m15_setups(f, inst)
    c = f["close"]
    if inst.asset_class == CRYPTO:
        liquid = pd.Series(True, index=f.index)
    else:
        h = f.index.hour + f.index.minute / 60
        lo_h, hi_h = {FOREX: (7, 17), INDICES: (13.5, 20), COMMODITIES: (7, 20)}[inst.asset_class]
        liquid = pd.Series((h >= lo_h) & (h < hi_h), index=f.index)
    out = []
    for d in (1, -1):
        setup = _recent(setups[d], f.index)
        if inst.asset_class == INDICES:
            if d == 1:
                touched = (f["low"].rolling(3).min() <= f.ema20)
                entry = touched & (c > f["high"].shift(1))
            else:
                touched = (f["high"].rolling(3).max() >= f.ema20)
                entry = touched & (c < f["low"].shift(1))
        else:
            entry = ((c > f["high"].shift(1)) & (c > f.ema20)) if d == 1 \
                else ((c < f["low"].shift(1)) & (c < f.ema20))
        regime = f.regime == d
        floor = np.maximum(ATR_STOP * f.m15_atr, c * min_stop_frac)
        stop = c - d * floor
        rsi_ok = (f.m15_rsi < 75) if d == 1 else (f.m15_rsi > 25)
        score = 3 * regime + 3 * setup + 2 * entry + (f.h1_adx > 25) + rsi_ok
        ok = regime & setup & entry & liquid & floor.notna() & (floor > 0)
        sig = pd.DataFrame({"direction": d, "stop": stop, "score": score.astype(int)},
                           index=f.index)[ok]
        sig["take_profit"] = c[ok] + rr * (c[ok] - sig["stop"])
        sig["strategy"] = f"playbook_{inst.asset_class}"
        out.append(sig)
    return pd.concat(out).sort_index()
