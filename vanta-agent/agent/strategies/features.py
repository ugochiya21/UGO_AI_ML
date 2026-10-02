"""Shared multi-timeframe features. Computed once per instrument, vectorized,
and aligned so every value is what a trader would have seen at the close of
the 1-hour bar - no peeking."""
import pandas as pd

from agent.indicators import adx, align, atr, ema, resample, rsi
from agent.instruments import CRYPTO, EQUITIES, Instrument


def build(df: pd.DataFrame, inst: Instrument) -> pd.DataFrame:
    f = df[["open", "high", "low", "close"]].copy()

    # Daily: the big-picture trend.
    d = resample(df, "1D")
    d_ema50, d_ema200 = ema(d["close"], 50), ema(d["close"], 200)
    f["d_close"] = align(d["close"], f.index)
    f["d_ema50"] = align(d_ema50, f.index)
    f["d_ema200"] = align(d_ema200, f.index)
    f["d_ema50_slope"] = align(d_ema50 - d_ema50.shift(5), f.index)

    # 4-hour: the swing structure we trade pullbacks in.
    h4 = resample(df, "4h")
    f["h4_ema20"] = align(ema(h4["close"], 20), f.index)
    f["h4_ema50"] = align(ema(h4["close"], 50), f.index)
    f["h4_adx"] = align(adx(h4), f.index)
    f["h4_atr"] = align(atr(h4), f.index)

    # 1-hour: entry timing.
    f["atr"] = atr(df)
    f["rsi"] = rsi(df["close"])
    f["atr_pct"] = f["atr"].rolling(500, min_periods=100).rank(pct=True)

    # Daily bias: +1 up-trend, -1 down-trend, 0 no clear trend.
    up = (f.d_close > f.d_ema50) & (f.d_ema50 > f.d_ema200)
    dn = (f.d_close < f.d_ema50) & (f.d_ema50 < f.d_ema200)
    f["bias"] = up.astype(int) - dn.astype(int)
    f["h4_trend"] = ((f.h4_ema20 > f.h4_ema50).astype(int)
                     - (f.h4_ema20 < f.h4_ema50).astype(int))

    # Liquid hours: London open to NY afternoon for most markets.
    h = f.index.hour
    if inst.asset_class == CRYPTO:
        f["liquid"] = True
    elif inst.asset_class == EQUITIES and inst.session_utc:
        f["liquid"] = (h >= inst.session_utc[0]) & (h < inst.session_utc[1])
    else:
        f["liquid"] = (h >= 7) & (h < 20)
    return f
