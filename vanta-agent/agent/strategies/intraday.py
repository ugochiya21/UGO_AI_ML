"""Intraday strategies on 15-minute bars - built so a trade can reach its
TP within hours, between high-impact news releases.

Both trade only WITH the bigger trend (daily + 4-hour), use reward:risk 1:2
and score setups out of 10 like the hourly strategies (8+ needed).

1. intraday_pullback - the trend-pullback idea one timeframe lower: daily
   and 4-hour trend agree, price dips to the 1-hour EMA20, and a 15-minute
   candle shows buyers (sellers) stepping back in.
     2 daily trend aligned          1 daily EMA50 sloping that way
     2 4-hour trend aligned         1 4-hour ADX > 20
     2 pullback + trigger (required) 1 15-min RSI healthy   1 liquid hours

2. orb - opening-range breakout. The first hour of the London and New York
   sessions sets a range; a 15-minute close beyond it, in the trend's
   direction, within 3 hours, is the entry. Stop at the middle of the range.
     2 daily trend aligned          1 daily EMA50 sloping that way
     2 4-hour trend aligned         1 4-hour ADX > 20
     2 breakout (required)          1 range a sensible size   1 RSI not stretched

Every value is what a trader would have seen at the close of the 15-minute
bar - no peeking.
"""
import numpy as np
import pandas as pd

from agent.indicators import adx, align, atr, ema, resample, rsi
from agent.instruments import COMMODITIES, CRYPTO, EQUITIES, INDICES, Instrument

BAR = pd.Timedelta("15min")


def features(df: pd.DataFrame, inst: Instrument) -> pd.DataFrame:
    f = df[["open", "high", "low", "close"]].copy()

    d = resample(df, "1D")
    d_ema50, d_ema200 = ema(d["close"], 50), ema(d["close"], 200)
    f["d_close"] = align(d["close"], f.index, BAR)
    f["d_ema50"] = align(d_ema50, f.index, BAR)
    f["d_ema200"] = align(d_ema200, f.index, BAR)
    f["d_ema50_slope"] = align(d_ema50 - d_ema50.shift(5), f.index, BAR)

    h4 = resample(df, "4h")
    f["h4_ema20"] = align(ema(h4["close"], 20), f.index, BAR)
    f["h4_ema50"] = align(ema(h4["close"], 50), f.index, BAR)
    f["h4_adx"] = align(adx(h4), f.index, BAR)

    h1 = resample(df, "1h")
    f["h1_ema20"] = align(ema(h1["close"], 20), f.index, BAR)
    f["h1_atr"] = align(atr(h1), f.index, BAR)

    f["atr"] = atr(df)
    f["rsi"] = rsi(df["close"])
    f["ema20"] = ema(df["close"], 20)
    f["atr_pct"] = f["atr"].rolling(2000, min_periods=400).rank(pct=True)

    up = (f.d_close > f.d_ema50) & (f.d_ema50 > f.d_ema200)
    dn = (f.d_close < f.d_ema50) & (f.d_ema50 < f.d_ema200)
    f["bias"] = up.astype(int) - dn.astype(int)
    f["h4_trend"] = ((f.h4_ema20 > f.h4_ema50).astype(int)
                     - (f.h4_ema20 < f.h4_ema50).astype(int))

    h = f.index.hour
    if inst.asset_class == CRYPTO:
        f["liquid"] = True
    elif inst.asset_class in (INDICES, EQUITIES):
        f["liquid"] = (h >= 13) & (h < 20)
    else:
        f["liquid"] = (h >= 7) & (h < 17)
    return f


def _signals(f, ok, d, stop, score, rr, name):
    entry = f["close"]
    sig = pd.DataFrame({"direction": d, "stop": stop, "score": score.astype(int)},
                       index=f.index)[ok]
    sig["take_profit"] = entry[ok] + rr * (entry[ok] - sig["stop"])
    sig["strategy"] = name
    return sig


def _trend_score(f, d):
    return (2 * (f.bias == d) + (np.sign(f.d_ema50_slope) == d)
            + 2 * (f.h4_trend == d) + (f.h4_adx > 20))


def intraday_pullback(f: pd.DataFrame, rr: float = 2.0, lookback: int = 8) -> pd.DataFrame:
    out = []
    for d in (1, -1):
        if d == 1:
            touched = f["low"].rolling(lookback).min() <= f.h1_ema20 + 0.2 * f.h1_atr
            intact = f["close"] > f.h4_ema50
            trigger = (f["close"] > f["high"].shift(1)) & (f["close"] > f["open"]) \
                & (f["close"] > f.ema20)
            extreme = f["low"].rolling(lookback).min()
            rsi_ok = f.rsi.between(45, 70)
        else:
            touched = f["high"].rolling(lookback).max() >= f.h1_ema20 - 0.2 * f.h1_atr
            intact = f["close"] < f.h4_ema50
            trigger = (f["close"] < f["low"].shift(1)) & (f["close"] < f["open"]) \
                & (f["close"] < f.ema20)
            extreme = f["high"].rolling(lookback).max()
            rsi_ok = f.rsi.between(30, 55)
        setup = touched & intact & trigger
        stop = extreme - d * 0.2 * f.atr
        dist = (f["close"] - stop) * d
        stop = np.where(dist < f.atr, f["close"] - d * f.atr, stop)
        dist = (f["close"] - stop) * d
        sane = (dist > 0) & (dist <= 4 * f.atr) & (f.atr_pct < 0.95)
        score = _trend_score(f, d) + 2 * setup + rsi_ok + f.liquid.astype(bool)
        ok = setup & sane & (f.bias == d) & f.liquid.astype(bool)
        out.append(_signals(f, ok, d, stop, score, rr, "intraday_pullback"))
    return pd.concat(out).sort_index()


# Sessions: (local timezone, open time, markets it applies to)
SESSIONS = {
    "london": ("Europe/London", "08:00", {"forex", COMMODITIES}),
    "newyork": ("America/New_York", "09:30", {"forex", COMMODITIES, INDICES}),
}


def orb(f: pd.DataFrame, inst: Instrument, rr: float = 2.0,
        range_minutes: int = 60, window_hours: int = 3) -> pd.DataFrame:
    if inst.asset_class == CRYPTO:
        return pd.DataFrame()
    out = []
    for tz, opens, classes in SESSIONS.values():
        if inst.asset_class not in classes:
            continue
        local = f.index.tz_convert(tz)
        mins = (local.hour * 60 + local.minute) - (int(opens[:2]) * 60 + int(opens[3:]))
        day = pd.Series(local.date, index=f.index)
        in_range = (mins >= 0) & (mins < range_minutes)
        after = (mins >= range_minutes) & (mins < range_minutes + window_hours * 60)
        hi = f["high"].where(in_range).groupby(day).transform("max")
        lo = f["low"].where(in_range).groupby(day).transform("min")
        # The range is only known once its last bar has closed.
        hi, lo = hi.where(after), lo.where(after)
        width = hi - lo
        size_ok = (width >= 1.5 * f.atr) & (width <= 6 * f.atr)
        for d in (1, -1):
            brk = (f["close"] > hi) if d == 1 else (f["close"] < lo)
            # Only the FIRST breakout of the session in this direction.
            first = brk & (brk.astype(int).groupby(day).cumsum() == 1)
            mid = (hi + lo) / 2
            stop = mid
            dist = (f["close"] - stop) * d
            stop = np.where(dist < f.atr, f["close"] - d * f.atr, stop)
            dist = (f["close"] - stop) * d
            sane = (dist > 0) & (dist <= 4 * f.atr)
            rsi_ok = f.rsi.between(40, 75) if d == 1 else f.rsi.between(25, 60)
            score = _trend_score(f, d) + 2 * first + size_ok + rsi_ok
            ok = first & sane & size_ok & (f.bias == d)
            out.append(_signals(f, ok, d, stop, score, rr, "orb"))
    return pd.concat(out).sort_index() if out else pd.DataFrame()





# ---------------------------------------------------------- news momentum
def news_momentum(f: pd.DataFrame, inst: Instrument, events, rr: float = 2.0,
                  wait_minutes: int = 30) -> pd.DataFrame:
    """3. news_momentum - trade WITH a high-impact surprise once the owner's
    30-minute wait is over and price has confirmed it.

    A release whose actual beats (misses) its forecast is good (bad) for
    that currency. Waiting out the 30 minutes, if price has moved at least
    half an hour-ATR in the direction the surprise points to, enter that way:
    stop beyond the pre-news price, TP 2R.
      2 surprise direction confirmed by price (required)
      2 move >= 1 hour-ATR (strong)   1 move >= 0.5 hour-ATR
      2 4-hour trend agrees           2 daily trend agrees   1 RSI not stretched
    Forex, gold and silver only (indices and crypto react unpredictably to
    the dollar)."""
    from agent.macro import LOWER_IS_BETTER, _num
    from agent.instruments import FOREX

    if inst.asset_class not in (FOREX, COMMODITIES) or inst.symbol == "WTIOILUSDC":
        return pd.DataFrame()
    if inst.asset_class == FOREX:
        base, quote = inst.news_currencies
        leg = {base: 1, quote: -1}
    else:
        leg = {"USD": -1}   # gold/silver fall when the dollar strengthens
    rows = []
    idx = f.index
    for e in events:
        if e.currency not in leg:
            continue
        a, fc = _num(e.actual), _num(e.forecast)
        if a is None or fc is None or a == fc:
            continue
        good = (1 if a > fc else -1) * (-1 if LOWER_IS_BETTER.search(e.title) else 1)
        d = good * leg[e.currency]
        # Bar that closes right when the news is out (pre-news price), and
        # the first bar that closes after the 30-minute wait.
        pre_i = idx.searchsorted(e.time - BAR)
        go_i = idx.searchsorted(e.time + pd.Timedelta(minutes=wait_minutes))
        if pre_i >= len(idx) or go_i >= len(idx) or idx[go_i] - e.time > pd.Timedelta("2h"):
            continue
        pre, now = f["close"].iloc[pre_i], f["close"].iloc[go_i]
        h1_atr = f.h1_atr.iloc[go_i]
        move = (now - pre) * d
        if not (h1_atr > 0) or move < 0.5 * h1_atr:
            continue
        a15 = f.atr.iloc[go_i]
        stop = pre - d * 0.2 * a15
        if (now - stop) * d < a15:
            stop = now - d * a15
        if (now - stop) * d > 4 * h1_atr:
            continue
        r = f.rsi.iloc[go_i]
        score = (2 + 2 * (move >= h1_atr) + (move >= 0.5 * h1_atr)
                 + 2 * (f.h4_trend.iloc[go_i] == d) + 2 * (f.bias.iloc[go_i] == d)
                 + ((30 < r < 75) if d == 1 else (25 < r < 70)))
        rows.append((idx[go_i], d, stop, int(score), now + rr * (now - stop)))
    if not rows:
        return pd.DataFrame()
    sig = pd.DataFrame(rows, columns=["time", "direction", "stop", "score", "take_profit"])
    sig = sig.set_index("time").sort_index()
    sig = sig[~sig.index.duplicated()]   # two releases at once: keep the first
    sig["strategy"] = "news_momentum"
    return sig


STRATEGIES = {"intraday_pullback": lambda f, inst, ev: intraday_pullback(f),
              "orb": lambda f, inst, ev: orb(f, inst),
              "news_momentum": lambda f, inst, ev: news_momentum(f, inst, ev)}
