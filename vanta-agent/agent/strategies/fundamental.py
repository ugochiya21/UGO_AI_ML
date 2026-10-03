"""Fundamentals-only strategies (Oct 2026). No chart patterns: the DIRECTION
comes only from economic data; price is used only to size the stop.
Rules fixed before any test, no tuning.

1. surprise_momentum - economic surprise momentum.
   For every currency, count the high-impact releases of the last 30 days
   that beat (+1) or missed (-1) their forecast (lower-is-better numbers
   like unemployment inverted). Pair edge = base score - quote score.
   Edge >= +4: buy the pair.  Edge <= -4: sell it.  Gold: buy when the USD
   score <= -4, sell when >= +4.

2. cb_divergence - central-bank policy divergence.
   For every central bank, add up its rate decisions of the last 180 days:
   hike +1, cut -1, hold 0, plus +1/-1 when the decision was more hawkish /
   dovish than forecast. Pair divergence = base stance - quote stance.
   Divergence >= +2: buy the pair.  <= -2: sell it.

Both: checked once a day at 08:00 UTC (London open) using only releases
already published; stop 1.5 x daily ATR(20); TP 2R. Run on 1-hour bars.
"""
import numpy as np
import pandas as pd

from agent.indicators import align, atr, resample
from agent.instruments import COMMODITIES, FOREX, Instrument
from agent.macro import LOWER_IS_BETTER, _num

BAR = pd.Timedelta("1h")
CHECK_HOUR = 8
RATE_DECISIONS = {
    "USD": ("Federal Funds Rate",), "EUR": ("Main Refinancing Rate",),
    "GBP": ("Official Bank Rate",), "JPY": ("BOJ Policy Rate", "Overnight Call Rate"),
    "AUD": ("Cash Rate",), "NZD": ("Official Cash Rate",), "CAD": ("Overnight Rate",),
    "CHF": ("SNB Policy Rate", "Libor Rate"),
}


def _points(events, kind: str) -> dict[str, pd.Series]:
    """{currency: Series of +/- points indexed by release time}."""
    rows = {}
    for e in events:
        a, fc = _num(e.actual), _num(e.forecast)
        if a is None:
            continue
        if kind == "surprise":
            if fc is None or a == fc:
                continue
            p = (1 if a > fc else -1) * (-1 if LOWER_IS_BETTER.search(e.title) else 1)
        else:
            if e.title not in RATE_DECISIONS.get(e.currency, ()):
                continue
            prev = _num(e.previous)
            p = 0
            if prev is not None and a != prev:
                p += 1 if a > prev else -1
            if fc is not None and a != fc:
                p += 1 if a > fc else -1
            if p == 0:
                continue
        rows.setdefault(e.currency, []).append((e.time, p))
    return {c: pd.Series([p for _, p in v], index=pd.DatetimeIndex([t for t, _ in v])).sort_index()
            for c, v in rows.items()}


def _score_at(points: pd.Series | None, times: pd.DatetimeIndex, window: pd.Timedelta) -> np.ndarray:
    """Sum of points released in (t - window, t] for each t."""
    if points is None or points.empty:
        return np.zeros(len(times))
    ts = points.index.as_unit("ns").asi8
    cum = np.r_[0, np.cumsum(points.to_numpy())]
    t = times.as_unit("ns").asi8
    hi = np.searchsorted(ts, t, side="right")
    lo = np.searchsorted(ts, t - window.value, side="right")
    return cum[hi] - cum[lo]


def _generate(df, inst, events, kind, window, threshold, name, rr=2.0, min_stop_frac=0.0):
    if inst.asset_class == FOREX:
        legs = {inst.news_currencies[0]: 1, inst.news_currencies[1]: -1}
    elif inst.symbol == "GOLDUSDC" and kind == "surprise":
        legs = {"USD": -1}
    else:
        return pd.DataFrame()
    f = df[["open", "high", "low", "close"]].copy()
    d = resample(df, "1D")
    d = d[(d["high"] - d["low"]) > 0]
    d_atr = align(atr(d, 20), f.index, BAR)
    check = (f.index.hour == CHECK_HOUR) & (f.index.weekday < 5)
    known_at = f.index + BAR                       # bar close: decision time
    pts = _points(events, kind)
    edge = sum(sign * _score_at(pts.get(c), known_at, window) for c, sign in legs.items())
    edge = pd.Series(edge, index=f.index)
    c = f["close"]
    out = []
    for dirn in (1, -1):
        want = (edge >= threshold) if dirn == 1 else (edge <= -threshold)
        dist = np.maximum(1.5 * d_atr, c * min_stop_frac)
        ok = check & want & dist.notna() & (dist > 0)
        sig = pd.DataFrame({"direction": dirn, "stop": c - dirn * dist, "score": 10},
                           index=f.index)[ok]
        sig["take_profit"] = c[ok] + rr * (c[ok] - sig["stop"])
        sig["strategy"] = name
        out.append(sig)
    return pd.concat(out).sort_index()


def surprise_momentum(df, inst, events, **kw):
    return _generate(df, inst, events, "surprise", pd.Timedelta(days=30), 4, "surprise_momentum", **kw)


def cb_divergence(df, inst, events, **kw):
    return _generate(df, inst, events, "rates", pd.Timedelta(days=180), 2, "cb_divergence", **kw)


STRATEGIES = {"surprise_momentum": surprise_momentum, "cb_divergence": cb_divergence}
