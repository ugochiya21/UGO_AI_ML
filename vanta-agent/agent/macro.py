"""Fundamental / macro layer.

1. CalendarSurprise (backtestable, no look-ahead):
   Each released high-impact number is compared with its forecast. Better
   than expected = +1 for that currency, worse = -1 (inverted for numbers
   where lower is better, like unemployment). Over the last N days this
   gives each currency a "fundamental momentum" score. A forex trade that
   bets AGAINST a currency with strongly positive momentum (or FOR one with
   strongly negative momentum) is vetoed.

2. BiasFile (live): a small JSON file the news/geopolitics analyst (an LLM
   reading headlines, or you) keeps up to date, e.g.
       {"risk_mode": "off", "avoid": ["SOLUSDC"],
        "bias": {"GOLDUSDC": 1, "USD": 1}, "note": "Middle East escalation"}
   Trades against a stated bias, or in an avoided market, are vetoed.
   risk_mode "off" blocks new longs in crypto, indices and stocks.
"""
import json
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import pandas as pd

from agent.instruments import CRYPTO, EQUITIES, FOREX, INDICES, Instrument
from agent.news.calendar import Event

LOWER_IS_BETTER = re.compile(r"unemployment|jobless|claimant|initial claims", re.I)


def _num(s: str) -> float | None:
    m = re.match(r"^\s*([-+]?\d*\.?\d+)\s*([KMBT%]?)", s or "")
    if not m:
        return None
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(m.group(2), 1)
    return float(m.group(1)) * mult


class CalendarSurprise:
    def __init__(self, events: list[Event], window_days: int = 14, threshold: int = 3):
        self.window = pd.Timedelta(days=window_days)
        self.threshold = threshold
        per = defaultdict(list)
        for e in events:
            a, f = _num(e.actual), _num(e.forecast)
            if a is None or f is None or a == f:
                continue
            sign = 1 if a > f else -1
            if LOWER_IS_BETTER.search(e.title):
                sign = -sign
            per[e.currency].append((e.time, sign))
        self._t = {c: [x[0] for x in sorted(v)] for c, v in per.items()}
        self._cum = {}
        for c, v in per.items():
            run, acc = [], 0
            for _, s in sorted(v):
                acc += s
                run.append(acc)
            self._cum[c] = run

    def score(self, ccy: str, now) -> int:
        """Net surprises for `ccy` released in (now - window, now]."""
        ts = self._t.get(ccy)
        if not ts:
            return 0
        hi = bisect_right(ts, now) - 1
        lo = bisect_right(ts, now - self.window) - 1
        if hi < 0:
            return 0
        return self._cum[ccy][hi] - (self._cum[ccy][lo] if lo >= 0 else 0)

    def veto(self, inst: Instrument, direction: int, now) -> str | None:
        if inst.asset_class != FOREX:
            return None
        base, quote = inst.news_currencies
        edge = (self.score(base, now) - self.score(quote, now)) * direction
        if edge <= -self.threshold:
            return f"fundamentals against ({base} vs {quote} surprises {edge:+d})"
        return None


class BiasFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def veto(self, inst: Instrument, direction: int, now) -> str | None:
        b = self._load()
        if inst.symbol in b.get("avoid", []):
            return "market on avoid list"
        bias = b.get("bias", {})
        if bias.get(inst.symbol, 0) * direction < 0:
            return f"against stated bias on {inst.symbol}"
        if inst.asset_class == FOREX:
            base, quote = inst.news_currencies
            if (bias.get(base, 0) - bias.get(quote, 0)) * direction < 0:
                return "against stated currency bias"
        if (b.get("risk_mode") == "off" and direction == 1
                and inst.asset_class in (CRYPTO, INDICES, EQUITIES)):
            return "risk-off: no new longs in risk assets"
        return None


class Combined:
    def __init__(self, *layers):
        self.layers = [x for x in layers if x is not None]

    def veto(self, inst, direction, now):
        for layer in self.layers:
            v = layer.veto(inst, direction, now)
            if v:
                return v
        return None
