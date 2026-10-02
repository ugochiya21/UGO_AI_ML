"""The owner's news rule:

  * Close every trade related to a currency 30 minutes BEFORE a high-impact
    event for that currency.
  * Open no new related trade until 30 minutes AFTER the event.

"Related" = the instrument lists the event's currency in news_currencies
(events tagged "All" affect everything).
"""
from bisect import bisect_left
from collections import defaultdict

import pandas as pd

from agent.instruments import Instrument
from agent.news.calendar import Event


class NewsGuard:
    def __init__(self, events: list[Event], close_before_min: int = 30,
                 block_after_min: int = 30):
        self.before = pd.Timedelta(minutes=close_before_min)
        self.after = pd.Timedelta(minutes=block_after_min)
        by_ccy = defaultdict(list)
        for e in events:
            by_ccy[e.currency].append(e)
        self._events = {c: sorted(v, key=lambda e: e.time) for c, v in by_ccy.items()}
        self._times = {c: [e.time for e in v] for c, v in self._events.items()}

    def _related(self, inst: Instrument):
        return (*inst.news_currencies, "All")

    def _between(self, ccy, lo, hi) -> list[Event]:
        times = self._times.get(ccy)
        if not times:
            return []
        i = bisect_left(times, lo)
        out = []
        while i < len(times) and times[i] <= hi:
            out.append(self._events[ccy][i])
            i += 1
        return out

    def events_between(self, inst: Instrument, lo, hi) -> list[Event]:
        return [e for c in self._related(inst) for e in self._between(c, lo, hi)]

    def must_flatten(self, inst: Instrument, now, horizon=pd.Timedelta(0)) -> list[Event]:
        """Events that require closing `inst` now. `horizon` widens the look-ahead
        for bar-based backtests (we only act once per bar)."""
        return self.events_between(inst, now, now + self.before + horizon)

    def blocks_entry(self, inst: Instrument, now) -> list[Event]:
        """Events that forbid opening `inst` now (from 30 min before to 30 min after)."""
        return self.events_between(inst, now - self.after, now + self.before)

    def next_event(self, inst: Instrument, now) -> Event | None:
        ups = self.events_between(inst, now, now + pd.Timedelta(days=7))
        return min(ups, key=lambda e: e.time) if ups else None
