"""Economic calendar - free sources only.

History (for backtests): Forex Factory archive 2007-today, rebuilt nightly by
https://github.com/janickfarrell/newfac and published as one CSV. Times are GMT.

Live (this week): Forex Factory's public weekly JSON feed.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

FF_HISTORY_URL = ("https://github.com/janickfarrell/newfac/releases/download/"
                  "calendar-data/forexfactory_calendar.csv")
FF_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


@dataclass(frozen=True)
class Event:
    time: pd.Timestamp        # UTC
    currency: str             # "USD", "EUR", ... or "All"
    impact: str               # "High" | "Medium" | "Low" | "Holiday"
    title: str
    actual: str = ""
    forecast: str = ""
    previous: str = ""


def download_history(dest: str | Path = "data/forexfactory_calendar.csv") -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(FF_HISTORY_URL, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load_history(path: str | Path, impacts=("High",), start=None, end=None) -> list[Event]:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    df = df[df["impact"].isin(impacts)]
    # Rows without a clock time ("All Day", "Tentative") can't be timed; skip.
    df = df[df["time"].str.match(r"^\d{1,2}:\d{2}$")]
    ts = pd.to_datetime(df["date"] + " " + df["time"], format="%a %b %d %Y %H:%M",
                        errors="coerce", utc=True)
    df = df.assign(ts=ts).dropna(subset=["ts"])
    if start is not None:
        df = df[df.ts >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.ts <= pd.Timestamp(end, tz="UTC")]
    return [Event(r.ts, r.currency, r.impact, r.event, r.actual, r.forecast, r.previous)
            for r in df.itertuples()]


def scheduled_only(events: list[Event]) -> list[Event]:
    """Drop events nobody could have known about in advance. The archive also
    records surprises (e.g. the Fed's emergency cut on Sunday 15 Mar 2020,
    unannounced speeches at 21:17); a backtest that closes trades 30 minutes
    before those is cheating. Kept: weekday events on a quarter hour, plus
    Sunday-evening Asia-Pacific releases (Monday morning local time)."""
    def known(e: Event) -> bool:
        wd = e.time.weekday()
        if e.time.minute % 15:
            return False
        if wd == 5:
            return False
        if wd == 6:
            return e.currency in ("AUD", "NZD", "JPY", "CNY") and e.time.hour >= 20
        return True
    return [e for e in events if known(e)]


def fetch_this_week(impacts=("High",)) -> list[Event]:
    r = requests.get(FF_WEEK_URL, timeout=30)
    r.raise_for_status()
    out = []
    for e in r.json():
        if e.get("impact") not in impacts:
            continue
        t = pd.to_datetime(e["date"], utc=True, errors="coerce")
        if pd.isna(t):
            continue
        out.append(Event(t, e.get("country", ""), e["impact"], e.get("title", ""),
                         str(e.get("actual", "") or ""), str(e.get("forecast", "") or ""),
                         str(e.get("previous", "") or "")))
    return out
