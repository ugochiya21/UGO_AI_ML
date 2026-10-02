"""Free historical price data, 1-hour bars, cached under data/prices/.

  * Forex, gold, silver, oil, US indices -> Dukascopy (free, 2003+)
  * Crypto                                -> Binance public data API (free);
                                             Bitstamp if Binance is blocked
  * US stocks                             -> Yahoo Finance via yfinance
                                             (free; hourly only for ~2 years)
  * Anything else                         -> your own CSV (time,open,high,low,close)

All bars are indexed by their OPEN time in UTC.
"""
import lzma
import struct
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from agent.instruments import Instrument

CACHE = Path("data/prices")
RAW = Path("data/raw/dukascopy")
COLS = ["open", "high", "low", "close"]

# Some data sites reject requests that don't look like a browser.
_HTTP = requests.Session()
_HTTP.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def _get(url: str, params=None, tries: int = 6) -> requests.Response | None:
    """GET with backoff on rate limits (429), server errors and network
    hiccups. Returns None for a definite 404 (file doesn't exist)."""
    for attempt in range(tries):
        try:
            r = _HTTP.get(url, params=params, timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return r
        except requests.ConnectionError:
            if attempt == tries - 1:
                raise
        _time.sleep(min(60, 3 * 2 ** attempt))
    r.raise_for_status()
    return r


def _cache_path(sym: str) -> Path:
    return CACHE / f"{sym}_1h.csv.gz"


def load(inst: Instrument, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    path = _cache_path(inst.symbol)
    df = read_csv(path) if path.exists() and not refresh else None
    # A cache from a shorter run (e.g. ends 2023) must not silently cut a
    # later run short. Yahoo only ever has ~2 years, so don't chase it.
    want_end = min(pd.Timestamp(end, tz="UTC"), pd.Timestamp.now(tz="UTC")) - pd.Timedelta(days=7)
    if df is not None and inst.data_source != "yahoo" and df.index[-1] < want_end:
        df = None
    if df is None:
        fetch = {"dukascopy": dukascopy, "binance": crypto, "yahoo": yahoo}[inst.data_source]
        df = fetch(inst.data_symbol, start, end)
        if df.empty:
            raise ValueError(f"no bars downloaded for {inst.data_symbol}")
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, compression="gzip")
    return df.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")]


def read_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = df.columns[0]
    df.index = pd.to_datetime(df[tcol], utc=True)
    df = df.rename(columns=str.lower)[COLS].astype(float)
    return df[~df.index.duplicated()].sort_index()


# ---------------------------------------------------------------- Binance
def binance(symbol: str, start: str, end: str) -> pd.DataFrame:
    url = "https://data-api.binance.vision/api/v3/klines"
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows = []
    while t0 < t1:
        r = _get(url, params={"symbol": symbol, "interval": "1h", "startTime": t0,
                              "endTime": t1, "limit": 1000}, tries=2)
        batch = r.json()
        if not batch:
            break
        rows += batch
        t0 = batch[-1][0] + 3_600_000
        _time.sleep(0.2)
    df = pd.DataFrame(rows).iloc[:, :5]
    df.columns = ["time", *COLS]
    df.index = pd.to_datetime(df.pop("time"), unit="ms", utc=True)
    return df.astype(float)


def crypto(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Binance first; Bitstamp when Binance can't be reached (some ISPs block it)."""
    try:
        return binance(symbol, start, end)
    except requests.ConnectionError:
        print(f"    Binance unreachable, using Bitstamp for {symbol}")
        return bitstamp(symbol, start, end)


def bitstamp(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Bitstamp public OHLC, 1000 hourly bars per request. Quoted in USD
    (Binance uses USDT; the two differ by well under 0.5%)."""
    pair = symbol.lower().replace("usdt", "usd")
    url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/"
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    t1 = int(pd.Timestamp(end, tz="UTC").timestamp())
    rows = []
    while t0 < t1:
        r = _get(url, params={"step": 3600, "limit": 1000, "start": t0})
        batch = r.json()["data"]["ohlc"] if r is not None else []
        batch = [b for b in batch if int(b["timestamp"]) >= t0]
        if not batch:
            # Pair not listed yet at t0 (e.g. SOL before 2022): skip ahead.
            t0 += 1000 * 3600
            continue
        rows += [(int(b["timestamp"]), *(float(b[c]) for c in COLS)) for b in batch]
        t0 = int(batch[-1]["timestamp"]) + 3600
        _time.sleep(0.3)
    df = pd.DataFrame(rows, columns=["time", *COLS])
    df.index = pd.to_datetime(df.pop("time"), unit="s", utc=True)
    df = df[df.index <= pd.Timestamp(end, tz="UTC")]
    return df[~df.index.duplicated()].sort_index()


# -------------------------------------------------------------- Dukascopy
# Prices are stored as integers; divide by this to get the real price.
_DUKA_POINT = {"JPY": 1e3, "XAUUSD": 1e3, "XAGUSD": 1e3, "IDX": 1e3, "CMD": 1e3}


def _point(symbol: str) -> float:
    for k, v in _DUKA_POINT.items():
        if k in symbol:
            return v
    return 1e5


def dukascopy(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Monthly files of hourly BID candles:
    https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM-1:02}/BID_candles_hour_1.bi5
    Each record: >IIIIIf = seconds-from-month-start, open, close, low, high, volume.

    Each finished month is kept under data/raw/ as soon as it arrives, so an
    interrupted download (Dukascopy blocks fast clients for a while) resumes
    where it stopped instead of starting the market over."""
    point = _point(symbol)
    raw_dir = RAW / symbol
    this_month = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")
    rows, missing = [], []
    for month in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        ym = month.strftime("%Y-%m")
        kept = raw_dir / f"{ym}.bi5"
        if kept.exists():
            content = kept.read_bytes()
        else:
            url = (f"https://datafeed.dukascopy.com/datafeed/{symbol}/{month.year}/"
                   f"{month.month - 1:02d}/BID_candles_hour_1.bi5")
            r = _get(url)
            _time.sleep(1.5)   # be polite: Dukascopy blocks fast clients
            content = r.content if r is not None else b""
            if content and ym < this_month:   # the current month isn't final yet
                raw_dir.mkdir(parents=True, exist_ok=True)
                kept.write_bytes(content)
        if not content:
            missing.append(ym)
            continue
        raw = lzma.decompress(content)
        base = datetime(month.year, month.month, 1, tzinfo=timezone.utc).timestamp()
        for off, o, c, lo, hi, vol in struct.iter_unpack(">IIIIIf", raw):
            if vol > 0:   # zero-volume hours are market-closed fillers
                rows.append((base + off, o / point, hi / point, lo / point, c / point))
    if missing:
        print(f"    {symbol}: no file for {len(missing)} month(s): {', '.join(missing)}")
    df = pd.DataFrame(rows, columns=["time", *COLS])
    df.index = pd.to_datetime(df.pop("time"), unit="s", utc=True)
    return df


# ------------------------------------------------------------------ Yahoo
def yahoo(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf   # optional dependency

    df = yf.download(symbol, interval="1h", period="730d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[COLS]
    df.index = pd.to_datetime(df.index, utc=True)
    return df
