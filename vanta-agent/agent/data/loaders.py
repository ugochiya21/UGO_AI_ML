"""Free historical price data, 1-hour bars, cached under data/prices/.

  * Forex, gold, silver, oil, US indices -> Dukascopy (free, 2003+)
  * Crypto                                -> Binance public data API (free)
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
COLS = ["open", "high", "low", "close"]


def _cache_path(sym: str) -> Path:
    return CACHE / f"{sym}_1h.csv.gz"


def load(inst: Instrument, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    path = _cache_path(inst.symbol)
    if path.exists() and not refresh:
        df = read_csv(path)
    else:
        fetch = {"dukascopy": dukascopy, "binance": binance, "yahoo": yahoo}[inst.data_source]
        df = fetch(inst.data_symbol, start, end)
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
        r = requests.get(url, params={"symbol": symbol, "interval": "1h", "startTime": t0,
                                      "endTime": t1, "limit": 1000}, timeout=30)
        r.raise_for_status()
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
    Each record: >IIIIIf = seconds-from-month-start, open, close, low, high, volume."""
    point = _point(symbol)
    rows = []
    for month in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        url = (f"https://datafeed.dukascopy.com/datafeed/{symbol}/{month.year}/"
               f"{month.month - 1:02d}/BID_candles_hour_1.bi5")
        for attempt in range(4):
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                break
            _time.sleep(2 ** attempt)
        if r.status_code != 200 or not r.content:
            continue
        raw = lzma.decompress(r.content)
        base = datetime(month.year, month.month, 1, tzinfo=timezone.utc).timestamp()
        for off, o, c, lo, hi, vol in struct.iter_unpack(">IIIIIf", raw):
            if vol > 0:   # zero-volume hours are market-closed fillers
                rows.append((base + off, o / point, hi / point, lo / point, c / point))
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
