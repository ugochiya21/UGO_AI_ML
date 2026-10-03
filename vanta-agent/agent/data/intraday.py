"""Free intraday bars (5- or 15-minute), built from 1-minute data, cached
under data/prices5/ and data/prices15/.

  * Forex                                     -> FXCM's free 1-minute candle
    files (weekly, bid AND ask - so the spread is known; no holes found)
  * Gold, silver, oil, S&P 500, Nasdaq        -> HistData.com free 1-minute
    files (one zip per year). WARNING: ~27 days of minutes missing per year
    in short holes, news hours included - treat results with care.
  (Dukascopy's minute data comes one day per request and blocks fast
   clients, so it isn't practical here.)
  * Crypto                                    -> Bitstamp public OHLC, 5/15-minute

HistData times are New York standard time all year (UTC-5, no daylight
saving); they are converted to UTC here. FXCM times are UTC. Prices are
BID; FXCM bars also carry `spread` (median ask - bid in the bar).
All bars are indexed by their OPEN time in UTC.
"""
import gzip
import io
import re
import time as _time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from agent.data.loaders import COLS, _HTTP, _get, bitstamp_ohlc
from agent.instruments import CRYPTO, FOREX, Instrument

CACHE = {"15min": Path("data/prices15"), "5min": Path("data/prices5")}
RAW = Path("data/raw/histdata")

# Vanta symbol -> HistData symbol (forex pairs use their own name).
HISTDATA = {"GOLDUSDC": "XAUUSD", "SILVERUSDC": "XAGUSD", "WTIOILUSDC": "WTIUSD",
            "SP500USDC": "SPXUSD", "XYZ100USDC": "NSXUSD"}
PAGE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{s}/{y}"


def load(inst: Instrument, start: str, end: str, refresh: bool = False,
         rule: str = "15min") -> pd.DataFrame:
    tag = {"15min": "15m", "5min": "5m"}[rule]
    path = CACHE[rule] / f"{inst.symbol}_{tag}.csv.gz"
    df = None
    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
    want_end = min(pd.Timestamp(end, tz="UTC"), pd.Timestamp.now(tz="UTC")) - pd.Timedelta(days=7)
    if df is not None and (df.index[-1] < want_end or
                           df.index[0] > pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=7)):
        df = None
    if df is None:
        if inst.asset_class == CRYPTO:
            df = bitstamp_ohlc(inst.data_symbol, start, end,
                               step=int(pd.Timedelta(rule).total_seconds()))
        elif inst.asset_class == FOREX:
            df = fxcm_bars(inst.symbol, start, end, rule)
            if df.empty:   # FXCM doesn't carry every cross (CHFJPY, EURCAD, GBPAUD)
                df = histdata_bars(inst.symbol, start, end, rule)
        else:
            df = histdata_bars(HISTDATA.get(inst.symbol, inst.symbol), start, end, rule)
        if df.empty:
            raise ValueError(f"no {tag} bars downloaded for {inst.symbol}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")   # write then rename: readers never see half a file
        df.to_csv(tmp, compression="gzip")
        tmp.replace(path)
    return df.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")]


def histdata_year(symbol: str, year: int, month: int | None = None) -> bytes:
    """One year (or, for the current year, one month) of 1-minute bars as
    the zip HistData serves. Kept under data/raw/histdata/ so it is only
    downloaded once - except the current month, which isn't final yet."""
    tag = f"{year}" if month is None else f"{year}{month:02d}"
    kept = RAW / f"{symbol}_{tag}.zip"
    if kept.exists():
        return kept.read_bytes()
    page_url = PAGE.format(s=symbol.lower(), y=year) + ("" if month is None else f"/{month}")
    page = _get(page_url)
    if page is None:
        return b""
    form = dict(re.findall(r'<input type="hidden" name="(\w+)" id="\w+" value="([^"]*)"', page.text))
    r = _HTTP.post("https://www.histdata.com/get.php", data=form,
                   headers={"Referer": page_url}, timeout=180)
    _time.sleep(2)
    if r.status_code != 200 or not r.content.startswith(b"PK"):
        return b""
    now = pd.Timestamp.now(tz="UTC")
    if (year, month or 12) < (now.year, now.month):
        RAW.mkdir(parents=True, exist_ok=True)
        kept.write_bytes(r.content)
    return r.content


def histdata_bars(symbol: str, start: str, end: str, rule: str = "15min") -> pd.DataFrame:
    parts, missing = [], []
    now = pd.Timestamp.now(tz="UTC")
    chunks = []
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        if year < now.year:
            chunks.append((year, None))
        else:   # the current year only exists as monthly files
            chunks += [(year, m) for m in range(1, now.month + 1)]
    for year, month in chunks:
        content = histdata_year(symbol, year, month)
        if not content:
            missing.append(year if month is None else f"{year}-{month:02d}")
            continue
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            m1 = pd.read_csv(z.open(name), sep=";", header=None,
                             names=["time", *COLS, "volume"], usecols=[0, 1, 2, 3, 4])
        m1.index = (pd.to_datetime(m1.pop("time"), format="%Y%m%d %H%M%S")
                    + pd.Timedelta(hours=5)).dt.tz_localize("UTC")
        parts.append(to_bars(m1, rule))
    if missing:
        print(f"    {symbol}: no HistData file for {missing}")
    if not parts:
        return pd.DataFrame(columns=COLS)
    df = pd.concat(parts)
    return df[~df.index.duplicated()].sort_index()


def to_bars(m1: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    df = m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return df.dropna()


# ------------------------------------------------------------------- FXCM
FXCM_RAW = Path("data/raw/fxcm")
FXCM_URL = "https://candledata.fxcorporate.com/m1/{s}/{y}/{w}.csv.gz"


def fxcm_week(symbol: str, year: int, week: int) -> bytes:
    kept = FXCM_RAW / symbol / f"{year}-{week:02d}.csv.gz"
    gone = kept.with_suffix(".missing")
    if kept.exists():
        return kept.read_bytes()
    if gone.exists():
        return b""
    r = _get(FXCM_URL.format(s=symbol, y=year, w=week))
    kept.parent.mkdir(parents=True, exist_ok=True)
    if r is None:
        gone.touch()   # 404: no such week; don't ask again
        return b""
    kept.write_bytes(r.content)
    return r.content


def fxcm_bars(symbol: str, start: str, end: str, rule: str = "15min",
              workers: int = 8) -> pd.DataFrame:
    weeks = [(y, w) for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1)
             for w in range(1, 54)]
    with ThreadPoolExecutor(workers) as ex:
        blobs = list(ex.map(lambda yw: fxcm_week(symbol, *yw), weeks))
    parts = []
    for (y, w), b in zip(weeks, blobs):
        if not b:
            continue
        try:
            d = pd.read_csv(io.BytesIO(gzip.decompress(b)))
        except (OSError, EOFError, ValueError):
            # Damaged download: fetch it once more, else skip that week.
            (FXCM_RAW / symbol / f"{y}-{w:02d}.csv.gz").unlink(missing_ok=True)
            try:
                d = pd.read_csv(io.BytesIO(gzip.decompress(fxcm_week(symbol, y, w))))
            except (OSError, EOFError, ValueError):
                print(f"    {symbol}: week {y}-{w:02d} unreadable, skipped")
                continue
        d.index = pd.to_datetime(d.pop("DateTime"), format="%m/%d/%Y %H:%M:%S.%f").dt.tz_localize("UTC")
        m1 = pd.DataFrame({"open": d.BidOpen, "high": d.BidHigh, "low": d.BidLow,
                           "close": d.BidClose, "spread": d.AskClose - d.BidClose})
        parts.append(m1)
    if not parts:
        return pd.DataFrame(columns=[*COLS, "spread"])
    m1 = pd.concat(parts)
    m1 = m1[~m1.index.duplicated()].sort_index()
    df = m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "median"})
    df = df.dropna(subset=["close"])
    return df.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
