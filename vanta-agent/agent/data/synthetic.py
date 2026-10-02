"""Fake but realistic-looking hourly prices, used only for tests and for
checking the machinery works when real data can't be downloaded.
Results on synthetic data say NOTHING about real performance."""
import numpy as np
import pandas as pd

from agent.instruments import CRYPTO, Instrument


def make(inst: Instrument, start="2021-01-01", end="2023-12-31", price=100.0,
         vol=0.0015, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1h", tz="UTC")
    if inst.asset_class != CRYPTO:
        idx = idx[idx.weekday < 5]
    n = len(idx)
    # Regimes of trend / range lasting ~2-8 weeks.
    drift = np.zeros(n)
    i = 0
    while i < n:
        length = rng.integers(300, 1300)
        drift[i:i + length] = rng.choice([-1, 0, 1]) * vol * rng.uniform(0.05, 0.2)
        i += length
    rets = drift + rng.standard_t(4, n) * vol / np.sqrt(2)
    close = price * np.exp(np.cumsum(rets))
    open_ = np.r_[price, close[:-1]]
    wick = np.abs(rng.normal(0, vol * 0.6, (2, n))) * close
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
