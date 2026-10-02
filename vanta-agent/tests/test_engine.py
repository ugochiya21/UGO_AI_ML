"""Scenario tests: hand-made price paths with one signal each, checking the
engine closes trades exactly when the rules say."""
import pandas as pd
import pytest

from agent.backtest.engine import Backtester
from agent.config import AgentConfig
from agent.indicators import align, resample
from agent.news.calendar import Event
from agent.news.guard import NewsGuard

START = pd.Timestamp("2024-03-05 00:00", tz="UTC")   # Tuesday


def bars(path):
    """path: list of (open, high, low, close)."""
    idx = pd.date_range(START, periods=len(path), freq="1h")
    return pd.DataFrame(path, columns=["open", "high", "low", "close"], index=idx)


def one_signal(at_bar, d=1, stop=0.995, tp=1.0125, score=9):
    t = START + pd.Timedelta(hours=at_bar)
    return pd.DataFrame({"direction": [d], "stop": [stop], "take_profit": [tp],
                         "score": [score], "strategy": ["test"]}, index=[t])


def run(path, sig, news=None):
    bt = Backtester(AgentConfig(), {"EURUSD": bars(path)}, {"EURUSD": sig}, news=news)
    return bt.run(back_to_back=False)


FLAT = (1.0, 1.0005, 0.9995, 1.0)


def test_target_hit():
    path = [FLAT] * 3 + [(1.0, 1.013, 0.999, 1.012)] + [FLAT] * 2
    res = run(path, one_signal(1))
    t = res.trades[0]
    assert t.reason == "target"
    assert t.r_multiple == pytest.approx(2.5, rel=0.01)


def test_stop_hit_when_both_touched():
    path = [FLAT] * 3 + [(1.0, 1.02, 0.99, 1.0)] + [FLAT]
    res = run(path, one_signal(1))
    assert res.trades[0].reason == "stop"
    assert res.trades[0].pnl == pytest.approx(-25, rel=0.01)


def test_breakeven_then_stopped_flat():
    path = [FLAT] * 3 + [(1.0, 1.006, 0.9995, 1.004), (1.004, 1.0045, 0.998, 0.999)]
    res = run(path, one_signal(1))
    assert res.trades[0].reason == "breakeven"
    assert abs(res.trades[0].pnl) < 1


def test_news_closes_trade_before_event():
    ev = [Event(START + pd.Timedelta(hours=4, minutes=15), "USD", "High", "NFP")]
    res = run([FLAT] * 6, one_signal(1), news=NewsGuard(ev))
    assert res.trades[0].reason == "news"
    assert res.trades[0].closed_at <= ev[0].time - pd.Timedelta(minutes=30)


def test_news_blocks_entry():
    ev = [Event(START + pd.Timedelta(hours=2, minutes=10), "USD", "High", "CPI")]
    res = run([FLAT] * 6, one_signal(1), news=NewsGuard(ev))
    assert res.trades == []


def test_align_never_uses_unfinished_bar():
    idx = pd.date_range(START, periods=8, freq="1h")
    df = pd.DataFrame({"open": range(8), "high": range(8), "low": range(8),
                       "close": [float(i) for i in range(8)]}, index=idx)
    h4 = resample(df, "4h")
    a = align(h4["close"], idx)
    # The first 4h bar (hours 0-3) closes at 04:00, i.e. at the close of the 03:00 bar.
    assert pd.isna(a.iloc[2])
    assert a.iloc[3] == 3.0
    assert a.iloc[6] == 3.0 and a.iloc[7] == 7.0
