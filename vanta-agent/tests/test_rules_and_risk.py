import pandas as pd
import pytest

from agent.config import AgentConfig
from agent.instruments import get
from agent.models import Position, Signal
from agent.risk import RiskManager
from agent.rules import FAILED, PASSED, ChallengeMonitor

T = pd.Timestamp("2024-03-05 10:00", tz="UTC")   # a Tuesday


def sig(symbol="EURUSD", d=1, entry=1.10, stop=1.095, tp=1.1125, score=9):
    return Signal(symbol, d, entry, stop, tp, "test", score, T)


def pos(symbol, d, entry, stop, notional=5000):
    return Position(symbol, d, entry, stop, entry + d * 0.02, notional, T, "test")


# ------------------------------------------------------------ Vanta rules
def test_static_drawdown_fails_below_4750():
    m = ChallengeMonitor(AgentConfig().rules)
    m.new_day_if_needed(T, 5000)
    assert m.check(T, 4751, 5000) == "active"
    assert m.check(T, 4749, 5000) == FAILED


def test_intraday_drawdown_from_day_open():
    m = ChallengeMonitor(AgentConfig().rules)
    m.new_day_if_needed(T, 5400)          # up 8%, day opens at 5,400
    assert m.check(T, 5131, 5400) == "active"
    assert m.check(T, 5129, 5400) == FAILED   # 5% below 5,400 although above 4,750
    assert "intraday" in m.reason


def test_pass_needs_realized_balance():
    m = ChallengeMonitor(AgentConfig().rules)
    m.new_day_if_needed(T, 5000)
    assert m.check(T, 5600, 5499) == "active"   # open profit doesn't count
    assert m.check(T, 5600, 5500) == PASSED


def test_inactivity_fails_after_60_days_without_an_order():
    m = ChallengeMonitor(AgentConfig().rules)
    m.new_day_if_needed(T, 5000)
    assert m.check(T, 5000, 5000) == "active"
    m.order_placed(T + pd.Timedelta(days=30))
    assert m.check(T + pd.Timedelta(days=89), 4790, 4790) == "active"
    assert m.check(T + pd.Timedelta(days=90), 4790, 4790) == FAILED
    assert "inactivity" in m.reason


# ------------------------------------------------------------ risk engine
@pytest.fixture
def rm():
    return RiskManager(AgentConfig())


def approve(rm, s, positions=(), balance=5000, equity=5000, day_open=5000, now=T, losses=0):
    return rm.approve(s, get(s.symbol), list(positions), balance=balance, equity=equity,
                      day_open_equity=day_open, now=now, consecutive_losses=losses)


def test_sizes_trade_to_half_percent(rm):
    d = approve(rm, sig())
    assert d.approved
    assert d.risk_usd == pytest.approx(25.0)
    # 25 / (0.005/1.10) = 5,500 notional
    assert d.notional == pytest.approx(5500, rel=1e-6)


def test_total_open_risk_never_above_one_percent(rm):
    # Two open trades already risking $25 each = $50 = 1%.
    p1 = pos("SP500USDC", 1, 5000, 4975, notional=5000)   # 0.5% of 5000 = 25
    p2 = pos("BTCUSDC", 1, 50000, 49500, notional=2500)   # 1% * 2500 = 25
    assert rm.open_risk([p1, p2]) == pytest.approx(50)
    d = approve(rm, sig(), [p1, p2])
    assert not d.approved and "1% rule" in d.reason


def test_breakeven_trade_frees_budget(rm):
    p1 = pos("SP500USDC", 1, 5000, 5000, notional=5000)   # stop at entry -> risk 0
    p2 = pos("BTCUSDC", 1, 50000, 49500, notional=2500)
    assert rm.open_risk([p1, p2]) == pytest.approx(25)
    d = approve(rm, sig(), [p1, p2])
    assert d.approved and d.risk_usd <= 25 + 1e-9


def test_rejects_low_score_and_poor_reward(rm):
    assert not approve(rm, sig(score=7)).approved
    assert not approve(rm, sig(tp=1.105)).approved   # only 1R


def test_rejects_correlated_trade(rm):
    # Long GBPUSD is short USD; long EURUSD is also short USD -> same bet.
    p = pos("GBPUSD", 1, 1.25, 1.245)
    d = approve(rm, sig(), [p])
    assert not d.approved and "correlated" in d.reason
    # Short EURUSD is long USD -> allowed alongside.
    assert approve(rm, sig(d=-1, stop=1.105, tp=1.0875), [p]).approved


def test_safety_lines(rm):
    assert not approve(rm, sig(), equity=4790).approved          # below $4,800
    small = approve(rm, sig(), equity=4860, day_open=4860)        # below $4,875
    assert small.approved and small.risk_usd == pytest.approx(12.5)
    assert not approve(rm, sig(), equity=5040, day_open=5100).approved  # -1.2% today


def test_protects_near_target(rm):
    d = approve(rm, sig(), balance=5420, equity=5420, day_open=5420)
    assert d.risk_usd == pytest.approx(12.5)


def test_can_still_trade_cents_from_target(rm):
    # 16 cents short used to size every trade below the minimum - no trades
    # ever again, then Vanta's 60-day inactivity rule fails the account.
    d = approve(rm, sig(), balance=5499.84, equity=5499.84, day_open=5499.84)
    assert d.approved
    assert d.risk_usd == pytest.approx(rm.min_trade_risk())


def test_weekend_rules(rm):
    sat = pd.Timestamp("2024-03-09 12:00", tz="UTC")
    assert not approve(rm, sig(), now=sat).approved
    btc = Signal("BTCUSDC", 1, 50000, 49500, 51250, "test", 9, sat)
    d = approve(rm, btc, now=sat)
    assert d.approved and d.risk_usd == pytest.approx(12.5)   # half size on weekends
