"""A faithful copy of how Vanta judges a standard (static) challenge account.

Source: taoshidev/vanta-network
  * vali_objects/vali_config.py      SUBACCOUNT_STATIC_* thresholds, 10% target
  * challenge_period/...manager.py   intraday drawdown from day-open equity
  * entity_manager.py                progress = realized balance / account size
"""
import pandas as pd

from agent.config import ChallengeRules

ACTIVE, PASSED, FAILED = "active", "passed", "failed"


class ChallengeMonitor:
    def __init__(self, rules: ChallengeRules):
        self.rules = rules
        self.start = rules.account_size
        self.status = ACTIVE
        self.reason = ""
        self.day = None
        self.day_open_equity = self.start
        self.ended_at = None
        self.worst_static_dd = 0.0
        self.worst_intraday_dd = 0.0

    @property
    def fail_equity(self) -> float:
        return self.start * (1 - self.rules.static_drawdown_pct)

    @property
    def target_balance(self) -> float:
        return self.start * (1 + self.rules.profit_target_pct)

    def intraday_floor(self) -> float:
        return self.day_open_equity * (1 - self.rules.intraday_drawdown_pct)

    def new_day_if_needed(self, t: pd.Timestamp, equity: float):
        d = t.date()
        if d != self.day:
            self.day = d
            self.day_open_equity = equity

    def check(self, t: pd.Timestamp, equity: float, balance: float) -> str:
        """`equity` must be the WORST equity seen since the last check
        (open P&L included). `balance` is realized balance."""
        if self.status != ACTIVE:
            return self.status
        self.worst_static_dd = max(self.worst_static_dd, 1 - equity / self.start)
        self.worst_intraday_dd = max(self.worst_intraday_dd, 1 - equity / self.day_open_equity)
        if equity < self.fail_equity:
            self.status, self.reason = FAILED, f"static drawdown: equity {equity:.2f} < {self.fail_equity:.2f}"
        elif equity < self.intraday_floor():
            self.status, self.reason = FAILED, (f"intraday drawdown: equity {equity:.2f} < "
                                                f"{self.intraday_floor():.2f} (day open {self.day_open_equity:.2f})")
        elif balance >= self.target_balance:
            self.status, self.reason = PASSED, f"target reached: balance {balance:.2f}"
        if self.status != ACTIVE:
            self.ended_at = t
        return self.status
