from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Signal:
    """A trade idea produced by a strategy. Nothing is traded until the risk
    engine approves it and sizes it."""
    symbol: str
    direction: int            # +1 long, -1 short
    entry: float
    stop: float
    take_profit: float
    strategy: str
    score: int                # 0-10 setup quality
    time: datetime
    reasons: list = field(default_factory=list)

    @property
    def reward_risk(self) -> float:
        risk = abs(self.entry - self.stop)
        return abs(self.take_profit - self.entry) / risk if risk else 0.0


@dataclass
class Position:
    symbol: str
    direction: int
    entry: float
    stop: float
    take_profit: float
    notional: float           # USD value of the position
    opened_at: datetime
    strategy: str
    score: int = 0
    initial_stop: float = 0.0
    fees: float = 0.0
    carry: float = 0.0
    # Sizing record: trades open (this one included) and their combined
    # risk when this trade was opened.
    opened_with: int = 1
    budget_used: float = 0.0
    # Intraday strategies: close at this time (end of session) if neither
    # the stop nor the TP has been hit.
    exit_at: object = None

    def __post_init__(self):
        if not self.initial_stop:
            self.initial_stop = self.stop

    def pnl(self, price: float) -> float:
        return self.direction * (price - self.entry) / self.entry * self.notional

    def risk_usd(self) -> float:
        """Money lost if the stop is hit now. Zero once the stop is at or
        beyond breakeven - that trade no longer uses risk budget."""
        loss = self.direction * (self.entry - self.stop) / self.entry * self.notional
        return max(0.0, loss)

    def initial_risk_usd(self) -> float:
        return abs(self.entry - self.initial_stop) / self.entry * self.notional

    def r_multiple(self, price: float) -> float:
        r = self.initial_risk_usd()
        return self.pnl(price) / r if r else 0.0


@dataclass
class ClosedTrade:
    symbol: str
    direction: int
    strategy: str
    score: int
    opened_at: datetime
    closed_at: datetime
    entry: float
    exit: float
    notional: float
    pnl: float                # net of fees and carry
    r_multiple: float
    reason: str               # "stop", "target", "breakeven", "news", "weekend", ...
    risk_usd: float = 0.0     # money at risk when opened (entry to stop)
    stop: float = 0.0
    take_profit: float = 0.0
    opened_with: int = 1      # open trades at entry, this one included
    budget_used: float = 0.0  # their combined risk at entry (<= 1% = $50)
