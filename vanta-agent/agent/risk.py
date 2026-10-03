"""Risk engine. Every trade idea must get through `approve()`; it can say no
for many reasons and only ever says yes with a size attached.

Owner's rules implemented here:
  * Total risk of all open trades <= 1% of the account, however many trades.
  * New trades only use budget that is actually free - a trade's slot frees
    when it closes (TP), not when its stop reaches breakeven - and only for
    top-quality setups.
  * After a stop-loss: wait for (or trade small until) two TPs.
"""
from dataclasses import dataclass

import pandas as pd

from agent.config import AgentConfig
from agent.instruments import CRYPTO, Instrument, get
from agent.models import Position, Signal


@dataclass
class Decision:
    approved: bool
    reason: str
    notional: float = 0.0
    risk_usd: float = 0.0


class RiskManager:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.start = cfg.rules.account_size

    # ---- budget -----------------------------------------------------------
    def open_risk(self, positions: list[Position]) -> float:
        if self.cfg.risk.breakeven_frees_budget:
            return sum(p.risk_usd() for p in positions)
        return sum(p.initial_risk_usd() for p in positions)

    def risk_budget(self) -> float:
        return self.cfg.risk.max_open_risk_pct * self.start

    def free_budget(self, positions) -> float:
        return max(0.0, self.risk_budget() - self.open_risk(positions))

    def per_trade_risk(self, balance: float, equity: float, consecutive_losses: int,
                       recovering: bool = False) -> float:
        r = self.cfg.risk
        pct = r.risk_per_trade_pct
        if (equity < self.start * r.reduce_risk_below          # in drawdown
                or balance >= self.start * r.protect_target_above  # protect a near-pass
                or consecutive_losses >= 2                      # cool off after losses
                or recovering):                                 # stop-loss, awaiting 2 TPs
            pct = r.reduced_risk_per_trade_pct
        # Never risk more than the distance left to the target needs - but
        # not less than the smallest trade we take, or a balance a few cents
        # short of the target can never trade again (and Vanta eliminates
        # accounts with no order for 60 days).
        to_target = self.start * (1 + self.cfg.rules.profit_target_pct) - balance
        return min(pct * self.start, max(to_target + 1.0, self.min_trade_risk()))

    def min_trade_risk(self) -> float:
        return 0.25 * self.cfg.risk.reduced_risk_per_trade_pct * self.start

    # ---- the gate -----------------------------------------------------------
    def approve(self, sig: Signal, inst: Instrument, positions: list[Position], *,
                balance: float, equity: float, day_open_equity: float,
                now: pd.Timestamp, consecutive_losses: int = 0,
                tps_owed: int = 0) -> Decision:
        """`tps_owed`: TPs still needed since the last stop-loss."""
        r = self.cfg.risk
        no = lambda why: Decision(False, why)  # noqa: E731
        if tps_owed and r.after_stop_loss == "wait_2tp":
            return no("after a stop-loss: waiting for two TPs")

        if sig.score < r.min_setup_score:
            return no(f"score {sig.score} < {r.min_setup_score}")
        if sig.reward_risk < r.min_reward_risk:
            return no(f"reward:risk {sig.reward_risk:.2f} < {r.min_reward_risk}")
        if sig.direction * (sig.entry - sig.stop) <= 0:
            return no("stop on wrong side of entry")
        if equity < self.start * r.stop_trading_below:
            return no("equity below safety line - trading halted")
        if equity < day_open_equity * (1 - r.daily_loss_stop_pct):
            return no("daily loss stop hit")
        if any(p.symbol == sig.symbol for p in positions):
            return no("already have a position in this market")

        # Correlation: never two open trades that are the same bet.
        new_exp = inst.signed_exposure(sig.direction)
        for p in positions:
            shared = new_exp & get(p.symbol).signed_exposure(p.direction)
            if shared:
                return no(f"correlated with open {p.symbol} via {sorted(shared)[0][0]}")

        if now.weekday() >= 5 and inst.asset_class != CRYPTO:
            return no("market closed (weekend)")
        if (now.weekday() == 4 and now.hour >= r.friday_flat_hour_utc - 4
                and inst.asset_class != CRYPTO):
            return no("too close to the Friday close")

        risk_usd = self.per_trade_risk(balance, equity, consecutive_losses,
                                       recovering=bool(tps_owed) and r.after_stop_loss == "half_2tp")
        if now.weekday() >= 5 and inst.asset_class == CRYPTO:
            risk_usd *= r.crypto_weekend_risk_factor
        risk_usd = min(risk_usd, self.free_budget(positions))
        if risk_usd < self.min_trade_risk():
            return no("no free risk budget (1% rule)")

        stop_frac = abs(sig.entry - sig.stop) / sig.entry
        notional = risk_usd / stop_frac
        cap = inst.max_leverage * self.start
        if notional > cap:  # Vanta would clamp anyway; risk ends up smaller
            notional = cap
            risk_usd = notional * stop_frac
        return Decision(True, "ok", notional, risk_usd)
