"""All tunable settings in one place.

Two groups:
  * ChallengeRules - Vanta's own rules (copied from taoshidev/vanta-network
    vali_objects/vali_config.py). Breaking these fails the account.
  * RiskConfig     - our own, stricter rules. They keep us far away from
    Vanta's limits.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChallengeRules:
    account_size: float = 5_000.0
    # Pass when *realized* balance reaches +10% (all-markets challenge).
    profit_target_pct: float = 0.10
    # Rule 1: equity incl. open P&L more than 5% below starting balance -> fail.
    static_drawdown_pct: float = 0.05
    # Rule 2: equity more than 5% below the day's opening equity -> fail.
    # The day starts at 00:00 UTC.
    intraday_drawdown_pct: float = 0.05
    # No order for 60 days -> eliminated.
    inactivity_days: int = 60


@dataclass(frozen=True)
class RiskConfig:
    # --- The owner's core rule -------------------------------------------
    # Combined risk of ALL open trades (distance to stop-loss) never above
    # 1% of the account. A trade whose stop is at breakeven risks nothing,
    # which frees budget for the next good setup.
    max_open_risk_pct: float = 0.01

    # Risk per single trade (fraction of starting balance).
    risk_per_trade_pct: float = 0.005
    # Smaller risk when protecting a near-pass or recovering from a loss.
    reduced_risk_per_trade_pct: float = 0.0025

    # Move stop to breakeven once price has moved this many R in our favour.
    breakeven_at_r: float = 1.0
    # Minimum reward:risk for a trade to be considered.
    min_reward_risk: float = 2.0
    # Only setups scoring at least this (out of 10) are traded.
    min_setup_score: int = 8

    # --- Safety lines (fractions of starting balance) ---------------------
    # Vanta fails us at -5% ($4,750). We act much earlier.
    reduce_risk_below: float = 0.975   # $4,875: switch to reduced risk
    stop_trading_below: float = 0.96   # $4,800: no new trades at all
    # Daily loss stop measured from the day's opening equity.
    daily_loss_stop_pct: float = 0.01
    # Near the target, shrink risk so the pass is not given back.
    protect_target_above: float = 1.08  # $5,400 realized

    # --- News (owner's rule) ----------------------------------------------
    news_close_before_min: int = 30   # flatten related trades 30 min before
    news_block_after_min: int = 30    # no new related trades until 30 min after
    news_impacts: tuple = ("High",)

    # --- Weekend -----------------------------------------------------------
    # Flatten non-crypto positions after this hour (UTC) on Friday.
    friday_flat_hour_utc: int = 20
    crypto_weekend_risk_factor: float = 0.5

    # Max positions sharing the same directional exposure (e.g. two trades
    # that are both short USD). 1 = never stack correlated bets.
    max_positions_per_exposure: int = 1


@dataclass
class AgentConfig:
    rules: ChallengeRules = field(default_factory=ChallengeRules)
    risk: RiskConfig = field(default_factory=RiskConfig)
