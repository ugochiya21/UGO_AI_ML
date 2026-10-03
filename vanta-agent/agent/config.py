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
    # 1% of the account. New trades only after a TP (see
    # breakeven_frees_budget and after_stop_loss below).
    max_open_risk_pct: float = 0.01

    # Risk per single trade (fraction of starting balance). Owner: many best
    # setups may share the 1% - 0.2% ($10) each, so up to 5 trades at once.
    risk_per_trade_pct: float = 0.002
    # Smaller risk when protecting a near-pass or recovering from a loss.
    reduced_risk_per_trade_pct: float = 0.001

    # Move stop to breakeven once price has moved this many R in our favour.
    # None = never move the stop (owner's decision after the backtests).
    breakeven_at_r: float | None = None
    # Owner's rule: a new trade only after a trade hits TP. A trade sitting
    # at breakeven still holds its slot until it closes. Set True to let a
    # breakeven trade free its budget early (the original design).
    breakeven_frees_budget: bool = False
    # Owner's rule after a stop-loss:
    #   "wait_tp"  - no new trade until one of the still-open trades hits TP;
    #                if none are left open, look for new setups again
    #   "wait_2tp" - no new trade until two trades hit TP (stalls: tested,
    #                fails every challenge on Vanta's 60-day inactivity rule)
    #   "half_2tp" - keep trading at reduced size until two TPs come in
    #   "none"     - no special treatment
    after_stop_loss: str = "wait_tp"
    tps_to_recover: int = 2
    # Minimum reward:risk for a trade to be considered.
    min_reward_risk: float = 2.0
    # Only setups scoring at least this (out of 10) are traded.
    min_setup_score: int = 8

    # --- Safety lines (fractions of starting balance) ---------------------
    # Vanta fails us at -5% ($4,750). We act much earlier.
    reduce_risk_below: float = 0.975   # $4,875: switch to reduced risk
    # $4,800: no new trades at all. Off (None) by the owner's rule "never stop
    # trading": a halted account is eliminated after 60 days without an order.
    stop_trading_below: float | None = None
    # Daily loss stop measured from the day's opening equity.
    daily_loss_stop_pct: float = 0.01
    # Near the target, shrink risk so the pass is not given back.
    protect_target_above: float = 1.08  # $5,400 realized

    # --- News (owner's rule) ----------------------------------------------
    news_close_before_min: int = 30   # flatten related trades 30 min before
    news_block_after_min: int = 30    # no new related trades until 30 min after
    news_impacts: tuple = ("High",)
    # Also flatten/block markets CORRELATED with the news currency, not
    # just the ones that contain it (owner's rule).
    news_include_correlated: bool = True
    # Owner: only enter if the next related high-impact event is at least
    # this many hours away (time a trade typically needs to reach TP).
    # None = off.
    min_hours_before_news: float | None = None

    # --- Weekend -----------------------------------------------------------
    # Flatten non-crypto positions after this hour (UTC) on Friday.
    friday_flat_hour_utc: int = 20
    # False = hold positions over the weekend (Vanta allows it).
    flat_on_weekend: bool = True
    crypto_weekend_risk_factor: float = 0.5

    # Max positions sharing the same directional exposure (e.g. two trades
    # that are both short USD). 1 = never stack correlated bets.
    max_positions_per_exposure: int = 1


@dataclass
class AgentConfig:
    rules: ChallengeRules = field(default_factory=ChallengeRules)
    risk: RiskConfig = field(default_factory=RiskConfig)
