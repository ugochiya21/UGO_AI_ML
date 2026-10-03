"""Event-driven backtester on 1-hour bars across many markets at once.

It replays history hour by hour exactly like the live agent would run:
  * news rule (flatten 30 min before, no entries until 30 min after)
  * Friday flatten for non-crypto
  * stop / target / breakeven management inside each bar
  * Vanta's fees, carry and both drawdown rules, checked on WORST-case
    intrabar equity (pessimistic on purpose)
  * risk engine approval and sizing for every entry

Run in "back-to-back challenge" mode: when a challenge passes or fails, all
trades are closed, the result is recorded, and a fresh $5,000 challenge
starts. The headline result is "N challenges: X passed, Y failed".
"""
import re
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from agent.config import AgentConfig
from agent.instruments import CRYPTO, get
from agent.models import ClosedTrade, Position, Signal
from agent.news.guard import NewsGuard
from agent.risk import RiskManager
from agent.rules import ACTIVE, ChallengeMonitor

BAR = pd.Timedelta("1h")


@dataclass
class ChallengeResult:
    number: int
    started: pd.Timestamp
    ended: pd.Timestamp | None
    status: str
    reason: str
    final_balance: float
    trades: int
    worst_static_dd: float
    worst_intraday_dd: float

    @property
    def days(self) -> float:
        end = self.ended if self.ended is not None else self.started
        return (end - self.started).total_seconds() / 86400


@dataclass
class BacktestResult:
    trades: list[ClosedTrade] = field(default_factory=list)
    challenges: list[ChallengeResult] = field(default_factory=list)
    equity: pd.Series | None = None
    rejections: dict = field(default_factory=dict)


class Backtester:
    def __init__(self, cfg: AgentConfig, bars: dict[str, pd.DataFrame],
                 signals: dict[str, pd.DataFrame], news: NewsGuard | None = None,
                 macro=None):
        self.cfg = cfg
        self.bars = bars
        self.signals = signals
        self.news = news
        self.macro = macro
        self.risk = RiskManager(cfg)

    # ------------------------------------------------------------------
    def run(self, start=None, end=None, back_to_back: bool = True) -> BacktestResult:
        cfg, start_bal = self.cfg, self.cfg.rules.account_size
        times = sorted(set().union(*[df.index for df in self.bars.values()]))
        times = [t for t in times if (start is None or t >= pd.Timestamp(start, tz="UTC"))
                 and (end is None or t <= pd.Timestamp(end, tz="UTC"))]
        # Fast row access: {symbol: {time: (open, high, low, close)}}
        lookup = {s: dict(zip(df.index, df[["open", "high", "low", "close"]]
                              .itertuples(index=False, name=None)))
                  for s, df in self.bars.items()}
        sigs = defaultdict(list)   # {time: [(symbol, row), ...]}
        for s, sdf in self.signals.items():
            for row in sdf.itertuples():
                sigs[row.Index].append((s, row))

        res = BacktestResult()
        positions: list[Position] = []
        last_close: dict[str, float] = {}
        balance = start_bal
        monitor = ChallengeMonitor(cfg.rules)
        ch_start, ch_trades, ch_no = None, 0, 1
        consecutive_losses = 0
        tps_owed = 0   # TPs still needed since the last stop-loss
        eq_curve = []

        def value(p: Position, price: float) -> float:
            # Open P&L net of fees and carry paid so far.
            return p.pnl(price) - p.fees - p.carry

        def close(p: Position, price: float, t, reason: str):
            nonlocal balance, ch_trades, consecutive_losses, tps_owed
            p.fees += p.notional * get(p.symbol).fee_rate   # exit fee
            net = value(p, price)
            balance += net
            r = net / p.initial_risk_usd() if p.initial_risk_usd() else 0.0
            res.trades.append(ClosedTrade(p.symbol, p.direction, p.strategy, p.score,
                                          p.opened_at, t, p.entry, price, p.notional,
                                          net, r, reason))
            positions.remove(p)
            ch_trades += 1
            consecutive_losses = consecutive_losses + 1 if net < 0 else 0
            if reason == "stop" and cfg.risk.after_stop_loss != "none":
                tps_owed = cfg.risk.tps_to_recover
            elif reason == "target":
                tps_owed = max(0, tps_owed - 1)

        def reject(why):
            key = re.sub(r"open \w+ via \w+", "open trade", re.sub(r"[-\d.]+", "#", why))
            res.rejections[key] = res.rejections.get(key, 0) + 1

        for t in times:
            if ch_start is None:
                ch_start = t
            # New UTC day: day-open equity is equity at the previous close.
            eq_now = balance + sum(value(p, last_close.get(p.symbol, p.entry)) for p in positions)
            monitor.new_day_if_needed(t, eq_now)

            worst_eq_adj = 0.0
            for sym, rows in lookup.items():
                bar = rows.get(t)
                if bar is None:
                    continue
                o, h, l, c = bar
                inst = get(sym)
                for p in [p for p in positions if p.symbol == sym]:
                    # Carry for the hour held.
                    p.carry += p.notional * inst.daily_carry / 24
                    # Weekend rule.
                    if (inst.asset_class != CRYPTO and t.weekday() == 4
                            and t.hour >= cfg.risk.friday_flat_hour_utc):
                        close(p, o, t, "weekend")
                        continue
                    # News rule.
                    if self.news and self.news.must_flatten(inst, t, horizon=BAR):
                        close(p, o, t, "news")
                        continue
                    # Gap through stop at the open.
                    if p.direction * (o - p.stop) <= 0:
                        close(p, o, t, "stop" if p.stop != p.entry else "breakeven")
                        continue
                    hit_stop = (l <= p.stop) if p.direction == 1 else (h >= p.stop)
                    hit_tp = (h >= p.take_profit) if p.direction == 1 else (l <= p.take_profit)
                    if hit_stop:   # if both touched in one bar, assume the worst
                        close(p, p.stop, t, "breakeven" if p.stop == p.entry else "stop")
                        continue
                    if hit_tp:
                        close(p, p.take_profit, t, "target")
                        continue
                    # Still open: worst point inside this bar counts for drawdown.
                    adverse = l if p.direction == 1 else h
                    worst_eq_adj += p.pnl(adverse) - p.pnl(c)
                    # Breakeven once +1R reached (takes effect from next bar).
                    best = h if p.direction == 1 else l
                    if (cfg.risk.breakeven_at_r is not None and p.stop != p.entry and
                            p.r_multiple(best) >= cfg.risk.breakeven_at_r):
                        p.stop = p.entry
                last_close[sym] = c

            equity = balance + sum(value(p, last_close[p.symbol]) for p in positions)
            status = monitor.check(t, equity + worst_eq_adj, balance)

            if status == ACTIVE:
                self._enter(t, sigs, positions, last_close, balance, equity, monitor,
                            consecutive_losses, reject, tps_owed)
                # Entry fees are paid immediately.
            eq_curve.append((t, balance + sum(value(p, last_close[p.symbol]) for p in positions)))

            if status != ACTIVE:
                for p in list(positions):
                    close(p, last_close[p.symbol], t, "challenge_end")
                res.challenges.append(ChallengeResult(
                    ch_no, ch_start, t, monitor.status, monitor.reason, balance,
                    ch_trades, monitor.worst_static_dd, monitor.worst_intraday_dd))
                if not back_to_back:
                    break
                balance, monitor = start_bal, ChallengeMonitor(cfg.rules)
                ch_start, ch_trades, ch_no, consecutive_losses = None, 0, ch_no + 1, 0
                tps_owed = 0

        if monitor.status == ACTIVE and ch_start is not None:
            res.challenges.append(ChallengeResult(
                ch_no, ch_start, times[-1] if times else None, "unfinished", "",
                balance + sum(value(p, last_close[p.symbol]) for p in positions), ch_trades,
                monitor.worst_static_dd, monitor.worst_intraday_dd))
        res.equity = pd.Series(dict(eq_curve))
        return res

    # ------------------------------------------------------------------
    def _enter(self, t, sigs, positions, last_close, balance, equity, monitor,
               consecutive_losses, reject, tps_owed=0):
        # Best setups first.
        candidates = sorted(sigs.get(t, ()), key=lambda x: -x[1].score)
        decision_time = t + BAR   # signal is known at the bar's close
        for sym, row in candidates:
            inst = get(sym)
            entry = last_close.get(sym)
            if entry is None:
                continue
            if self.news and (self.news.blocks_entry(inst, decision_time)
                              or self.news.must_flatten(inst, decision_time, horizon=BAR)):
                reject("news window")
                continue
            sig = Signal(sym, int(row.direction), entry, float(row.stop),
                         float(row.take_profit), row.strategy, int(row.score), decision_time)
            if self.macro is not None:
                veto = self.macro.veto(inst, sig.direction, decision_time)
                if veto:
                    reject(f"macro veto: {veto}")
                    continue
            dec = self.risk.approve(sig, inst, positions, balance=balance, equity=equity,
                                    day_open_equity=monitor.day_open_equity,
                                    now=decision_time, consecutive_losses=consecutive_losses,
                                    tps_owed=tps_owed)
            if not dec.approved:
                reject(dec.reason)
                continue
            p = Position(sym, sig.direction, entry, sig.stop, sig.take_profit, dec.notional,
                         decision_time, sig.strategy, sig.score)
            p.fees = dec.notional * inst.fee_rate
            positions.append(p)
            monitor.order_placed(decision_time)
