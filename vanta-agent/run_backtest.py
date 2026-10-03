"""Run the full backtest.

  python run_backtest.py                         # real data, 2019-2025, all markets
  python run_backtest.py --start 2024-01-01 --end 2025-12-31   # the hidden test period
  python run_backtest.py --symbols EURUSD GOLDUSDC BTCUSDC
  python run_backtest.py --synthetic             # fake data, checks the machinery only

Results are written to results/.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from agent import costs, instruments
from agent.backtest import montecarlo, report
from agent.backtest.engine import Backtester
from agent.config import AgentConfig
from agent.macro import CalendarSurprise
from agent.news import calendar
from agent.news.guard import NewsGuard
from agent.strategies import features, session_sweep, trend_pullback
from agent.strategies import intraday as intraday_strats
from agent.strategies import fundamental, mtf, playbook, research, trend

CAL_PATH = Path("data/forexfactory_calendar.csv")

def spread_cost(df: pd.DataFrame, inst) -> float:
    """Vanta spread + slippage per round trip (see agent/costs.py)."""
    measured = float((df["spread"] / df["close"]).median()) if "spread" in df else None
    return costs.round_trip(inst, measured)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--symbols", nargs="*", default=list(instruments.UNIVERSE))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-news", action="store_true", help="disable the news rule (for comparison)")
    ap.add_argument("--no-macro", action="store_true", help="disable the fundamental filter")
    ap.add_argument("--tf", choices=["1h", "1hx", "15m", "5m"], default="5m",
                    help="5m: owner's intraday 1h/15m/5m strategies (default); "
                         "15m: earlier intraday; 1h: earlier swing; "
                         "1hx: hourly bars built from the 5m data (Claude's trend strategy)")
    ap.add_argument("--hold-weekends", action="store_true",
                    help="keep positions open over the weekend (Vanta allows it)")
    ap.add_argument("--strategies", nargs="*",
                    help="1h: trend_pullback session_sweep; 15m: intraday_pullback orb; "
                         "5m: mtf_pullback orb5 playbook late_momentum nr7_orb")
    ap.add_argument("--min-hours-before-news", type=float,
                    help="skip entries when related high-impact news is due sooner")
    ap.add_argument("--warmup-days", type=int, default=300,
                    help="price history loaded before --start so indicators are ready on day 1")
    ap.add_argument("--breakeven", choices=["on", "off"], help="move stop to entry at +1R")
    ap.add_argument("--after-sl", choices=["wait_tp", "half_2tp", "wait_2tp", "none"],
                    help="what happens after a stop-loss (see config.py)")
    ap.add_argument("--min-score", type=int, help="minimum setup score (default 8)")
    ap.add_argument("--risk-per-trade", type=float,
                    help="risk per trade as a fraction, e.g. 0.0025 = 0.25%% (default 0.005)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    cfg = AgentConfig()
    over = {}
    if args.breakeven:
        over["breakeven_at_r"] = 1.0 if args.breakeven == "on" else None
    if args.after_sl:
        over["after_stop_loss"] = args.after_sl
    if args.risk_per_trade:
        over["risk_per_trade_pct"] = args.risk_per_trade
        over["reduced_risk_per_trade_pct"] = args.risk_per_trade / 2
    if args.hold_weekends:
        over["flat_on_weekend"] = False
    if args.min_hours_before_news:
        over["min_hours_before_news"] = args.min_hours_before_news
    if not args.strategies:
        args.strategies = {"1h": ["trend_pullback", "session_sweep"], "1hx": ["trend"],
                           "15m": list(intraday_strats.STRATEGIES),
                           "5m": list(mtf.STRATEGIES)}[args.tf]
    if args.min_score:
        over["min_setup_score"] = args.min_score
    cfg.risk = replace(cfg.risk, **over)
    data_start = str((pd.Timestamp(args.start) - pd.Timedelta(days=args.warmup_days)).date())

    # 1. Economic calendar
    events = []
    if not args.no_news or not args.no_macro or set(args.strategies) & set(fundamental.STRATEGIES):
        if not CAL_PATH.exists():
            print("Downloading Forex Factory calendar history ...")
            calendar.download_history(CAL_PATH)
        events = calendar.load_history(CAL_PATH, impacts=cfg.risk.news_impacts)
        print(f"Loaded {len(events)} high-impact events")
    news = None if args.no_news else NewsGuard(calendar.scheduled_only(events),
                                               cfg.risk.news_close_before_min,
                                               cfg.risk.news_block_after_min,
                                               cfg.risk.news_include_correlated)
    macro = None if args.no_macro else CalendarSurprise(events)

    # 2. Prices, features and signals
    bars, signals = {}, {}
    for i, sym in enumerate(args.symbols):
        inst = instruments.get(sym)
        try:
            if args.synthetic:
                from agent.data import synthetic
                df = synthetic.make(inst, data_start, args.end, seed=i)
            elif args.tf in ("15m", "5m", "1hx"):
                from agent.data import intraday
                df = intraday.load(inst, data_start, args.end,
                                   rule={"15m": "15min", "5m": "5min", "1hx": "5min"}[args.tf])
                if args.tf == "1hx":
                    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
                    if "spread" in df:
                        agg["spread"] = "median"
                    df = df.resample("1h", label="left", closed="left").agg(agg).dropna(
                        subset=["close"])
            else:
                from agent.data import loaders
                df = loaders.load(inst, data_start, args.end)
        except Exception as e:  # noqa: BLE001 - keep going with other markets
            print(f"  ! {sym}: could not load data ({e})")
            continue
        if len(df) < 2000:
            print(f"  ! {sym}: only {len(df)} bars, skipped")
            continue
        parts = []
        if args.tf == "1hx":
            min_stop = spread_cost(df, inst) / 0.08
            if "trend" in args.strategies:
                parts.append(trend.generate(df, inst, min_stop_frac=min_stop))
            parts += [fundamental.STRATEGIES[s](df, inst, events, min_stop_frac=min_stop)
                      for s in args.strategies if s in fundamental.STRATEGIES]
            parts = [p for p in parts if len(p)]
        elif args.tf == "5m":
            f = mtf.features(df, inst)
            # Stop never so tight that Vanta's spread + slippage exceed 8% of the risk.
            min_stop = spread_cost(df, inst) / 0.08
            parts = [mtf.STRATEGIES[s](f, inst, min_stop_frac=min_stop)
                     for s in args.strategies if s in mtf.STRATEGIES]
            parts += [research.STRATEGIES[s](df, inst, min_stop_frac=min_stop)
                      for s in args.strategies if s in research.STRATEGIES]
            parts = [p for p in parts if len(p)]
            if "playbook" in args.strategies:
                parts.append(playbook.generate(playbook.features(df, inst), inst,
                                               min_stop_frac=min_stop))
        elif args.tf == "15m":
            f = intraday_strats.features(df, inst)
            parts = [intraday_strats.STRATEGIES[s](f, inst, events) for s in args.strategies]
        else:
            f = features.build(df, inst)
            if "trend_pullback" in args.strategies:
                parts.append(trend_pullback.generate(f))
            if "session_sweep" in args.strategies:
                parts.append(session_sweep.generate(f, inst))
        signals[sym] = pd.concat([p for p in parts if len(p)]) if any(len(p) for p in parts) \
            else pd.DataFrame()
        bars[sym] = df
        print(f"  {sym:12s} {len(df):6d} bars  {len(signals[sym]):5d} raw setups")

    if not bars:
        raise SystemExit("No price data loaded.")

    # 3. Backtest
    bar = pd.Timedelta({"1h": "1h", "1hx": "1h", "15m": "15min", "5m": "5min"}[args.tf])
    bt = Backtester(cfg, bars, signals, news=news, macro=macro, bar=bar,
                    costs={s: spread_cost(df, instruments.get(s)) for s, df in bars.items()})
    res = bt.run(args.start, args.end)
    print(report.summary(res))

    trades = report.trades_frame(res)
    if not trades.empty:
        mc = montecarlo.simulate(trades.r_multiple.values, cfg)
        print("\nMonte Carlo (10,000 reshuffles of these trades):")
        for k, v in mc.items():
            print(f"  {k}: {v:.1%}" if isinstance(v, float) and k.endswith("prob") else f"  {k}: {v}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    trades.to_csv(out / "trades.csv", index=False)
    report.challenges_frame(res).to_csv(out / "challenges.csv", index=False)
    res.equity.to_csv(out / "equity.csv")
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
