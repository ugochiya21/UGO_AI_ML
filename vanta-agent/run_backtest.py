"""Run the full backtest.

  python run_backtest.py                         # real data, 2019-2025, all markets
  python run_backtest.py --start 2024-01-01 --end 2025-12-31   # the hidden test period
  python run_backtest.py --symbols EURUSD GOLDUSDC BTCUSDC
  python run_backtest.py --synthetic             # fake data, checks the machinery only

Results are written to results/.
"""
import argparse
from pathlib import Path

import pandas as pd

from agent import instruments
from agent.backtest import montecarlo, report
from agent.backtest.engine import Backtester
from agent.config import AgentConfig
from agent.macro import CalendarSurprise
from agent.news import calendar
from agent.news.guard import NewsGuard
from agent.strategies import features, session_sweep, trend_pullback

CAL_PATH = Path("data/forexfactory_calendar.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--symbols", nargs="*", default=list(instruments.UNIVERSE))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--no-news", action="store_true", help="disable the news rule (for comparison)")
    ap.add_argument("--no-macro", action="store_true", help="disable the fundamental filter")
    ap.add_argument("--strategies", nargs="*", default=["trend_pullback", "session_sweep"])
    args = ap.parse_args()
    cfg = AgentConfig()

    # 1. Economic calendar
    events = []
    if not args.no_news or not args.no_macro:
        if not CAL_PATH.exists():
            print("Downloading Forex Factory calendar history ...")
            calendar.download_history(CAL_PATH)
        events = calendar.load_history(CAL_PATH, impacts=cfg.risk.news_impacts)
        print(f"Loaded {len(events)} high-impact events")
    news = None if args.no_news else NewsGuard(events, cfg.risk.news_close_before_min,
                                               cfg.risk.news_block_after_min)
    macro = None if args.no_macro else CalendarSurprise(events)

    # 2. Prices, features and signals
    bars, signals = {}, {}
    for i, sym in enumerate(args.symbols):
        inst = instruments.get(sym)
        try:
            if args.synthetic:
                from agent.data import synthetic
                df = synthetic.make(inst, args.start, args.end, seed=i)
            else:
                from agent.data import loaders
                df = loaders.load(inst, args.start, args.end)
        except Exception as e:  # noqa: BLE001 - keep going with other markets
            print(f"  ! {sym}: could not load data ({e})")
            continue
        if len(df) < 2000:
            print(f"  ! {sym}: only {len(df)} bars, skipped")
            continue
        f = features.build(df, inst)
        parts = []
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
    bt = Backtester(cfg, bars, signals, news=news, macro=macro)
    res = bt.run(args.start, args.end)
    print(report.summary(res))

    trades = report.trades_frame(res)
    if not trades.empty:
        mc = montecarlo.simulate(trades.r_multiple.values, cfg)
        print("\nMonte Carlo (10,000 reshuffles of these trades):")
        for k, v in mc.items():
            print(f"  {k}: {v:.1%}" if isinstance(v, float) and k.endswith("prob") else f"  {k}: {v}")

    out = Path("results")
    out.mkdir(exist_ok=True)
    trades.to_csv(out / "trades.csv", index=False)
    report.challenges_frame(res).to_csv(out / "challenges.csv", index=False)
    res.equity.to_csv(out / "equity.csv")
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
