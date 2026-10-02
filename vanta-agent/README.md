# Vanta Challenge Agent

An automated trading agent built to pass the **$5,000 Vanta Trading
all-markets challenge** with disciplined risk management.

## Status

| Part | State |
|---|---|
| Vanta rules simulator (5% static DD, 5% intraday DD, +10% realized target, fees) | ✅ built + tested |
| Risk engine (1% total open risk, correlation, safety lines, target protection) | ✅ built + tested |
| News rule (close 30 min before, no entries until 30 min after) | ✅ built + tested |
| Fundamental filter (economic surprise momentum, no look-ahead) | ✅ built + tested |
| Strategy 1: multi-timeframe trend pullback | ✅ built |
| Strategy 2: London/NY liquidity sweep | ✅ built |
| Backtester (back-to-back challenges) + Monte Carlo | ✅ built + tested |
| Free data downloaders (Dukascopy, Binance, Yahoo) | ✅ built, **needs a run on your PC** |
| Vanta Trading Desk API client | ✅ built (dry-run by default) |
| **Real-data backtest 2019–2025** | ⏳ next — run on your PC |
| Live headline / geopolitics analyst (LLM) | ⏳ after backtest |
| Live runner + Telegram alerts | ⏳ after backtest |
| 2–4 weeks paper trading | ⏳ |

## The rules this agent follows

**Vanta's rules** (from Vanta's own source code, `taoshidev/vanta-network`):
- Equity (including open trades) must never fall below **$4,750** (5% static).
- Equity must never fall **5% below the day's opening equity** (day starts 00:00 UTC).
- Pass = **realized** balance reaches **$5,500** (+10%).

**Owner's rules:**
- All open trades together never risk more than **1% ($50)**.
- New trades only when budget is free (a trade hit target or its stop is at
  breakeven) **and** the setup scores 8/10 or better. No setup → no trade.
- Close every related trade **30 minutes before** high-impact news; no new
  related trade until **30 minutes after**.

**Extra safety (ours):**
- 0.5% ($25) per trade; 0.25% when below $4,875, after 2 losses in a row, or
  once realized balance is above $5,400.
- No new trades below **$4,800** or after **−1%** on the day.
- Never two trades that are the same bet (e.g. long EURUSD + long GBPUSD).
- Non-crypto positions closed Friday 20:00 UTC; crypto at half size on weekends.

## How to run it on your computer (Windows/Mac)

1. Install Python 3.11+ from python.org (tick "Add Python to PATH").
2. Open a terminal in this `vanta-agent` folder and run:
   ```
   pip install -r requirements.txt
   python -m pytest            # should say: 23 passed
   python run_backtest.py      # downloads free data, then backtests 2019-2025
   ```
   The first run downloads ~7 years of prices and takes a while. Results go
   to `results/` (trades.csv, challenges.csv, equity.csv).
3. Hidden test period (run only once, at the very end):
   `python run_backtest.py --start 2024-01-01 --end 2025-12-31`

## Free data sources

| Data | Source |
|---|---|
| Economic calendar history 2007–today (actual/forecast) | Forex Factory archive via github.com/janickfarrell/newfac |
| Economic calendar this week (live) | Forex Factory public weekly feed |
| Forex, gold, silver, oil, US indices | Dukascopy |
| Crypto | Binance public data API |
| US stocks | Yahoo Finance (yfinance) |

## API keys

Copy `.env.example` to `.env` and fill it in yourself. Never paste keys into
a chat. `VANTA_DRY_RUN=1` means orders are only logged, never sent.

## Honest note

No system can guarantee a pass. This agent is built to make a pass likely and
a blown account very unlikely, and nothing goes live until the real-data
backtest and the paper-trading period both look good.
