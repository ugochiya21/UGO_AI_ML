"""Monte Carlo: reshuffle the backtest's real trade outcomes (in R) thousands
of times and replay the challenge with our sizing rules. This answers "what
if the same trades came in a luckier or unluckier order?"
"""
import numpy as np

from agent.config import AgentConfig


def simulate(r_multiples, cfg: AgentConfig, runs: int = 10_000, max_trades: int = 400,
             seed: int = 0) -> dict:
    r_multiples = np.asarray(r_multiples, dtype=float)
    if len(r_multiples) < 20:
        return {"error": "need at least 20 trades for a meaningful simulation"}
    rng = np.random.default_rng(seed)
    start = cfg.rules.account_size
    target = start * (1 + cfg.rules.profit_target_pct)
    # Our own stop line, not Vanta's - trading halts here.
    halt = start * cfg.risk.stop_trading_below
    fail = start * (1 - cfg.rules.static_drawdown_pct)
    passed = failed = halted = 0
    trades_to_pass = []
    for _ in range(runs):
        bal, losses = start, 0
        for n in range(1, max_trades + 1):
            pct = cfg.risk.risk_per_trade_pct
            if bal < start * cfg.risk.reduce_risk_below or bal >= start * cfg.risk.protect_target_above \
                    or losses >= 2:
                pct = cfg.risk.reduced_risk_per_trade_pct
            r = rng.choice(r_multiples)
            bal += r * pct * start
            losses = losses + 1 if r < 0 else 0
            if bal >= target:
                passed += 1
                trades_to_pass.append(n)
                break
            if bal < fail:
                failed += 1
                break
            if bal < halt:
                halted += 1
                break
    return {
        "runs": runs,
        "pass_prob": passed / runs,
        "fail_prob": failed / runs,
        "halted_prob": halted / runs,          # stopped by our own safety line
        "unfinished_prob": (runs - passed - failed - halted) / runs,
        "median_trades_to_pass": float(np.median(trades_to_pass)) if trades_to_pass else None,
    }
