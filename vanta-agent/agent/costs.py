"""Trading costs on Vanta, per round trip (open + close), as a fraction of
the position's value - on top of the transaction fees in instruments.py.

From vantatrading.io/rules section 07 (Oct 2026):
  * Forex: bid-ask spread from Massive + slippage 0.5 bps per fill on USD
    majors, 1 bp on other pairs (doubled 5-6 pm New York; we use the
    normal rate - the agent rarely trades then).
  * Hyperliquid pairs (crypto, commodities, indices): spread and slippage
    from the Hyperliquid order book. Not published as a number, so these
    are cautious estimates.
"""
from agent.instruments import COMMODITIES, CRYPTO, EQUITIES, FOREX, INDICES, Instrument

USD_MAJORS = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"}
BP = 1e-4

# Estimated spread + slippage per round trip where we have no measured spread.
ESTIMATE = {
    CRYPTO: 3 * BP,        # BTC/ETH/SOL books are deep; alts would cost more
    COMMODITIES: 4 * BP,   # gold/silver/oil perps on Hyperliquid (xyz:)
    INDICES: 3 * BP,
    EQUITIES: 3 * BP,
    FOREX: 1 * BP,         # spread only; slippage added below
}


def round_trip(inst: Instrument, measured_spread: float | None = None) -> float:
    if inst.asset_class == FOREX:
        spread = measured_spread if measured_spread is not None else ESTIMATE[FOREX]
        slip = 0.5 * BP if inst.symbol in USD_MAJORS else 1.0 * BP
        return spread + 2 * slip
    return ESTIMATE[inst.asset_class]
