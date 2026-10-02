"""Vanta Trading Desk API client.

Docs: taoshidev/vanta-starter (app/docs/trading, app/docs/api-keys).
Auth: one header, `X-Api-Key: <key_id>.<key_secret>` - mint it on the
Trading Desk's API keys page. The key can trade and read only; it cannot
withdraw or create more keys.

Set in .env (never commit it):
    VANTA_API_URL=https://...        # from the Trading Desk docs page
    VANTA_API_KEY=hskk_xxx.yyyy
    VANTA_PROP_ACCOUNT=prop_...      # optional if the key is bound to one account
    VANTA_DRY_RUN=1                  # 1 = log orders, don't send them
"""
import logging
import os

import requests

log = logging.getLogger("vanta")


class VantaClient:
    def __init__(self, base_url=None, api_key=None, prop_account=None, dry_run=None):
        self.base = (base_url or os.environ["VANTA_API_URL"]).rstrip("/")
        self.headers = {"X-Api-Key": api_key or os.environ["VANTA_API_KEY"],
                        "Content-Type": "application/json"}
        acct = prop_account or os.environ.get("VANTA_PROP_ACCOUNT")
        if acct:
            self.headers["X-Prop-Account"] = acct
        self.dry_run = (os.environ.get("VANTA_DRY_RUN", "1") == "1") if dry_run is None else dry_run

    def _req(self, method, path, **kw):
        r = requests.request(method, self.base + path, headers=self.headers, timeout=90, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Vanta {method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else {}

    # ------------------------------------------------------------- reads
    def desk(self) -> dict:
        """Positions, resting orders, history and balance in one call."""
        return self._req("GET", "/v2/trading/desk-poll")

    def balance(self) -> dict:
        return self._req("GET", "/v2/trading/balance")

    def positions(self) -> list:
        return self._req("GET", "/v2/trading/positions")

    # ------------------------------------------------------------ writes
    def _write(self, path, body):
        if self.dry_run:
            log.info("DRY RUN %s %s", path, body)
            return {"success": True, "dry_run": True, "body": body}
        return self._req("POST", path, json=body)

    def market_order(self, symbol: str, direction: int, value_usd: float,
                     stop_loss: float, take_profit: float) -> dict:
        """Every order goes in WITH its stop-loss, so it is protected on
        Vanta's side even if this program crashes."""
        return self._write("/v2/trading/orders", {
            "trade_pair": symbol,
            "order_type": "LONG" if direction == 1 else "SHORT",
            "execution_type": "MARKET",
            "value": round(value_usd, 2),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        })

    def close(self, symbol: str) -> dict:
        return self._write("/v2/trading/orders/close", {"trade_pair": symbol})

    def set_tp_sl(self, symbol: str, stop_loss=None, take_profit=None) -> dict:
        body = {"trade_pair": symbol}
        if stop_loss is not None:
            body["stop_loss"] = stop_loss
        if take_profit is not None:
            body["take_profit"] = take_profit
        return self._write("/v2/trading/orders/tp-sl", body)
