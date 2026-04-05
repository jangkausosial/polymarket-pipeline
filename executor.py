from __future__ import annotations

import asyncio
import time
import logging

import config
import logger
from edge import Signal
from markets import get_token_id

log = logging.getLogger(__name__)

# FIX: Track gross spend per day (not net PnL) to correctly enforce the daily loss limit.
# The original code used abs(get_daily_pnl()), which returns 0 if wins and losses cancel out,
# allowing the bot to keep spending even after the limit should have triggered.
_daily_gross_spend: float = 0.0
_daily_reset_date: str = ""


def _get_today() -> str:
        from datetime import date
        return date.today().isoformat()


def _check_and_update_daily_spend(amount: float) -> bool:
        """
            Returns True if the trade is within the daily loss limit and records the spend.
                Returns False if it would breach the limit.
                    Resets the counter automatically at the start of each new day.
                        """
        global _daily_gross_spend, _daily_reset_date
        today = _get_today()
        if today != _daily_reset_date:
                    _daily_gross_spend = 0.0
                    _daily_reset_date = today
                    log.info(f"[executor] Daily spend counter reset for {today}")

        if _daily_gross_spend + amount > config.DAILY_LOSS_LIMIT_USD:
                    log.warning(
                                    f"[executor] Daily limit reached. Spent ${_daily_gross_spend:.2f} today, "
                                    f"limit is ${config.DAILY_LOSS_LIMIT_USD:.2f}. Rejecting trade of ${amount:.2f}."
                    )
                    return False

        _daily_gross_spend += amount
        return True


def execute_trade(signal: Signal) -> dict:
        """Execute a trade on Polymarket or log a dry-run. Synchronous."""
        # FIX: Use gross spend tracking instead of abs(PnL)
        if not _check_and_update_daily_spend(signal.bet_amount):
                    return _log_and_return(signal, status="rejected_daily_limit", order_id=None)

        if config.DRY_RUN:
                    return _log_and_return(signal, status="dry_run", order_id=None)

        return _execute_live_with_retry(signal)


async def execute_trade_async(signal: Signal) -> dict:
        """Async wrapper around execute_trade."""
        # FIX: Use get_running_loop() — get_event_loop() is deprecated in Python 3.10+
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, execute_trade, signal)


def _execute_live_with_retry(signal: Signal) -> dict:
        """Attempt live execution with configurable retries on transient failures."""
        last_error: Exception | None = None
        for attempt in range(1, config.EXECUTOR_MAX_RETRIES + 1):
                    try:
                                    return _execute_live(signal)
except Exception as e:
            last_error = e
            log.warning(
                                f"[executor] Attempt {attempt}/{config.EXECUTOR_MAX_RETRIES} failed: "
                                f"{type(e).__name__}: {e}"
            )
            if attempt < config.EXECUTOR_MAX_RETRIES:
                                time.sleep(config.EXECUTOR_RETRY_DELAY_SECONDS)

    log.error(f"[executor] All retries exhausted. Last error: {last_error}")
    return _log_and_return(
                signal,
                status=f"error_after_{config.EXECUTOR_MAX_RETRIES}_retries_{type(last_error).__name__}",
                order_id=None,
    )


def _execute_live(signal: Signal) -> dict:
        """Place a real order via Polymarket CLOB client."""
        try:
                    from py_clob_client.client import ClobClient
                    from py_clob_client.clob_types import OrderArgs, OrderType
except ImportError:
        return _log_and_return(signal, status="error_no_clob_client", order_id=None)

    client = ClobClient(
                host=config.POLYMARKET_HOST,
                key=config.POLYMARKET_API_KEY,
                chain_id=137,
                funder=config.POLYMARKET_PRIVATE_KEY,
    )

    client.set_api_creds(client.create_or_derive_api_creds())

    token_id = get_token_id(signal.market, signal.side)
    if not token_id:
                return _log_and_return(signal, status="error_no_token", order_id=None)

    price = signal.market.yes_price if signal.side == "YES" else signal.market.no_price

    order_args = OrderArgs(
                price=price,
                size=signal.bet_amount,
                side="BUY",
                token_id=token_id,
    )

    signed_order = client.create_order(order_args)
    resp = client.post_order(signed_order, OrderType.GTC)

    order_id = resp.get("orderID", resp.get("id", "unknown"))
    return _log_and_return(signal, status="executed", order_id=order_id)


def _log_and_return(signal: Signal, status: str, order_id: str | None) -> dict:
        """Log trade to SQLite and return result dict."""
        trade_id = logger.log_trade(
            market_id=signal.market.condition_id,
            market_question=signal.market.question,
            claude_score=signal.claude_score,
            market_price=signal.market_price,
            edge=signal.edge,
            side=signal.side,
            amount_usd=signal.bet_amount,
            order_id=order_id,
            status=status,
            reasoning=signal.reasoning,
            headlines=signal.headlines,
            news_source=signal.news_source,
            classification=signal.classification,
            materiality=signal.materiality,
            news_latency_ms=signal.news_latency_ms,
            classification_latency_ms=signal.classification_latency_ms,
            total_latency_ms=signal.total_latency_ms,
        )

    return {
                "trade_id": trade_id,
                "market": signal.market.question,
                "side": signal.side,
                "amount": signal.bet_amount,
                "edge": signal.edge,
                "status": status,
                "order_id": order_id,
                "classification": signal.classification,
                "materiality": signal.materiality,
                "latency_ms": signal.total_latency_ms,
    }
