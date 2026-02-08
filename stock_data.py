"""Fetch stock data from yfinance and Twelve Data."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console

from config import (
    STOCK_UNIVERSE,
    TWELVE_DATA_API_KEY,
    YFINANCE_SUFFIX,
)

console = Console()

# Thread pool for yfinance calls (which are synchronous)
_executor = ThreadPoolExecutor(max_workers=8)


def get_yfinance_symbol(ticker: str, exchange: str) -> str:
    """Format ticker for yfinance (e.g., ASML -> ASML.AS)."""
    suffix = YFINANCE_SUFFIX.get(exchange, "")
    return f"{ticker}{suffix}"


async def fetch_all_stocks(client: httpx.AsyncClient) -> list[dict]:
    """Fetch data for all stocks in the universe."""
    all_stocks = []
    total = sum(len(v["tickers"]) for v in STOCK_UNIVERSE.values())
    console.print(f"  Fetching data for {total} stocks across {len(STOCK_UNIVERSE)} indices...")

    for index_name, index_info in STOCK_UNIVERSE.items():
        exchange = index_info["exchange"]
        currency = index_info["currency"]
        tickers = index_info["tickers"]

        console.print(f"  [{index_name}] {len(tickers)} stocks ({exchange})...")

        # Fetch yfinance data in parallel using thread pool
        loop = asyncio.get_event_loop()
        tasks = []
        for ticker in tickers:
            symbol = get_yfinance_symbol(ticker, exchange)
            task = loop.run_in_executor(
                _executor,
                _fetch_single_stock,
                ticker, symbol, index_name, exchange, currency,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            if result is not None:
                all_stocks.append(result)

    console.print(f"  [green]Fetched {len(all_stocks)} / {total} stocks successfully[/green]")
    return all_stocks


def _fetch_single_stock(
    ticker: str,
    symbol: str,
    index_name: str,
    exchange: str,
    currency: str,
) -> dict | None:
    """Fetch data for a single stock (runs in thread pool)."""
    try:
        yf_ticker = yf.Ticker(symbol)
        info = yf_ticker.info

        # Skip if no price data available
        if not info or info.get("regularMarketPrice") is None:
            return None

        # Get historical data for technicals (1 year)
        history = yf_ticker.history(period="1y")
        if history.empty:
            return None

        technicals = calculate_technicals(history)

        return build_stock_dict(
            ticker=ticker,
            index_name=index_name,
            exchange=exchange,
            currency=currency,
            info=info,
            technicals=technicals,
        )

    except Exception:
        return None


def calculate_technicals(history: pd.DataFrame) -> dict:
    """Calculate technical indicators from price history."""
    if history.empty or len(history) < 14:
        return {}

    close = history["Close"]
    volume = history["Volume"]

    # SMAs
    sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

    current_price = close.iloc[-1]

    # RSI-14
    rsi_14 = _calculate_rsi(close, 14)

    # Momentum returns
    momentum_3m = _pct_change(close, 63) if len(close) >= 63 else None
    momentum_6m = _pct_change(close, 126) if len(close) >= 126 else None
    momentum_12m = _pct_change(close, 252) if len(close) >= 252 else None

    # Price vs SMAs
    price_vs_sma50 = (current_price / sma_50 - 1) * 100 if sma_50 and sma_50 > 0 else None
    price_vs_sma200 = (current_price / sma_200 - 1) * 100 if sma_200 and sma_200 > 0 else None

    # Volume
    volume_avg_30d = volume.tail(30).mean() if len(volume) >= 30 else volume.mean()
    volume_ratio = volume.iloc[-1] / volume_avg_30d if volume_avg_30d > 0 else None

    # ATR-14
    atr_14 = None
    if len(history) >= 15:
        high = history["High"]
        low = history["Low"]
        prev_close = close.shift()
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_14 = float(tr.rolling(14).mean().iloc[-1])

    # 52-week high
    high_52w = close.max()
    pct_from_52w_high = ((current_price / high_52w) - 1) * 100 if high_52w > 0 else None

    return {
        "sma_50": sma_50,
        "sma_200": sma_200,
        "rsi_14": rsi_14,
        "momentum_3m": momentum_3m,
        "momentum_6m": momentum_6m,
        "momentum_12m": momentum_12m,
        "price_vs_sma50": price_vs_sma50,
        "price_vs_sma200": price_vs_sma200,
        "volume_avg_30d": volume_avg_30d,
        "volume_ratio": volume_ratio,
        "atr_14": atr_14,
        "fifty_two_week_high": high_52w,
        "pct_from_52w_high": pct_from_52w_high,
    }


def _calculate_rsi(prices: pd.Series, period: int = 14) -> float | None:
    """Calculate RSI-14."""
    if len(prices) < period + 1:
        return None

    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))

    avg_gain = gain.rolling(period).mean().iloc[-1]
    avg_loss = loss.rolling(period).mean().iloc[-1]

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _pct_change(prices: pd.Series, lookback: int) -> float | None:
    """Calculate percentage change over lookback period."""
    if len(prices) < lookback:
        return None
    current = prices.iloc[-1]
    past = prices.iloc[-lookback]
    if past == 0:
        return None
    return ((current / past) - 1) * 100


def build_stock_dict(
    ticker: str,
    index_name: str,
    exchange: str,
    currency: str,
    info: dict,
    technicals: dict,
) -> dict:
    """Build unified stock dictionary from yfinance info and technicals."""
    # Parse earnings date
    earnings_date = None
    raw_earnings = info.get("earningsTimestamp")
    if raw_earnings:
        try:
            earnings_date = datetime.fromtimestamp(raw_earnings).isoformat()
        except (ValueError, TypeError, OSError):
            pass

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "exchange": exchange,
        "index": index_name,
        "currency": currency,
        "sector": info.get("sector"),
        "price": info.get("regularMarketPrice") or info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "pb_ratio": info.get("priceToBook"),
        "roe": _pct_if_decimal(info.get("returnOnEquity")),
        "debt_to_equity": info.get("debtToEquity"),
        "dividend_yield": _pct_if_decimal(info.get("dividendYield")),
        "earnings_date": earnings_date,
        "momentum_3m": technicals.get("momentum_3m"),
        "momentum_6m": technicals.get("momentum_6m"),
        "momentum_12m": technicals.get("momentum_12m"),
        "rsi_14": technicals.get("rsi_14"),
        "price_vs_sma200": technicals.get("price_vs_sma200"),
        "price_vs_sma50": technicals.get("price_vs_sma50"),
        "volume_avg_30d": technicals.get("volume_avg_30d"),
        "volume_ratio": technicals.get("volume_ratio"),
        "atr_14": technicals.get("atr_14"),
        "fifty_two_week_high": technicals.get("fifty_two_week_high"),
        "pct_from_52w_high": technicals.get("pct_from_52w_high"),
    }


def _pct_if_decimal(value) -> float | None:
    """Convert decimal ratio to percentage (0.15 -> 15.0), pass through None."""
    if value is None:
        return None
    try:
        v = float(value)
        # yfinance returns ROE as 0.15 for 15%, dividend yield as 0.03 for 3%
        if abs(v) < 5:
            return v * 100
        return v
    except (ValueError, TypeError):
        return None
