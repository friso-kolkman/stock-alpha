"""Multi-factor alpha scoring for stocks."""

from datetime import datetime, timedelta
from statistics import median

from rich.console import Console

from config import PRIORITY_SECTORS, SCORING_WEIGHTS

console = Console()


def rank_stocks(stocks: list[dict], weight_overrides: dict | None = None) -> list[dict]:
    """Score all stocks and return sorted by alpha score descending.

    Args:
        stocks: List of stock dicts with fundamental/technical data.
        weight_overrides: Optional dict of adjusted max scores per factor
                          (e.g. {"value": 27, "momentum": 23}). If None,
                          uses SCORING_WEIGHTS defaults.
    """
    if not stocks:
        return []

    weights = weight_overrides or SCORING_WEIGHTS
    sector_medians = calculate_sector_medians(stocks)

    for stock in stocks:
        score = 0
        breakdown = {}
        sector = stock.get("sector") or "Unknown"

        # Value
        value = calculate_value_score(stock, sector_medians.get(sector, {}),
                                      max_score=weights.get("value", 25))
        breakdown["value"] = value
        score += value

        # Quality
        quality = calculate_quality_score(stock,
                                          max_score=weights.get("quality", 25))
        breakdown["quality"] = quality
        score += quality

        # Momentum
        momentum = calculate_momentum_score(stock,
                                            max_score=weights.get("momentum", 25))
        breakdown["momentum"] = momentum
        score += momentum

        # Catalyst
        catalyst = calculate_catalyst_score(stock,
                                            max_score=weights.get("catalyst", 15))
        breakdown["catalyst"] = catalyst
        score += catalyst

        # Liquidity
        liquidity = calculate_liquidity_score(stock,
                                              max_score=weights.get("liquidity", 10))
        breakdown["liquidity"] = liquidity
        score += liquidity

        stock["alpha_score"] = score
        stock["score_breakdown"] = breakdown

    ranked = sorted(stocks, key=lambda s: s["alpha_score"], reverse=True)
    console.print(f"  Scored {len(ranked)} stocks (top: {ranked[0]['alpha_score']}, median: {ranked[len(ranked)//2]['alpha_score']})")
    return ranked


def calculate_sector_medians(stocks: list[dict]) -> dict:
    """Calculate median P/E and P/B per sector for relative valuation."""
    sector_data: dict[str, dict[str, list]] = {}

    for stock in stocks:
        sector = stock.get("sector") or "Unknown"
        if sector not in sector_data:
            sector_data[sector] = {"pe": [], "pb": []}

        pe = stock.get("pe_ratio")
        pb = stock.get("pb_ratio")
        if pe is not None and 0 < pe < 200:
            sector_data[sector]["pe"].append(pe)
        if pb is not None and 0 < pb < 100:
            sector_data[sector]["pb"].append(pb)

    result = {}
    for sector, data in sector_data.items():
        result[sector] = {
            "pe_median": median(data["pe"]) if data["pe"] else None,
            "pb_median": median(data["pb"]) if data["pb"] else None,
        }
    return result


def calculate_value_score(stock: dict, sector_med: dict, max_score: int = 25) -> int:
    """Value score: P/E vs sector, P/B vs sector, dividend yield."""
    score = 0
    pe = stock.get("pe_ratio")
    pb = stock.get("pb_ratio")
    div_yield = stock.get("dividend_yield")
    pe_median = sector_med.get("pe_median")
    pb_median = sector_med.get("pb_median")

    # P/E vs sector median (0-12)
    if pe is not None and pe > 0 and pe_median is not None and pe_median > 0:
        ratio = pe / pe_median
        if ratio <= 0.50:
            score += 12
        elif ratio <= 0.75:
            score += 8
        elif ratio < 1.0:
            score += 4

    # P/B vs sector median (0-8)
    if pb is not None and pb > 0 and pb_median is not None and pb_median > 0:
        ratio = pb / pb_median
        if ratio <= 0.50:
            score += 8
        elif ratio <= 0.75:
            score += 5
        elif ratio < 1.0:
            score += 2

    # Dividend yield (0-5)
    if div_yield is not None:
        if div_yield > 3:
            score += 5
        elif div_yield > 2:
            score += 3
        elif div_yield > 1:
            score += 1

    return min(score, max_score)


def calculate_quality_score(stock: dict, max_score: int = 25) -> int:
    """Quality score: ROE, D/E, dividend, earnings growth."""
    score = 0
    roe = stock.get("roe")
    de = stock.get("debt_to_equity")
    div_yield = stock.get("dividend_yield")
    pe = stock.get("pe_ratio")
    forward_pe = stock.get("forward_pe")

    # ROE (0-10)
    if roe is not None:
        if roe > 20:
            score += 10
        elif roe > 15:
            score += 7
        elif roe > 10:
            score += 4

    # Debt-to-Equity (0-8)
    if de is not None:
        # yfinance returns D/E as percentage (e.g., 50 for 0.5x)
        de_ratio = de / 100 if de > 5 else de
        if de_ratio < 0.5:
            score += 8
        elif de_ratio < 1.0:
            score += 5
        elif de_ratio < 1.5:
            score += 2

    # Dividend exists (0-4)
    if div_yield is not None and div_yield > 0:
        score += 4

    # Forward PE < Trailing PE (earnings growing) (0-3)
    if pe is not None and forward_pe is not None and pe > 0 and forward_pe > 0:
        if forward_pe < pe:
            score += 3

    return min(score, max_score)


def calculate_momentum_score(stock: dict, max_score: int = 25) -> int:
    """Momentum score: returns, trend, RSI."""
    score = 0
    mom_12m = stock.get("momentum_12m")
    mom_6m = stock.get("momentum_6m")
    price_vs_sma200 = stock.get("price_vs_sma200")
    price_vs_sma50 = stock.get("price_vs_sma50")
    rsi = stock.get("rsi_14")

    # 12m return (0-8)
    if mom_12m is not None:
        if mom_12m > 20:
            score += 8
        elif mom_12m > 10:
            score += 5
        elif mom_12m > 0:
            score += 2

    # 6m return (0-7)
    if mom_6m is not None:
        if mom_6m > 10:
            score += 7
        elif mom_6m > 5:
            score += 4
        elif mom_6m > 0:
            score += 2

    # Price > SMA200 (0-5)
    if price_vs_sma200 is not None and price_vs_sma200 > 0:
        score += 5

    # Price > SMA50 (0-3)
    if price_vs_sma50 is not None and price_vs_sma50 > 0:
        score += 3

    # RSI in healthy zone 40-70 (0-2)
    if rsi is not None and 40 <= rsi <= 70:
        score += 2

    return min(score, max_score)


def calculate_catalyst_score(stock: dict, max_score: int = 15) -> int:
    """Catalyst score: upcoming earnings, sector, recovery potential."""
    score = 0
    earnings_date = stock.get("earnings_date")
    sector = stock.get("sector") or ""
    pct_from_high = stock.get("pct_from_52w_high")

    # Earnings within 30/60 days (0-8)
    if earnings_date:
        try:
            ed = datetime.fromisoformat(earnings_date)
            days_until = (ed - datetime.now()).days
            if 0 <= days_until <= 30:
                score += 8
            elif 0 <= days_until <= 60:
                score += 4
        except (ValueError, TypeError):
            pass

    # Priority sector (0-5)
    if sector in PRIORITY_SECTORS:
        score += 5

    # Recovery potential: > 20% below 52-week high (0-2)
    if pct_from_high is not None and pct_from_high < -20:
        score += 2

    return min(score, max_score)


def calculate_liquidity_score(stock: dict, max_score: int = 10) -> int:
    """Liquidity score: market cap, volume, index membership."""
    score = 0
    market_cap = stock.get("market_cap")
    volume_ratio = stock.get("volume_ratio")

    # Market cap (0-5)
    if market_cap is not None:
        if market_cap > 10_000_000_000:  # > 10B
            score += 5
        elif market_cap > 2_000_000_000:  # > 2B
            score += 3
        elif market_cap > 500_000_000:  # > 500M
            score += 1

    # Above-average volume (0-3)
    if volume_ratio is not None and volume_ratio > 1.0:
        score += 3

    # Major index member (0-2) -- all stocks in our universe are index members
    score += 2

    return min(score, max_score)
