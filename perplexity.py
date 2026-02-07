"""Research stocks via Perplexity API."""

import httpx
from rich.console import Console

from config import PERPLEXITY_API_KEY, PERPLEXITY_API_BASE, PERPLEXITY_MODEL
from usage_tracker import can_make_request, record_usage

console = Console()


async def research_stock(client: httpx.AsyncClient, stock: dict) -> dict:
    """
    Research a stock using Perplexity's Sonar Pro.

    Returns structured research with:
    - Company overview
    - Key developments
    - Investment thesis (BULLISH/BEARISH/NEUTRAL)
    - Catalysts and risk factors
    """
    if not PERPLEXITY_API_KEY:
        console.print("[red]Error: PERPLEXITY_API_KEY not set[/red]")
        return create_error_response("API key not configured")

    # Check budget before making request
    can_proceed, remaining = can_make_request(estimated_tokens=2000)
    if not can_proceed:
        console.print(f"[red]Budget exceeded! Remaining: ${remaining:.2f}[/red]")
        return create_error_response(f"Monthly budget exceeded (${remaining:.2f} remaining)")

    prompt = build_research_prompt(stock)

    try:
        response = await client.post(
            f"{PERPLEXITY_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": PERPLEXITY_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1200
            },
            timeout=60.0
        )
        response.raise_for_status()
        data = response.json()

        # Extract response
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        citations = data.get("citations", [])

        # Record usage for budget tracking
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        ticker = stock.get("ticker", "")
        cost = record_usage(input_tokens, output_tokens, f"Stock: {ticker}")
        console.print(f"    [dim]Cost: ${cost:.4f}[/dim]")

        return {
            "success": True,
            "content": content,
            "citations": citations,
            "model": PERPLEXITY_MODEL,
            "tokens_used": usage.get("total_tokens", 0),
            "cost": cost
        }

    except httpx.HTTPStatusError as e:
        console.print(f"[red]Perplexity API error: {e.response.status_code}[/red]")
        return create_error_response(f"API error: {e.response.status_code}")
    except httpx.RequestError as e:
        console.print(f"[red]Request error: {e}[/red]")
        return create_error_response(f"Request failed: {str(e)}")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return create_error_response(f"Unexpected error: {str(e)}")


def get_system_prompt() -> str:
    """System prompt for stock research."""
    return """You are a European equity research analyst covering both European and American stocks. Your job is to:

1. Find the most recent and relevant news about the company
2. Assess the company's current business trajectory
3. Identify upcoming catalysts and risk factors
4. Provide a clear investment thesis direction
5. Be objective and cite your sources

Provide your analysis in this structured format:

COMPANY OVERVIEW:
[2-3 sentences about the company, its market position, and recent performance]

KEY DEVELOPMENTS:
- [Recent development 1]
- [Recent development 2]
- [Recent development 3]

INVESTMENT THESIS:
[BULLISH/BEARISH/NEUTRAL] - [Confidence: HIGH/MEDIUM/LOW]
[2-3 sentences explaining the thesis]

CATALYSTS:
- [Upcoming catalyst 1]
- [Upcoming catalyst 2]

RISK FACTORS:
- [Key risk 1]
- [Key risk 2]"""


def build_research_prompt(stock: dict) -> str:
    """Build the research prompt for a specific stock."""
    ticker = stock.get("ticker", "")
    name = stock.get("name", "")
    exchange = stock.get("exchange", "")
    sector = stock.get("sector", "Unknown")
    price = stock.get("price", 0)
    pe = stock.get("pe_ratio")
    roe = stock.get("roe")
    mom_6m = stock.get("momentum_6m")
    mom_12m = stock.get("momentum_12m")
    score = stock.get("alpha_score", 0)
    breakdown = stock.get("score_breakdown", {})

    metrics = []
    if pe is not None:
        metrics.append(f"P/E: {pe:.1f}")
    if roe is not None:
        metrics.append(f"ROE: {roe:.1f}%")
    if mom_6m is not None:
        metrics.append(f"6m return: {mom_6m:+.1f}%")
    if mom_12m is not None:
        metrics.append(f"12m return: {mom_12m:+.1f}%")

    metrics_str = " | ".join(metrics) if metrics else "Limited data available"

    score_parts = []
    for factor, pts in breakdown.items():
        score_parts.append(f"{factor}: {pts}")
    score_str = ", ".join(score_parts) if score_parts else "N/A"

    return f"""Research the following stock for investment potential:

STOCK: {ticker} -- {name}
Exchange: {exchange} | Sector: {sector}
Current Price: {price}
Key Metrics: {metrics_str}
Alpha Score: {score}/100 (breakdown: {score_str})

Please research the latest news and developments for {name} ({ticker}):
1. What is the company's current business situation?
2. What recent developments (last 30 days) affect the stock?
3. What is the investment outlook for the next 3-6 months?
4. What upcoming catalysts could move the stock?
5. What are the key risks?

Focus on recent news and verifiable facts. Consider both European and global market context."""


def create_error_response(error_msg: str) -> dict:
    """Create a standardized error response."""
    return {
        "success": False,
        "content": "",
        "citations": [],
        "error": error_msg,
        "tokens_used": 0
    }
