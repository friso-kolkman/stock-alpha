#!/usr/bin/env python3
"""
Stock Alpha Scanner

Scans European and American stocks for high-alpha opportunities,
researches them via Perplexity, and outputs actionable signals
to a GitHub Pages dashboard.

Signal generation only -- no automated trading.
"""

import asyncio
import sys

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import TOP_N_STOCKS, MIN_ALPHA_SCORE, PERPLEXITY_API_KEY, YFINANCE_SUFFIX
from stock_data import fetch_all_stocks
from alpha_scorer import rank_stocks
from perplexity import research_stock
from evaluator import evaluate_research, ThesisResult
from publisher import publish_results
from usage_tracker import print_budget_status, get_remaining_budget

console = Console()


def build_yahoo_url(ticker: str, exchange: str) -> str:
    """Build Yahoo Finance URL for a stock."""
    suffix = YFINANCE_SUFFIX.get(exchange, "")
    return f"https://finance.yahoo.com/quote/{ticker}{suffix}"


async def main():
    """Main entry point for the stock alpha scanner."""
    console.print(Panel.fit(
        "[bold cyan]Stock Alpha Scanner[/bold cyan]\n"
        "Scanning for high-alpha stock opportunities...\n"
        "[dim]Signal generation only. Not financial advice.[/dim]",
        border_style="cyan"
    ))

    # Check for API key and show budget
    if not PERPLEXITY_API_KEY:
        console.print("[yellow]Warning: PERPLEXITY_API_KEY not set. Research will be skipped.[/yellow]")
        console.print("[dim]Set it in .env file or as environment variable.[/dim]\n")
    else:
        print_budget_status()
        remaining = get_remaining_budget()
        if remaining < 0.10:
            console.print("[red]Warning: Very low budget remaining! Research may be limited.[/red]\n")

    async with httpx.AsyncClient() as client:
        # Step 1: Fetch all stock data
        console.print("\n[bold]Step 1: Fetching stock data[/bold]")
        stocks = await fetch_all_stocks(client)

        if not stocks:
            console.print("[red]No stocks fetched. Exiting.[/red]")
            return

        # Step 2: Score and rank stocks
        console.print("\n[bold]Step 2: Scoring stocks[/bold]")
        ranked_stocks = rank_stocks(stocks)

        # Get top N above threshold
        top_stocks = [
            s for s in ranked_stocks[:TOP_N_STOCKS * 2]
            if s.get("alpha_score", 0) >= MIN_ALPHA_SCORE
        ][:TOP_N_STOCKS]

        if not top_stocks:
            console.print(f"[yellow]No stocks found with alpha score >= {MIN_ALPHA_SCORE}[/yellow]")
            publish_results([], len(stocks), push_to_github=False)
            return

        # Display top stocks
        display_top_stocks(top_stocks)

        # Step 3: Research top stocks
        console.print(f"\n[bold]Step 3: Researching top {len(top_stocks)} stocks[/bold]")
        results = []

        for i, stock in enumerate(top_stocks, 1):
            ticker = stock["ticker"]
            name = stock.get("name", "")[:30]
            console.print(f"\n  [{i}/{len(top_stocks)}] {ticker} -- {name}...")

            if PERPLEXITY_API_KEY:
                research = await research_stock(client, stock)
                thesis = evaluate_research(stock, research)

                if research.get("success"):
                    console.print(f"    [green]Research complete[/green] - {thesis.signal}")
                else:
                    console.print(f"    [yellow]Research failed[/yellow]")
            else:
                thesis = ThesisResult(
                    signal=ThesisResult.HOLD,
                    confidence="LOW",
                    reasoning="Research skipped - no API key",
                    company_overview="",
                    key_developments=[],
                    catalysts=[],
                    risk_factors=[],
                    sources=[],
                )

            # Compile result
            result = {
                "ticker": stock["ticker"],
                "name": stock.get("name", ""),
                "exchange": stock.get("exchange", ""),
                "index": stock.get("index", ""),
                "sector": stock.get("sector", ""),
                "price": stock.get("price"),
                "currency": stock.get("currency", ""),
                "market_cap": stock.get("market_cap"),
                "pe_ratio": stock.get("pe_ratio"),
                "forward_pe": stock.get("forward_pe"),
                "roe": stock.get("roe"),
                "dividend_yield": stock.get("dividend_yield"),
                "momentum_6m": stock.get("momentum_6m"),
                "momentum_12m": stock.get("momentum_12m"),
                "rsi_14": stock.get("rsi_14"),
                "alpha_score": stock["alpha_score"],
                "score_breakdown": stock.get("score_breakdown", {}),
                "yahoo_url": build_yahoo_url(stock["ticker"], stock.get("exchange", "")),
                **thesis.to_dict()
            }
            results.append(result)

        # Step 4: Display results
        console.print("\n[bold]Step 4: Results Summary[/bold]")
        display_results(results)

        # Step 5: Publish
        console.print("\n[bold]Step 5: Publishing[/bold]")
        publish_results(results, len(stocks), push_to_github=False)

        console.print("\n[bold green]Scan complete![/bold green]")
        console.print(f"  Results saved to docs/index.html and docs/data.json")
        console.print(f"  Open docs/index.html in a browser to view the dashboard")

        # Show final budget status
        if PERPLEXITY_API_KEY:
            console.print("")
            print_budget_status()


def display_top_stocks(stocks: list[dict]):
    """Display top stocks in a table."""
    table = Table(title="Top Alpha Opportunities", show_header=True)
    table.add_column("Score", style="cyan", width=6)
    table.add_column("Ticker", style="white", width=8)
    table.add_column("Sector", style="dim", width=14)
    table.add_column("Price", width=8)
    table.add_column("P/E", width=6)
    table.add_column("ROE", width=6)
    table.add_column("6m Mom", width=7)

    for s in stocks:
        pe = f"{s['pe_ratio']:.1f}" if s.get("pe_ratio") else "N/A"
        roe = f"{s['roe']:.0f}%" if s.get("roe") else "N/A"
        mom = f"{s['momentum_6m']:+.0f}%" if s.get("momentum_6m") is not None else "N/A"
        price = f"{s.get('price', 0):.2f}"
        sector = (s.get("sector") or "")[:14]

        table.add_row(
            str(s["alpha_score"]),
            s["ticker"],
            sector,
            price,
            pe,
            roe,
            mom,
        )

    console.print(table)


def display_results(results: list[dict]):
    """Display research results."""
    for r in results:
        signal = r.get("signal", "HOLD")
        color = {
            "BUY": "green",
            "HOLD": "blue",
            "AVOID": "red",
        }.get(signal, "white")

        ticker = r.get("ticker", "???")
        name = r.get("name", "")[:30]
        price = r.get("price", 0)
        score = r.get("alpha_score", 0)

        console.print(f"\n  [{color}]{signal}[/{color}] {ticker} -- {name}")
        console.print(f"    Price: {price} | Score: {score} | Confidence: {r.get('confidence', 'LOW')}")
        if r.get("reasoning"):
            reasoning = r["reasoning"][:120] + "..." if len(r.get("reasoning", "")) > 120 else r["reasoning"]
            console.print(f"    {reasoning}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
