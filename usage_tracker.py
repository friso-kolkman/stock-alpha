"""Track Perplexity API usage and enforce monthly budget."""

import json
from datetime import datetime
from pathlib import Path
from rich.console import Console

from config import (
    PERPLEXITY_MONTHLY_BUDGET,
    PERPLEXITY_PRICE_PER_1K_INPUT,
    PERPLEXITY_PRICE_PER_1K_OUTPUT,
)

console = Console()

USAGE_FILE = Path(__file__).parent / "usage.json"


def load_usage() -> dict:
    """Load usage data from file."""
    if not USAGE_FILE.exists():
        return create_new_usage()

    try:
        with open(USAGE_FILE, "r") as f:
            data = json.load(f)

        # Check if we need to reset for new month
        current_month = datetime.now().strftime("%Y-%m")
        if data.get("month") != current_month:
            console.print(f"[cyan]New month detected, resetting usage tracker[/cyan]")
            return create_new_usage()

        return data
    except (json.JSONDecodeError, KeyError):
        return create_new_usage()


def create_new_usage() -> dict:
    """Create fresh usage data for current month."""
    return {
        "month": datetime.now().strftime("%Y-%m"),
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "requests": 0,
        "history": []
    }


def save_usage(data: dict) -> None:
    """Save usage data to file."""
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a request."""
    input_cost = (input_tokens / 1000) * PERPLEXITY_PRICE_PER_1K_INPUT
    output_cost = (output_tokens / 1000) * PERPLEXITY_PRICE_PER_1K_OUTPUT
    return input_cost + output_cost


def get_remaining_budget() -> float:
    """Get remaining budget for current month."""
    usage = load_usage()
    return max(0, PERPLEXITY_MONTHLY_BUDGET - usage["total_cost"])


def can_make_request(estimated_tokens: int = 2000) -> tuple[bool, float]:
    """
    Check if we have budget for another request.

    Args:
        estimated_tokens: Estimated total tokens (input + output)

    Returns:
        (can_proceed, remaining_budget)
    """
    remaining = get_remaining_budget()
    # Estimate cost (assume 50/50 split input/output for safety margin)
    estimated_cost = calculate_cost(estimated_tokens // 2, estimated_tokens // 2)

    return remaining >= estimated_cost, remaining


def record_usage(input_tokens: int, output_tokens: int, question: str = "") -> float:
    """
    Record usage from a completed request.

    Returns the cost of this request.
    """
    usage = load_usage()
    cost = calculate_cost(input_tokens, output_tokens)

    usage["total_input_tokens"] += input_tokens
    usage["total_output_tokens"] += output_tokens
    usage["total_cost"] += cost
    usage["requests"] += 1

    # Keep last 100 requests in history
    usage["history"].append({
        "timestamp": datetime.now().isoformat(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "question": question[:100] if question else ""
    })
    usage["history"] = usage["history"][-100:]

    save_usage(usage)
    return cost


def get_usage_summary() -> str:
    """Get a formatted summary of current usage."""
    usage = load_usage()
    remaining = PERPLEXITY_MONTHLY_BUDGET - usage["total_cost"]

    return (
        f"Month: {usage['month']} | "
        f"Spent: ${usage['total_cost']:.2f} / ${PERPLEXITY_MONTHLY_BUDGET:.2f} | "
        f"Remaining: ${remaining:.2f} | "
        f"Requests: {usage['requests']}"
    )


def print_budget_status() -> None:
    """Print current budget status to console."""
    usage = load_usage()
    remaining = PERPLEXITY_MONTHLY_BUDGET - usage["total_cost"]
    pct_used = (usage["total_cost"] / PERPLEXITY_MONTHLY_BUDGET) * 100

    if pct_used >= 90:
        color = "red"
    elif pct_used >= 70:
        color = "yellow"
    else:
        color = "green"

    console.print(f"[{color}]Budget: ${usage['total_cost']:.2f} / ${PERPLEXITY_MONTHLY_BUDGET:.2f} ({pct_used:.1f}% used)[/{color}]")
