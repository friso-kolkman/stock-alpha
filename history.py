"""Historical tracking, backtesting, and adaptive scoring for Stock Alpha Scanner."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import yfinance as yf
from rich.console import Console

from config import (
    EVAL_MIN_DAYS,
    EVAL_MAX_DAYS,
    BUY_CORRECT_THRESHOLD,
    BUY_INCORRECT_THRESHOLD,
    HOLD_UPPER_THRESHOLD,
    HOLD_LOWER_THRESHOLD,
    AVOID_CORRECT_THRESHOLD,
    AVOID_INCORRECT_THRESHOLD,
    MAX_WEIGHT_ADJUSTMENT,
    MIN_EVALUATIONS_FOR_ADAPT,
    SCORING_WEIGHTS,
    YFINANCE_SUFFIX,
)

console = Console()

HISTORY_PATH = Path(__file__).parent / "docs" / "history.json"

_executor = ThreadPoolExecutor(max_workers=4)


def _empty_history() -> dict:
    """Return an empty history structure."""
    return {
        "predictions": [],
        "model_stats": {
            "total_predictions": 0,
            "evaluated": 0,
            "pending": 0,
            "correct": 0,
            "incorrect": 0,
            "inconclusive": 0,
            "win_rate": None,
            "per_signal": {},
            "factor_correlations": {},
        },
        "weight_overrides": {},
        "weight_history": [],
        "performance_snapshots": [],
    }


def load_history() -> dict:
    """Load history from disk, returning empty structure if missing."""
    if not HISTORY_PATH.exists():
        return _empty_history()
    try:
        with open(HISTORY_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"  [yellow]Warning: Could not load history: {e}[/yellow]")
        return _empty_history()


def save_history(history: dict) -> None:
    """Persist history to disk."""
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, default=str)


def record_predictions(results: list[dict], scan_date: str) -> None:
    """Store signals from the current scan run."""
    history = load_history()
    existing_ids = {p["id"] for p in history["predictions"]}
    added = 0

    for r in results:
        pred_id = f"{scan_date}_{r['ticker']}"
        if pred_id in existing_ids:
            continue

        history["predictions"].append({
            "id": pred_id,
            "scan_date": scan_date,
            "ticker": r["ticker"],
            "exchange": r.get("exchange", ""),
            "signal": r.get("signal", "HOLD"),
            "price_at_signal": r.get("price"),
            "alpha_score": r.get("alpha_score", 0),
            "score_breakdown": r.get("score_breakdown", {}),
            "evaluated": False,
            "outcome": None,
            "return_pct": None,
            "eval_date": None,
            "eval_price": None,
        })
        added += 1

    _recalculate_model_stats(history)
    _record_performance_snapshot(history, scan_date[:10])
    save_history(history)
    console.print(f"  Recorded {added} predictions ({len(history['predictions'])} total)")


def evaluate_past_predictions() -> dict:
    """Check predictions older than EVAL_MIN_DAYS against current prices."""
    history = load_history()
    now = datetime.now(timezone.utc)
    evaluated_count = 0

    for pred in history["predictions"]:
        if pred["evaluated"]:
            continue

        scan_dt = datetime.fromisoformat(pred["scan_date"]).replace(tzinfo=timezone.utc)
        days_elapsed = (now - scan_dt).days

        if days_elapsed < EVAL_MIN_DAYS:
            continue

        if days_elapsed > EVAL_MAX_DAYS:
            pred["evaluated"] = True
            pred["outcome"] = "EXPIRED"
            pred["eval_date"] = now.isoformat()
            evaluated_count += 1
            continue

        price_at_signal = pred.get("price_at_signal")
        if price_at_signal is None or price_at_signal <= 0:
            pred["evaluated"] = True
            pred["outcome"] = "NO_DATA"
            pred["eval_date"] = now.isoformat()
            evaluated_count += 1
            continue

        current_price = _fetch_current_price(pred["ticker"], pred["exchange"])
        if current_price is None:
            continue

        return_pct = ((current_price - price_at_signal) / price_at_signal) * 100
        outcome = _determine_outcome(pred["signal"], return_pct)

        pred["evaluated"] = True
        pred["outcome"] = outcome
        pred["return_pct"] = round(return_pct, 2)
        pred["eval_date"] = now.isoformat()
        pred["eval_price"] = round(current_price, 4)
        evaluated_count += 1

    if evaluated_count > 0:
        _recalculate_model_stats(history)
        save_history(history)
        console.print(f"  Evaluated {evaluated_count} past predictions")
    else:
        console.print("  No predictions ready for evaluation yet")

    return history


def get_adaptive_weights(history: dict | None = None) -> dict:
    """Return SCORING_WEIGHTS + any weight_overrides from history."""
    if history is None:
        history = load_history()

    overrides = history.get("weight_overrides", {})
    weights = {}
    for factor, base in SCORING_WEIGHTS.items():
        weights[factor] = base + overrides.get(factor, 0)
    return weights


def update_adaptive_weights(history: dict) -> dict:
    """Recalculate weight overrides based on factor-return correlations."""
    evaluated = [
        p for p in history["predictions"]
        if p["evaluated"] and p.get("return_pct") is not None
        and p["outcome"] in ("CORRECT", "INCORRECT", "INCONCLUSIVE")
    ]

    if len(evaluated) < MIN_EVALUATIONS_FOR_ADAPT:
        console.print(f"  Not enough evaluated predictions for adaptation "
                      f"({len(evaluated)}/{MIN_EVALUATIONS_FOR_ADAPT})")
        return history

    stats = history.get("model_stats", {})
    _calculate_factor_correlations(evaluated, stats)

    correlations = stats.get("factor_correlations", {})
    if not correlations:
        return history

    # Find best and worst performing factors
    factors_by_spread = sorted(
        correlations.items(),
        key=lambda x: x[1].get("return_spread", 0),
        reverse=True,
    )

    old_overrides = dict(history.get("weight_overrides", {}))
    new_overrides = {}

    if len(factors_by_spread) >= 2:
        best_factor, best_data = factors_by_spread[0]
        worst_factor, worst_data = factors_by_spread[-1]

        best_spread = best_data.get("return_spread", 0)
        worst_spread = worst_data.get("return_spread", 0)

        # Boost best factor (+1 to +3 based on spread magnitude)
        if best_spread > 0:
            boost = min(MAX_WEIGHT_ADJUSTMENT, max(1, int(best_spread / 3)))
            new_overrides[best_factor] = boost

        # Reduce worst factor (-1 to -3 based on negative spread)
        if worst_spread < 0:
            penalty = max(-MAX_WEIGHT_ADJUSTMENT, min(-1, int(worst_spread / 3)))
            new_overrides[worst_factor] = penalty

    if new_overrides != old_overrides:
        history["weight_overrides"] = new_overrides
        history["weight_history"].append({
            "date": datetime.now(timezone.utc).isoformat(),
            "overrides": new_overrides,
            "evaluated_count": len(evaluated),
            "reason": f"Best: {factors_by_spread[0][0]} "
                      f"(spread {factors_by_spread[0][1].get('return_spread', 0):+.1f}%), "
                      f"Worst: {factors_by_spread[-1][0]} "
                      f"(spread {factors_by_spread[-1][1].get('return_spread', 0):+.1f}%)",
        })
        history["model_stats"] = stats
        save_history(history)
        console.print(f"  Adaptive weights updated: {new_overrides}")
    else:
        history["model_stats"] = stats
        console.print("  Adaptive weights unchanged")

    return history


def get_streaks(results: list[dict], history: dict | None = None) -> dict[str, int]:
    """Count consecutive weekly scan appearances for each ticker in results.

    Looks at historical predictions grouped by scan_date and counts how many
    of the most recent consecutive scan dates each ticker appeared in.
    A stock appearing 3 weeks in a row shows higher conviction than a first-timer.

    The current scan (represented by ``results``) always counts as 1. Previous
    scan dates found in history extend the streak further.

    Args:
        results: Current scan results (list of dicts with at least a "ticker" key).
        history: Pre-loaded history dict, or None to load from disk.

    Returns:
        Dict mapping ticker -> consecutive scan count (minimum 1 for current scan).
    """
    if history is None:
        history = load_history()

    predictions = history.get("predictions", [])

    # Group tickers by scan_date
    date_tickers: dict[str, set[str]] = {}
    for p in predictions:
        sd = p.get("scan_date", "")[:10]  # normalize to YYYY-MM-DD
        if sd:
            date_tickers.setdefault(sd, set()).add(p["ticker"])

    # Sorted unique scan dates (oldest first)
    sorted_dates = sorted(date_tickers.keys())

    # Build streak for each ticker in the current results
    current_tickers = {r["ticker"] for r in results}
    streaks: dict[str, int] = {}

    if not sorted_dates:
        # First scan ever -- every ticker gets streak of 1
        for ticker in current_tickers:
            streaks[ticker] = 1
        return streaks

    for ticker in current_tickers:
        streak = 0
        # Walk backwards through scan dates
        for date in reversed(sorted_dates):
            if ticker in date_tickers[date]:
                streak += 1
            else:
                break
        # streak==0 means ticker never appeared before; current scan counts as 1
        streaks[ticker] = max(streak, 1)

    return streaks


def get_dashboard_stats(history: dict | None = None) -> dict | None:
    """Return formatted stats for the dashboard template."""
    if history is None:
        history = load_history()

    stats = history.get("model_stats", {})
    if stats.get("total_predictions", 0) == 0:
        return None

    # Recent evaluations (last 10)
    recent = sorted(
        [p for p in history["predictions"] if p["evaluated"] and p.get("outcome") in
         ("CORRECT", "INCORRECT", "INCONCLUSIVE")],
        key=lambda p: p.get("eval_date", ""),
        reverse=True,
    )[:10]

    weight_overrides = history.get("weight_overrides", {})
    active_weights = get_adaptive_weights(history)

    # Compute streaks for the most recent scan's tickers
    predictions = history.get("predictions", [])
    if predictions:
        latest_date = max(p.get("scan_date", "")[:10] for p in predictions)
        latest_results = [
            {"ticker": p["ticker"]}
            for p in predictions if p.get("scan_date", "")[:10] == latest_date
        ]
        streaks = get_streaks(latest_results, history)
    else:
        streaks = {}

    return {
        "has_history": True,
        "total_predictions": stats.get("total_predictions", 0),
        "evaluated": stats.get("evaluated", 0),
        "pending": stats.get("pending", 0),
        "correct": stats.get("correct", 0),
        "incorrect": stats.get("incorrect", 0),
        "inconclusive": stats.get("inconclusive", 0),
        "win_rate": stats.get("win_rate"),
        "per_signal": stats.get("per_signal", {}),
        "streaks": streaks,
        "weight_overrides": weight_overrides,
        "active_weights": active_weights,
        "base_weights": dict(SCORING_WEIGHTS),
        "recent_evaluations": [
            {
                "ticker": p["ticker"],
                "signal": p["signal"],
                "outcome": p["outcome"],
                "return_pct": p.get("return_pct"),
                "scan_date": p["scan_date"][:10],
                "eval_date": (p.get("eval_date") or "")[:10],
            }
            for p in recent
        ],
        "performance_timeline": history.get("performance_snapshots", []),
    }


def _record_performance_snapshot(history: dict, date_label: str) -> None:
    """Append a snapshot of current model stats for the timeline chart."""
    snapshots = history.setdefault("performance_snapshots", [])

    # Avoid duplicate snapshots for the same date
    if snapshots and snapshots[-1].get("date") == date_label:
        snapshots[-1] = _build_snapshot(history, date_label)
        return

    snapshots.append(_build_snapshot(history, date_label))


def _build_snapshot(history: dict, date_label: str) -> dict:
    stats = history.get("model_stats", {})
    per_signal = stats.get("per_signal", {})

    # Average return across all evaluated predictions with returns
    all_returns = [
        p["return_pct"] for p in history["predictions"]
        if p["evaluated"] and p.get("return_pct") is not None
    ]
    avg_return = round(sum(all_returns) / len(all_returns), 2) if all_returns else None

    return {
        "date": date_label,
        "win_rate": stats.get("win_rate"),
        "total_predictions": stats.get("total_predictions", 0),
        "evaluated": stats.get("evaluated", 0),
        "correct": stats.get("correct", 0),
        "avg_return": avg_return,
        "buy_accuracy": (per_signal.get("BUY") or {}).get("accuracy"),
        "hold_accuracy": (per_signal.get("HOLD") or {}).get("accuracy"),
        "avoid_accuracy": (per_signal.get("AVOID") or {}).get("accuracy"),
    }


def _fetch_current_price(ticker: str, exchange: str) -> float | None:
    """Fetch current price via yfinance."""
    suffix = YFINANCE_SUFFIX.get(exchange, "")
    symbol = f"{ticker}{suffix}"
    try:
        t = yf.Ticker(symbol)
        price = t.info.get("currentPrice") or t.info.get("regularMarketPrice")
        return float(price) if price else None
    except Exception:
        return None


def _determine_outcome(signal: str, return_pct: float) -> str:
    """Determine if a prediction was CORRECT, INCORRECT, or INCONCLUSIVE."""
    if signal == "BUY":
        if return_pct > BUY_CORRECT_THRESHOLD:
            return "CORRECT"
        elif return_pct < BUY_INCORRECT_THRESHOLD:
            return "INCORRECT"
    elif signal == "HOLD":
        if HOLD_LOWER_THRESHOLD <= return_pct <= HOLD_UPPER_THRESHOLD:
            return "CORRECT"
        elif return_pct < HOLD_LOWER_THRESHOLD or return_pct > HOLD_UPPER_THRESHOLD:
            return "INCORRECT"
    elif signal == "AVOID":
        if return_pct < AVOID_CORRECT_THRESHOLD:
            return "CORRECT"
        elif return_pct > AVOID_INCORRECT_THRESHOLD:
            return "INCORRECT"

    return "INCONCLUSIVE"


def _recalculate_model_stats(history: dict) -> None:
    """Recalculate overall accuracy and per-signal stats."""
    predictions = history["predictions"]
    stats = history.get("model_stats", {})

    total = len(predictions)
    evaluated = [p for p in predictions if p["evaluated"]]
    pending = [p for p in predictions if not p["evaluated"]]

    # Only count decisive outcomes
    decisive = [p for p in evaluated if p.get("outcome") in ("CORRECT", "INCORRECT")]
    correct = [p for p in decisive if p["outcome"] == "CORRECT"]
    incorrect = [p for p in decisive if p["outcome"] == "INCORRECT"]
    inconclusive = [p for p in evaluated if p.get("outcome") == "INCONCLUSIVE"]

    stats["total_predictions"] = total
    stats["evaluated"] = len(evaluated)
    stats["pending"] = len(pending)
    stats["correct"] = len(correct)
    stats["incorrect"] = len(incorrect)
    stats["inconclusive"] = len(inconclusive)
    stats["win_rate"] = round(len(correct) / len(decisive) * 100, 1) if decisive else None

    # Per-signal breakdown
    per_signal = {}
    for signal in ("BUY", "HOLD", "AVOID"):
        signal_preds = [p for p in evaluated if p["signal"] == signal
                        and p.get("outcome") in ("CORRECT", "INCORRECT", "INCONCLUSIVE")]
        signal_decisive = [p for p in signal_preds if p["outcome"] in ("CORRECT", "INCORRECT")]
        signal_correct = [p for p in signal_decisive if p["outcome"] == "CORRECT"]
        returns = [p["return_pct"] for p in signal_preds if p.get("return_pct") is not None]

        per_signal[signal] = {
            "total": len(signal_preds),
            "correct": len(signal_correct),
            "decisive": len(signal_decisive),
            "accuracy": round(len(signal_correct) / len(signal_decisive) * 100, 1) if signal_decisive else None,
            "avg_return": round(sum(returns) / len(returns), 2) if returns else None,
        }

    stats["per_signal"] = per_signal
    history["model_stats"] = stats


def _calculate_factor_correlations(evaluated: list[dict], stats: dict) -> None:
    """Above/below median return analysis per factor."""
    correlations = {}

    for factor in SCORING_WEIGHTS:
        # Get factor scores and returns
        pairs = []
        for p in evaluated:
            breakdown = p.get("score_breakdown", {})
            factor_score = breakdown.get(factor)
            return_pct = p.get("return_pct")
            if factor_score is not None and return_pct is not None:
                pairs.append((factor_score, return_pct))

        if len(pairs) < 4:
            continue

        scores = [s for s, _ in pairs]
        med_score = median(scores)

        above = [r for s, r in pairs if s > med_score]
        below = [r for s, r in pairs if s <= med_score]

        if not above or not below:
            continue

        avg_above = sum(above) / len(above)
        avg_below = sum(below) / len(below)

        correlations[factor] = {
            "avg_return_above_median": round(avg_above, 2),
            "avg_return_below_median": round(avg_below, 2),
            "return_spread": round(avg_above - avg_below, 2),
            "sample_size": len(pairs),
        }

    stats["factor_correlations"] = correlations
