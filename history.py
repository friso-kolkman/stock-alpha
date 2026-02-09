"""Historical tracking, backtesting, and adaptive scoring for Stock Alpha Scanner."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median

import yfinance as yf
from rich.console import Console

from config import (
    EVAL_TIMEFRAMES,
    MAX_WEIGHT_ADJUSTMENT,
    MIN_EVALUATIONS_FOR_ADAPT,
    SCORING_WEIGHTS,
    YFINANCE_SUFFIX,
)

console = Console()

HISTORY_PATH = Path(__file__).parent / "docs" / "history.json"

_executor = ThreadPoolExecutor(max_workers=4)

# Ordered timeframe keys (shortest first)
_TF_KEYS = sorted(EVAL_TIMEFRAMES.keys(), key=lambda k: EVAL_TIMEFRAMES[k]["min_days"])


def _empty_history() -> dict:
    """Return an empty history structure."""
    return {
        "predictions": [],
        "model_stats": {},
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
            data = json.load(f)
        _migrate_predictions(data)
        return data
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"  [yellow]Warning: Could not load history: {e}[/yellow]")
        return _empty_history()


def _migrate_predictions(history: dict) -> None:
    """Migrate old flat-format predictions to multi-timeframe format."""
    for pred in history.get("predictions", []):
        if "evaluations" in pred:
            continue
        # Convert old flat fields to new structure
        evals = {}
        if pred.get("evaluated") and pred.get("outcome"):
            evals["90d"] = {
                "outcome": pred["outcome"],
                "return_pct": pred.get("return_pct"),
                "eval_date": pred.get("eval_date"),
                "eval_price": pred.get("eval_price"),
            }
        pred["evaluations"] = evals
        # Clean up old fields
        for key in ("evaluated", "outcome", "return_pct", "eval_date", "eval_price"):
            pred.pop(key, None)


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
            "evaluations": {},
        })
        added += 1

    _recalculate_model_stats(history)
    _record_performance_snapshot(history, scan_date[:10])
    save_history(history)
    console.print(f"  Recorded {added} predictions ({len(history['predictions'])} total)")


def evaluate_past_predictions() -> dict:
    """Evaluate predictions at all eligible timeframes."""
    history = load_history()
    now = datetime.now(timezone.utc)
    eval_count = 0

    for pred in history["predictions"]:
        scan_dt = datetime.fromisoformat(pred["scan_date"]).replace(tzinfo=timezone.utc)
        days_elapsed = (now - scan_dt).days

        price_at_signal = pred.get("price_at_signal")
        evals = pred.setdefault("evaluations", {})

        for tf_key in _TF_KEYS:
            if tf_key in evals:
                continue  # already evaluated at this timeframe

            tf = EVAL_TIMEFRAMES[tf_key]

            if days_elapsed < tf["min_days"]:
                continue

            if days_elapsed > tf["max_days"] and tf_key != "90d":
                # Missed the window for shorter timeframes -- mark expired
                evals[tf_key] = {
                    "outcome": "EXPIRED",
                    "return_pct": None,
                    "eval_date": now.isoformat(),
                    "eval_price": None,
                }
                eval_count += 1
                continue

            if price_at_signal is None or price_at_signal <= 0:
                evals[tf_key] = {
                    "outcome": "NO_DATA",
                    "return_pct": None,
                    "eval_date": now.isoformat(),
                    "eval_price": None,
                }
                eval_count += 1
                continue

            current_price = _fetch_current_price(pred["ticker"], pred["exchange"])
            if current_price is None:
                continue

            return_pct = ((current_price - price_at_signal) / price_at_signal) * 100
            outcome = _determine_outcome(pred["signal"], return_pct, tf)

            evals[tf_key] = {
                "outcome": outcome,
                "return_pct": round(return_pct, 2),
                "eval_date": now.isoformat(),
                "eval_price": round(current_price, 4),
            }
            eval_count += 1

    if eval_count > 0:
        _recalculate_model_stats(history)
        save_history(history)
        console.print(f"  Evaluated {eval_count} timeframe checks")
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
    """Recalculate weight overrides using shortest timeframe with enough data."""
    # Try timeframes shortest-first
    best_tf = None
    best_evaluated = []

    for tf_key in _TF_KEYS:
        evaluated = _get_evaluated_for_tf(history, tf_key)
        if len(evaluated) >= MIN_EVALUATIONS_FOR_ADAPT:
            best_tf = tf_key
            best_evaluated = evaluated
            break  # use shortest that qualifies

    if not best_tf:
        total = sum(len(_get_evaluated_for_tf(history, k)) for k in _TF_KEYS)
        console.print(f"  Not enough evaluated predictions for adaptation "
                      f"({total} total across timeframes, need {MIN_EVALUATIONS_FOR_ADAPT} in one)")
        return history

    console.print(f"  Using {best_tf} timeframe for adaptation ({len(best_evaluated)} predictions)")

    stats = history.get("model_stats", {})
    tf_stats = stats.get(best_tf, {})
    _calculate_factor_correlations(best_evaluated, tf_stats, best_tf)
    stats[best_tf] = tf_stats

    correlations = tf_stats.get("factor_correlations", {})
    if not correlations:
        return history

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

        if best_spread > 0:
            boost = min(MAX_WEIGHT_ADJUSTMENT, max(1, int(best_spread / 3)))
            new_overrides[best_factor] = boost

        if worst_spread < 0:
            penalty = max(-MAX_WEIGHT_ADJUSTMENT, min(-1, int(worst_spread / 3)))
            new_overrides[worst_factor] = penalty

    if new_overrides != old_overrides:
        history["weight_overrides"] = new_overrides
        history["weight_history"].append({
            "date": datetime.now(timezone.utc).isoformat(),
            "overrides": new_overrides,
            "timeframe": best_tf,
            "evaluated_count": len(best_evaluated),
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
    """Count consecutive scan appearances for each ticker."""
    if history is None:
        history = load_history()

    predictions = history.get("predictions", [])
    date_tickers: dict[str, set[str]] = {}
    for p in predictions:
        sd = p.get("scan_date", "")[:10]
        if sd:
            date_tickers.setdefault(sd, set()).add(p["ticker"])

    sorted_dates = sorted(date_tickers.keys())
    current_tickers = {r["ticker"] for r in results}
    streaks: dict[str, int] = {}

    if not sorted_dates:
        for ticker in current_tickers:
            streaks[ticker] = 1
        return streaks

    for ticker in current_tickers:
        streak = 0
        for date in reversed(sorted_dates):
            if ticker in date_tickers[date]:
                streak += 1
            else:
                break
        streaks[ticker] = max(streak, 1)

    return streaks


def get_scan_diff(current_results: list[dict], history: dict | None = None) -> dict:
    """Compare current scan results with the previous scan.

    Returns a dict with:
        - new_entries: stocks in current scan but not in previous
        - exits: stocks in previous scan but not in current
        - score_changes: stocks in both scans with score deltas
        - signal_changes: stocks whose signal changed between scans
        - previous_date: the date of the scan we're comparing against (or None)
    """
    if history is None:
        history = load_history()

    predictions = history.get("predictions", [])
    if not predictions:
        return {"new_entries": [], "exits": [], "score_changes": [],
                "signal_changes": [], "previous_date": None}

    # Group predictions by scan date and find the most recent one
    date_preds: dict[str, list[dict]] = {}
    for p in predictions:
        sd = p.get("scan_date", "")[:10]
        if sd:
            date_preds.setdefault(sd, []).append(p)

    if not date_preds:
        return {"new_entries": [], "exits": [], "score_changes": [],
                "signal_changes": [], "previous_date": None}

    previous_date = max(date_preds.keys())
    prev_preds = date_preds[previous_date]

    # Build lookup from previous scan
    prev_by_ticker = {}
    for p in prev_preds:
        prev_by_ticker[p["ticker"]] = {
            "alpha_score": p.get("alpha_score", 0),
            "signal": p.get("signal", "HOLD"),
            "name": p.get("ticker"),
        }

    current_by_ticker = {}
    for r in current_results:
        current_by_ticker[r["ticker"]] = {
            "alpha_score": r.get("alpha_score", 0),
            "signal": r.get("signal", "HOLD"),
            "name": r.get("name", r["ticker"]),
        }

    prev_tickers = set(prev_by_ticker.keys())
    curr_tickers = set(current_by_ticker.keys())

    # New entries
    new_entries = []
    for t in sorted(curr_tickers - prev_tickers):
        c = current_by_ticker[t]
        new_entries.append({
            "ticker": t, "name": c["name"],
            "signal": c["signal"], "alpha_score": c["alpha_score"],
        })

    # Exits
    exits = []
    for t in sorted(prev_tickers - curr_tickers):
        p = prev_by_ticker[t]
        exits.append({
            "ticker": t, "signal": p["signal"],
            "alpha_score": p["alpha_score"],
        })

    # Score and signal changes for stocks in both scans
    score_changes = []
    signal_changes = []
    for t in sorted(curr_tickers & prev_tickers):
        c = current_by_ticker[t]
        p = prev_by_ticker[t]
        delta = c["alpha_score"] - p["alpha_score"]
        if delta != 0:
            score_changes.append({
                "ticker": t, "name": c["name"],
                "old_score": p["alpha_score"], "new_score": c["alpha_score"],
                "delta": delta,
            })
        if c["signal"] != p["signal"]:
            signal_changes.append({
                "ticker": t, "name": c["name"],
                "old_signal": p["signal"], "new_signal": c["signal"],
            })

    # Sort score changes by absolute delta descending
    score_changes.sort(key=lambda x: abs(x["delta"]), reverse=True)

    console.print(f"  Scan diff vs {previous_date}: "
                  f"{len(new_entries)} new, {len(exits)} exited, "
                  f"{len(signal_changes)} signal changes")

    return {
        "new_entries": new_entries,
        "exits": exits,
        "score_changes": score_changes,
        "signal_changes": signal_changes,
        "previous_date": previous_date,
    }


def get_dashboard_stats(history: dict | None = None) -> dict | None:
    """Return formatted stats for the dashboard template, per timeframe."""
    if history is None:
        history = load_history()

    predictions = history.get("predictions", [])
    if not predictions:
        return None

    # Build stats per timeframe
    timeframe_stats = {}
    for tf_key in _TF_KEYS:
        timeframe_stats[tf_key] = _build_tf_stats(history, tf_key)

    # Use the default timeframe (shortest with any evaluations, else 90d)
    default_tf = "90d"
    for tf_key in _TF_KEYS:
        if timeframe_stats[tf_key]["evaluated"] > 0:
            default_tf = tf_key
            break

    # Recent evaluations across all timeframes
    recent = _get_recent_evaluations(history, default_tf)

    weight_overrides = history.get("weight_overrides", {})
    active_weights = get_adaptive_weights(history)

    # Streaks
    latest_date = max(p.get("scan_date", "")[:10] for p in predictions)
    latest_results = [
        {"ticker": p["ticker"]}
        for p in predictions if p.get("scan_date", "")[:10] == latest_date
    ]
    streaks = get_streaks(latest_results, history)

    return {
        "has_history": True,
        "total_predictions": len(predictions),
        "timeframes": list(_TF_KEYS),
        "default_timeframe": default_tf,
        "timeframe_stats": timeframe_stats,
        # Flat stats for the default timeframe (backward compat)
        **timeframe_stats[default_tf],
        "streaks": streaks,
        "weight_overrides": weight_overrides,
        "active_weights": active_weights,
        "base_weights": dict(SCORING_WEIGHTS),
        "recent_evaluations": recent,
        "performance_timeline": history.get("performance_snapshots", []),
    }


def _build_tf_stats(history: dict, tf_key: str) -> dict:
    """Build stats for a single timeframe."""
    predictions = history.get("predictions", [])
    total = len(predictions)

    evaluated_preds = []
    for p in predictions:
        ev = p.get("evaluations", {}).get(tf_key)
        if ev and ev.get("outcome"):
            evaluated_preds.append((p, ev))

    pending = total - len(evaluated_preds)
    decisive = [(p, ev) for p, ev in evaluated_preds
                if ev["outcome"] in ("CORRECT", "INCORRECT")]
    correct = [(p, ev) for p, ev in decisive if ev["outcome"] == "CORRECT"]
    inconclusive = [(p, ev) for p, ev in evaluated_preds
                    if ev["outcome"] == "INCONCLUSIVE"]

    win_rate = round(len(correct) / len(decisive) * 100, 1) if decisive else None

    # Per-signal
    per_signal = {}
    for signal in ("BUY", "HOLD", "AVOID"):
        sig_evals = [(p, ev) for p, ev in evaluated_preds
                     if p["signal"] == signal
                     and ev["outcome"] in ("CORRECT", "INCORRECT", "INCONCLUSIVE")]
        sig_decisive = [(p, ev) for p, ev in sig_evals
                        if ev["outcome"] in ("CORRECT", "INCORRECT")]
        sig_correct = [(p, ev) for p, ev in sig_decisive if ev["outcome"] == "CORRECT"]
        returns = [ev["return_pct"] for p, ev in sig_evals
                   if ev.get("return_pct") is not None]

        per_signal[signal] = {
            "total": len(sig_evals),
            "correct": len(sig_correct),
            "decisive": len(sig_decisive),
            "accuracy": round(len(sig_correct) / len(sig_decisive) * 100, 1) if sig_decisive else None,
            "avg_return": round(sum(returns) / len(returns), 2) if returns else None,
        }

    return {
        "evaluated": len(evaluated_preds),
        "pending": pending,
        "correct": len(correct),
        "incorrect": len(decisive) - len(correct),
        "inconclusive": len(inconclusive),
        "win_rate": win_rate,
        "per_signal": per_signal,
    }


def _get_evaluated_for_tf(history: dict, tf_key: str) -> list[dict]:
    """Get predictions with usable evaluations for a timeframe."""
    result = []
    for p in history.get("predictions", []):
        ev = p.get("evaluations", {}).get(tf_key)
        if ev and ev.get("return_pct") is not None and ev["outcome"] in (
            "CORRECT", "INCORRECT", "INCONCLUSIVE"
        ):
            result.append({**p, "_eval": ev})
    return result


def _get_recent_evaluations(history: dict, tf_key: str) -> list[dict]:
    """Get last 10 evaluations for a timeframe."""
    items = []
    for p in history.get("predictions", []):
        ev = p.get("evaluations", {}).get(tf_key)
        if ev and ev.get("outcome") in ("CORRECT", "INCORRECT", "INCONCLUSIVE"):
            items.append({
                "ticker": p["ticker"],
                "signal": p["signal"],
                "outcome": ev["outcome"],
                "return_pct": ev.get("return_pct"),
                "scan_date": p["scan_date"][:10],
                "eval_date": (ev.get("eval_date") or "")[:10],
                "timeframe": tf_key,
            })
    items.sort(key=lambda x: x.get("eval_date", ""), reverse=True)
    return items[:10]


def _record_performance_snapshot(history: dict, date_label: str) -> None:
    """Append a snapshot of current model stats for the timeline chart."""
    snapshots = history.setdefault("performance_snapshots", [])

    if snapshots and snapshots[-1].get("date") == date_label:
        snapshots[-1] = _build_snapshot(history, date_label)
        return

    snapshots.append(_build_snapshot(history, date_label))


def _build_snapshot(history: dict, date_label: str) -> dict:
    # Use shortest timeframe with data for the snapshot
    snapshot = {"date": date_label, "total_predictions": len(history.get("predictions", []))}

    for tf_key in _TF_KEYS:
        tf_stats = _build_tf_stats(history, tf_key)
        snapshot[f"{tf_key}_win_rate"] = tf_stats["win_rate"]
        snapshot[f"{tf_key}_evaluated"] = tf_stats["evaluated"]

    # Average return across all evaluated predictions (any timeframe)
    all_returns = []
    for p in history.get("predictions", []):
        for ev in p.get("evaluations", {}).values():
            if ev and ev.get("return_pct") is not None:
                all_returns.append(ev["return_pct"])
                break  # one return per prediction for the avg
    snapshot["avg_return"] = round(sum(all_returns) / len(all_returns), 2) if all_returns else None

    return snapshot


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


def _determine_outcome(signal: str, return_pct: float, tf: dict) -> str:
    """Determine outcome using timeframe-specific thresholds."""
    if signal == "BUY":
        if return_pct > tf["buy_correct"]:
            return "CORRECT"
        elif return_pct < tf["buy_incorrect"]:
            return "INCORRECT"
    elif signal == "HOLD":
        if tf["hold_lower"] <= return_pct <= tf["hold_upper"]:
            return "CORRECT"
        elif return_pct < tf["hold_lower"] or return_pct > tf["hold_upper"]:
            return "INCORRECT"
    elif signal == "AVOID":
        if return_pct < tf["avoid_correct"]:
            return "CORRECT"
        elif return_pct > tf["avoid_incorrect"]:
            return "INCORRECT"

    return "INCONCLUSIVE"


def _recalculate_model_stats(history: dict) -> None:
    """Recalculate stats for all timeframes."""
    stats = history.setdefault("model_stats", {})
    for tf_key in _TF_KEYS:
        stats[tf_key] = _build_tf_stats(history, tf_key)
    history["model_stats"] = stats


def _calculate_factor_correlations(evaluated: list[dict], stats: dict, tf_key: str) -> None:
    """Above/below median return analysis per factor."""
    correlations = {}

    for factor in SCORING_WEIGHTS:
        pairs = []
        for p in evaluated:
            breakdown = p.get("score_breakdown", {})
            factor_score = breakdown.get(factor)
            ev = p.get("_eval", {})
            return_pct = ev.get("return_pct") if ev else p.get("evaluations", {}).get(tf_key, {}).get("return_pct")
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
