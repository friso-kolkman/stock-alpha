"""Generate HTML dashboard and publish to GitHub Pages."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from rich.console import Console

console = Console()

# Paths
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
TEMPLATES_DIR = BASE_DIR / "templates"


def publish_results(
    results: list[dict],
    stocks_scanned: int,
    push_to_github: bool = False,
    dashboard_stats: dict | None = None,
) -> bool:
    """
    Generate HTML dashboard and data.json, optionally push to GitHub.

    Returns True if successful.
    """
    console.print("[cyan]Publishing results...[/cyan]")

    # Ensure docs directory exists
    DOCS_DIR.mkdir(exist_ok=True)

    # Prepare data
    run_date = datetime.now(timezone.utc).isoformat()

    data = {
        "run_date": run_date,
        "stocks_scanned": stocks_scanned,
        "top_picks": results,
        "dashboard_stats": dashboard_stats,
    }

    # Write data.json
    data_path = DOCS_DIR / "data.json"
    try:
        with open(data_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        console.print(f"  [green]Wrote {data_path}[/green]")
    except Exception as e:
        console.print(f"  [red]Error writing data.json: {e}[/red]")
        return False

    # Render all HTML pages
    try:
        pages = render_all_pages(data)
        for filename, html_content in pages.items():
            html_path = DOCS_DIR / filename
            with open(html_path, "w") as f:
                f.write(html_content)
            console.print(f"  [green]Wrote {html_path}[/green]")
    except Exception as e:
        console.print(f"  [red]Error rendering HTML: {e}[/red]")
        return False

    # Optionally push to GitHub
    if push_to_github:
        return push_to_git()

    return True


def _build_template_context(data: dict) -> dict:
    """Build the shared template context from scan data."""
    picks = data["top_picks"]
    avg_score = 0
    if picks:
        avg_score = sum(p.get("alpha_score", 0) for p in picks) / len(picks)

    buy_count = sum(1 for p in picks if p.get("signal") == "BUY")
    hold_count = sum(1 for p in picks if p.get("signal") == "HOLD")
    avoid_count = sum(1 for p in picks if p.get("signal") == "AVOID")

    # Sector distribution for heatmap
    sector_dist = {}
    for p in picks:
        sector = p.get("sector") or "Other"
        if sector not in sector_dist:
            sector_dist[sector] = {
                "count": 0, "total_score": 0,
                "signals": {"BUY": 0, "HOLD": 0, "AVOID": 0},
            }
        sector_dist[sector]["count"] += 1
        sector_dist[sector]["total_score"] += p.get("alpha_score", 0)
        sig = p.get("signal", "HOLD")
        if sig in sector_dist[sector]["signals"]:
            sector_dist[sector]["signals"][sig] += 1
    for s in sector_dist.values():
        s["avg_score"] = round(s["total_score"] / s["count"]) if s["count"] > 0 else 0

    # Index distribution for donut chart
    index_dist = {}
    for p in picks:
        idx = p.get("index") or "Unknown"
        index_dist[idx] = index_dist.get(idx, 0) + 1

    stats = data.get("dashboard_stats") or {}

    return {
        "run_date": data["run_date"],
        "stocks_scanned": data["stocks_scanned"],
        "picks": picks,
        "avg_score": round(avg_score),
        "buy_count": buy_count,
        "hold_count": hold_count,
        "avoid_count": avoid_count,
        "has_history": stats.get("has_history", False),
        "model_stats": stats,
        "streaks": stats.get("streaks", {}),
        "sector_dist": sector_dist,
        "index_dist": index_dist,
        "top_n": len(picks),
    }


def render_all_pages(data: dict) -> dict[str, str]:
    """Render all dashboard pages. Returns {filename: html_content}."""
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    ctx = _build_template_context(data)

    pages = {
        "index.html": ("index.html", "overview"),
        "picks.html": ("picks.html", "picks"),
        "performance.html": ("performance.html", "performance"),
        "methodology.html": ("methodology.html", "methodology"),
    }

    result = {}
    for output_file, (template_name, active_page) in pages.items():
        try:
            template = env.get_template(template_name)
            result[output_file] = template.render(**ctx, active_page=active_page)
        except Exception as e:
            console.print(f"  [yellow]Template {template_name} failed: {e}[/yellow]")
            if output_file == "index.html":
                result[output_file] = render_default_dashboard(data)

    return result


def render_default_dashboard(data: dict) -> str:
    """Render a default dashboard if template is missing."""
    picks_html = ""
    for pick in data.get("top_picks", []):
        signal = pick.get("signal", "HOLD")
        signal_class = signal.lower()

        picks_html += f"""
        <div class="pick-card {signal_class}">
            <div class="pick-header">
                <span class="signal-badge {signal_class}">{signal}</span>
                <span class="alpha-score">Score: {pick.get('alpha_score', 0)}</span>
            </div>
            <h3>{pick.get('ticker', '???')} -- {pick.get('name', 'Unknown')}</h3>
            <div class="metrics">{pick.get('sector', '')} | Price: {pick.get('price', 'N/A')}</div>
            <div class="reasoning">{pick.get('reasoning', 'No reasoning provided')}</div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Alpha Scanner</title>
    <style>
        :root {{ --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3;
            --buy: #238636; --hold: #1f6feb; --avoid: #da3633; }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, sans-serif; background: var(--bg); color: var(--text); padding: 2rem; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .pick-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }}
        .pick-card.buy {{ border-left: 4px solid var(--buy); }}
        .pick-card.hold {{ border-left: 4px solid var(--hold); }}
        .pick-card.avoid {{ border-left: 4px solid var(--avoid); }}
        .signal-badge {{ padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
        .signal-badge.buy {{ background: rgba(35,134,54,0.2); color: #3fb950; }}
        .signal-badge.hold {{ background: rgba(31,111,235,0.2); color: #58a6ff; }}
        .signal-badge.avoid {{ background: rgba(218,54,51,0.2); color: #f85149; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Stock Alpha Scanner</h1>
        <p>Updated: {data['run_date'][:19].replace('T', ' ')} UTC | Scanned: {data['stocks_scanned']}</p>
        {picks_html if picks_html else '<p>No high-alpha opportunities found.</p>'}
        <footer><p>Data from Yahoo Finance & Twelve Data | Research via Perplexity</p>
        <p>Signal generation only. Not financial advice.</p></footer>
    </div>
</body>
</html>"""


def push_to_git() -> bool:
    """Commit and push docs/ to GitHub."""
    console.print("  [cyan]Pushing to GitHub...[/cyan]")

    try:
        result = subprocess.run(
            ["git", "status"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            console.print("  [yellow]Not a git repository, skipping push[/yellow]")
            return True

        subprocess.run(
            ["git", "add", "docs/"],
            cwd=BASE_DIR,
            check=True
        )

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        subprocess.run(
            ["git", "commit", "-m", f"Weekly stock scan - {timestamp}"],
            cwd=BASE_DIR,
            capture_output=True
        )

        result = subprocess.run(
            ["git", "push"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            console.print("  [green]Pushed to GitHub successfully[/green]")
            return True
        else:
            console.print(f"  [yellow]Push failed: {result.stderr}[/yellow]")
            return False

    except subprocess.CalledProcessError as e:
        console.print(f"  [red]Git error: {e}[/red]")
        return False
    except FileNotFoundError:
        console.print("  [yellow]Git not found, skipping push[/yellow]")
        return True
