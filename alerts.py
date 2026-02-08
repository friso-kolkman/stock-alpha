"""Send Telegram alerts for new signals."""

import os

import httpx
from rich.console import Console

console = Console()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


async def send_telegram_alert(
    results: list[dict],
    client: httpx.AsyncClient,
) -> bool:
    """Send a Telegram message summarizing BUY signals. Returns True on success."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        console.print("  [dim]Telegram not configured (set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID)[/dim]")
        return False

    buys = [r for r in results if r.get("signal") == "BUY"]
    if not buys:
        console.print("  No BUY signals to alert")
        return True

    lines = ["\U0001f4c8 *Stock Alpha Scanner*\n"]
    for r in buys:
        ticker = r.get("ticker", "?")
        price = r.get("price", 0)
        score = r.get("alpha_score", 0)
        sl = r.get("stop_loss")
        tp = r.get("take_profit")
        pct = r.get("suggested_pct", 0)

        lines.append(f"*BUY {ticker}* @ {price}")
        parts = [f"Score: {score}"]
        if pct:
            parts.append(f"Size: {pct}%")
        lines.append(f"  {' | '.join(parts)}")
        if sl is not None and tp is not None:
            lines.append(f"  Stop: {sl} | Target: {tp}")
        lines.append("")

    text = "\n".join(lines)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    try:
        resp = await client.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
        })
        if resp.status_code == 200:
            console.print(f"  [green]Telegram alert sent ({len(buys)} BUY signals)[/green]")
            return True
        else:
            console.print(f"  [yellow]Telegram error: {resp.status_code}[/yellow]")
            return False
    except Exception as e:
        console.print(f"  [yellow]Telegram failed: {e}[/yellow]")
        return False
