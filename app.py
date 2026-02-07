#!/usr/bin/env python3
"""
Simple web UI for Stock Alpha Scanner.
Run with: python app.py
Then open: http://localhost:5050
"""

import subprocess
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template_string, jsonify

from usage_tracker import load_usage
from config import PERPLEXITY_MONTHLY_BUDGET

app = Flask(__name__)

# State
scan_status = {
    "running": False,
    "last_run": None,
    "last_output": ""
}

DOCS_DIR = Path(__file__).parent / "docs"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Alpha Scanner</title>
    <style>
        :root {
            --bg: #0d1117;
            --card: #161b22;
            --border: #30363d;
            --text: #e6edf3;
            --text-dim: #8b949e;
            --accent: #58a6ff;
            --green: #3fb950;
            --red: #f85149;
            --yellow: #d29922;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 2rem;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
        .subtitle { color: var(--text-dim); margin-bottom: 2rem; }

        .card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        .budget-bar {
            height: 8px;
            background: var(--border);
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .budget-fill {
            height: 100%;
            background: var(--green);
            transition: width 0.3s;
        }
        .budget-fill.warning { background: var(--yellow); }
        .budget-fill.danger { background: var(--red); }

        .btn {
            background: var(--accent);
            color: var(--bg);
            border: none;
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            transition: opacity 0.2s;
        }
        .btn:hover { opacity: 0.9; }
        .btn:disabled {
            background: var(--border);
            color: var(--text-dim);
            cursor: not-allowed;
        }

        .status {
            text-align: center;
            padding: 1rem;
            color: var(--text-dim);
        }
        .status.running { color: var(--yellow); }

        .results-link {
            display: block;
            text-align: center;
            margin-top: 1rem;
            color: var(--accent);
            text-decoration: none;
        }
        .results-link:hover { text-decoration: underline; }

        .log {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.85rem;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
            color: var(--text-dim);
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            text-align: center;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent);
        }
        .stat-label {
            font-size: 0.8rem;
            color: var(--text-dim);
        }

        .disclaimer {
            text-align: center;
            color: var(--text-dim);
            font-size: 0.75rem;
            font-style: italic;
            margin-top: 1rem;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .running .btn { animation: pulse 1.5s infinite; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Stock Alpha Scanner</h1>
        <p class="subtitle">Find high-alpha stock opportunities across EU & US markets</p>

        <div class="card">
            <div class="stats">
                <div>
                    <div class="stat-value" id="spent">${{ "%.2f"|format(spent) }}</div>
                    <div class="stat-label">Spent this month</div>
                </div>
                <div>
                    <div class="stat-value" id="remaining">${{ "%.2f"|format(remaining) }}</div>
                    <div class="stat-label">Remaining</div>
                </div>
                <div>
                    <div class="stat-value" id="requests">{{ requests }}</div>
                    <div class="stat-label">API calls</div>
                </div>
            </div>
            <div class="budget-bar">
                <div class="budget-fill {{ budget_class }}" style="width: {{ pct_used }}%"></div>
            </div>
        </div>

        <div class="card" id="scan-card">
            <button class="btn" id="scan-btn" onclick="runScan()">
                Run Stock Scan
            </button>
            <div class="status" id="status">
                {% if last_run %}
                Last scan: {{ last_run }}
                {% else %}
                Ready to scan
                {% endif %}
            </div>
            <a href="/results" class="results-link" target="_blank">View Latest Results</a>
        </div>

        <div class="card">
            <h3 style="margin-bottom: 0.5rem;">Output Log</h3>
            <div class="log" id="log">{{ last_output if last_output else "No output yet. Click the button above to run a scan." }}</div>
        </div>

        <p class="disclaimer">Signal generation only. Not financial advice. Do your own research.</p>
    </div>

    <script>
        let pollInterval;

        async function runScan() {
            const btn = document.getElementById('scan-btn');
            const status = document.getElementById('status');
            const card = document.getElementById('scan-card');
            const log = document.getElementById('log');

            btn.disabled = true;
            btn.textContent = 'Scanning...';
            card.classList.add('running');
            status.textContent = 'Scan in progress...';
            status.classList.add('running');
            log.textContent = 'Starting scan...\\n';

            try {
                const response = await fetch('/run', { method: 'POST' });
                pollInterval = setInterval(pollStatus, 1000);
            } catch (err) {
                btn.disabled = false;
                btn.textContent = 'Run Stock Scan';
                status.textContent = 'Error: ' + err.message;
            }
        }

        async function pollStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();

                document.getElementById('log').textContent = data.output || 'Running...';
                document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;

                // Update budget
                document.getElementById('spent').textContent = '$' + data.spent.toFixed(2);
                document.getElementById('remaining').textContent = '$' + data.remaining.toFixed(2);
                document.getElementById('requests').textContent = data.requests;

                if (!data.running) {
                    clearInterval(pollInterval);
                    document.getElementById('scan-btn').disabled = false;
                    document.getElementById('scan-btn').textContent = 'Run Stock Scan';
                    document.getElementById('scan-card').classList.remove('running');
                    document.getElementById('status').textContent = 'Last scan: ' + new Date().toLocaleString();
                    document.getElementById('status').classList.remove('running');
                }
            } catch (err) {
                console.error('Poll error:', err);
            }
        }
    </script>
</body>
</html>
"""

def get_budget_info():
    usage = load_usage()
    spent = usage.get("total_cost", 0)
    remaining = max(0, PERPLEXITY_MONTHLY_BUDGET - spent)
    pct_used = (spent / PERPLEXITY_MONTHLY_BUDGET) * 100

    if pct_used >= 90:
        budget_class = "danger"
    elif pct_used >= 70:
        budget_class = "warning"
    else:
        budget_class = ""

    return {
        "spent": spent,
        "remaining": remaining,
        "pct_used": pct_used,
        "budget_class": budget_class,
        "requests": usage.get("requests", 0)
    }


@app.route("/")
def index():
    budget = get_budget_info()
    return render_template_string(
        HTML_TEMPLATE,
        **budget,
        last_run=scan_status["last_run"],
        last_output=scan_status["last_output"]
    )


@app.route("/results")
def results():
    """Serve the generated dashboard."""
    html_path = DOCS_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "No results yet. Run a scan first.", 404


@app.route("/run", methods=["POST"])
def run_scan():
    if scan_status["running"]:
        return jsonify({"error": "Scan already running"}), 400

    def do_scan():
        scan_status["running"] = True
        scan_status["last_output"] = ""

        try:
            process = subprocess.Popen(
                ["python", "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=Path(__file__).parent
            )

            output_lines = []
            for line in process.stdout:
                output_lines.append(line)
                scan_status["last_output"] = "".join(output_lines[-100:])

            process.wait()
            scan_status["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        except Exception as e:
            scan_status["last_output"] += f"\nError: {e}"
        finally:
            scan_status["running"] = False

    thread = threading.Thread(target=do_scan)
    thread.start()

    return jsonify({"status": "started"})


@app.route("/status")
def status():
    budget = get_budget_info()
    return jsonify({
        "running": scan_status["running"],
        "output": scan_status["last_output"],
        "last_run": scan_status["last_run"],
        **budget
    })


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Stock Alpha Scanner")
    print("  Open: http://localhost:5050")
    print("="*50 + "\n")
    app.run(host="127.0.0.1", port=5050, debug=False)
