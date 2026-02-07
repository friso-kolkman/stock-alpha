"""Evaluate research results and generate stock signals."""

import re
from rich.console import Console

console = Console()


class ThesisResult:
    """Result of stock thesis evaluation."""

    BUY = "BUY"
    HOLD = "HOLD"
    AVOID = "AVOID"

    def __init__(
        self,
        signal: str,
        confidence: str,
        reasoning: str,
        company_overview: str,
        key_developments: list[str],
        catalysts: list[str],
        risk_factors: list[str],
        sources: list[str],
        target_timeframe: str = "3-6 months",
    ):
        self.signal = signal
        self.confidence = confidence
        self.reasoning = reasoning
        self.company_overview = company_overview
        self.key_developments = key_developments
        self.catalysts = catalysts
        self.risk_factors = risk_factors
        self.sources = sources
        self.target_timeframe = target_timeframe

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "company_overview": self.company_overview,
            "key_developments": self.key_developments,
            "catalysts": self.catalysts,
            "risk_factors": self.risk_factors,
            "sources": self.sources,
            "target_timeframe": self.target_timeframe,
        }


def evaluate_research(stock: dict, research: dict) -> ThesisResult:
    """
    Evaluate Perplexity research and generate a stock signal.

    Determines BUY/HOLD/AVOID based on research direction, confidence, and alpha score.
    """
    if not research.get("success"):
        return ThesisResult(
            signal=ThesisResult.HOLD,
            confidence="LOW",
            reasoning=f"Research failed: {research.get('error', 'Unknown error')}",
            company_overview="",
            key_developments=[],
            catalysts=[],
            risk_factors=[],
            sources=[],
        )

    content = research.get("content", "")
    citations = research.get("citations", [])
    alpha_score = stock.get("alpha_score", 0)

    # Parse structured response
    parsed = parse_research_content(content)

    # Determine signal
    research_direction = parsed.get("direction", "NEUTRAL").upper()
    research_confidence = parsed.get("confidence", "LOW").upper()

    signal, confidence = determine_signal(
        research_direction, research_confidence, alpha_score
    )

    return ThesisResult(
        signal=signal,
        confidence=confidence,
        reasoning=parsed.get("reasoning", "No clear reasoning provided"),
        company_overview=parsed.get("overview", ""),
        key_developments=parsed.get("developments", []),
        catalysts=parsed.get("catalysts", []),
        risk_factors=parsed.get("risks", []),
        sources=citations if isinstance(citations, list) else [],
    )


def parse_research_content(content: str) -> dict:
    """Parse structured research response from Perplexity."""
    result = {
        "overview": "",
        "developments": [],
        "direction": "NEUTRAL",
        "confidence": "LOW",
        "reasoning": "",
        "catalysts": [],
        "risks": [],
    }

    if not content:
        return result

    # Extract sections using regex
    sections = {
        "overview": r"COMPANY OVERVIEW:\s*\n?(.*?)(?=\n\n|KEY DEVELOPMENTS:|$)",
        "developments": r"KEY DEVELOPMENTS:\s*\n?(.*?)(?=\n\n|INVESTMENT THESIS:|$)",
        "thesis": r"INVESTMENT THESIS:\s*\n?(.*?)(?=\n\n|CATALYSTS:|$)",
        "catalysts": r"CATALYSTS:\s*\n?(.*?)(?=\n\n|RISK FACTORS:|$)",
        "risks": r"RISK FACTORS:\s*\n?(.*?)(?=\n\n|$)",
    }

    for key, pattern in sections.items():
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
            if key in ["developments", "catalysts", "risks"]:
                items = re.findall(r'[-•]\s*(.+)', text)
                result[key] = [item.strip() for item in items if item.strip()]
            elif key == "thesis":
                _parse_thesis(text, result)
            else:
                result[key] = text

    return result


def _parse_thesis(text: str, result: dict) -> None:
    """Parse the INVESTMENT THESIS section for direction, confidence, and reasoning."""
    # Look for BULLISH/BEARISH/NEUTRAL
    if re.search(r'\bBULLISH\b', text, re.IGNORECASE):
        result["direction"] = "BULLISH"
    elif re.search(r'\bBEARISH\b', text, re.IGNORECASE):
        result["direction"] = "BEARISH"
    else:
        result["direction"] = "NEUTRAL"

    # Look for confidence level
    if re.search(r'\bHIGH\b', text, re.IGNORECASE):
        result["confidence"] = "HIGH"
    elif re.search(r'\bMEDIUM\b', text, re.IGNORECASE):
        result["confidence"] = "MEDIUM"
    else:
        result["confidence"] = "LOW"

    # Extract reasoning (everything after the direction/confidence line)
    lines = text.split('\n')
    if len(lines) > 1:
        result["reasoning"] = ' '.join(lines[1:]).strip()
    elif result["overview"]:
        result["reasoning"] = result["overview"]


def determine_signal(
    research_direction: str,
    confidence: str,
    alpha_score: int,
) -> tuple[str, str]:
    """
    Determine trading signal based on research direction, confidence, and alpha score.

    Returns (signal, confidence) tuple.
    """
    # BEARISH -> AVOID regardless of score
    if research_direction == "BEARISH" and confidence in ("HIGH", "MEDIUM"):
        return ThesisResult.AVOID, confidence

    # BULLISH + HIGH confidence + decent score -> BUY
    if research_direction == "BULLISH":
        if confidence == "HIGH" and alpha_score >= 60:
            return ThesisResult.BUY, "HIGH"
        if confidence == "MEDIUM" and alpha_score >= 70:
            return ThesisResult.BUY, "MEDIUM"
        if confidence == "HIGH" and alpha_score >= 50:
            return ThesisResult.BUY, "MEDIUM"

    # Everything else -> HOLD
    return ThesisResult.HOLD, confidence


def format_thesis_summary(stock: dict, thesis: ThesisResult) -> str:
    """Format a human-readable thesis summary."""
    ticker = stock.get("ticker", "???")
    name = stock.get("name", "Unknown")
    price = stock.get("price", 0)
    score = stock.get("alpha_score", 0)

    signal_color = {
        ThesisResult.BUY: "green",
        ThesisResult.HOLD: "blue",
        ThesisResult.AVOID: "red",
    }.get(thesis.signal, "white")

    summary = f"""
[{signal_color}]{thesis.signal}[/{signal_color}] {ticker} -- {name}
   Price: {price} | Alpha Score: {score} | Confidence: {thesis.confidence}
   Timeframe: {thesis.target_timeframe}

   {thesis.reasoning}
"""

    if thesis.catalysts:
        summary += "\n   Catalysts:\n"
        for cat in thesis.catalysts[:3]:
            summary += f"   + {cat}\n"

    if thesis.risk_factors:
        summary += "\n   Risks:\n"
        for risk in thesis.risk_factors[:2]:
            summary += f"   - {risk}\n"

    return summary
