"""AI-powered portfolio insights using the Anthropic SDK."""
import os
from typing import Optional

import anthropic

from models import PortfolioSummary


def generate_portfolio_insights(summary: PortfolioSummary, warnings: list[str]) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "⚠️  ANTHROPIC_API_KEY not set — skipping AI insights."

    client = anthropic.Anthropic(api_key=api_key)

    # Build a compact, token-efficient summary dict
    positions_data = []
    for p in sorted(summary.positions, key=lambda x: x.weight_pct or 0, reverse=True)[:15]:
        positions_data.append({
            "ticker": p.ticker,
            "name": p.name,
            "weight_pct": round(p.weight_pct or 0, 1),
            "unrealised_pnl_pct": round(p.unrealised_pnl_pct or 0, 1),
            "unrealised_pnl_gbp": round(p.unrealised_pnl_gbp or 0, 2),
            "current_value_gbp": round(p.current_value_gbp or 0, 2),
            "days_held": p.days_held,
            "sector": p.sector,
            "beta": p.beta,
        })

    sector_summary = {k: round(v["pct"], 1) for k, v in summary.sector_breakdown.items()}
    geo_summary = {k: round(v["pct"], 1) for k, v in summary.geo_breakdown.items()}

    portfolio_data = {
        "total_value_gbp": round(summary.total_value_gbp, 2),
        "total_invested_gbp": round(summary.total_invested_gbp, 2),
        "total_unrealised_pnl_gbp": round(summary.total_unrealised_pnl_gbp, 2),
        "total_realised_pnl_gbp": round(summary.total_realised_pnl_gbp, 2),
        "total_pnl_pct": round(summary.total_pnl_pct, 2),
        "total_dividends_gbp": round(summary.total_dividends_gbp, 2),
        "portfolio_beta": summary.portfolio_beta,
        "sharpe_ratio": summary.sharpe_ratio,
        "volatility_30d_pct": summary.volatility_30d,
        "max_drawdown_pct": summary.max_drawdown_pct,
        "var_95_gbp": summary.var_95,
        "var_99_gbp": summary.var_99,
        "sector_allocation_pct": sector_summary,
        "geographic_allocation_pct": geo_summary,
        "top_15_positions": positions_data,
        "risk_warnings": warnings,
    }

    prompt = f"""You are a professional portfolio manager and financial analyst. Analyse this UK retail investor's Trading 212 portfolio and provide a structured report.

Portfolio data (values in GBP):
{portfolio_data}

Provide EXACTLY this structure — be specific, cite actual tickers and numbers:

## Executive Summary
3-4 sentences covering overall portfolio health, performance, and key characteristics.

## Top 3 Strengths
1. [Specific strength with data]
2. [Specific strength with data]
3. [Specific strength with data]

## Top 3 Risks & Concerns
1. [Specific risk with data]
2. [Specific risk with data]
3. [Specific risk with data]

## 3 Actionable Recommendations
1. [Specific action — which ticker to buy/sell/rebalance and why]
2. [Specific action — which ticker to buy/sell/rebalance and why]
3. [Specific action — which ticker to buy/sell/rebalance and why]

## Stocks/ETFs to Watch This Week
Brief bullet points on 2-3 holdings requiring attention.

Write with Goldman Sachs analyst-level precision. Be direct, not generic."""

    try:
        message = client.messages.create(
            model=os.environ.get("CLAUDE_MODEL", "claude-opus-4-5"),
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"⚠️  AI insights failed: {e}"
