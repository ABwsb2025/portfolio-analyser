#!/usr/bin/env python3
"""
Deep single-stock analysis for AMD (Advanced Micro Devices).
Fetches live data, computes technicals + fundamentals, peer comparison,
then uses Claude to generate a Goldman Sachs-style research report.

Usage:
    python amd_analysis.py
    python amd_analysis.py --output-dir .
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(v, decimals=2, prefix="", suffix=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{prefix}{v:,.{decimals}f}{suffix}"

def _pct(v, decimals=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}%"

def _billions(v):
    if v is None:
        return "N/A"
    return f"${v/1e9:.2f}B"


# ── data fetch ────────────────────────────────────────────────────────────────

def fetch_stock_data(ticker: str) -> dict:
    """Pull everything yfinance offers for one ticker."""
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    fast = t.fast_info
    price = getattr(fast, "last_price", None) or info.get("currentPrice") or info.get("regularMarketPrice")

    # historical prices
    hist_1y  = t.history(period="1y")
    hist_6mo = t.history(period="6mo")
    hist_3mo = t.history(period="3mo")

    # quarterly financials
    try:
        quarterly_income = t.quarterly_income_stmt
    except Exception:
        quarterly_income = pd.DataFrame()

    try:
        quarterly_balance = t.quarterly_balance_sheet
    except Exception:
        quarterly_balance = pd.DataFrame()

    try:
        quarterly_cashflow = t.quarterly_cashflow
    except Exception:
        quarterly_cashflow = pd.DataFrame()

    # analyst targets
    try:
        analyst_price_targets = t.analyst_price_targets
    except Exception:
        analyst_price_targets = {}

    try:
        recommendations_summary = t.recommendations_summary
    except Exception:
        recommendations_summary = pd.DataFrame()

    try:
        earnings_dates = t.earnings_dates
    except Exception:
        earnings_dates = pd.DataFrame()

    return {
        "info": info,
        "price": price,
        "hist_1y": hist_1y,
        "hist_6mo": hist_6mo,
        "hist_3mo": hist_3mo,
        "quarterly_income": quarterly_income,
        "quarterly_balance": quarterly_balance,
        "quarterly_cashflow": quarterly_cashflow,
        "analyst_price_targets": analyst_price_targets,
        "recommendations_summary": recommendations_summary,
        "earnings_dates": earnings_dates,
    }


# ── technicals ────────────────────────────────────────────────────────────────

def compute_technicals(hist: pd.DataFrame) -> dict:
    if hist is None or hist.empty or len(hist) < 10:
        return {}

    close = hist["Close"]
    volume = hist["Volume"]
    high = hist["High"]
    low = hist["Low"]

    def sma(n):
        return close.rolling(n).mean().iloc[-1] if len(close) >= n else None

    def ema(n):
        return close.ewm(span=n, adjust=False).mean().iloc[-1] if len(close) >= n else None

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi_series = 100 - (100 / (1 + rs))
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else None

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    macd_val = float(macd_line.iloc[-1]) if not macd_line.empty else None
    signal_val = float(signal_line.iloc[-1]) if not signal_line.empty else None
    macd_hist_val = float(macd_hist.iloc[-1]) if not macd_hist.empty else None

    # Bollinger Bands (20, 2σ)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pct = ((close - bb_lower) / (bb_upper - bb_lower)).iloc[-1]

    # ATR (14-day)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    # Volume
    avg_vol_30 = float(volume.rolling(30).mean().iloc[-1]) if len(volume) >= 30 else None
    last_vol = float(volume.iloc[-1])
    vol_ratio = last_vol / avg_vol_30 if avg_vol_30 and avg_vol_30 > 0 else None

    # Price vs MAs
    current = float(close.iloc[-1])
    sma20 = sma(20)
    sma50 = sma(50)
    sma200 = sma(200)

    # 52-week high/low from the 1y hist
    hi52 = float(high.max())
    lo52 = float(low.min())
    pct_from_hi = ((current - hi52) / hi52) * 100
    pct_from_lo = ((current - lo52) / lo52) * 100

    # Momentum
    ret_1m  = ((current / float(close.iloc[-21])) - 1) * 100  if len(close) > 21  else None
    ret_3m  = ((current / float(close.iloc[-63])) - 1) * 100  if len(close) > 63  else None
    ret_6m  = ((current / float(close.iloc[-126])) - 1) * 100 if len(close) > 126 else None
    ret_1y  = ((current / float(close.iloc[0])) - 1) * 100

    # 30-day realised vol (annualised)
    daily_rets = close.pct_change(fill_method=None).dropna()
    vol_30 = float(daily_rets.iloc[-30:].std(ddof=1) * math.sqrt(252) * 100) if len(daily_rets) >= 30 else None

    # Golden/death cross
    cross_signal = None
    if sma50 and sma200:
        if sma50 > sma200:
            cross_signal = "Golden Cross (bullish)"
        else:
            cross_signal = "Death Cross (bearish)"

    # RSI interpretation
    rsi_signal = None
    if rsi is not None:
        if rsi > 70:
            rsi_signal = "Overbought"
        elif rsi < 30:
            rsi_signal = "Oversold"
        else:
            rsi_signal = "Neutral"

    # MACD signal
    macd_signal = None
    if macd_val is not None and signal_val is not None:
        macd_signal = "Bullish" if macd_val > signal_val else "Bearish"

    return {
        "current_price": current,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "rsi": rsi,
        "rsi_signal": rsi_signal,
        "macd": macd_val,
        "macd_signal_line": signal_val,
        "macd_histogram": macd_hist_val,
        "macd_signal": macd_signal,
        "bb_pct_b": float(bb_pct) if bb_pct is not None and not math.isnan(float(bb_pct)) else None,
        "bb_upper": float(bb_upper.iloc[-1]),
        "bb_lower": float(bb_lower.iloc[-1]),
        "atr": atr,
        "avg_volume_30d": avg_vol_30,
        "last_volume": last_vol,
        "volume_ratio": vol_ratio,
        "hi_52w": hi52,
        "lo_52w": lo52,
        "pct_from_52w_high": pct_from_hi,
        "pct_from_52w_low": pct_from_lo,
        "return_1m": ret_1m,
        "return_3m": ret_3m,
        "return_6m": ret_6m,
        "return_1y": ret_1y,
        "volatility_30d_ann": vol_30,
        "cross_signal": cross_signal,
    }


# ── fundamentals ──────────────────────────────────────────────────────────────

def extract_fundamentals(data: dict) -> dict:
    info = data["info"]
    qi = data["quarterly_income"]
    qb = data["quarterly_balance"]
    qcf = data["quarterly_cashflow"]

    def _get(key, default=None):
        v = info.get(key, default)
        return v if v not in (None, "N/A", float("inf"), float("-inf")) else default

    # Valuation
    pe_ttm       = _get("trailingPE")
    pe_fwd       = _get("forwardPE")
    ps_ttm       = _get("priceToSalesTrailing12Months")
    pb           = _get("priceToBook")
    peg          = _get("pegRatio")
    ev_ebitda    = _get("enterpriseToEbitda")
    ev_rev       = _get("enterpriseToRevenue")
    market_cap   = _get("marketCap")
    enterprise_v = _get("enterpriseValue")

    # Growth
    rev_growth_yoy = _get("revenueGrowth")
    eps_growth     = _get("earningsGrowth")
    rev_qoq        = _get("revenueQuarterlyGrowth")

    # Profitability
    gross_margin   = _get("grossMargins")
    op_margin      = _get("operatingMargins")
    net_margin     = _get("profitMargins")
    roe            = _get("returnOnEquity")
    roa            = _get("returnOnAssets")

    # Per share
    eps_ttm = _get("trailingEps")
    eps_fwd = _get("forwardEps")
    bvps    = _get("bookValue")

    # Balance sheet
    total_cash  = _get("totalCash")
    total_debt  = _get("totalDebt")
    current_ratio = _get("currentRatio")
    debt_equity   = _get("debtToEquity")
    fcf           = _get("freeCashflow")

    # Revenue/earnings quarterly trend
    quarterly_rev = []
    quarterly_eps = []
    if not qi.empty:
        rev_row = None
        for idx_name in ["Total Revenue", "Revenue"]:
            if idx_name in qi.index:
                rev_row = qi.loc[idx_name]
                break
        if rev_row is not None:
            for col in rev_row.index[:4]:  # last 4 quarters
                v = rev_row[col]
                try:
                    quarterly_rev.append({
                        "quarter": str(col)[:10],
                        "revenue_usd": float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None,
                    })
                except Exception:
                    pass

        # Net income
        ni_row = None
        for idx_name in ["Net Income", "Net Income Common Stockholders"]:
            if idx_name in qi.index:
                ni_row = qi.loc[idx_name]
                break
        if ni_row is not None:
            for col in ni_row.index[:4]:
                v = ni_row[col]
                try:
                    quarterly_eps.append({
                        "quarter": str(col)[:10],
                        "net_income_usd": float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None,
                    })
                except Exception:
                    pass

    # Analyst data
    analyst = data.get("analyst_price_targets", {})
    rec = data.get("recommendations_summary")
    rec_counts = {}
    if rec is not None and not (isinstance(rec, pd.DataFrame) and rec.empty):
        try:
            if isinstance(rec, pd.DataFrame):
                latest = rec.iloc[0] if len(rec) > 0 else {}
                rec_counts = latest.to_dict() if hasattr(latest, "to_dict") else {}
        except Exception:
            pass

    return {
        # valuation
        "pe_ttm": pe_ttm,
        "pe_fwd": pe_fwd,
        "ps_ttm": ps_ttm,
        "pb": pb,
        "peg": peg,
        "ev_ebitda": ev_ebitda,
        "ev_revenue": ev_rev,
        "market_cap": market_cap,
        "enterprise_value": enterprise_v,
        # growth
        "revenue_growth_yoy": rev_growth_yoy,
        "eps_growth_yoy": eps_growth,
        "revenue_growth_qoq": rev_qoq,
        # profitability
        "gross_margin": gross_margin,
        "operating_margin": op_margin,
        "net_margin": net_margin,
        "roe": roe,
        "roa": roa,
        # per share
        "eps_ttm": eps_ttm,
        "eps_fwd": eps_fwd,
        "bvps": bvps,
        # balance sheet
        "total_cash": total_cash,
        "total_debt": total_debt,
        "current_ratio": current_ratio,
        "debt_to_equity": debt_equity,
        "free_cash_flow": fcf,
        # quarterly detail
        "quarterly_revenue": quarterly_rev,
        "quarterly_net_income": quarterly_eps,
        # analyst
        "analyst_target_mean": analyst.get("mean") if isinstance(analyst, dict) else None,
        "analyst_target_high": analyst.get("high") if isinstance(analyst, dict) else None,
        "analyst_target_low": analyst.get("low") if isinstance(analyst, dict) else None,
        "analyst_rec_counts": rec_counts,
        "target_median_consensus": info.get("targetMedianPrice"),
        "number_of_analysts": info.get("numberOfAnalystOpinions"),
        "recommendation": info.get("recommendationKey"),
    }


# ── peer comparison ───────────────────────────────────────────────────────────

def fetch_peer_comparison(peers: list[str]) -> list[dict]:
    rows = []
    for sym in peers:
        try:
            t = yf.Ticker(sym)
            info = t.info or {}
            finfo = t.fast_info
            price = getattr(finfo, "last_price", None) or info.get("currentPrice")
            rows.append({
                "ticker": sym,
                "price": price,
                "market_cap": info.get("marketCap"),
                "pe_ttm": info.get("trailingPE"),
                "pe_fwd": info.get("forwardPE"),
                "ps_ttm": info.get("priceToSalesTrailing12Months"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                "revenue_growth": info.get("revenueGrowth"),
                "gross_margin": info.get("grossMargins"),
                "op_margin": info.get("operatingMargins"),
                "beta": info.get("beta"),
                "return_1y": None,  # computed below
            })
        except Exception as e:
            rows.append({"ticker": sym, "error": str(e)})

    # 1-year returns for each peer
    for row in rows:
        if "error" in row:
            continue
        try:
            h = yf.Ticker(row["ticker"]).history(period="1y")
            if h is not None and len(h) > 5:
                ret = ((float(h["Close"].iloc[-1]) / float(h["Close"].iloc[0])) - 1) * 100
                row["return_1y"] = round(ret, 1)
        except Exception:
            pass

    return rows


# ── Claude AI report ──────────────────────────────────────────────────────────

def generate_amd_report(
    fundamentals: dict,
    technicals: dict,
    peers: list[dict],
    info: dict,
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "ANTHROPIC_API_KEY not set — skipping AI report."

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # build compact but rich data packet for the prompt
    amd_peer = next((p for p in peers if p["ticker"] == "AMD"), {})

    prompt_data = {
        "ticker": "AMD",
        "company": "Advanced Micro Devices Inc.",
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "technicals": {
            "current_price_usd": round(technicals.get("current_price", 0), 2),
            "52w_high": round(technicals.get("hi_52w", 0), 2),
            "52w_low": round(technicals.get("lo_52w", 0), 2),
            "pct_from_52w_high": round(technicals.get("pct_from_52w_high", 0), 1),
            "sma20": round(technicals.get("sma20") or 0, 2),
            "sma50": round(technicals.get("sma50") or 0, 2),
            "sma200": round(technicals.get("sma200") or 0, 2),
            "rsi_14": round(technicals.get("rsi") or 0, 1),
            "rsi_signal": technicals.get("rsi_signal"),
            "macd_signal": technicals.get("macd_signal"),
            "cross_signal": technicals.get("cross_signal"),
            "bb_pct_b": round(technicals.get("bb_pct_b") or 0, 2),
            "atr": round(technicals.get("atr") or 0, 2),
            "volume_ratio_vs_30d_avg": round(technicals.get("volume_ratio") or 1, 2),
            "volatility_30d_ann_pct": round(technicals.get("volatility_30d_ann") or 0, 1),
            "returns": {
                "1m": round(technicals.get("return_1m") or 0, 1),
                "3m": round(technicals.get("return_3m") or 0, 1),
                "6m": round(technicals.get("return_6m") or 0, 1),
                "1y": round(technicals.get("return_1y") or 0, 1),
            },
        },
        "fundamentals": {
            "market_cap": _billions(fundamentals.get("market_cap")),
            "enterprise_value": _billions(fundamentals.get("enterprise_value")),
            "valuation": {
                "pe_ttm": fundamentals.get("pe_ttm"),
                "pe_forward": fundamentals.get("pe_fwd"),
                "ps_ttm": fundamentals.get("ps_ttm"),
                "ev_ebitda": fundamentals.get("ev_ebitda"),
                "peg_ratio": fundamentals.get("peg"),
            },
            "growth": {
                "revenue_yoy_pct": (
                    round(fundamentals["revenue_growth_yoy"] * 100, 1)
                    if fundamentals.get("revenue_growth_yoy") else None
                ),
                "eps_yoy_pct": (
                    round(fundamentals["eps_growth_yoy"] * 100, 1)
                    if fundamentals.get("eps_growth_yoy") else None
                ),
            },
            "margins": {
                "gross_pct": (
                    round(fundamentals["gross_margin"] * 100, 1)
                    if fundamentals.get("gross_margin") else None
                ),
                "operating_pct": (
                    round(fundamentals["operating_margin"] * 100, 1)
                    if fundamentals.get("operating_margin") else None
                ),
                "net_pct": (
                    round(fundamentals["net_margin"] * 100, 1)
                    if fundamentals.get("net_margin") else None
                ),
            },
            "returns": {
                "roe_pct": (
                    round(fundamentals["roe"] * 100, 1)
                    if fundamentals.get("roe") else None
                ),
                "roa_pct": (
                    round(fundamentals["roa"] * 100, 1)
                    if fundamentals.get("roa") else None
                ),
            },
            "balance_sheet": {
                "cash_usd": _billions(fundamentals.get("total_cash")),
                "debt_usd": _billions(fundamentals.get("total_debt")),
                "current_ratio": fundamentals.get("current_ratio"),
                "debt_to_equity": fundamentals.get("debt_to_equity"),
                "free_cash_flow": _billions(fundamentals.get("free_cash_flow")),
            },
            "eps": {
                "ttm": fundamentals.get("eps_ttm"),
                "forward": fundamentals.get("eps_fwd"),
            },
            "quarterly_revenue_last_4": fundamentals.get("quarterly_revenue", []),
        },
        "analyst_consensus": {
            "recommendation": fundamentals.get("recommendation"),
            "mean_price_target": fundamentals.get("analyst_target_mean"),
            "high_target": fundamentals.get("analyst_target_high"),
            "low_target": fundamentals.get("analyst_target_low"),
            "median_consensus": fundamentals.get("target_median_consensus"),
            "num_analysts": fundamentals.get("number_of_analysts"),
        },
        "peer_comparison": [
            {
                "ticker": p["ticker"],
                "market_cap": _billions(p.get("market_cap")),
                "pe_fwd": p.get("pe_fwd"),
                "ev_ebitda": p.get("ev_ebitda"),
                "gross_margin_pct": (
                    round(p["gross_margin"] * 100, 1) if p.get("gross_margin") else None
                ),
                "op_margin_pct": (
                    round(p["op_margin"] * 100, 1) if p.get("op_margin") else None
                ),
                "revenue_growth_pct": (
                    round(p["revenue_growth"] * 100, 1) if p.get("revenue_growth") else None
                ),
                "return_1y_pct": p.get("return_1y"),
                "beta": p.get("beta"),
            }
            for p in peers if "error" not in p
        ],
        "sector_context": info.get("sector"),
        "industry": info.get("industry"),
        "description_snippet": (info.get("longBusinessSummary", "") or "")[:600],
    }

    prompt = f"""You are a senior equity research analyst at a top-tier investment bank (Goldman Sachs / Morgan Stanley tier). Produce a full, institutional-quality research report on AMD (Advanced Micro Devices) using the real market data below.

Today's date: {prompt_data['analysis_date']}

DATA:
{json.dumps(prompt_data, indent=2)}

Write EXACTLY this structure. Be specific: cite actual numbers, percentages, price levels. No waffle.

---

# AMD (Advanced Micro Devices) — Equity Research Report
**Date:** {prompt_data['analysis_date']} | **Rating:** [Your rating: BUY / HOLD / SELL] | **12-Month Price Target:** $[X]

---

## 1. Executive Summary (3-4 sentences)
Overall investment thesis. Current price vs. target. Key catalyst near-term.

## 2. Business Overview (2-3 sentences)
What AMD does, key segments (Data Center, Client, Gaming, Embedded), competitive moat.

## 3. Recent Performance
- Q1/Q4 earnings highlights: revenue, EPS beat/miss, guidance
- Price action: where is stock relative to 52w range, MAs, key moves
- What drove the most recent +18% post-earnings surge (AI infrastructure demand, data center)

## 4. Technical Analysis
### Price Structure
- Current price vs. SMA 20/50/200 — is it in uptrend?
- 52-week range context
- Key support and resistance levels

### Momentum Indicators
- RSI reading and what it implies
- MACD status (bullish/bearish crossover?)
- Bollinger Band position (%B)
- Volume analysis

### Outlook
Bull case / bear case based on technicals. Near-term price levels to watch.

## 5. Fundamental Analysis
### Valuation
Compare AMD's P/E (fwd), EV/EBITDA, P/S to peers (NVDA, INTC, QCOM). Is AMD cheap, fair, or expensive?

### Growth Profile
Revenue growth rate YoY. EPS trajectory. What's driving it (MI300X AI chip ramp, EPYC server CPUs)?

### Margins
Gross / operating / net margins vs. history and vs. peers. Margin expansion thesis.

### Balance Sheet & Cash Flow
Net cash/debt position. Free cash flow generation. Financial health score.

## 6. Competitive Landscape
### AMD vs NVDA vs INTC
Direct comparison table (from peer data above). Where AMD wins, where it loses.

### Market Share
Data Center GPU: MI300X vs H100/H200. Server CPU: EPYC vs Xeon.
Gaming: Radeon vs GeForce trajectory.

## 7. Catalysts & Risks
### Bull Catalysts (3 specific)
1.
2.
3.

### Key Risks (3 specific)
1.
2.
3.

## 8. Valuation & Price Target
Build a simple narrative DCF / multiple-based target. Justify your 12-month price target with a specific number and the assumptions behind it.

## 9. Investment Recommendation
**Rating: BUY / HOLD / SELL** — one decisive paragraph. State the risk/reward clearly.

---

Write at Goldman Sachs analyst precision. No generic statements. Every claim must be anchored to a specific data point."""

    try:
        message = client.messages.create(
            model=os.environ.get("CLAUDE_MODEL", "claude-opus-4-5"),
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"AI report failed: {e}"


# ── HTML output ────────────────────────────────────────────────────────────────

def generate_html(
    report_md: str,
    technicals: dict,
    fundamentals: dict,
    peers: list[dict],
    hist_1y: pd.DataFrame,
    output_dir: str = ".",
) -> str:
    """Generate a self-contained HTML report with Chart.js charts."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"amd_analysis_{date_str}.html"
    filepath = Path(output_dir) / filename

    # Price history for chart
    if hist_1y is not None and not hist_1y.empty:
        price_dates = [str(d)[:10] for d in hist_1y.index]
        price_closes = [round(float(c), 2) for c in hist_1y["Close"]]
        price_vols = [int(v) for v in hist_1y["Volume"]]

        # SMA overlays
        close_s = hist_1y["Close"]
        sma20_vals = [round(float(v), 2) if not math.isnan(float(v)) else None
                      for v in close_s.rolling(20).mean()]
        sma50_vals = [round(float(v), 2) if not math.isnan(float(v)) else None
                      for v in close_s.rolling(50).mean()]
        sma200_vals = [round(float(v), 2) if not math.isnan(float(v)) else None
                       for v in close_s.rolling(200).mean()]
    else:
        price_dates = price_closes = price_vols = sma20_vals = sma50_vals = sma200_vals = []

    # Peer comparison data
    peer_tickers = [p["ticker"] for p in peers if "error" not in p]
    peer_pe_fwd = [p.get("pe_fwd") for p in peers if "error" not in p]
    peer_ev_ebitda = [p.get("ev_ebitda") for p in peers if "error" not in p]
    peer_gm = [round(p["gross_margin"] * 100, 1) if p.get("gross_margin") else None
               for p in peers if "error" not in p]
    peer_ret1y = [p.get("return_1y") for p in peers if "error" not in p]

    # Pre-build peer table rows (avoids nested f-string issue with Python < 3.12)
    peer_rows_list = []
    for p in peers:
        if "error" in p:
            continue
        is_amd = p.get("ticker") == "AMD"
        row_style = 'style="background:rgba(237,28,36,0.06);font-weight:600;"' if is_amd else ""
        td_color = "var(--amd-red)" if is_amd else "var(--blue)"
        gm_class = "green" if (p.get("gross_margin") or 0) > 0.45 else ""
        op_class = "green" if (p.get("op_margin") or 0) > 0.15 else "red"
        r1y_class = "green" if (p.get("return_1y") or 0) >= 0 else "red"
        peer_rows_list.append(
            f'<tr {row_style}>'
            f'<td style="font-weight:600;color:{td_color};">{p.get("ticker","")}</td>'
            f'<td>{_billions(p.get("market_cap"))}</td>'
            f'<td>{_fmt(p.get("pe_fwd"),1)}x</td>'
            f'<td>{_fmt(p.get("ev_ebitda"),1)}x</td>'
            f'<td class="{gm_class}">{_pct((p.get("gross_margin") or 0)*100)}</td>'
            f'<td class="{op_class}">{_pct((p.get("op_margin") or 0)*100)}</td>'
            f'<td class="green">{_pct((p.get("revenue_growth") or 0)*100)}</td>'
            f'<td class="{r1y_class}">{_pct(p.get("return_1y"))}</td>'
            f'<td>{_fmt(p.get("beta"),2)}</td>'
            f'</tr>'
        )
    peer_rows_html = "\n".join(peer_rows_list)

    # Quarterly revenue
    q_rev = fundamentals.get("quarterly_revenue", [])
    q_labels = [q["quarter"] for q in q_rev]
    q_values = [round(q["revenue_usd"] / 1e9, 2) if q.get("revenue_usd") else None for q in q_rev]

    # Scorecard items
    curr = technicals.get("current_price", 0)
    sma50 = technicals.get("sma50")
    sma200 = technicals.get("sma200")
    rsi = technicals.get("rsi")
    hi52 = technicals.get("hi_52w", 0)
    lo52 = technicals.get("lo_52w", 0)

    def _score_color(condition):
        return "#28a745" if condition else "#dc3545"

    # convert markdown report to basic HTML
    import re
    html_report = report_md
    # h1
    html_report = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_report, flags=re.MULTILINE)
    # h2
    html_report = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_report, flags=re.MULTILINE)
    # h3
    html_report = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_report, flags=re.MULTILINE)
    # bold
    html_report = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_report)
    # bullets
    html_report = re.sub(r'^- (.+)$', r'<li>\1</li>', html_report, flags=re.MULTILINE)
    # numbered
    html_report = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', html_report, flags=re.MULTILINE)
    # hr
    html_report = html_report.replace('---', '<hr>')
    # paragraphs
    lines = html_report.split('\n')
    result = []
    in_li = False
    for line in lines:
        if line.strip().startswith('<li>'):
            if not in_li:
                result.append('<ul>')
                in_li = True
            result.append(line)
        else:
            if in_li:
                result.append('</ul>')
                in_li = False
            if line.strip() and not any(line.strip().startswith(t) for t in ['<h', '<hr', '<ul', '</ul', '<li']):
                result.append(f'<p>{line}</p>')
            else:
                result.append(line)
    if in_li:
        result.append('</ul>')
    html_report = '\n'.join(result)

    target_mean = fundamentals.get("analyst_target_mean")
    upside = ((target_mean - curr) / curr * 100) if target_mean and curr else None

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AMD Deep Analysis — {date_str}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --green: #3fb950; --red: #f85149;
    --blue: #58a6ff; --orange: #d29922; --purple: #bc8cff;
    --amd-red: #ed1c24;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 14px; line-height: 1.6; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  header {{ border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 28px; display: flex; align-items: center; gap: 20px; }}
  .amd-logo {{ font-size: 28px; font-weight: 900; color: var(--amd-red); letter-spacing: -1px; }}
  .header-meta {{ flex: 1; }}
  .header-meta h1 {{ font-size: 22px; color: var(--text); }}
  .header-meta .sub {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  .price-hero {{ text-align: right; }}
  .price-hero .price {{ font-size: 32px; font-weight: 700; color: var(--text); }}
  .price-hero .change {{ font-size: 14px; margin-top: 4px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
  .card h3 {{ font-size: 12px; text-transform: uppercase; color: var(--muted); letter-spacing: 0.5px; margin-bottom: 12px; }}
  .metric-row {{ display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid var(--border); }}
  .metric-row:last-child {{ border-bottom: none; }}
  .metric-label {{ color: var(--muted); font-size: 12px; }}
  .metric-value {{ font-weight: 600; font-size: 13px; }}
  .green {{ color: var(--green); }}
  .red {{ color: var(--red); }}
  .blue {{ color: var(--blue); }}
  .orange {{ color: var(--orange); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
  .badge-green {{ background: rgba(63,185,80,0.15); color: var(--green); }}
  .badge-red {{ background: rgba(248,81,73,0.15); color: var(--red); }}
  .badge-orange {{ background: rgba(210,153,34,0.15); color: var(--orange); }}
  .badge-blue {{ background: rgba(88,166,255,0.15); color: var(--blue); }}
  .chart-wrap {{ position: relative; height: 280px; }}
  .chart-wrap-sm {{ position: relative; height: 200px; }}
  .full-width {{ grid-column: 1 / -1; }}
  .report-section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 28px; margin-bottom: 20px; }}
  .report-section h1 {{ font-size: 20px; margin-bottom: 12px; color: var(--blue); border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
  .report-section h2 {{ font-size: 16px; margin: 20px 0 8px; color: var(--text); }}
  .report-section h3 {{ font-size: 14px; margin: 14px 0 6px; color: var(--orange); }}
  .report-section p {{ margin-bottom: 10px; color: var(--text); line-height: 1.7; }}
  .report-section ul {{ margin: 8px 0 12px 20px; }}
  .report-section li {{ margin-bottom: 4px; color: var(--text); }}
  .report-section hr {{ border: none; border-top: 1px solid var(--border); margin: 16px 0; }}
  .report-section strong {{ color: var(--blue); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ background: rgba(88,166,255,0.08); color: var(--muted); text-align: left; padding: 8px 12px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; border-bottom: 1px solid var(--border); }}
  td {{ padding: 8px 12px; border-bottom: 1px solid rgba(48,54,61,0.5); }}
  tr:hover td {{ background: rgba(88,166,255,0.04); }}
  .section-title {{ font-size: 14px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }}
  .scorecard-item {{ display: flex; align-items: center; gap: 8px; padding: 6px 0; border-bottom: 1px solid rgba(48,54,61,0.4); }}
  .scorecard-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}
  .scorecard-label {{ flex: 1; font-size: 12px; color: var(--muted); }}
  .scorecard-value {{ font-size: 12px; font-weight: 600; }}
  @media (max-width: 900px) {{ .grid-2,.grid-3,.grid-4 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">

<header>
  <div class="amd-logo">AMD</div>
  <div class="header-meta">
    <h1>Advanced Micro Devices, Inc.</h1>
    <div class="sub">NASDAQ: AMD &nbsp;|&nbsp; Semiconductors &nbsp;|&nbsp; Deep Analysis Report &nbsp;|&nbsp; {date_str}</div>
  </div>
  <div class="price-hero">
    <div class="price">${_fmt(curr)}</div>
    <div class="change {'green' if (technicals.get('return_1m') or 0) >= 0 else 'red'}">
      1M: {_pct(technicals.get('return_1m'))} &nbsp;|&nbsp; 1Y: {_pct(technicals.get('return_1y'))}
    </div>
  </div>
</header>

<!-- Scorecard row -->
<div class="grid-4" style="margin-bottom:20px;">
  <div class="card">
    <h3>52-Week Range</h3>
    <div class="metric-row"><span class="metric-label">High</span><span class="metric-value">${_fmt(hi52)}</span></div>
    <div class="metric-row"><span class="metric-label">Low</span><span class="metric-value">${_fmt(lo52)}</span></div>
    <div class="metric-row"><span class="metric-label">From High</span><span class="metric-value {'red' if (technicals.get('pct_from_52w_high') or 0) < -15 else 'green'}">{_pct(technicals.get('pct_from_52w_high'))}</span></div>
    <div class="metric-row"><span class="metric-label">From Low</span><span class="metric-value green">{_pct(technicals.get('pct_from_52w_low'))}</span></div>
  </div>
  <div class="card">
    <h3>Moving Averages</h3>
    <div class="metric-row"><span class="metric-label">SMA 20</span><span class="metric-value {'green' if sma50 and curr > sma50 else 'red'}">${_fmt(sma20_vals[-1] if sma20_vals else None)}</span></div>
    <div class="metric-row"><span class="metric-label">SMA 50</span><span class="metric-value {'green' if sma50 and curr > sma50 else 'red'}">${_fmt(sma50)}</span></div>
    <div class="metric-row"><span class="metric-label">SMA 200</span><span class="metric-value {'green' if sma200 and curr > sma200 else 'red'}">${_fmt(sma200)}</span></div>
    <div class="metric-row"><span class="metric-label">Trend</span><span class="metric-value">{technicals.get('cross_signal', 'N/A')}</span></div>
  </div>
  <div class="card">
    <h3>Momentum</h3>
    <div class="metric-row"><span class="metric-label">RSI (14)</span><span class="metric-value {'red' if (rsi or 50) > 70 else ('green' if (rsi or 50) < 30 else 'blue')}">{_fmt(rsi, 1)} — {technicals.get('rsi_signal', '')}</span></div>
    <div class="metric-row"><span class="metric-label">MACD</span><span class="metric-value {'green' if technicals.get('macd_signal') == 'Bullish' else 'red'}">{technicals.get('macd_signal', 'N/A')}</span></div>
    <div class="metric-row"><span class="metric-label">BB %B</span><span class="metric-value">{_fmt(technicals.get('bb_pct_b'), 2)}</span></div>
    <div class="metric-row"><span class="metric-label">Vol Ratio</span><span class="metric-value">{_fmt(technicals.get('volume_ratio'), 2)}x</span></div>
  </div>
  <div class="card">
    <h3>Analyst Consensus</h3>
    <div class="metric-row"><span class="metric-label">Rating</span><span class="metric-value badge {'badge-green' if fundamentals.get('recommendation','').lower() in ('buy','strong_buy') else 'badge-orange'}">{(fundamentals.get('recommendation') or 'N/A').upper()}</span></div>
    <div class="metric-row"><span class="metric-label">Avg Target</span><span class="metric-value">${_fmt(fundamentals.get('analyst_target_mean'))}</span></div>
    <div class="metric-row"><span class="metric-label">Upside</span><span class="metric-value {'green' if (upside or 0) > 0 else 'red'}">{_pct(upside)}</span></div>
    <div class="metric-row"><span class="metric-label"># Analysts</span><span class="metric-value">{fundamentals.get('number_of_analysts', 'N/A')}</span></div>
  </div>
</div>

<!-- Price chart + Fundamentals -->
<div class="grid-2">
  <div class="card full-width" style="grid-column:1/2;">
    <h3>1-Year Price Chart with Moving Averages</h3>
    <div class="chart-wrap">
      <canvas id="priceChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h3>Valuation Multiples</h3>
    <div class="metric-row"><span class="metric-label">P/E TTM</span><span class="metric-value">{_fmt(fundamentals.get('pe_ttm'), 1)}x</span></div>
    <div class="metric-row"><span class="metric-label">P/E Forward</span><span class="metric-value">{_fmt(fundamentals.get('pe_fwd'), 1)}x</span></div>
    <div class="metric-row"><span class="metric-label">EV/EBITDA</span><span class="metric-value">{_fmt(fundamentals.get('ev_ebitda'), 1)}x</span></div>
    <div class="metric-row"><span class="metric-label">P/S TTM</span><span class="metric-value">{_fmt(fundamentals.get('ps_ttm'), 1)}x</span></div>
    <div class="metric-row"><span class="metric-label">PEG</span><span class="metric-value">{_fmt(fundamentals.get('peg'), 2)}</span></div>
    <div class="metric-row"><span class="metric-label">P/B</span><span class="metric-value">{_fmt(fundamentals.get('pb'), 1)}x</span></div>
    <div class="metric-row"><span class="metric-label">Market Cap</span><span class="metric-value">{_billions(fundamentals.get('market_cap'))}</span></div>
    <div class="metric-row"><span class="metric-label">EV</span><span class="metric-value">{_billions(fundamentals.get('enterprise_value'))}</span></div>
  </div>
  <div class="card">
    <h3>Profitability & Growth</h3>
    <div class="metric-row"><span class="metric-label">Gross Margin</span><span class="metric-value green">{_pct((fundamentals.get('gross_margin') or 0)*100)}</span></div>
    <div class="metric-row"><span class="metric-label">Op. Margin</span><span class="metric-value {'green' if (fundamentals.get('operating_margin') or 0) > 0 else 'red'}">{_pct((fundamentals.get('operating_margin') or 0)*100)}</span></div>
    <div class="metric-row"><span class="metric-label">Net Margin</span><span class="metric-value {'green' if (fundamentals.get('net_margin') or 0) > 0 else 'red'}">{_pct((fundamentals.get('net_margin') or 0)*100)}</span></div>
    <div class="metric-row"><span class="metric-label">Rev Growth YoY</span><span class="metric-value green">{_pct((fundamentals.get('revenue_growth_yoy') or 0)*100)}</span></div>
    <div class="metric-row"><span class="metric-label">EPS Growth YoY</span><span class="metric-value green">{_pct((fundamentals.get('eps_growth_yoy') or 0)*100)}</span></div>
    <div class="metric-row"><span class="metric-label">ROE</span><span class="metric-value">{_pct((fundamentals.get('roe') or 0)*100)}</span></div>
    <div class="metric-row"><span class="metric-label">ROA</span><span class="metric-value">{_pct((fundamentals.get('roa') or 0)*100)}</span></div>
    <div class="metric-row"><span class="metric-label">Free Cash Flow</span><span class="metric-value">{_billions(fundamentals.get('free_cash_flow'))}</span></div>
  </div>
</div>

<!-- Quarterly Revenue -->
<div class="grid-2" style="margin-top:0;">
  <div class="card">
    <h3>Quarterly Revenue (last 4 quarters)</h3>
    <div class="chart-wrap-sm">
      <canvas id="revenueChart"></canvas>
    </div>
  </div>
  <div class="card">
    <h3>Balance Sheet</h3>
    <div class="metric-row"><span class="metric-label">Total Cash</span><span class="metric-value green">{_billions(fundamentals.get('total_cash'))}</span></div>
    <div class="metric-row"><span class="metric-label">Total Debt</span><span class="metric-value red">{_billions(fundamentals.get('total_debt'))}</span></div>
    <div class="metric-row"><span class="metric-label">Current Ratio</span><span class="metric-value">{_fmt(fundamentals.get('current_ratio'), 2)}</span></div>
    <div class="metric-row"><span class="metric-label">Debt/Equity</span><span class="metric-value">{_fmt(fundamentals.get('debt_to_equity'), 1)}</span></div>
    <div class="metric-row"><span class="metric-label">EPS TTM</span><span class="metric-value">${_fmt(fundamentals.get('eps_ttm'), 2)}</span></div>
    <div class="metric-row"><span class="metric-label">EPS Forward</span><span class="metric-value blue">${_fmt(fundamentals.get('eps_fwd'), 2)}</span></div>
    <div class="metric-row"><span class="metric-label">52w Volatility</span><span class="metric-value">{_pct(technicals.get('volatility_30d_ann'))}</span></div>
    <div class="metric-row"><span class="metric-label">Beta</span><span class="metric-value">{_fmt(next((p.get('beta') for p in peers if p.get('ticker')=='AMD'), None), 2)}</span></div>
  </div>
</div>

<!-- Peer comparison -->
<div class="card" style="margin-bottom:20px;">
  <h3>Peer Comparison — AMD vs NVDA vs INTC vs QCOM</h3>
  <table>
    <thead>
      <tr>
        <th>Ticker</th><th>Market Cap</th><th>P/E Fwd</th><th>EV/EBITDA</th>
        <th>Gross Margin</th><th>Op Margin</th><th>Rev Growth</th><th>1Y Return</th><th>Beta</th>
      </tr>
    </thead>
    <tbody>
      {peer_rows_html}
    </tbody>
  </table>
</div>

<!-- Peer charts -->
<div class="grid-2" style="margin-bottom:20px;">
  <div class="card">
    <h3>Forward P/E Comparison</h3>
    <div class="chart-wrap-sm"><canvas id="peerPeChart"></canvas></div>
  </div>
  <div class="card">
    <h3>1-Year Return Comparison</h3>
    <div class="chart-wrap-sm"><canvas id="peerRetChart"></canvas></div>
  </div>
</div>

<!-- Returns momentum -->
<div class="card" style="margin-bottom:20px;">
  <h3>Price Returns — AMD</h3>
  <div style="display:flex;gap:24px;flex-wrap:wrap;">
    {' '.join(f'''<div style="text-align:center;padding:12px 20px;background:rgba(88,166,255,0.05);border-radius:6px;border:1px solid var(--border)">
      <div style="font-size:11px;color:var(--muted);margin-bottom:4px;">{period}</div>
      <div style="font-size:20px;font-weight:700;color:{'var(--green)' if (val or 0) >= 0 else 'var(--red)'};">{_pct(val)}</div>
    </div>'''
    for period, val in [
      ("1 Month", technicals.get("return_1m")),
      ("3 Months", technicals.get("return_3m")),
      ("6 Months", technicals.get("return_6m")),
      ("1 Year", technicals.get("return_1y")),
    ])}
  </div>
</div>

<!-- AI Report -->
<div class="report-section">
  <div style="font-size:11px;color:var(--muted);margin-bottom:16px;">GENERATED BY CLAUDE AI · INSTITUTIONAL RESEARCH FORMAT · {date_str}</div>
  {html_report}
</div>

<footer style="text-align:center;color:var(--muted);font-size:11px;padding:20px 0;border-top:1px solid var(--border);margin-top:8px;">
  AMD Deep Analysis · Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} · Data via yfinance · AI via Claude · Not financial advice
</footer>

</div><!-- /container -->

<script>
const priceCtx = document.getElementById('priceChart').getContext('2d');
new Chart(priceCtx, {{
  type: 'line',
  data: {{
    labels: {json.dumps(price_dates)},
    datasets: [
      {{
        label: 'AMD Close',
        data: {json.dumps(price_closes)},
        borderColor: '#ed1c24',
        backgroundColor: 'rgba(237,28,36,0.05)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.2,
        yAxisID: 'y',
      }},
      {{
        label: 'SMA 20',
        data: {json.dumps(sma20_vals)},
        borderColor: '#f28e2b',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        spanGaps: true,
        yAxisID: 'y',
      }},
      {{
        label: 'SMA 50',
        data: {json.dumps(sma50_vals)},
        borderColor: '#58a6ff',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        spanGaps: true,
        yAxisID: 'y',
      }},
      {{
        label: 'SMA 200',
        data: {json.dumps(sma200_vals)},
        borderColor: '#bc8cff',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        spanGaps: true,
        yAxisID: 'y',
      }},
      {{
        label: 'Volume',
        data: {json.dumps(price_vols)},
        type: 'bar',
        backgroundColor: 'rgba(88,166,255,0.15)',
        borderWidth: 0,
        yAxisID: 'y2',
      }},
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }},
      tooltip: {{ backgroundColor: '#1c2128', borderColor: '#30363d', borderWidth: 1 }},
    }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 8, font:{{size:10}} }}, grid: {{ color: 'rgba(48,54,61,0.3)' }} }},
      y: {{ position:'left', ticks: {{ color: '#8b949e', font:{{size:10}}, callback: v => '$'+v }}, grid: {{ color: 'rgba(48,54,61,0.4)' }} }},
      y2: {{ position:'right', ticks:{{ display:false }}, grid:{{ drawOnChartArea:false }} }},
    }},
  }}
}});

const revCtx = document.getElementById('revenueChart').getContext('2d');
new Chart(revCtx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(q_labels)},
    datasets: [{{
      label: 'Revenue ($B)',
      data: {json.dumps(q_values)},
      backgroundColor: ['rgba(237,28,36,0.7)','rgba(237,28,36,0.55)','rgba(237,28,36,0.4)','rgba(237,28,36,0.25)'],
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ backgroundColor: '#1c2128' }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', font:{{size:10}} }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#8b949e', font:{{size:10}}, callback: v => '$'+v+'B' }}, grid: {{ color: 'rgba(48,54,61,0.4)' }} }},
    }},
  }}
}});

const peerPeCtx = document.getElementById('peerPeChart').getContext('2d');
const peColors = {json.dumps(peer_tickers)}.map(t => t === 'AMD' ? 'rgba(237,28,36,0.8)' : 'rgba(88,166,255,0.6)');
new Chart(peerPeCtx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(peer_tickers)},
    datasets: [{{
      label: 'Forward P/E',
      data: {json.dumps(peer_pe_fwd)},
      backgroundColor: peColors,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ backgroundColor: '#1c2128' }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => v+'x' }}, grid: {{ color: 'rgba(48,54,61,0.4)' }} }},
    }},
  }}
}});

const peerRetCtx = document.getElementById('peerRetChart').getContext('2d');
const retColors = {json.dumps(peer_ret1y)}.map(v => v >= 0 ? 'rgba(63,185,80,0.7)' : 'rgba(248,81,73,0.7)');
new Chart(peerRetCtx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(peer_tickers)},
    datasets: [{{
      label: '1Y Return %',
      data: {json.dumps(peer_ret1y)},
      backgroundColor: retColors,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ backgroundColor: '#1c2128' }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => v+'%' }}, grid: {{ color: 'rgba(48,54,61,0.4)' }} }},
    }},
  }}
}});
</script>
</body>
</html>"""

    filepath.write_text(html, encoding="utf-8")
    return str(filepath)


# ── terminal output ────────────────────────────────────────────────────────────

def print_summary(technicals: dict, fundamentals: dict, peers: list[dict], report: str):
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich import box

    console = Console()

    console.rule("[bold red]AMD — Deep Stock Analysis[/bold red]", style="red")
    console.print()

    # price panel
    curr = technicals.get("current_price", 0)
    hi52 = technicals.get("hi_52w", 0)
    lo52 = technicals.get("lo_52w", 0)
    ret1y = technicals.get("return_1y", 0)

    console.print(Panel(
        f"[bold white]${curr:.2f}[/bold white]   "
        f"[dim]52W:[/dim] [green]${lo52:.2f}[/green] — [red]${hi52:.2f}[/red]   "
        f"[dim]1Y Return:[/dim] [{'green' if ret1y >= 0 else 'red'}]{ret1y:+.1f}%[/{'green' if ret1y >= 0 else 'red'}]",
        title="[bold red]Advanced Micro Devices (NASDAQ: AMD)[/bold red]",
        border_style="red",
    ))

    # technicals table
    t1 = Table(title="Technical Indicators", box=box.SIMPLE_HEAVY, show_header=True)
    t1.add_column("Indicator", style="dim", width=18)
    t1.add_column("Value", width=14)
    t1.add_column("Signal", width=20)

    def _add(label, val, signal, sig_color="white"):
        t1.add_row(label, str(val), f"[{sig_color}]{signal}[/{sig_color}]")

    _add("RSI (14)", f"{technicals.get('rsi', 0):.1f}", technicals.get("rsi_signal", ""), "yellow" if technicals.get("rsi_signal") == "Neutral" else "green")
    _add("MACD", f"{technicals.get('macd', 0):.2f}", technicals.get("macd_signal", ""), "green" if technicals.get("macd_signal") == "Bullish" else "red")
    sma50 = technicals.get("sma50") or 0
    sma200 = technicals.get("sma200") or 0
    _add("vs SMA 50", f"${sma50:.2f}", f"{'ABOVE ▲' if curr > sma50 else 'BELOW ▼'}", "green" if curr > sma50 else "red")
    _add("vs SMA 200", f"${sma200:.2f}", f"{'ABOVE ▲' if curr > sma200 else 'BELOW ▼'}", "green" if curr > sma200 else "red")
    _add("MA Cross", "", technicals.get("cross_signal", ""), "green" if "Golden" in (technicals.get("cross_signal") or "") else "red")
    _add("Vol 30d Ann.", f"{technicals.get('volatility_30d_ann', 0):.1f}%", "Moderate" if (technicals.get('volatility_30d_ann') or 0) < 50 else "High", "yellow")
    _add("BB %B", f"{technicals.get('bb_pct_b', 0):.2f}", "Upper zone" if (technicals.get("bb_pct_b") or 0) > 0.8 else ("Lower zone" if (technicals.get("bb_pct_b") or 0) < 0.2 else "Mid range"), "white")
    _add("ATR", f"${technicals.get('atr', 0):.2f}", "Daily range estimate", "dim")

    # fundamentals table
    t2 = Table(title="Fundamentals", box=box.SIMPLE_HEAVY, show_header=True)
    t2.add_column("Metric", style="dim", width=18)
    t2.add_column("AMD", width=12)
    t2.add_column("Note", width=22)

    t2.add_row("P/E TTM", f"{fundamentals.get('pe_ttm') or 0:.1f}x", "")
    t2.add_row("P/E Fwd", f"{fundamentals.get('pe_fwd') or 0:.1f}x", "")
    t2.add_row("EV/EBITDA", f"{fundamentals.get('ev_ebitda') or 0:.1f}x", "")
    t2.add_row("P/S TTM", f"{fundamentals.get('ps_ttm') or 0:.1f}x", "")
    t2.add_row("Gross Margin", f"{(fundamentals.get('gross_margin') or 0)*100:.1f}%", "")
    t2.add_row("Op Margin", f"{(fundamentals.get('operating_margin') or 0)*100:.1f}%", "")
    t2.add_row("Rev Growth", f"{(fundamentals.get('revenue_growth_yoy') or 0)*100:.1f}%", "YoY")
    t2.add_row("FCF", _billions(fundamentals.get("free_cash_flow")), "")

    console.print(Columns([t1, t2]))

    # peer table
    pt = Table(title="Peer Comparison", box=box.SIMPLE_HEAVY)
    pt.add_column("Ticker", width=8)
    pt.add_column("Mkt Cap", width=10)
    pt.add_column("Fwd P/E", width=10)
    pt.add_column("EV/EBITDA", width=12)
    pt.add_column("Gross Mgn", width=12)
    pt.add_column("Rev Grwth", width=12)
    pt.add_column("1Y Ret", width=10)

    for p in peers:
        if "error" in p:
            continue
        color = "red" if p["ticker"] == "AMD" else "white"
        gm_str = f"{p.get('gross_margin', 0)*100:.1f}%" if p.get("gross_margin") else "N/A"
        rg_str = f"{p.get('revenue_growth', 0)*100:.1f}%" if p.get("revenue_growth") else "N/A"
        r1y = p.get("return_1y")
        r1y_str = f"{r1y:+.1f}%" if r1y is not None else "N/A"
        r1y_color = "green" if (r1y or 0) >= 0 else "red"
        pt.add_row(
            f"[{color}]{p['ticker']}[/{color}]",
            _billions(p.get("market_cap")),
            f"{p.get('pe_fwd') or 0:.1f}x" if p.get("pe_fwd") else "N/A",
            f"{p.get('ev_ebitda') or 0:.1f}x" if p.get("ev_ebitda") else "N/A",
            gm_str,
            rg_str,
            f"[{r1y_color}]{r1y_str}[/{r1y_color}]",
        )

    console.print(pt)
    console.print()

    # AI report
    if report and not report.startswith("ANTHROPIC"):
        console.rule("[bold blue]AI Research Report[/bold blue]", style="blue")
        console.print()
        # simple strip of markdown for terminal
        import re
        clean = re.sub(r'#+ ', '', report)
        clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
        clean = re.sub(r'\*(.+?)\*', r'\1', clean)
        console.print(clean)
    else:
        console.print(f"[yellow]{report}[/yellow]")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AMD Deep Stock Analysis")
    parser.add_argument("--output-dir", default=".", help="Output directory for HTML report")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report generation")
    args = parser.parse_args()

    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()

    PEERS = ["AMD", "NVDA", "INTC", "QCOM"]

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:

        task = progress.add_task("Fetching AMD live market data...", total=None)
        data = fetch_stock_data("AMD")
        progress.remove_task(task)
        console.print("[green]✓[/green] AMD data fetched")

        task = progress.add_task("Computing technical indicators...", total=None)
        technicals = compute_technicals(data["hist_1y"])
        progress.remove_task(task)
        console.print("[green]✓[/green] Technicals computed")

        task = progress.add_task("Extracting fundamental metrics...", total=None)
        fundamentals = extract_fundamentals(data)
        progress.remove_task(task)
        console.print("[green]✓[/green] Fundamentals extracted")

        task = progress.add_task(f"Fetching peer data ({', '.join(PEERS)})...", total=None)
        peers = fetch_peer_comparison(PEERS)
        progress.remove_task(task)
        console.print(f"[green]✓[/green] Peer data for {len(peers)} companies")

        task = progress.add_task("Generating AI research report via Claude...", total=None)
        report = generate_amd_report(fundamentals, technicals, peers, data["info"])
        progress.remove_task(task)
        console.print("[green]✓[/green] AI report generated")

    print_summary(technicals, fundamentals, peers, report)

    if not args.no_html:
        path = generate_html(report, technicals, fundamentals, peers, data["hist_1y"], args.output_dir)
        console.print(f"\n[green]✓[/green] HTML report saved: [bold]{path}[/bold]")
        # open in browser
        import subprocess
        try:
            subprocess.run(["open", path], check=True)
            console.print("[green]✓[/green] Opened in browser")
        except Exception:
            pass


if __name__ == "__main__":
    main()
