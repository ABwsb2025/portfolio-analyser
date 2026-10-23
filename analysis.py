"""Core analysis modules: positions, risk, sectors, dividends, performance."""
import math
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd

from models import Position, PortfolioSummary
from market_data import fetch_quote, fetch_historical_prices, fetch_benchmark_history, _get_gbpusd_rate, ISIN_TICKER_OVERRIDES

# Risk-free rate (annualised)
DEFAULT_RISK_FREE_RATE = 0.05

# Sector overrides for ETFs and funds that yfinance won't categorise
ETF_SECTOR_MAP = {
    "VUAG": "ETF - US Equities",
    "VWRP": "ETF - Global Equities",
    "VJPB": "ETF - Japan Equities",
    "SGLN": "ETF - Gold",
    "SLVP": "ETF - Silver",
    "GSPX": "ETF - US Equities",
    "VFEG": "ETF - Emerging Markets",
    "IGUS": "ETF - US Equities (Hedged)",
    "CSP1": "ETF - US Equities",
    "EWSP": "ETF - US Equities",
    "INAA": "ETF - North America",
    "IEUX": "ETF - Europe ex-UK",
    "EUXS": "ETF - Europe ex-UK (Hedged)",
    "EEDG": "ETF - US ESG",
    "CPJ1": "ETF - Pacific ex-Japan",
    "EEJG": "ETF - Japan ESG",
    "IWDP": "ETF - Property",
    "IGTM": "ETF - US Treasuries",
    "EMHG": "ETF - EM Bonds",
    "EHYG": "ETF - HY Bonds",
    "UESD": "ETF - Short Duration Bonds",
    "SAEM": "ETF - Emerging Markets",
    "QQQ5": "ETF - Leveraged Tech",
    "CUKX": "ETF - UK Equities",
}

GEO_MAP = {
    "VUAG": "United States", "GSPX": "United States", "IGUS": "United States",
    "CSP1": "United States", "EWSP": "United States", "INAA": "North America",
    "EEDG": "United States", "QQQ5": "United States",
    "VWRP": "Global", "SAEM": "Emerging Markets", "VFEG": "Emerging Markets",
    "EMHG": "Emerging Markets", "CPJ1": "Asia Pacific",
    "VJPB": "Japan", "EEJG": "Japan",
    "IEUX": "Europe ex-UK", "EUXS": "Europe ex-UK",
    "SGLN": "Global (Gold)", "SLVP": "Global (Silver)",
    "CUKX": "United Kingdom", "IWDP": "Global",
    "IGTM": "United States", "EHYG": "Europe",
    "UESD": "United Kingdom",
    # Individual stocks
    "PLTR": "United States", "NVDA": "United States", "GOOGL": "United States",
    "AAPL": "United States", "JPM": "United States", "ORCL": "United States",
    "CDTX": "United States",
}


def build_positions(raw_positions: dict, fetch_live: bool = True) -> list[Position]:
    """Convert raw position book to Position objects with live market data."""
    positions = []
    for isin, b in raw_positions.items():
        ticker = b["ticker"]
        name = b["name"] or ticker
        shares = b["shares"]
        total_cost = b["total_cost_gbp"]
        avg_cost = b["avg_cost_gbp"]

        pos = Position(
            ticker=ticker,
            isin=isin,
            name=name,
            shares=shares,
            avg_cost_gbp=avg_cost,
            total_cost_gbp=total_cost,
            realised_pnl_gbp=b.get("realised_pnl_gbp", 0.0),
            first_buy_date=b.get("first_buy_date"),
        )

        if b.get("first_buy_date"):
            pos.days_held = (datetime.now() - b["first_buy_date"]).days

        if fetch_live:
            quote = fetch_quote(isin, ticker)
            pos.current_price = quote.get("price")
            pos.current_price_currency = quote.get("currency")
            pos.current_price_gbp = quote.get("price_gbp")
            pos.price_52w_high = quote.get("52w_high")
            pos.price_52w_low = quote.get("52w_low")
            pos.beta = quote.get("beta")
            pos.sector = quote.get("sector") or ETF_SECTOR_MAP.get(ticker)
            pos.industry = quote.get("industry")
            pos.country = quote.get("country") or GEO_MAP.get(ticker)
            pos.market_cap = quote.get("market_cap")
            pos.dividend_rate = quote.get("dividend_rate")
            pos.dividend_yield = quote.get("dividend_yield")

            if pos.current_price_gbp:
                pos.current_value_gbp = shares * pos.current_price_gbp
                pos.unrealised_pnl_gbp = pos.current_value_gbp - total_cost
                if total_cost > 0:
                    pos.unrealised_pnl_pct = (pos.unrealised_pnl_gbp / total_cost) * 100

                # Annualised return
                if pos.days_held and pos.days_held > 0 and total_cost > 0:
                    total_return = pos.current_value_gbp / total_cost
                    years = pos.days_held / 365.25
                    pos.annualised_return_pct = (total_return ** (1 / years) - 1) * 100

        if not pos.sector:
            pos.sector = ETF_SECTOR_MAP.get(ticker, "Unknown")
        if not pos.country:
            pos.country = GEO_MAP.get(ticker, "Unknown")

        positions.append(pos)

    return positions


def compute_portfolio_summary(
    positions: list[Position],
    cash_flows: dict,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    benchmark: str = "SPY",
    fetch_live: bool = True,
) -> PortfolioSummary:
    total_value = sum(p.current_value_gbp or p.total_cost_gbp for p in positions)
    total_invested = sum(p.total_cost_gbp for p in positions)
    total_unrealised = sum(p.unrealised_pnl_gbp or 0.0 for p in positions)
    total_realised = cash_flows["realised_pnl"]
    total_pnl = total_unrealised + total_realised
    total_pnl_pct = (total_pnl / cash_flows["deposits"]) * 100 if cash_flows["deposits"] > 0 else 0.0

    # Weights
    for p in positions:
        if total_value > 0:
            p.weight_pct = ((p.current_value_gbp or p.total_cost_gbp) / total_value) * 100

    # Sector breakdown
    sector_breakdown: dict[str, dict] = {}
    geo_breakdown: dict[str, dict] = {}
    for p in positions:
        val = p.current_value_gbp or p.total_cost_gbp
        s = p.sector or "Unknown"
        g = p.country or "Unknown"
        sector_breakdown.setdefault(s, {"value": 0.0, "pct": 0.0})["value"] += val
        geo_breakdown.setdefault(g, {"value": 0.0, "pct": 0.0})["value"] += val

    if total_value > 0:
        for s in sector_breakdown:
            sector_breakdown[s]["pct"] = sector_breakdown[s]["value"] / total_value * 100
        for g in geo_breakdown:
            geo_breakdown[g]["pct"] = geo_breakdown[g]["value"] / total_value * 100

    # Risk metrics (need price history)
    portfolio_beta = _compute_portfolio_beta(positions, benchmark)
    if fetch_live:
        vol_30d, sharpe, max_dd, var_95, var_99 = _compute_risk_metrics(
            positions, risk_free_rate, benchmark
        )
    else:
        vol_30d = sharpe = max_dd = var_95 = var_99 = None

    # Cash = deposits - withdrawals - amount currently invested
    cash = max(0.0, cash_flows["deposits"] - cash_flows["withdrawals"] - total_invested)

    return PortfolioSummary(
        total_value_gbp=total_value,
        total_invested_gbp=total_invested,
        total_realised_pnl_gbp=total_realised,
        total_unrealised_pnl_gbp=total_unrealised,
        total_pnl_gbp=total_pnl,
        total_pnl_pct=total_pnl_pct,
        total_dividends_gbp=cash_flows["dividends"],
        total_deposits_gbp=cash_flows["deposits"],
        total_withdrawals_gbp=cash_flows["withdrawals"],
        cash_gbp=cash,
        positions=positions,
        portfolio_beta=portfolio_beta,
        sharpe_ratio=sharpe,
        volatility_30d=vol_30d,
        max_drawdown_pct=max_dd,
        var_95=var_95,
        var_99=var_99,
        sector_breakdown=sector_breakdown,
        geo_breakdown=geo_breakdown,
    )


def _compute_portfolio_beta(positions: list[Position], benchmark: str = "SPY") -> Optional[float]:
    """Weighted average beta using position weights."""
    total_val = sum(p.current_value_gbp or p.total_cost_gbp for p in positions)
    if total_val == 0:
        return None
    weighted_beta = 0.0
    for p in positions:
        if p.beta is not None:
            w = (p.current_value_gbp or p.total_cost_gbp) / total_val
            weighted_beta += w * p.beta
        # ETFs without beta — approximate: equity ETFs ~1.0, bond ~0.1, commodity ~0.3
        else:
            w = (p.current_value_gbp or p.total_cost_gbp) / total_val
            sector = p.sector or ""
            if "Bond" in sector or "bond" in sector:
                weighted_beta += w * 0.1
            elif "Gold" in sector or "Silver" in sector:
                weighted_beta += w * 0.2
            else:
                weighted_beta += w * 1.0
    return round(weighted_beta, 3)


def _compute_risk_metrics(
    positions: list[Position],
    risk_free_rate: float,
    benchmark: str,
) -> tuple:
    """Return (vol_30d, sharpe, max_drawdown_pct, var_95, var_99)."""
    from market_data import fetch_historical_prices, ISIN_TICKER_OVERRIDES

    # Build portfolio daily returns from the largest positions
    total_val = sum(p.current_value_gbp or p.total_cost_gbp for p in positions)
    if total_val == 0:
        return None, None, None, None, None

    # Collect weighted daily returns from top holdings
    all_returns: list[pd.Series] = []
    weights: list[float] = []

    # Sort by weight, take top 10 for efficiency
    sorted_pos = sorted(
        positions,
        key=lambda p: p.current_value_gbp or p.total_cost_gbp,
        reverse=True,
    )[:10]

    for p in sorted_pos:
        from market_data import _resolve_ticker as _rt
        yf_sym = _rt(p.isin, p.ticker)
        hist = fetch_historical_prices(yf_sym, "3mo")
        if hist is not None and len(hist) > 5:
            returns = hist["Close"].pct_change(fill_method=None).dropna()
            # GBX → GBP doesn't affect returns
            all_returns.append(returns)
            w = (p.current_value_gbp or p.total_cost_gbp) / total_val
            weights.append(w)

    if not all_returns:
        return None, None, None, None, None

    # Align on common dates
    df = pd.concat(all_returns, axis=1, join="inner")
    df.columns = range(len(df.columns))

    if df.empty or len(df) < 2:
        return None, None, None, None, None

    w_arr = np.array(weights[: len(df.columns)])
    w_arr = w_arr / w_arr.sum()

    port_returns = df.values @ w_arr
    port_returns = port_returns[~np.isnan(port_returns)]

    if len(port_returns) < 2:
        return None, None, None, None, None

    # 30-day volatility (annualised)
    tail = port_returns[-30:] if len(port_returns) >= 30 else port_returns
    std_30 = np.std(tail, ddof=1) if len(tail) > 1 else None
    vol_30d = float(std_30 * math.sqrt(252) * 100) if std_30 is not None and not math.isnan(std_30) else None

    # Sharpe ratio
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = port_returns - daily_rf
    std_ex = np.std(excess, ddof=1)
    sharpe = float(np.mean(excess) / std_ex * math.sqrt(252)) if std_ex > 0 and not math.isnan(std_ex) else None

    # Max drawdown
    cum = np.cumprod(1 + port_returns)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    max_dd = float(drawdown.min() * 100) if len(drawdown) > 0 else None

    # VaR
    var_95 = float(np.percentile(port_returns, 5) * total_val)
    var_99 = float(np.percentile(port_returns, 1) * total_val)

    return (
        round(vol_30d, 2) if vol_30d is not None else None,
        round(sharpe, 3) if sharpe is not None else None,
        round(max_dd, 2) if max_dd is not None else None,
        round(var_95, 2),
        round(var_99, 2),
    )


def get_concentration_warnings(summary: PortfolioSummary) -> list[str]:
    warnings = []
    for p in summary.positions:
        if p.weight_pct and p.weight_pct > 20:
            warnings.append(
                f"HIGH CONCENTRATION: {p.ticker} is {p.weight_pct:.1f}% of portfolio (>20% threshold)"
            )
    for sector, data in summary.sector_breakdown.items():
        if data["pct"] > 40:
            warnings.append(
                f"SECTOR CONCENTRATION: {sector} is {data['pct']:.1f}% of portfolio (>40% threshold)"
            )
    for geo, data in summary.geo_breakdown.items():
        if data["pct"] > 60:
            warnings.append(
                f"GEO CONCENTRATION: {geo} is {data['pct']:.1f}% of portfolio (>60% threshold)"
            )
    return warnings


def performance_attribution(positions: list[Position]) -> dict:
    ranked = sorted(
        [p for p in positions if p.unrealised_pnl_pct is not None],
        key=lambda p: p.unrealised_pnl_pct or 0,
    )
    losers = ranked[:3]
    winners = ranked[-3:][::-1]
    winners_count = sum(1 for p in positions if (p.unrealised_pnl_gbp or 0) > 0)
    losers_count = sum(1 for p in positions if (p.unrealised_pnl_gbp or 0) < 0)
    return {
        "top_winners": winners,
        "top_losers": losers,
        "winners_count": winners_count,
        "losers_count": losers_count,
    }


def dividend_analysis(positions: list[Position]) -> dict:
    """Project annual dividend income."""
    payers = [p for p in positions if p.dividend_rate and p.dividend_rate > 0]
    gbpusd = _get_gbpusd_rate() if payers else 1.0
    annual_income_gbp = 0.0
    div_positions = []
    for p in positions:
        if p.dividend_rate and p.dividend_rate > 0:
            # dividend_rate is in native currency per share
            rate_gbp = p.dividend_rate / gbpusd if (p.current_price_currency == "USD") else p.dividend_rate
            annual = rate_gbp * p.shares
            annual_income_gbp += annual
            yield_on_cost = (rate_gbp / p.avg_cost_gbp * 100) if p.avg_cost_gbp else 0
            div_positions.append({
                "ticker": p.ticker,
                "name": p.name,
                "annual_income_gbp": round(annual, 2),
                "yield_on_cost_pct": round(yield_on_cost, 2),
                "current_yield_pct": round((p.dividend_yield or 0) * 100, 2),
            })
    return {
        "annual_income_gbp": round(annual_income_gbp, 2),
        "positions": sorted(div_positions, key=lambda x: x["annual_income_gbp"], reverse=True),
    }
