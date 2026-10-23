"""Fetch live market data via yfinance with caching and retry logic."""
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf

CACHE_DIR = Path(".cache")
CACHE_TTL_SECONDS = 3600  # 1 hour

# GBX tickers on T212 need the LSE suffix for yfinance
# Map known problematic ISINs to correct yfinance tickers
ISIN_TICKER_OVERRIDES: dict[str, str] = {
    "IE00BFMXXD54": "VUAG.L",     # Vanguard S&P 500 Acc
    "IE00BFMXYX26": "VJPB.L",
    "IE00BK5BQT80": "VWRP.L",
    "IE00BD8KRH84": "GSPX.L",
    "IE00B4ND3602": "SGLN.L",
    "IE00B43VDT70": "SLVP.L",
    "IE00BK5BR733": "VFEG.L",
    "IE00B3Y8X563": "IGUS.L",
    "IE00B53HP851": "CUKX.L",
    "IE00B14X4M10": "INAA.L",
    "IE00B5BMR087": "CSP1.L",
    "IE000MLMNYS0": "EWSP.L",
    "IE00B14X4N27": "IEUX.L",
    "IE00BVDPJP67": "EUXS.L",
    "IE00BHZPJ890": "EEDG.L",
    "IE00B52MJY50": "CPJ1.L",
    "IE00BHZPJ346": "EEJG.L",
    "IE00B1FZS350": "IWDP.L",
    "IE00BGPP6580": "IGTM.L",
    "IE00BDFGJ734": "EMHG.L",
    "IE000YZVB4R7": "EHYG.L",
    "IE00BJP26F04": "UESD.L",
    "IE00BFNM3P36": "SAEM.L",
    "XS2399364152": "QQQ5.L",
}

# US tickers as-is
US_TICKERS = {"PLTR", "NVDA", "GOOGL", "AAPL", "JPM", "ORCL", "CDTX"}

GBX_LSE_ISINS = set(ISIN_TICKER_OVERRIDES.keys())


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{h}.json"


def _cache_read(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if time.time() - data.get("_ts", 0) < CACHE_TTL_SECONDS:
            return data
    except Exception:
        pass
    return None


def _cache_write(key: str, data: dict) -> None:
    data["_ts"] = time.time()
    _cache_path(key).write_text(json.dumps(data, default=str))


def _resolve_ticker(isin: str, ticker: str) -> str:
    """Map ISIN/ticker to a yfinance-compatible symbol."""
    if isin in ISIN_TICKER_OVERRIDES:
        return ISIN_TICKER_OVERRIDES[isin]
    # UK ETFs generally need .L suffix; US stocks are fine as-is
    if ticker in US_TICKERS:
        return ticker
    # Heuristic: if ticker has no dot and ISIN starts with IE/GB, add .L
    if "." not in ticker and (isin.startswith("IE") or isin.startswith("GB")):
        return f"{ticker}.L"
    return ticker


def fetch_quote(isin: str, ticker: str) -> dict:
    """
    Return a dict with: price, currency, price_gbp, 52w_high, 52w_low,
    beta, sector, industry, country, market_cap, dividend_rate, dividend_yield,
    yf_ticker (the symbol used)
    """
    yf_ticker = _resolve_ticker(isin, ticker)
    cache_key = f"quote:{yf_ticker}"
    cached = _cache_read(cache_key)
    if cached:
        return cached

    result = _fetch_with_retry(yf_ticker)
    _cache_write(cache_key, result)
    return result


def _fetch_with_retry(symbol: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            t = yf.Ticker(symbol)
            info = t.fast_info
            full_info = {}
            try:
                full_info = t.info or {}
            except Exception:
                pass

            price = getattr(info, "last_price", None) or full_info.get("currentPrice") or full_info.get("regularMarketPrice")
            currency = getattr(info, "currency", None) or full_info.get("currency", "GBP")
            high_52 = getattr(info, "year_high", None) or full_info.get("fiftyTwoWeekHigh")
            low_52 = getattr(info, "year_low", None) or full_info.get("fiftyTwoWeekLow")

            # GBX → GBP conversion (LSE prices in pence)
            price_gbp = price
            if currency == "GBp" or currency == "GBX":
                price_gbp = price / 100.0 if price else None
                if high_52:
                    high_52 = high_52 / 100.0
                if low_52:
                    low_52 = low_52 / 100.0
                currency = "GBP"
            elif currency == "USD" and price:
                fx = _get_gbpusd_rate()
                price_gbp = price / fx if fx else price

            return {
                "price": price,
                "currency": currency,
                "price_gbp": price_gbp,
                "52w_high": high_52,
                "52w_low": low_52,
                "beta": full_info.get("beta"),
                "sector": full_info.get("sector"),
                "industry": full_info.get("industry"),
                "country": full_info.get("country"),
                "market_cap": full_info.get("marketCap"),
                "dividend_rate": full_info.get("dividendRate"),
                "dividend_yield": full_info.get("dividendYield"),
                "yf_ticker": symbol,
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"price": None, "currency": "GBP", "price_gbp": None,
                        "52w_high": None, "52w_low": None, "beta": None,
                        "sector": None, "industry": None, "country": None,
                        "market_cap": None, "dividend_rate": None,
                        "dividend_yield": None, "yf_ticker": symbol,
                        "error": str(e)}
    return {}


def _get_gbpusd_rate() -> float:
    """Return live GBP/USD exchange rate."""
    cached = _cache_read("fx:gbpusd")
    if cached:
        return cached.get("rate", 1.27)
    try:
        t = yf.Ticker("GBPUSD=X")
        rate = t.fast_info.last_price or 1.27
        _cache_write("fx:gbpusd", {"rate": rate})
        return rate
    except Exception:
        return 1.27


def fetch_historical_prices(symbol: str, period: str = "1y") -> Optional[object]:
    """Fetch OHLCV history for risk calculations."""
    cache_key = f"hist:{symbol}:{period}"
    # Don't cache DataFrames in JSON; just fetch fresh (they're cheap)
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period)
        return hist if not hist.empty else None
    except Exception:
        return None


def fetch_benchmark_history(symbol: str = "SPY", period: str = "1y"):
    """Fetch benchmark historical prices."""
    return fetch_historical_prices(symbol, period)
