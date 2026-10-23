"""Parse Trading 212 transaction history CSV exports."""
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from models import Transaction

# Trading 212 CSV column names (exact)
_COL_ACTION = "Action"
_COL_TIME = "Time"
_COL_ISIN = "ISIN"
_COL_TICKER = "Ticker"
_COL_NAME = "Name"
_COL_NOTES = "Notes"
_COL_ID = "ID"
_COL_SHARES = "No. of shares"
_COL_PRICE = "Price / share"
_COL_PRICE_CCY = "Currency (Price / share)"
_COL_FX = "Exchange rate"
_COL_RESULT = "Result"
_COL_RESULT_CCY = "Currency (Result)"
_COL_TOTAL = "Total"
_COL_TOTAL_CCY = "Currency (Total)"
_COL_WHT = "Withholding tax"
_COL_WHT_CCY = "Currency (Withholding tax)"
_COL_STAMP = "Stamp duty reserve tax"
_COL_STAMP_CCY = "Currency (Stamp duty reserve tax)"
_COL_FXFEE = "Currency conversion fee"
_COL_FXFEE_CCY = "Currency (Currency conversion fee)"


def _float_or_none(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        s = str(val).strip()
        return float(s) if s else None
    except (ValueError, TypeError):
        return None


def _parse_time(val) -> datetime:
    s = str(val).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}")


def load_transactions(filepath: str) -> list[Transaction]:
    """Load and parse a Trading 212 transaction history CSV."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, dtype=str, keep_default_na=False)
    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    transactions = []
    for _, row in df.iterrows():
        def g(col, default=None):
            v = row.get(col, default)
            if isinstance(v, str):
                v = v.strip()
            return v if v != "" else default

        action = g(_COL_ACTION, "")
        if not action:
            continue

        try:
            time = _parse_time(g(_COL_TIME, "1970-01-01 00:00:00"))
        except ValueError:
            continue

        # Exchange rate — T212 stores "1.00000000" for GBP assets
        fx = _float_or_none(g(_COL_FX))
        price_ccy = g(_COL_PRICE_CCY)

        # Price is in native currency; convert to GBP
        raw_price = _float_or_none(g(_COL_PRICE))
        # GBX (pence) → GBP
        if price_ccy == "GBX" and raw_price is not None:
            price_gbp_equiv = raw_price / 100.0
        else:
            price_gbp_equiv = raw_price

        # Total is already in GBP for buys/sells
        total_raw = _float_or_none(g(_COL_TOTAL))

        t = Transaction(
            action=action,
            time=time,
            isin=g(_COL_ISIN),
            ticker=g(_COL_TICKER),
            name=g(_COL_NAME),
            notes=g(_COL_NOTES),
            transaction_id=g(_COL_ID),
            shares=_float_or_none(g(_COL_SHARES)),
            price_per_share=price_gbp_equiv,
            price_currency=price_ccy,
            exchange_rate=fx,
            result=_float_or_none(g(_COL_RESULT)),
            total=total_raw,
            total_currency=g(_COL_TOTAL_CCY, "GBP"),
            withholding_tax=_float_or_none(g(_COL_WHT)),
            withholding_tax_currency=g(_COL_WHT_CCY),
            stamp_duty=_float_or_none(g(_COL_STAMP)),
            stamp_duty_currency=g(_COL_STAMP_CCY),
            fx_fee=_float_or_none(g(_COL_FXFEE)),
            fx_fee_currency=g(_COL_FXFEE_CCY),
        )
        transactions.append(t)

    transactions.sort(key=lambda x: x.time)
    return transactions


def compute_positions(transactions: list[Transaction]) -> dict:
    """
    Build current open positions from transaction history.

    Returns a dict keyed by ISIN → dict with:
        ticker, name, isin, shares, total_cost_gbp, avg_cost_gbp,
        first_buy_date, realised_pnl_gbp
    """
    # Running book: isin → {shares, total_cost_gbp, first_buy_date, realised_pnl}
    book: dict[str, dict] = {}

    for t in transactions:
        if not t.isin or not t.ticker:
            continue

        action_lower = t.action.lower()

        if "buy" in action_lower:
            cost = t.total or 0.0  # GBP spent (positive in CSV)
            shares = t.shares or 0.0
            if t.isin not in book:
                book[t.isin] = {
                    "ticker": t.ticker,
                    "name": t.name,
                    "isin": t.isin,
                    "shares": 0.0,
                    "total_cost_gbp": 0.0,
                    "realised_pnl_gbp": 0.0,
                    "first_buy_date": t.time,
                }
            b = book[t.isin]
            b["shares"] += shares
            b["total_cost_gbp"] += cost
            if t.time < b["first_buy_date"]:
                b["first_buy_date"] = t.time

        elif "sell" in action_lower:
            shares_sold = t.shares or 0.0
            result = t.result or 0.0  # P&L on this trade
            if t.isin in book:
                b = book[t.isin]
                # Reduce cost basis proportionally
                if b["shares"] > 0:
                    cost_per_share = b["total_cost_gbp"] / b["shares"]
                    b["total_cost_gbp"] -= cost_per_share * shares_sold
                b["shares"] -= shares_sold
                b["realised_pnl_gbp"] += result
                # Clean up fully-closed positions
                if b["shares"] <= 1e-9:
                    b["shares"] = 0.0
                    b["total_cost_gbp"] = 0.0

    # Filter out empty positions and compute avg cost
    open_positions = {}
    for isin, b in book.items():
        if b["shares"] > 1e-9:
            b["avg_cost_gbp"] = b["total_cost_gbp"] / b["shares"] if b["shares"] > 0 else 0.0
            open_positions[isin] = b

    return open_positions


def compute_cash_flows(transactions: list[Transaction]) -> dict:
    """Summarise deposits, withdrawals, dividends."""
    deposits = 0.0
    withdrawals = 0.0
    dividends = 0.0
    realised_pnl = 0.0

    for t in transactions:
        action_lower = t.action.lower()
        if "deposit" in action_lower:
            deposits += t.total or 0.0
        elif "withdrawal" in action_lower:
            withdrawals += abs(t.total or 0.0)
        elif "dividend" in action_lower:
            dividends += t.total or 0.0
        elif "sell" in action_lower:
            realised_pnl += t.result or 0.0

    return {
        "deposits": deposits,
        "withdrawals": withdrawals,
        "dividends": dividends,
        "realised_pnl": realised_pnl,
    }
