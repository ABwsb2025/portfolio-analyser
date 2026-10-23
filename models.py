"""Internal data models for the Trading 212 portfolio analyser."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Transaction:
    action: str          # Market buy, Market sell, Dividend, Deposit, Withdrawal, etc.
    time: datetime
    isin: Optional[str]
    ticker: Optional[str]
    name: Optional[str]
    notes: Optional[str]
    transaction_id: Optional[str]
    shares: Optional[float]
    price_per_share: Optional[float]
    price_currency: Optional[str]
    exchange_rate: Optional[float]   # to GBP
    result: Optional[float]          # GBP P&L on closed trade
    total: Optional[float]           # GBP cash flow (negative = spent, positive = received)
    total_currency: str = "GBP"
    withholding_tax: Optional[float] = None
    withholding_tax_currency: Optional[str] = None
    stamp_duty: Optional[float] = None
    stamp_duty_currency: Optional[str] = None
    fx_fee: Optional[float] = None
    fx_fee_currency: Optional[str] = None


@dataclass
class Position:
    ticker: str
    isin: str
    name: str
    shares: float
    avg_cost_gbp: float          # average cost per share in GBP
    total_cost_gbp: float        # total amount invested in GBP
    current_price: Optional[float] = None
    current_price_currency: Optional[str] = None
    current_price_gbp: Optional[float] = None
    current_value_gbp: Optional[float] = None
    unrealised_pnl_gbp: Optional[float] = None
    unrealised_pnl_pct: Optional[float] = None
    # Market data
    price_52w_high: Optional[float] = None
    price_52w_low: Optional[float] = None
    beta: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    dividend_rate: Optional[float] = None   # annual dividend per share (USD)
    dividend_yield: Optional[float] = None
    market_cap: Optional[float] = None
    # Computed
    weight_pct: Optional[float] = None
    first_buy_date: Optional[datetime] = None
    days_held: Optional[int] = None
    annualised_return_pct: Optional[float] = None
    realised_pnl_gbp: float = 0.0


@dataclass
class PortfolioSummary:
    total_value_gbp: float
    total_invested_gbp: float
    total_realised_pnl_gbp: float
    total_unrealised_pnl_gbp: float
    total_pnl_gbp: float
    total_pnl_pct: float
    total_dividends_gbp: float
    total_deposits_gbp: float
    total_withdrawals_gbp: float
    cash_gbp: float
    positions: list = field(default_factory=list)
    # Risk
    portfolio_beta: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    volatility_30d: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    # Sector breakdown: {sector: {"value": float, "pct": float}}
    sector_breakdown: dict = field(default_factory=dict)
    geo_breakdown: dict = field(default_factory=dict)
