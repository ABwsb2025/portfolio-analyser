# Portfolio Analytics & LLM Research Tool

A modular Python CLI that turns a Trading 212 transaction export into a full portfolio analysis: reconstructed positions, risk metrics, concentration checks, and an optional analyst-style report written by Claude. A second script applies the same approach to deep single-stock analysis.

> Educational project. Nothing here is financial advice.

## What it does

**Portfolio analyser (`main.py`)**

The pipeline parses the brokerage CSV with pandas, replays every buy and sell to rebuild open positions with average cost basis and realised P&L, then enriches each holding with live market data from yfinance (price, beta, sector, country, dividends). Quotes are cached to disk for an hour and fetched with exponential-backoff retries, and LSE prices quoted in pence are converted to GBP.

Risk analytics are computed in NumPy from a weighted daily-returns series of the largest holdings: annualised volatility, Sharpe ratio, maximum drawdown, historical 1-day VaR at 95% and 99%, and position-weighted beta. Concentration flags fire when a single position exceeds 20%, a sector exceeds 40%, or a region exceeds 60% of the portfolio.

Output goes to a Rich terminal dashboard, a self-contained HTML report with Chart.js charts, and a JSON export. With `--insights`, a compact JSON summary is sent to the Claude API, which returns a structured report covering strengths, risks and recommendations.

**Single-stock analysis (`amd_analysis.py`)**

Pulls one year of price history, fundamentals, quarterly financials and analyst consensus, then computes technical indicators (RSI, MACD, Bollinger Bands, ATR, SMA 20/50/200 with golden/death cross detection) and a peer comparison table. Claude turns the data into an equity-research style report, rendered as HTML alongside price and peer charts.

## Project structure

```
main.py            CLI entry point and orchestration
data_loader.py     CSV parsing, position reconstruction, cash flows
market_data.py     yfinance wrapper with file cache and retries
analysis.py        Risk metrics, concentration, attribution, dividends
ai_insights.py     Claude API integration
terminal_ui.py     Rich terminal dashboard
html_report.py     Chart.js HTML report
models.py          Dataclasses for transactions, positions, summary
amd_analysis.py    Single-stock technical + fundamental analysis
examples/          Synthetic sample data
```

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/ABwsb2025/portfolio-analyser.git
cd portfolio-analyser
pip install -r requirements.txt

# Offline run on the synthetic sample (no network or API key needed)
python main.py -t examples/sample_transactions.csv --no-live --report

# Full run with live market data
python main.py -t examples/sample_transactions.csv --report --json
```

To use your own data, export your transaction history from Trading 212 (History → Export) and pass the CSV with `-t`.

### AI reports

```bash
cp .env.example .env        # add your key, or just export it
export ANTHROPIC_API_KEY=sk-ant-...
python main.py -t examples/sample_transactions.csv --report --insights
python amd_analysis.py
```

Set `CLAUDE_MODEL` to override the default model.

### Options

| Flag | Purpose |
|---|---|
| `-t, --transactions` | Path to the transaction CSV (required) |
| `-r, --report` | Write the HTML report |
| `-i, --insights` | Generate the Claude report |
| `--json` | Export the full analysis as JSON |
| `--no-live` | Skip market data and use cost basis only |
| `--benchmark` | Benchmark ticker (default SPY) |
| `--risk-free-rate` | Annual rate used in Sharpe (default 0.05) |
| `--output-dir` | Where reports are written |

## Design notes

Risk metrics use the top ten holdings by value, aligned on common trading days, which keeps API calls bounded at the cost of slightly understating diversification in very long portfolios. Where yfinance has no beta for an ETF, the beta calculation falls back to sector-based approximations (equity 1.0, gold 0.2, bonds 0.1). The LLM receives only aggregated metrics and the top 15 positions, which keeps prompts small and avoids sending raw transaction history.

## Privacy

Brokerage exports, generated reports and the cache are gitignored. The sample CSV in `examples/` is entirely synthetic.

## Licence

MIT
