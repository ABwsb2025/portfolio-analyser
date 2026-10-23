#!/usr/bin/env python3
"""
Trading 212 Portfolio Analyser — Main CLI entry point.

Usage:
  python main.py --transactions transactions.csv
  python main.py --transactions transactions.csv --report
  python main.py --transactions transactions.csv --insights
  python main.py --transactions transactions.csv --no-live
  python main.py --transactions transactions.csv --report --insights --benchmark SPY
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def parse_args():
    p = argparse.ArgumentParser(
        description="Trading 212 Portfolio Analyser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--transactions", "-t",
        required=True,
        help="Path to Trading 212 transaction history CSV",
    )
    p.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate HTML report",
    )
    p.add_argument(
        "--insights", "-i",
        action="store_true",
        help="Generate AI insights (requires ANTHROPIC_API_KEY)",
    )
    p.add_argument(
        "--benchmark",
        default="SPY",
        help="Benchmark ticker for comparison (default: SPY)",
    )
    p.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.05,
        help="Annualised risk-free rate for Sharpe (default: 0.05)",
    )
    p.add_argument(
        "--no-live",
        action="store_true",
        help="Skip live market data fetching (faster, uses cost basis only)",
    )
    p.add_argument(
        "--output-dir",
        default=".",
        help="Directory for HTML report output (default: current dir)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Also export full analysis as JSON",
    )
    return p.parse_args()


def main():
    args = parse_args()

    fetch_live = not args.no_live

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:

        # Step 1: Load transactions
        task = progress.add_task("Loading transaction history...", total=None)
        from data_loader import load_transactions, compute_positions, compute_cash_flows
        try:
            transactions = load_transactions(args.transactions)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        console.print(f"[green]✓[/green] Loaded [bold]{len(transactions)}[/bold] transactions")
        progress.remove_task(task)

        # Step 2: Compute positions
        task = progress.add_task("Computing positions from transaction history...", total=None)
        raw_positions = compute_positions(transactions)
        cash_flows = compute_cash_flows(transactions)
        console.print(f"[green]✓[/green] Found [bold]{len(raw_positions)}[/bold] open positions")
        progress.remove_task(task)

        # Step 3: Fetch live market data
        if fetch_live:
            task = progress.add_task(
                f"Fetching live quotes for {len(raw_positions)} positions...", total=None
            )
        from analysis import build_positions, compute_portfolio_summary, get_concentration_warnings, dividend_analysis
        positions = build_positions(raw_positions, fetch_live=fetch_live)
        if fetch_live:
            progress.remove_task(task)
            console.print(f"[green]✓[/green] Live market data fetched and cached")

        # Step 4: Build summary
        task = progress.add_task("Running analysis modules...", total=None)
        summary = compute_portfolio_summary(
            positions,
            cash_flows,
            risk_free_rate=args.risk_free_rate,
            benchmark=args.benchmark,
            fetch_live=fetch_live,
        )
        warnings = get_concentration_warnings(summary)
        div_data = dividend_analysis(positions)
        progress.remove_task(task)
        console.print(f"[green]✓[/green] Analysis complete")

        # Step 5: AI Insights (optional)
        insights = ""
        if args.insights:
            task = progress.add_task("Generating AI insights via Claude...", total=None)
            from ai_insights import generate_portfolio_insights
            insights = generate_portfolio_insights(summary, warnings)
            progress.remove_task(task)
            console.print(f"[green]✓[/green] AI insights generated")

    # --- Render terminal dashboard ---
    from terminal_ui import render_dashboard
    render_dashboard(summary, warnings, insights)

    # --- Dividend summary ---
    if div_data["annual_income_gbp"] > 0:
        console.print()
        console.rule("[bold yellow]Dividend Analysis[/bold yellow]", style="yellow")
        console.print(f"  Projected annual dividend income: [bold yellow]£{div_data['annual_income_gbp']:,.2f}[/bold yellow]")
        for d in div_data["positions"][:5]:
            console.print(
                f"  • {d['ticker']}: £{d['annual_income_gbp']:.2f}/yr  "
                f"| Yield on cost: {d['yield_on_cost_pct']:.2f}%  "
                f"| Current yield: {d['current_yield_pct']:.2f}%"
            )

    # --- HTML report (optional) ---
    if args.report:
        console.print()
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p:
            t = p.add_task("Generating HTML report...", total=None)
            from html_report import generate_html_report
            report_path = generate_html_report(summary, warnings, insights, transactions, args.output_dir)
        console.print(f"[green]✓[/green] HTML report saved: [bold]{report_path}[/bold]")

    # --- JSON export (optional) ---
    if args.json:
        json_path = Path(args.output_dir) / f"portfolio_analysis_{datetime.now().strftime('%Y-%m-%d')}.json"
        export = _build_json_export(summary, warnings, div_data)
        json_path.write_text(json.dumps(export, indent=2, default=str))
        console.print(f"[green]✓[/green] JSON export saved: [bold]{json_path}[/bold]")


def _build_json_export(summary, warnings, div_data) -> dict:
    from dataclasses import asdict
    pos_list = []
    for p in summary.positions:
        pos_list.append({
            "ticker": p.ticker,
            "isin": p.isin,
            "name": p.name,
            "shares": p.shares,
            "avg_cost_gbp": p.avg_cost_gbp,
            "total_cost_gbp": p.total_cost_gbp,
            "current_price_gbp": p.current_price_gbp,
            "current_value_gbp": p.current_value_gbp,
            "unrealised_pnl_gbp": p.unrealised_pnl_gbp,
            "unrealised_pnl_pct": p.unrealised_pnl_pct,
            "weight_pct": p.weight_pct,
            "days_held": p.days_held,
            "annualised_return_pct": p.annualised_return_pct,
            "beta": p.beta,
            "sector": p.sector,
            "country": p.country,
            "realised_pnl_gbp": p.realised_pnl_gbp,
            "first_buy_date": p.first_buy_date.isoformat() if p.first_buy_date else None,
        })

    return {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_value_gbp": summary.total_value_gbp,
            "total_invested_gbp": summary.total_invested_gbp,
            "total_unrealised_pnl_gbp": summary.total_unrealised_pnl_gbp,
            "total_realised_pnl_gbp": summary.total_realised_pnl_gbp,
            "total_pnl_gbp": summary.total_pnl_gbp,
            "total_pnl_pct": summary.total_pnl_pct,
            "total_dividends_gbp": summary.total_dividends_gbp,
            "total_deposits_gbp": summary.total_deposits_gbp,
            "total_withdrawals_gbp": summary.total_withdrawals_gbp,
            "cash_gbp": summary.cash_gbp,
            "portfolio_beta": summary.portfolio_beta,
            "sharpe_ratio": summary.sharpe_ratio,
            "volatility_30d": summary.volatility_30d,
            "max_drawdown_pct": summary.max_drawdown_pct,
            "var_95": summary.var_95,
            "var_99": summary.var_99,
        },
        "positions": pos_list,
        "sector_breakdown": summary.sector_breakdown,
        "geo_breakdown": summary.geo_breakdown,
        "risk_warnings": warnings,
        "dividend_analysis": div_data,
    }


if __name__ == "__main__":
    main()
