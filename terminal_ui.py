"""Rich terminal dashboard for portfolio analysis."""
from datetime import datetime

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich import print as rprint

from models import PortfolioSummary

console = Console()


def _pnl_colour(val: float) -> str:
    return "green" if val >= 0 else "red"


def _fmt_gbp(val: float, sign: bool = True) -> str:
    prefix = ("+" if val >= 0 else "") if sign else ""
    return f"{prefix}£{val:,.2f}"


def _fmt_pct(val: float, sign: bool = True) -> str:
    prefix = ("+" if val >= 0 else "") if sign else ""
    return f"{prefix}{val:.2f}%"


def render_dashboard(summary: PortfolioSummary, warnings: list[str], insights: str = "") -> None:
    console.print()
    console.print(Rule("[bold cyan]Trading 212 Portfolio Analyser[/bold cyan]", style="cyan"))
    console.print(f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
    console.print()

    _render_overview_cards(summary)
    console.print()
    _render_positions_table(summary)
    console.print()
    _render_risk_metrics(summary)
    console.print()
    _render_sector_geo(summary)
    console.print()
    _render_performance_attribution(summary)

    if warnings:
        console.print()
        _render_warnings(warnings)

    if insights:
        console.print()
        _render_insights(insights)


def _render_overview_cards(summary: PortfolioSummary) -> None:
    pnl_col = _pnl_colour(summary.total_pnl_gbp)
    unreal_col = _pnl_colour(summary.total_unrealised_pnl_gbp)
    real_col = _pnl_colour(summary.total_realised_pnl_gbp)

    cards = [
        Panel(
            f"[bold white]£{summary.total_value_gbp:,.2f}[/bold white]\n[dim]Invested: £{summary.total_invested_gbp:,.2f}[/dim]",
            title="[bold]Portfolio Value[/bold]",
            border_style="blue",
        ),
        Panel(
            f"[bold {pnl_col}]{_fmt_gbp(summary.total_pnl_gbp)}[/bold {pnl_col}]\n[{pnl_col}]{_fmt_pct(summary.total_pnl_pct)}[/{pnl_col}]",
            title="[bold]Total P&L[/bold]",
            border_style=pnl_col,
        ),
        Panel(
            f"[{unreal_col}]{_fmt_gbp(summary.total_unrealised_pnl_gbp)}[/{unreal_col}]",
            title="[bold]Unrealised P&L[/bold]",
            border_style=unreal_col,
        ),
        Panel(
            f"[{real_col}]{_fmt_gbp(summary.total_realised_pnl_gbp)}[/{real_col}]",
            title="[bold]Realised P&L[/bold]",
            border_style=real_col,
        ),
        Panel(
            f"[bold yellow]£{summary.total_dividends_gbp:,.2f}[/bold yellow]",
            title="[bold]Dividends Received[/bold]",
            border_style="yellow",
        ),
        Panel(
            f"[bold]£{summary.total_deposits_gbp:,.2f}[/bold]\n[dim]Withdrawn: £{summary.total_withdrawals_gbp:,.2f}[/dim]",
            title="[bold]Total Deposited[/bold]",
            border_style="magenta",
        ),
    ]
    console.print(Columns(cards, equal=True, expand=True))


def _render_positions_table(summary: PortfolioSummary) -> None:
    console.print(Rule("[bold]Open Positions[/bold]", style="blue"))

    t = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        expand=True,
    )
    t.add_column("Ticker", style="bold white", no_wrap=True)
    t.add_column("Name", max_width=30)
    t.add_column("Shares", justify="right")
    t.add_column("Avg Cost", justify="right")
    t.add_column("Curr Price", justify="right")
    t.add_column("Value (£)", justify="right")
    t.add_column("Weight", justify="right")
    t.add_column("Unreal P&L", justify="right")
    t.add_column("Unreal %", justify="right")
    t.add_column("Days", justify="right")
    t.add_column("Beta", justify="right")
    t.add_column("Sector", max_width=25)

    sorted_pos = sorted(summary.positions, key=lambda p: p.weight_pct or 0, reverse=True)
    for p in sorted_pos:
        pnl_col = _pnl_colour(p.unrealised_pnl_gbp or 0)
        curr_price_str = f"£{p.current_price_gbp:.4f}" if p.current_price_gbp else "N/A"
        avg_cost_str = f"£{p.avg_cost_gbp:.4f}"
        value_str = f"£{p.current_value_gbp:,.2f}" if p.current_value_gbp else f"£{p.total_cost_gbp:,.2f}"
        weight_str = f"{p.weight_pct:.1f}%" if p.weight_pct else "N/A"
        pnl_str = f"[{pnl_col}]{_fmt_gbp(p.unrealised_pnl_gbp or 0)}[/{pnl_col}]"
        pnl_pct_str = f"[{pnl_col}]{_fmt_pct(p.unrealised_pnl_pct or 0)}[/{pnl_col}]"
        beta_str = f"{p.beta:.2f}" if p.beta else "-"
        days_str = str(p.days_held) if p.days_held else "-"
        sector_str = (p.sector or "Unknown")[:25]

        t.add_row(
            p.ticker,
            p.name[:30] if p.name else "",
            f"{p.shares:.4f}",
            avg_cost_str,
            curr_price_str,
            value_str,
            weight_str,
            pnl_str,
            pnl_pct_str,
            days_str,
            beta_str,
            sector_str,
        )

    console.print(t)


def _render_risk_metrics(summary: PortfolioSummary) -> None:
    console.print(Rule("[bold]Risk Metrics[/bold]", style="red"))

    t = Table(box=box.SIMPLE, show_header=False, expand=False)
    t.add_column("Metric", style="bold")
    t.add_column("Value")

    def _add(metric, val, colour="white"):
        t.add_row(metric, f"[{colour}]{val}[/{colour}]" if val != "N/A" else "[dim]N/A[/dim]")

    _add("Portfolio Beta", f"{summary.portfolio_beta:.3f}" if summary.portfolio_beta else "N/A",
         "yellow" if summary.portfolio_beta and summary.portfolio_beta > 1.2 else "green")
    _add("30-Day Volatility (ann.)", f"{summary.volatility_30d:.2f}%" if summary.volatility_30d else "N/A",
         "red" if summary.volatility_30d and summary.volatility_30d > 20 else "green")
    _add("Sharpe Ratio", f"{summary.sharpe_ratio:.3f}" if summary.sharpe_ratio else "N/A",
         "green" if summary.sharpe_ratio and summary.sharpe_ratio > 1 else "yellow")
    _add("Max Drawdown", f"{summary.max_drawdown_pct:.2f}%" if summary.max_drawdown_pct else "N/A",
         "red" if summary.max_drawdown_pct and summary.max_drawdown_pct < -15 else "yellow")
    _add("VaR 95% (1-day)", _fmt_gbp(summary.var_95) if summary.var_95 else "N/A", "yellow")
    _add("VaR 99% (1-day)", _fmt_gbp(summary.var_99) if summary.var_99 else "N/A", "red")

    console.print(t)


def _render_sector_geo(summary: PortfolioSummary) -> None:
    console.print(Rule("[bold]Diversification[/bold]", style="magenta"))

    # Sector table
    st = Table(title="Sector Allocation", box=box.SIMPLE, title_style="bold")
    st.add_column("Sector")
    st.add_column("Value (£)", justify="right")
    st.add_column("%", justify="right")
    st.add_column("Bar", no_wrap=True)

    for sector, data in sorted(summary.sector_breakdown.items(), key=lambda x: -x[1]["pct"]):
        bar_len = int(data["pct"] / 2)
        bar = "█" * bar_len
        colour = "red" if data["pct"] > 40 else ("yellow" if data["pct"] > 25 else "green")
        st.add_row(sector[:40], f"£{data['value']:,.2f}", f"{data['pct']:.1f}%", f"[{colour}]{bar}[/{colour}]")

    # Geo table
    gt = Table(title="Geographic Allocation", box=box.SIMPLE, title_style="bold")
    gt.add_column("Region")
    gt.add_column("Value (£)", justify="right")
    gt.add_column("%", justify="right")
    gt.add_column("Bar", no_wrap=True)

    for geo, data in sorted(summary.geo_breakdown.items(), key=lambda x: -x[1]["pct"]):
        bar_len = int(data["pct"] / 2)
        bar = "█" * bar_len
        colour = "red" if data["pct"] > 60 else ("yellow" if data["pct"] > 40 else "green")
        gt.add_row(geo[:30], f"£{data['value']:,.2f}", f"{data['pct']:.1f}%", f"[{colour}]{bar}[/{colour}]")

    console.print(Columns([st, gt], expand=True))


def _render_performance_attribution(summary: PortfolioSummary) -> None:
    from analysis import performance_attribution
    attr = performance_attribution(summary.positions)

    console.print(Rule("[bold]Performance Attribution[/bold]", style="green"))

    wt = Table(title=f"Top Winners ({attr['winners_count']} positions profitable)", box=box.SIMPLE, title_style="bold green")
    wt.add_column("Ticker")
    wt.add_column("Name", max_width=25)
    wt.add_column("Unreal P&L (£)", justify="right")
    wt.add_column("Unreal P&L %", justify="right")
    for p in attr["top_winners"]:
        wt.add_row(
            p.ticker, (p.name or "")[:25],
            f"[green]{_fmt_gbp(p.unrealised_pnl_gbp or 0)}[/green]",
            f"[green]{_fmt_pct(p.unrealised_pnl_pct or 0)}[/green]",
        )

    lt = Table(title=f"Top Losers ({attr['losers_count']} positions underwater)", box=box.SIMPLE, title_style="bold red")
    lt.add_column("Ticker")
    lt.add_column("Name", max_width=25)
    lt.add_column("Unreal P&L (£)", justify="right")
    lt.add_column("Unreal P&L %", justify="right")
    for p in attr["top_losers"]:
        lt.add_row(
            p.ticker, (p.name or "")[:25],
            f"[red]{_fmt_gbp(p.unrealised_pnl_gbp or 0)}[/red]",
            f"[red]{_fmt_pct(p.unrealised_pnl_pct or 0)}[/red]",
        )

    console.print(Columns([wt, lt], expand=True))


def _render_warnings(warnings: list[str]) -> None:
    console.print(Rule("[bold yellow]Risk Warnings[/bold yellow]", style="yellow"))
    for w in warnings:
        console.print(f"  [bold yellow]⚠[/bold yellow]  {w}")


def _render_insights(insights: str) -> None:
    console.print(Rule("[bold cyan]AI Portfolio Insights[/bold cyan]", style="cyan"))
    console.print(Panel(insights, border_style="cyan", padding=(1, 2)))
