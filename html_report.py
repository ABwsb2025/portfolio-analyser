"""Generate self-contained HTML report with Chart.js visualisations."""
import json
from datetime import datetime, date
from pathlib import Path

from models import PortfolioSummary


def _to_json(obj):
    """JSON serialiser that handles dates and None."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Not serialisable: {type(obj)}")


def generate_html_report(
    summary: PortfolioSummary,
    warnings: list[str],
    insights: str,
    transactions: list,
    output_dir: str = ".",
) -> str:
    """Build and save a self-contained HTML report. Returns the file path."""
    report_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"portfolio_report_{report_date}.html"
    filepath = Path(output_dir) / filename

    # --- Data for charts ---
    sector_labels = list(summary.sector_breakdown.keys())
    sector_values = [round(v["value"], 2) for v in summary.sector_breakdown.values()]
    sector_pcts = [round(v["pct"], 1) for v in summary.sector_breakdown.values()]

    geo_labels = list(summary.geo_breakdown.keys())
    geo_values = [round(v["value"], 2) for v in summary.geo_breakdown.values()]

    positions_sorted = sorted(summary.positions, key=lambda p: p.weight_pct or 0, reverse=True)
    pos_labels = [p.ticker for p in positions_sorted]
    pos_pnl = [round(p.unrealised_pnl_gbp or 0, 2) for p in positions_sorted]
    pos_values = [round(p.current_value_gbp or p.total_cost_gbp, 2) for p in positions_sorted]
    pos_weights = [round(p.weight_pct or 0, 1) for p in positions_sorted]

    # Portfolio value over time from transactions
    timeline_data = _build_portfolio_timeline(transactions)

    # Colours for charts
    COLOURS = [
        "#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f",
        "#edc948","#b07aa1","#ff9da7","#9c755f","#bab0ac",
        "#54a24b","#f58518","#e45756","#72b7b2","#a2c8ec",
    ]

    pnl_colours = ["rgba(40,167,69,0.7)" if v >= 0 else "rgba(220,53,69,0.7)" for v in pos_pnl]

    pnl_col = "#28a745" if summary.total_pnl_gbp >= 0 else "#dc3545"
    unreal_col = "#28a745" if summary.total_unrealised_pnl_gbp >= 0 else "#dc3545"

    # Positions table rows
    pos_rows_html = ""
    for p in positions_sorted:
        pnl = p.unrealised_pnl_gbp or 0
        pnl_pct = p.unrealised_pnl_pct or 0
        pnl_style = "color:#28a745;font-weight:600" if pnl >= 0 else "color:#dc3545;font-weight:600"
        curr_val = p.current_value_gbp or p.total_cost_gbp
        curr_price = f"£{p.current_price_gbp:.4f}" if p.current_price_gbp else "N/A"
        sign = "+" if pnl >= 0 else ""
        pos_rows_html += f"""
        <tr>
            <td><strong>{p.ticker}</strong></td>
            <td>{(p.name or '')[:35]}</td>
            <td>{p.shares:.4f}</td>
            <td>£{p.avg_cost_gbp:.4f}</td>
            <td>{curr_price}</td>
            <td>£{curr_val:,.2f}</td>
            <td>{p.weight_pct:.1f}%</td>
            <td style="{pnl_style}">{sign}£{pnl:,.2f}</td>
            <td style="{pnl_style}">{sign}{pnl_pct:.2f}%</td>
            <td>{p.days_held or '-'}</td>
            <td>{f"{p.beta:.2f}" if p.beta is not None else "-"}</td>
            <td>{(p.sector or 'Unknown')[:30]}</td>
        </tr>"""

    # Warnings HTML
    warnings_html = ""
    if warnings:
        for w in warnings:
            warnings_html += f'<div class="alert alert-warning">⚠️  {w}</div>'

    # Insights HTML (convert markdown-ish to HTML)
    insights_html = insights.replace("\n## ", "\n<h3>").replace("\n### ", "\n<h4>")
    insights_html = insights_html.replace("\n## ", "\n</h3>\n<h3>")
    insights_html = insights_html.replace("\n", "<br>")

    # Risk metric rows
    def risk_row(label, val, good_threshold=None, bad_threshold=None):
        badge = ""
        if val is not None:
            badge = f'<span class="badge bg-secondary">{val}</span>'
        return f"<tr><td>{label}</td><td>{badge}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Report — {report_date}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ background: #0f1117; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }}
  .card {{ background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 12px; }}
  .card-title {{ color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }}
  .metric-value {{ font-size: 1.6rem; font-weight: 700; }}
  .positive {{ color: #28a745; }}
  .negative {{ color: #dc3545; }}
  h2, h3 {{ color: #58a6ff; }}
  .section-title {{ border-bottom: 2px solid #2a2d3a; padding-bottom: 8px; margin-bottom: 20px; color: #58a6ff; }}
  table {{ color: #e0e0e0 !important; }}
  .table-dark th {{ background: #1f2230 !important; }}
  .table tbody tr:hover {{ background: #1f2230; }}
  .table td, .table th {{ border-color: #2a2d3a !important; vertical-align: middle; font-size: 0.85rem; }}
  .alert-warning {{ background: #332a00; border-color: #664d00; color: #ffc107; }}
  .badge {{ font-size: 0.9rem; }}
  .insights-box {{ background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 20px; white-space: pre-wrap; line-height: 1.7; }}
  canvas {{ max-height: 300px; }}
  footer {{ color: #555; font-size: 0.75rem; text-align: center; padding: 20px; }}
</style>
</head>
<body>
<div class="container-fluid px-4 py-4">

  <!-- Header -->
  <div class="d-flex justify-content-between align-items-center mb-4">
    <div>
      <h1 class="mb-0" style="color:#58a6ff;">📊 Portfolio Analysis Report</h1>
      <small class="text-muted">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
    </div>
  </div>

  <!-- KPI Cards -->
  <div class="row g-3 mb-4">
    <div class="col-md-2">
      <div class="card p-3 h-100">
        <div class="card-title">Portfolio Value</div>
        <div class="metric-value text-white">£{summary.total_value_gbp:,.2f}</div>
        <small class="text-muted">Invested: £{summary.total_invested_gbp:,.2f}</small>
      </div>
    </div>
    <div class="col-md-2">
      <div class="card p-3 h-100">
        <div class="card-title">Total P&amp;L</div>
        <div class="metric-value" style="color:{pnl_col};">{"+" if summary.total_pnl_gbp >= 0 else ""}£{summary.total_pnl_gbp:,.2f}</div>
        <small style="color:{pnl_col};">{"+" if summary.total_pnl_pct >= 0 else ""}{summary.total_pnl_pct:.2f}% on invested</small>
      </div>
    </div>
    <div class="col-md-2">
      <div class="card p-3 h-100">
        <div class="card-title">Unrealised P&amp;L</div>
        <div class="metric-value" style="color:{unreal_col};">{"+" if summary.total_unrealised_pnl_gbp >= 0 else ""}£{summary.total_unrealised_pnl_gbp:,.2f}</div>
      </div>
    </div>
    <div class="col-md-2">
      <div class="card p-3 h-100">
        <div class="card-title">Realised P&amp;L</div>
        <div class="metric-value" style="color:{"#28a745" if summary.total_realised_pnl_gbp >= 0 else "#dc3545"};">{"+" if summary.total_realised_pnl_gbp >= 0 else ""}£{summary.total_realised_pnl_gbp:,.2f}</div>
      </div>
    </div>
    <div class="col-md-2">
      <div class="card p-3 h-100">
        <div class="card-title">Dividends Received</div>
        <div class="metric-value text-warning">£{summary.total_dividends_gbp:,.2f}</div>
      </div>
    </div>
    <div class="col-md-2">
      <div class="card p-3 h-100">
        <div class="card-title">Total Deposited</div>
        <div class="metric-value text-info">£{summary.total_deposits_gbp:,.2f}</div>
        <small class="text-muted">Withdrawn: £{summary.total_withdrawals_gbp:,.2f}</small>
      </div>
    </div>
  </div>

  <!-- Charts Row 1 -->
  <div class="row g-3 mb-4">
    <div class="col-md-4">
      <div class="card p-3">
        <h5 class="section-title">Sector Allocation</h5>
        <canvas id="sectorChart"></canvas>
      </div>
    </div>
    <div class="col-md-4">
      <div class="card p-3">
        <h5 class="section-title">Geographic Allocation</h5>
        <canvas id="geoChart"></canvas>
      </div>
    </div>
    <div class="col-md-4">
      <div class="card p-3">
        <h5 class="section-title">Portfolio Value Over Time</h5>
        <canvas id="timelineChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Position P&L Bar Chart -->
  <div class="row g-3 mb-4">
    <div class="col-12">
      <div class="card p-3">
        <h5 class="section-title">Position P&amp;L (£)</h5>
        <canvas id="pnlChart" style="max-height:250px;"></canvas>
      </div>
    </div>
  </div>

  <!-- Risk Metrics -->
  <div class="row g-3 mb-4">
    <div class="col-md-4">
      <div class="card p-3">
        <h5 class="section-title">Risk Metrics</h5>
        <table class="table table-sm table-dark table-borderless">
          <tbody>
            <tr><td>Portfolio Beta</td><td class="text-end"><strong>{f"{summary.portfolio_beta:.3f}" if summary.portfolio_beta else "N/A"}</strong></td></tr>
            <tr><td>30d Volatility (ann.)</td><td class="text-end"><strong>{f"{summary.volatility_30d:.2f}%" if summary.volatility_30d else "N/A"}</strong></td></tr>
            <tr><td>Sharpe Ratio</td><td class="text-end"><strong>{f"{summary.sharpe_ratio:.3f}" if summary.sharpe_ratio else "N/A"}</strong></td></tr>
            <tr><td>Max Drawdown</td><td class="text-end" style="color:#dc3545;"><strong>{f"{summary.max_drawdown_pct:.2f}%" if summary.max_drawdown_pct else "N/A"}</strong></td></tr>
            <tr><td>VaR 95% (1-day)</td><td class="text-end" style="color:#ffc107;"><strong>{f"£{summary.var_95:,.2f}" if summary.var_95 else "N/A"}</strong></td></tr>
            <tr><td>VaR 99% (1-day)</td><td class="text-end" style="color:#dc3545;"><strong>{f"£{summary.var_99:,.2f}" if summary.var_99 else "N/A"}</strong></td></tr>
          </tbody>
        </table>
      </div>
    </div>
    <div class="col-md-8">
      {f'<div class="card p-3"><h5 class="section-title">Risk Warnings</h5>{warnings_html}</div>' if warnings else '<div class="card p-3 d-flex align-items-center justify-content-center"><span class="text-success fs-5">✅ No concentration risk warnings</span></div>'}
    </div>
  </div>

  <!-- Positions Table -->
  <div class="card p-3 mb-4">
    <h5 class="section-title">Open Positions</h5>
    <div class="table-responsive">
      <table class="table table-dark table-hover table-sm">
        <thead>
          <tr>
            <th>Ticker</th><th>Name</th><th>Shares</th><th>Avg Cost</th>
            <th>Curr Price</th><th>Value (£)</th><th>Weight</th>
            <th>Unreal P&amp;L</th><th>Unreal %</th><th>Days</th><th>Beta</th><th>Sector</th>
          </tr>
        </thead>
        <tbody>
          {pos_rows_html}
        </tbody>
      </table>
    </div>
  </div>

  <!-- AI Insights -->
  {f'''<div class="card p-3 mb-4">
    <h5 class="section-title">🤖 AI Portfolio Insights</h5>
    <div class="insights-box">{insights_html}</div>
  </div>''' if insights and "not set" not in insights else ""}

  <footer>Generated by Trading 212 Portfolio Analyser · {report_date}</footer>
</div>

<script>
const COLOURS = {json.dumps(COLOURS)};

// Sector Doughnut
new Chart(document.getElementById('sectorChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(sector_labels)},
    datasets: [{{ data: {json.dumps(sector_values)}, backgroundColor: COLOURS, borderWidth: 1, borderColor: '#0f1117' }}]
  }},
  options: {{ plugins: {{ legend: {{ labels: {{ color: '#e0e0e0', font: {{ size: 10 }} }}, position: 'bottom' }} }}, cutout: '60%' }}
}});

// Geo Doughnut
new Chart(document.getElementById('geoChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(geo_labels)},
    datasets: [{{ data: {json.dumps(geo_values)}, backgroundColor: COLOURS, borderWidth: 1, borderColor: '#0f1117' }}]
  }},
  options: {{ plugins: {{ legend: {{ labels: {{ color: '#e0e0e0', font: {{ size: 10 }} }}, position: 'bottom' }} }}, cutout: '60%' }}
}});

// P&L Bar
new Chart(document.getElementById('pnlChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(pos_labels)},
    datasets: [{{
      label: 'Unrealised P&L (£)',
      data: {json.dumps(pos_pnl)},
      backgroundColor: {json.dumps(pnl_colours)},
      borderRadius: 4,
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => '£' + v.toFixed(0) }} }}
    }}
  }}
}});

// Timeline Line Chart
const timelineData = {json.dumps(timeline_data)};
new Chart(document.getElementById('timelineChart'), {{
  type: 'line',
  data: {{
    labels: timelineData.labels,
    datasets: [{{
      label: 'Portfolio Value (£)',
      data: timelineData.values,
      borderColor: '#58a6ff',
      backgroundColor: 'rgba(88,166,255,0.1)',
      fill: true,
      tension: 0.3,
      pointRadius: 2,
    }}]
  }},
  options: {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 8, font: {{ size: 10 }} }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => '£' + v.toLocaleString() }} }}
    }}
  }}
}});
</script>
</body>
</html>"""

    filepath.write_text(html, encoding="utf-8")
    return str(filepath)


def _build_portfolio_timeline(transactions: list) -> dict:
    """Build cumulative deposit-based timeline from transactions."""
    if not transactions:
        return {"labels": [], "values": []}

    # Compute running cumulative invested from deposits
    events = []
    running = 0.0
    for t in transactions:
        if "deposit" in t.action.lower():
            running += t.total or 0.0
            events.append((t.time.strftime("%Y-%m-%d"), round(running, 2)))
        elif "withdrawal" in t.action.lower():
            running -= abs(t.total or 0.0)
            events.append((t.time.strftime("%Y-%m-%d"), round(running, 2)))

    if not events:
        return {"labels": [], "values": []}

    labels = [e[0] for e in events]
    values = [e[1] for e in events]
    return {"labels": labels, "values": values}
