## Implementation Plan: Portfolio Dashboard (Risk/Return Metrics)

Goal: add a portfolio dashboard showing Sortino, Sharpe, VaR, CVaR, and an overall quality gauge.

### Scope & Requirements
- Inputs: portfolio weights + tickers (and optionally cash); lookback window selection (e.g., 6M/1Y/3Y); frequency (daily).
- Outputs: metrics (Sharpe, Sortino, VaR, CVaR), annualized return/vol, max drawdown, hit rate; visualizations: gauge for quality score, risk/return scatter, drawdown chart, VaR/CVaR distribution.
- Data source: consistent with current stack — prefer yahooquery price history.

### Steps
1) Backend data model
   - Add portfolio endpoint `/api/portfolio/metrics` (POST) accepting: tickers[], weights[], period, maybe risk-free rate override.
   - Validate weights (sum ≈ 1), support optional cash weight.
   - Fetch prices via yahooquery (aligned dates, utc→naive like correlation), build returns, compute portfolio returns.

2) Metrics calculations
   - Annualized return, annualized volatility (assume 252 trading days).
   - Sharpe: (mean – rf) / vol.
   - Sortino: downside stdev (<0) in denominator.
   - VaR (e.g., 95%/99%) historical; CVaR (expected shortfall) at same alpha.
   - Max drawdown, Calmar, hit ratio.
   - Compose a “quality score” (e.g., weighted z-scores of Sharpe, Sortino, drawdown).

3) Data schema for UI
   - JSON: {metrics: {...}, series: {portfolioReturns[], drawdown[], distribution[]}, meta: {period, rf}}.
   - Include percentiles for VaR/CVaR and bounds for the gauge.

4) Frontend UI
   - New section/tab “Portfolio Dashboard”.
   - Inputs: tickers + weights table, quick presets, period selector.
   - Outputs: cards for Sharpe/Sortino/VaR/CVaR, gauge for quality, mini charts: drawdown area, return distribution, risk-return scatter (maybe vs. equal-weight benchmark).

5) Error handling & UX
   - Surface validation errors (weights, insufficient data) via toast.
   - Show loading spinners and empty states.

6) Testing
   - Unit tests for metric functions (edge cases: zero vol, negative returns).
   - Manual: compare with a known portfolio (e.g., equal-weight QQQ top names) against a reference (numpy/pandas quick script).

7) Performance & caching (later)
   - Cache recent price pulls per ticker/period.
   - Optional: memoize portfolio results keyed by payload.

### Alternative approaches
- Use `yfinance` instead of yahooquery (slower, but simpler); or polygon/finnhub if keys/rate limits permit.
- Compute metrics client-side after sending aligned price matrix (less backend CPU, more network).
- Use MC simulation for VaR/CVaR (parametric/normal) if speed trumps fidelity.

