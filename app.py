"""
Agentic Financial Analysis Web Application
A modern web interface for stock analysis and global macro outlook
"""

import json
import pandas as pd
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib not installed. Technical indicators will be limited.")

import numpy as np
from typing import Dict, List
import yahooquery as yq
import requests
import os
from dotenv import load_dotenv
from datetime import datetime
from datetime import date as _date
from fredapi import Fred
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# LangChain imports
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# Load environment variables (handle cases where .env is not accessible)
try:
    load_dotenv()
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

openai_api_key = os.getenv("OPENAI_KEY")
fred_api_key = os.getenv("FRED_API")
polygon_key = os.getenv("POLYGON")
finhub_key = os.getenv("finhub")
mistral_key = os.getenv("MISTRAL")
groqK = os.getenv("groq")

# Global reference date
CURRENT_REF_DATE: _date | None = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_company_tickers() -> tuple[list[dict], dict[str, str]]:
    """Load the SEC company tickers file into memory."""
    company_list: list[dict] = []
    ticker_to_name: dict[str, str] = {}
    ticker_file = os.path.join(BASE_DIR, "company_tickers.json")

    if not os.path.exists(ticker_file):
        return company_list, ticker_to_name

    try:
        with open(ticker_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            for entry in raw.values():
                ticker = entry.get("ticker")
                title = entry.get("title")
                if not ticker or not title:
                    continue
                ticker_upper = ticker.upper().strip()
                title_clean = title.strip()
                company_list.append(
                    {
                        "ticker": ticker_upper,
                        "company": title_clean,
                        "company_lower": title_clean.lower(),
                    }
                )
                ticker_to_name[ticker_upper] = title_clean
    except (json.JSONDecodeError, OSError) as exc:
        print(f"⚠️ Unable to load company_tickers.json: {exc}")

    company_list.sort(key=lambda item: item["ticker"])
    return company_list, ticker_to_name


COMPANY_TICKER_LIST, TICKER_TO_NAME = _load_company_tickers()


def resolve_ticker_symbol(query: str) -> str | None:
    """Resolve a user provided ticker or company name to an actual ticker."""
    if not query:
        return None

    candidate = query.strip()
    upto = candidate.upper()
    if upto in TICKER_TO_NAME:
        return upto

    candidate_lower = candidate.lower()

    # Exact company name match
    for entry in COMPANY_TICKER_LIST:
        if candidate_lower == entry["company_lower"]:
            return entry["ticker"]

    # Partial match
    for entry in COMPANY_TICKER_LIST:
        if candidate_lower in entry["company_lower"]:
            return entry["ticker"]

    return None


def search_company_tickers(term: str, limit: int = 8) -> list[dict]:
    """Search company tickers by ticker symbol prefix or company name."""
    if not term:
        return []

    term_clean = term.strip()
    term_upper = term_clean.upper()
    term_lower = term_clean.lower()
    results: list[dict] = []

    # Prioritize ticker prefix matches
    for entry in COMPANY_TICKER_LIST:
        if entry["ticker"].startswith(term_upper):
            results.append({"ticker": entry["ticker"], "company": entry["company"]})
            if len(results) >= limit:
                return results

    # Then company name matches
    for entry in COMPANY_TICKER_LIST:
        if term_lower in entry["company_lower"]:
            results.append({"ticker": entry["ticker"], "company": entry["company"]})
            if len(results) >= limit:
                break

    return results


def build_correlation_matrix(
    raw_inputs: list[str] | str,
    period: str = "1y",
) -> dict:
    """
    Build a price return correlation matrix for a list of tickers.
    Uses yahooquery for pricing data and returns JSON-ready structure.
    """
    if not raw_inputs:
        raise ValueError("No tickers provided")

    # Normalize input to a clean list of tickers
    if isinstance(raw_inputs, str):
        candidates = [
            item.strip()
            for item in raw_inputs.replace(";", ",").split(",")
            if item.strip()
        ]
    else:
        candidates = [str(item).strip() for item in raw_inputs if str(item).strip()]

    if len(candidates) < 2:
        raise ValueError("Please provide at least two tickers")

    resolved: list[str] = []
    invalid: list[str] = []

    for item in candidates:
        ticker = resolve_ticker_symbol(item)
        if ticker:
            resolved.append(ticker)
        else:
            # If resolver fails, fall back to the cleaned uppercase symbol
            fallback = item.upper()
            resolved.append(fallback)
            invalid.append(item)

    # De-duplicate while preserving order
    seen = set()
    unique_tickers: list[str] = []
    for t in resolved:
        if t not in seen:
            unique_tickers.append(t)
            seen.add(t)

    if len(unique_tickers) < 2:
        raise ValueError("Need at least two distinct tickers after cleaning")
    if len(unique_tickers) > 25:
        raise ValueError("Please limit requests to 25 tickers")

    # Fetch historical prices
    ticker_obj = yq.Ticker(" ".join(unique_tickers))
    hist = ticker_obj.history(period=period, interval="1d")
    if hist is None or getattr(hist, "empty", True):
        raise ValueError("No price data returned for the requested tickers/period")

    # Reset index to columns
    hist = hist.reset_index()

    symbol_col = "symbol" if "symbol" in hist.columns else None
    date_col = "date" if "date" in hist.columns else None
    value_col = "adjclose" if "adjclose" in hist.columns else (
        "close" if "close" in hist.columns else None
    )

    if not symbol_col or not date_col or not value_col:
        raise ValueError("Unexpected price data format from provider")

    # Normalize date column to consistent timezone-naive pandas datetime
    hist[date_col] = (
        pd.to_datetime(hist[date_col], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    hist = hist.dropna(subset=[date_col])

    price_table = (
        hist[[symbol_col, date_col, value_col]]
        .pivot_table(index=date_col, columns=symbol_col, values=value_col)
        .sort_index()
    )

    # Forward-fill to handle occasional gaps, drop all-NaN columns
    price_table = price_table.ffill().dropna(axis=1, how="all")
    if price_table.shape[1] < 2:
        raise ValueError("Insufficient overlapping data to compute correlation")

    returns = price_table.pct_change().dropna(how="all")
    returns = returns.dropna(axis=1, how="all")

    if returns.shape[1] < 2:
        raise ValueError("Not enough return data to compute correlation matrix")

    corr = returns.corr().replace([np.inf, -np.inf], np.nan).round(4)
    corr = corr.fillna(0)

    last_date = price_table.index.max()
    last_date_str = last_date.strftime("%Y-%m-%d") if hasattr(last_date, "strftime") else str(last_date)

    return {
        "tickers": corr.columns.tolist(),
        "matrix": corr.to_numpy().tolist(),
        "period": period,
        "asOf": last_date_str,
        "warnings": invalid if invalid else None,
    }


# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# ========== TOOLS ==========

@tool
def get_financial_metrics(ticker: str) -> str:
    """
    Fetches comprehensive and validated financial ratios and metrics for a given ticker symbol.
    Uses same data source as technical analysis for consistency. Validates data quality.
    """
    time.sleep(1)
    import urllib.error
    try:
        ticker = ticker.strip().lstrip('$').upper()
        
        retries = 3
        delay = 1
        info = None
        
        for attempt in range(retries):
            try:
                stock = yq.Ticker(ticker, formatted=False)
                modules = stock.all_modules
                ticker_modules = modules.get(ticker, {})
                info = {}
                # Fetch comprehensive modules
                for module_name in ['summaryDetail', 'financialData', 'defaultKeyStatistics', 
                                   'price', 'quoteType', 'assetProfile']:
                    module_data = ticker_modules.get(module_name)
                    if isinstance(module_data, dict):
                        info.update(module_data)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    return f"❌ Error fetching metrics for {ticker}: {e}"
        
        if info is None:
            return f"❌ Error fetching metrics for {ticker}: rate limited or no data"

        # Get historical data for validation and additional metrics
        hist = None
        for attempt in range(retries):
            try:
                hist_full = stock.history(period="1y", interval="1d")
                hist = hist_full.tail(252) if not hist_full.empty else pd.DataFrame()
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    hist = pd.DataFrame()
                    break
        
        # Extract and validate metrics
        current_price = info.get('currentPrice') or (hist['close'].iloc[-1] if not hist.empty else None)
        if current_price is None:
            return f"❌ Unable to determine current price for {ticker}"
        
        # === VALUATION METRICS ===
        result = f"📊 Fundamental Analysis for {ticker} (Price: ${current_price:.2f}):\n\n"
        result += "🔹 Valuation Metrics:\n"
        
        pe_forward = info.get('forwardPE')
        pe_trailing = info.get('trailingPE')
        if pe_forward:
            result += f"  • Forward P/E: {pe_forward:.2f}"
            if pe_forward < 15:
                result += " ✅ Undervalued"
            elif pe_forward > 25:
                result += " ⚠️ Potentially overvalued"
            result += "\n"
        if pe_trailing:
            result += f"  • Trailing P/E: {pe_trailing:.2f}\n"
        
        pb_ratio = info.get('priceToBook')
        if pb_ratio:
            result += f"  • Price-to-Book (P/B): {pb_ratio:.2f}"
            if pb_ratio < 1:
                result += " ✅ Trading below book value"
            elif pb_ratio > 3:
                result += " ⚠️ High premium to book"
            result += "\n"
        
        ps_ratio = info.get('priceToSalesTrailing12Months')
        if ps_ratio:
            result += f"  • Price-to-Sales (P/S): {ps_ratio:.2f}\n"
        
        market_cap = info.get('marketCap')
        if market_cap:
            if market_cap >= 1e12:
                result += f"  • Market Cap: ${market_cap/1e12:.2f}T (Mega Cap)\n"
            elif market_cap >= 1e9:
                result += f"  • Market Cap: ${market_cap/1e9:.2f}B\n"
            else:
                result += f"  • Market Cap: ${market_cap/1e6:.2f}M\n"
        
        ev = info.get('enterpriseValue')
        ev_revenue = info.get('enterpriseToRevenue')
        ev_ebitda = info.get('enterpriseToEbitda')
        if ev:
            result += f"  • Enterprise Value: ${ev/1e9:.2f}B\n"
        if ev_revenue:
            result += f"  • EV/Revenue: {ev_revenue:.2f}\n"
        if ev_ebitda:
            result += f"  • EV/EBITDA: {ev_ebitda:.2f}\n"
        
        # === PROFITABILITY METRICS ===
        result += "\n🔹 Profitability Metrics:\n"
        
        profit_margin = info.get('profitMargins')
        if profit_margin is not None:
            result += f"  • Profit Margin: {profit_margin*100:.2f}%"
            if profit_margin > 0.20:
                result += " ✅ Excellent (>20%)"
            elif profit_margin > 0.10:
                result += " ✅ Good (10-20%)"
            elif profit_margin > 0:
                result += " ⚠️ Low (<10%)"
            else:
                result += " ❌ Negative"
            result += "\n"
        
        gross_margin = info.get('grossMargins')
        if gross_margin is not None:
            result += f"  • Gross Margin: {gross_margin*100:.2f}%\n"
        
        operating_margin = info.get('operatingMargins')
        if operating_margin is not None:
            result += f"  • Operating Margin: {operating_margin*100:.2f}%\n"
        
        roe = info.get('returnOnEquity')
        if roe is not None:
            result += f"  • Return on Equity (ROE): {roe*100:.2f}%"
            if roe > 0.15:
                result += " ✅ Excellent (>15%)"
            elif roe > 0.10:
                result += " ✅ Good (10-15%)"
            result += "\n"
        
        roa = info.get('returnOnAssets')
        if roa is not None:
            result += f"  • Return on Assets (ROA): {roa*100:.2f}%\n"
        
        # === GROWTH METRICS ===
        result += "\n🔹 Growth Metrics:\n"
        
        earnings_growth = info.get('earningsGrowth')
        if earnings_growth is not None:
            result += f"  • Earnings Growth: {earnings_growth*100:.2f}%"
            if earnings_growth > 0.20:
                result += " 📈 Strong growth"
            elif earnings_growth > 0:
                result += " 📈 Positive growth"
            else:
                result += " 📉 Declining earnings"
            result += "\n"
        
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth is not None:
            result += f"  • Revenue Growth: {revenue_growth*100:.2f}%\n"
        
        # === FINANCIAL HEALTH METRICS ===
        result += "\n🔹 Financial Health:\n"
        
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity is not None:
            result += f"  • Debt-to-Equity: {debt_to_equity:.2f}"
            if debt_to_equity < 0.5:
                result += " ✅ Low debt"
            elif debt_to_equity < 1.0:
                result += " ✅ Moderate debt"
            elif debt_to_equity < 2.0:
                result += " ⚠️ High debt"
            else:
                result += " ❌ Very high debt"
            result += "\n"
        
        current_ratio = info.get('currentRatio')
        if current_ratio is not None:
            result += f"  • Current Ratio: {current_ratio:.2f}"
            if current_ratio > 2.0:
                result += " ✅ Strong liquidity"
            elif current_ratio > 1.0:
                result += " ✅ Adequate liquidity"
            else:
                result += " ⚠️ Low liquidity"
            result += "\n"
        
        quick_ratio = info.get('quickRatio')
        if quick_ratio is not None:
            result += f"  • Quick Ratio: {quick_ratio:.2f}\n"
        
        cash_per_share = info.get('totalCashPerShare')
        if cash_per_share is not None:
            result += f"  • Cash per Share: ${cash_per_share:.2f}\n"
        
        # === MARKET METRICS ===
        result += "\n🔹 Market Metrics:\n"
        
        beta = info.get('beta')
        if beta is not None:
            result += f"  • Beta: {beta:.2f}"
            if beta > 1.2:
                result += " ⚠️ High volatility (more volatile than market)"
            elif beta < 0.8:
                result += " ✅ Low volatility (less volatile than market)"
            else:
                result += " 📊 Market-like volatility"
            result += "\n"
        
        dividend_yield = info.get('dividendYield')
        if dividend_yield is not None:
            result += f"  • Dividend Yield: {dividend_yield*100:.2f}%\n"
        
        # 52-week range
        week_52_high = info.get('fiftyTwoWeekHigh')
        week_52_low = info.get('fiftyTwoWeekLow')
        if week_52_high and week_52_low:
            result += f"  • 52-Week High: ${week_52_high:.2f}\n"
            result += f"  • 52-Week Low: ${week_52_low:.2f}\n"
            if current_price:
                price_position = ((current_price - week_52_low) / (week_52_high - week_52_low)) * 100
                result += f"  • Price Position: {price_position:.1f}% of 52-week range\n"
        
        # === PRICE PERFORMANCE (from historical data) ===
        if not hist.empty and len(hist) > 0:
            result += "\n🔹 Price Performance:\n"
            hist_close = hist['close']
            
            if len(hist_close) >= 1:
                price_change_1d = ((hist_close.iloc[-1] - hist_close.iloc[-2]) / hist_close.iloc[-2] * 100) if len(hist_close) >= 2 else 0
                result += f"  • 1-Day Change: {price_change_1d:+.2f}%\n"
            
            if len(hist_close) >= 6:
                price_change_5d = ((hist_close.iloc[-1] - hist_close.iloc[-6]) / hist_close.iloc[-6] * 100)
                result += f"  • 5-Day Change: {price_change_5d:+.2f}%\n"
            
            if len(hist_close) >= 21:
                price_change_20d = ((hist_close.iloc[-1] - hist_close.iloc[-21]) / hist_close.iloc[-21] * 100)
                result += f"  • 20-Day Change: {price_change_20d:+.2f}%\n"
            
            if len(hist_close) >= 63:
                price_change_3m = ((hist_close.iloc[-1] - hist_close.iloc[-63]) / hist_close.iloc[-63] * 100)
                result += f"  • 3-Month Change: {price_change_3m:+.2f}%\n"
            
            if len(hist_close) >= 252:
                price_change_1y = ((hist_close.iloc[-1] - hist_close.iloc[-252]) / hist_close.iloc[-252] * 100)
                result += f"  • 1-Year Change: {price_change_1y:+.2f}%\n"
            
            # Volatility
            if len(hist_close) >= 30:
                volatility_30d = (hist_close.rolling(30).std().iloc[-1] / hist_close.iloc[-1]) * 100
                result += f"  • 30-Day Volatility: {volatility_30d:.2f}%\n"
        
        # === OVERALL ASSESSMENT ===
        result += "\n💡 Fundamental Summary: "
        strong_points = []
        concerns = []
        
        if profit_margin and profit_margin > 0.15:
            strong_points.append("high profitability")
        if roe and roe > 0.15:
            strong_points.append("strong ROE")
        if debt_to_equity and debt_to_equity < 1.0:
            strong_points.append("low debt")
        if earnings_growth and earnings_growth > 0.15:
            strong_points.append("strong earnings growth")
        
        if profit_margin and profit_margin < 0:
            concerns.append("negative margins")
        if debt_to_equity and debt_to_equity > 2.0:
            concerns.append("high debt")
        if earnings_growth and earnings_growth < -0.10:
            concerns.append("declining earnings")
        
        if strong_points:
            result += f"✅ Strengths: {', '.join(strong_points)}. "
        if concerns:
            result += f"⚠️ Concerns: {', '.join(concerns)}."
        if not strong_points and not concerns:
            result += "Mixed fundamentals."
        
        return result
        
    except Exception as e:
        return f"❌ Error fetching metrics for {ticker}: {str(e)}"


@tool
def get_market_news(ticker: str) -> str:
    """Fetches recent news for a given ticker symbol with sentiment analysis."""
    try:
        ticker = ticker.upper().strip().lstrip('$')
        news_items: List[Dict] = []

        try:
            stock = yq.Ticker(ticker)
            raw_news = stock.news() if callable(stock.news) else stock.news
            if isinstance(raw_news, dict):
                raw_news = raw_news.get(ticker) or next(iter(raw_news.values()), [])
            if isinstance(raw_news, list):
                news_items = raw_news[:5]
        except Exception as e:
            print(f"yahooquery news fetch failed: {e}")

        if not news_items and finhub_key:
            url = f"https://finnhub.io/api/v1/news?symbol={ticker}&token={finhub_key}"
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    for art in resp.json()[:5]:
                        news_items.append({
                            "title": art.get("headline", ""),
                            "summary": art.get("summary", "")
                        })
            except Exception as e:
                print(f"Finnhub news fetch failed: {e}")

        if not news_items and polygon_key:
            url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=5&apiKey={polygon_key}"
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    poly_news = resp.json().get("results", [])
                    for art in poly_news:
                        news_items.append({
                            "title": art.get("title", ""),
                            "summary": art.get("description", "")
                        })
            except Exception as e:
                print(f"Polygon news fetch failed: {e}")

        news_items = [n for n in news_items if isinstance(n, dict)]

        if not news_items:
            return f"No recent news found for {ticker}"

        positive_words = ['growth', 'profit', 'beat', 'exceed', 'strong', 'bullish', 'upgrade', 'buy']
        negative_words = ['loss', 'decline', 'miss', 'weak', 'bearish', 'downgrade', 'sell', 'concern']

        formatted_news = []
        for art in news_items:
            if not isinstance(art, dict):
                continue
            title = art.get('title', '')
            summary = art.get('summary', '')
            text = f"{title} {summary}".lower()

            pos_count = sum(1 for w in positive_words if w in text)
            neg_count = sum(1 for w in negative_words if w in text)
            sentiment = '🟢 Positive' if pos_count > neg_count else '🔴 Negative' if neg_count > pos_count else '⚪ Neutral'

            formatted_news.append(f"• {title} [{sentiment}]\n  {summary[:200]}...")

        return f"📰 Recent news for {ticker}:\n" + "\n".join(formatted_news)

    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


@tool
def get_economic_indicators(indicator_type: str) -> str:
    """Fetches key economic indicators from FRED API. Options: 'general', 'inflation', 'employment', 'rates', 'gdp'"""
    
    if not fred_api_key:
        return "❌ FRED_API_KEY not set"
    fred = Fred(api_key=fred_api_key)

    indicators_map = {
        "general": {
            'Federal Funds Rate': 'FEDFUNDS',
            '10-Year Treasury': 'GS10',
            'VIX': 'VIXCLS',
            'Unemployment Rate': 'UNRATE',
            'CPI': 'CPIAUCSL'
        },
        'inflation': {
            'CPI All Items': 'CPIAUCSL',
            'Core CPI': 'CPILFESL',
            'PCE': 'PCEPI',
            'Core PCE': 'PCEPILFE',
            '5Y5Y Inflation Expect': 'T5YIE'
        },
        'employment': {
            'Unemployment Rate': 'UNRATE',
            'Labor Force Participation': 'CIVPART',
            'Nonfarm Payrolls': 'PAYEMS',
            'Initial Jobless Claims': 'ICSA',
            'Job Openings': 'JTSJOL'
        },
        'rates': {
            'Federal Funds Rate': 'FEDFUNDS',
            '2-Year Treasury': 'GS2',
            '10-Year Treasury': 'GS10',
            '30-Year Treasury': 'GS30',
            'Yield Curve (10Y-2Y)': None
        },
        'gdp': {
            'Real GDP': 'GDPC1',
            'GDP Growth Rate': 'GDPC1',
            'Personal Consumption': 'PCEC',
            'Business Investment': 'GPDI',
            'Government Spending': 'GCE'
        }
    }

    sel = indicators_map.get(indicator_type.lower(), indicators_map['general'])
    lines = [f"📈 Economic Indicators ({indicator_type}):"]

    for name, sid in sel.items():
        try:
            if sid is None:
                y10 = fred.get_series('GS10')
                y2 = fred.get_series('GS2')
                if y10.empty or y2.empty:
                    lines.append(f"• {name}: no data")
                else:
                    val = y10.iloc[-1] - y2.iloc[-1]
                    date = y10.index[-1].strftime('%Y-%m-%d')
                    lines.append(f"• {name}: {val:.2f}% (as of {date})")
                continue

            data = fred.get_series(sid)
            if data.empty:
                lines.append(f"• {name}: no data")
                continue
            latest = data.iloc[-1]
            date = data.index[-1].strftime('%Y-%m-%d')

            if name in ['CPI', 'CPI All Items', 'Core CPI', 'PCE', 'Core PCE']:
                if len(data) >= 12:
                    year_data = data[-12:]
                    start = year_data.iloc[0]
                    end = year_data.iloc[-1]
                    yoy = (end / start - 1) * 100
                    lines.append(f"• {name}: {yoy:.1f}% YoY (as of {date})")
                else:
                    lines.append(f"• {name}: {latest:.1f} (as of {date})")

            elif name == 'GDP Growth Rate':
                if len(data) >= 2:
                    prev = data.iloc[-2]
                    gr = (latest / prev - 1) * 100
                    lines.append(f"• {name}: {gr:.1f}% qtr (as of {date})")
                else:
                    lines.append(f"• {name}: {latest:.1f} (as of {date})")

            elif 'Rate' in name or 'Treasury' in name or 'VIX' in name:
                lines.append(f"• {name}: {latest:.2f}% (as of {date})")

            else:
                lines.append(f"• {name}: {latest:,.1f} (as of {date})")

        except Exception as e:
            lines.append(f"• {name}: error fetching ({e})")

    return "\n".join(lines)


@tool
def get_fed_policy_info() -> str:
    """Get Federal Reserve policy-related indicators including rates and balance sheet."""
    if not fred_api_key:
        return "❌ FRED_API_KEY not set"
    fred = Fred(api_key=fred_api_key)
    policy = {
        'Federal Funds Rate': 'FEDFUNDS',
        'Upper Target': 'DFEDTARU',
        'Lower Target': 'DFEDTARL',
        'Total Assets': 'WALCL',
        'Excess Reserves': 'EXCSRESNS'
    }
    out = ["🏛️ Fed Policy Indicators:"]
    for name, sid in policy.items():
        try:
            data = fred.get_series(sid, limit=3)
            if data.empty:
                out.append(f"• {name}: no data")
                continue
            latest = data.iloc[-1]
            date = data.index[-1].strftime("%Y-%m-%d")
            if len(data) >= 2:
                prev = data.iloc[-2]
                diff = latest - prev
                arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            else:
                arrow = ""
            val = f"{latest:.2f}%" if "Rate" in name else f"{int(latest):,}"
            out.append(f"• {name}: {val} {arrow} (as of {date})")
        except Exception as e:
            out.append(f"• {name}: error fetching ({e})")
    return "\n".join(out)


@tool
def get_technical_indicators(ticker: str) -> str:
    """
    Fetches comprehensive technical indicators for a stock using the same data source as TradingView charts.
    Returns: RSI, MACD, Bollinger Bands, ATR, Moving Averages, Volume indicators, Support/Resistance levels.
    """
    try:
        t = ticker.strip().lstrip('$').upper()
        stock = yq.Ticker(t)
        
        # Use same data source as chart endpoint for consistency
        hist_full = stock.history(period="1y", interval="1d")
        if hist_full.empty:
            return f"❌ No price data available for {t}"
        
        # Get last 252 trading days (1 year)
        hist = hist_full.tail(252)
        
        if hist.empty or len(hist) < 50:
            return f"❌ Not enough price data for {t} (need at least 50 days, got {len(hist)})"
        
        # Extract price arrays
        close = hist["close"].values
        high = hist["high"].values
        low = hist["low"].values
        volume = hist["volume"].values if "volume" in hist.columns else None
        
        current_price = float(close[-1])
        result = f"📊 Technical Analysis for {t} (Current Price: ${current_price:.2f}):\n\n"
        
        # === MOMENTUM INDICATORS ===
        result += "🔹 Momentum Indicators:\n"
        
        if TALIB_AVAILABLE:
            # RSI (14-day)
            rsi = talib.RSI(close, timeperiod=14)
            rsi_val = float(rsi[-1])
            result += f"  • RSI (14): {rsi_val:.2f}"
            if rsi_val < 30:
                result += " ⚠️ OVERSOLD"
            elif rsi_val > 70:
                result += " ⚠️ OVERBOUGHT"
            elif 30 <= rsi_val <= 50:
                result += " 📉 Bearish"
            elif 50 < rsi_val <= 70:
                result += " 📈 Bullish"
            result += "\n"
            
            # MACD (12, 26, 9)
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            macd_val = float(macd[-1])
            macd_sig_val = float(macd_signal[-1])
            macd_hist_val = float(macd_hist[-1])
            result += f"  • MACD: {macd_val:.4f}\n"
            result += f"  • MACD Signal: {macd_sig_val:.4f}\n"
            result += f"  • MACD Histogram: {macd_hist_val:.4f}"
            if macd_hist_val > 0:
                result += " 📈 Bullish momentum"
            else:
                result += " 📉 Bearish momentum"
            result += "\n"
            
            # Stochastic Oscillator (14, 3, 3)
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            stoch_k = float(slowk[-1])
            stoch_d = float(slowd[-1])
            result += f"  • Stochastic %K: {stoch_k:.2f}\n"
            result += f"  • Stochastic %D: {stoch_d:.2f}\n"
            
        else:
            # Fallback: Simplified calculations
            # RSI calculation
            delta = np.diff(close)
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            
            # Calculate RSI using Wilder's smoothing
            if len(gains) >= 14:
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                rs = avg_gain / avg_loss if avg_loss > 0 else 100
                rsi_val = 100 - (100 / (1 + rs))
            else:
                rsi_val = 50.0
            
            result += f"  • RSI (14): {rsi_val:.2f}\n"
            
            # Simple MACD
            ema12 = np.mean(close[-12:]) if len(close) >= 12 else current_price
            ema26 = np.mean(close[-26:]) if len(close) >= 26 else current_price
            macd_val = ema12 - ema26
            result += f"  • MACD (approx): {macd_val:.4f}\n"
        
        # === TREND INDICATORS ===
        result += "\n🔹 Trend Indicators:\n"
        
        if TALIB_AVAILABLE:
            # Moving Averages
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            sma_200 = talib.SMA(close, timeperiod=200) if len(close) >= 200 else None
            ema_12 = talib.EMA(close, timeperiod=12)
            ema_26 = talib.EMA(close, timeperiod=26)
            
            result += f"  • SMA (20): ${sma_20[-1]:.2f}\n"
            result += f"  • SMA (50): ${sma_50[-1]:.2f}\n"
            if sma_200 is not None:
                result += f"  • SMA (200): ${sma_200[-1]:.2f}\n"
            result += f"  • EMA (12): ${ema_12[-1]:.2f}\n"
            result += f"  • EMA (26): ${ema_26[-1]:.2f}\n"
            
            # Trend analysis
            if len(close) >= 200 and sma_200 is not None:
                if current_price > sma_200[-1]:
                    result += "  ✅ Price above 200-day SMA (Long-term uptrend)\n"
                else:
                    result += "  ⚠️ Price below 200-day SMA (Long-term downtrend)\n"
            
            if current_price > sma_50[-1] > sma_20[-1]:
                result += "  📈 Short-term uptrend (Price > SMA50 > SMA20)\n"
            elif current_price < sma_50[-1] < sma_20[-1]:
                result += "  📉 Short-term downtrend (Price < SMA50 < SMA20)\n"
            else:
                result += "  ⚡ Sideways/consolidation\n"
                
        else:
            # Fallback moving averages
            sma_20 = np.mean(close[-20:]) if len(close) >= 20 else current_price
            sma_50 = np.mean(close[-50:]) if len(close) >= 50 else current_price
            result += f"  • SMA (20): ${sma_20:.2f}\n"
            result += f"  • SMA (50): ${sma_50:.2f}\n"
        
        # === VOLATILITY INDICATORS ===
        result += "\n🔹 Volatility Indicators:\n"
        
        if TALIB_AVAILABLE:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            bb_upper_val = float(bb_upper[-1])
            bb_middle_val = float(bb_middle[-1])
            bb_lower_val = float(bb_lower[-1])
            bb_position = ((current_price - bb_lower_val) / (bb_upper_val - bb_lower_val)) * 100 if (bb_upper_val - bb_lower_val) > 0 else 50
            
            result += f"  • Bollinger Upper: ${bb_upper_val:.2f}\n"
            result += f"  • Bollinger Middle: ${bb_middle_val:.2f}\n"
            result += f"  • Bollinger Lower: ${bb_lower_val:.2f}\n"
            result += f"  • BB Position: {bb_position:.1f}%"
            if bb_position > 80:
                result += " ⚠️ Near upper band (overbought)"
            elif bb_position < 20:
                result += " ⚠️ Near lower band (oversold)"
            result += "\n"
            
            # ATR (Average True Range)
            atr = talib.ATR(high, low, close, timeperiod=14)
            atr_val = float(atr[-1])
            atr_pct = (atr_val / current_price) * 100
            result += f"  • ATR (14): ${atr_val:.2f} ({atr_pct:.2f}% of price)\n"
            
        else:
            # Fallback Bollinger Bands
            sma_20_bb = np.mean(close[-20:]) if len(close) >= 20 else current_price
            std_20 = np.std(close[-20:]) if len(close) >= 20 else 0
            bb_upper_val = sma_20_bb + 2 * std_20
            bb_lower_val = sma_20_bb - 2 * std_20
            bb_position = ((current_price - bb_lower_val) / (bb_upper_val - bb_lower_val)) * 100 if (bb_upper_val - bb_lower_val) > 0 else 50
            
            result += f"  • Bollinger Upper: ${bb_upper_val:.2f}\n"
            result += f"  • Bollinger Lower: ${bb_lower_val:.2f}\n"
            result += f"  • BB Position: {bb_position:.1f}%\n"
        
        # === VOLUME INDICATORS ===
        if volume is not None and len(volume) > 0:
            result += "\n🔹 Volume Indicators:\n"
            avg_volume_20 = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
            current_volume = float(volume[-1])
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
            
            result += f"  • Current Volume: {current_volume:,.0f}\n"
            result += f"  • Avg Volume (20d): {avg_volume_20:,.0f}\n"
            result += f"  • Volume Ratio: {volume_ratio:.2f}x"
            if volume_ratio > 1.5:
                result += " 📈 High volume (strong interest)"
            elif volume_ratio < 0.5:
                result += " 📉 Low volume (weak interest)"
            result += "\n"
        
        # === SUPPORT & RESISTANCE ===
        result += "\n🔹 Support & Resistance Levels:\n"
        recent_high = float(np.max(high[-20:]))  # 20-day high
        recent_low = float(np.min(low[-20:]))   # 20-day low
        result += f"  • 20-Day High (Resistance): ${recent_high:.2f}\n"
        result += f"  • 20-Day Low (Support): ${recent_low:.2f}\n"
        
        if len(hist) >= 50:
            medium_high = float(np.max(high[-50:]))
            medium_low = float(np.min(low[-50:]))
            result += f"  • 50-Day High: ${medium_high:.2f}\n"
            result += f"  • 50-Day Low: ${medium_low:.2f}\n"
        
        # === PRICE ACTION SUMMARY ===
        result += "\n🔹 Price Action Summary:\n"
        price_change_1d = ((close[-1] - close[-2]) / close[-2] * 100) if len(close) >= 2 else 0
        price_change_5d = ((close[-1] - close[-6]) / close[-6] * 100) if len(close) >= 6 else 0
        price_change_20d = ((close[-1] - close[-21]) / close[-21] * 100) if len(close) >= 21 else 0
        
        result += f"  • 1-Day Change: {price_change_1d:+.2f}%\n"
        result += f"  • 5-Day Change: {price_change_5d:+.2f}%\n"
        result += f"  • 20-Day Change: {price_change_20d:+.2f}%\n"
        
        # Overall technical score
        result += "\n💡 Technical Outlook: "
        bullish_signals = 0
        bearish_signals = 0
        
        if TALIB_AVAILABLE:
            if rsi_val > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
            if macd_hist_val > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
            if current_price > sma_20[-1]:
                bullish_signals += 1
            else:
                bearish_signals += 1
            if bb_position > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
        else:
            # Fallback scoring when TA-Lib not available
            if rsi_val > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
            if macd_val > 0:
                bullish_signals += 1
            else:
                bearish_signals += 1
            if current_price > sma_20:
                bullish_signals += 1
            else:
                bearish_signals += 1
            if bb_position > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            result += "📈 BULLISH - More bullish signals than bearish\n"
        elif bearish_signals > bullish_signals:
            result += "📉 BEARISH - More bearish signals than bullish\n"
        else:
            result += "⚡ NEUTRAL - Mixed signals\n"
        
        return result

    except Exception as e:
        return f"❌ Error getting technical indicators for {ticker}: {str(e)}"


@tool
def get_bond_yields() -> str:
    """Fetches current US Treasury bond yields and yield curve analysis."""
    if not fred_api_key:
        return "❌ FRED_API_KEY not set"
    fred = Fred(api_key=fred_api_key)
    
    yields = {
        '3-Month Treasury': 'GS3M',
        '6-Month Treasury': 'GS6M',
        '1-Year Treasury': 'GS1',
        '2-Year Treasury': 'GS2',
        '5-Year Treasury': 'GS5',
        '10-Year Treasury': 'GS10',
        '30-Year Treasury': 'GS30',
    }
    
    out = ["📉 Bond Yields:"]
    yield_values = {}
    
    for name, sid in yields.items():
        try:
            data = fred.get_series(sid)
            if not data.empty:
                latest = data.iloc[-1]
                date = data.index[-1].strftime("%Y-%m-%d")
                out.append(f"• {name}: {latest:.2f}% (as of {date})")
                yield_values[name] = latest
        except Exception as e:
            out.append(f"• {name}: error ({e})")
    
    # Calculate spreads
    if '10-Year Treasury' in yield_values and '2-Year Treasury' in yield_values:
        spread_10_2 = yield_values['10-Year Treasury'] - yield_values['2-Year Treasury']
        out.append("\n📊 Yield Curve Analysis:")
        out.append(f"• 10Y-2Y Spread: {spread_10_2:.2f}%")
        if spread_10_2 < 0:
            out.append("  ⚠️ Inverted yield curve - potential recession signal")
        elif spread_10_2 < 0.5:
            out.append("  ⚡ Flat yield curve - economic uncertainty")
        else:
            out.append("  ✅ Normal yield curve - healthy economy")
    
    return "\n".join(out)


# ========== AGENT SETUP ==========

STOCK_SYSTEM_PROMPT = """You are an expert stock analyst with deep knowledge of financial markets and technical analysis.
You have access to comprehensive, validated data sources that match TradingView chart data for consistency.

When analyzing stocks, you MUST:
1. Fetch comprehensive financial metrics using get_financial_metrics (includes valuation, profitability, growth, financial health)
2. Get detailed technical indicators using get_technical_indicators (includes RSI, MACD, Bollinger Bands, moving averages, volume, support/resistance)
3. Check recent news and sentiment using get_market_news

IMPORTANT DATA ACCURACY NOTES:
- Technical indicators use the same data source as TradingView charts (Yahoo Finance, 1-year daily data)
- All metrics are validated and include interpretation flags (✅ ⚠️ ❌)
- Price data is consistent between technical analysis and chart visualization
- Fundamental metrics include comprehensive ratios with industry context

Provide a clear, structured analysis with:
1. **Valuation Assessment**: 
   - Is the stock undervalued, fairly valued, or overvalued?
   - Compare P/E, P/B, P/S ratios to industry standards
   - Consider enterprise value metrics
   
2. **Technical Outlook**: 
   - Analyze momentum indicators (RSI, MACD, Stochastic)
   - Evaluate trend indicators (moving averages, price position)
   - Assess volatility (Bollinger Bands, ATR)
   - Review support/resistance levels
   - Determine overall technical bias (bullish/bearish/neutral)
   
3. **Fundamental Strength**:
   - Profitability metrics (margins, ROE, ROA)
   - Growth trajectory (earnings growth, revenue growth)
   - Financial health (debt levels, liquidity ratios)
   - Cash position and balance sheet strength
   
4. **Sentiment Analysis**: 
   - Recent news impact
   - Market sentiment from news
   
5. **Investment Recommendation**: 
   - Buy/Hold/Sell with clear reasoning
   - Risk assessment
   - Price targets or key levels to watch

Be concise but thorough. Always reference specific data points and indicators. Use the interpretation flags provided in the data to guide your analysis."""

MACRO_SYSTEM_PROMPT = """You are an expert macroeconomist and fixed income strategist.
Analyze global macro conditions and provide insights.

When analyzing macro topics, you should:
1. Get economic indicators using get_economic_indicators (options: 'general', 'inflation', 'employment', 'rates', 'gdp')
2. Check Fed policy using get_fed_policy_info
3. Get bond yields using get_bond_yields

Provide a clear, structured analysis with:
- Current economic conditions
- Policy outlook
- Market implications
- Investment positioning recommendations

Be concise but thorough. Use data to support your analysis."""

# Initialize LLM
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,
    api_key=groqK,
    max_tokens=2048,
    max_retries=3,
)


def analyze_stock_with_tools(ticker: str, company_name: str | None = None) -> str:
    """Analyze a stock by gathering data from tools and synthesizing with LLM."""
    try:
        # Gather data from tools
        metrics = get_financial_metrics.invoke(ticker)
        technical = get_technical_indicators.invoke(ticker)
        news = get_market_news.invoke(ticker)
        
        # Create a comprehensive prompt for the LLM
        display_name = company_name or TICKER_TO_NAME.get(ticker)
        header = f"{display_name} ({ticker})" if display_name else ticker

        data_summary = f"""
Here is the data gathered for {header}:

## Financial Metrics
{metrics}

## Technical Indicators
{technical}

## Recent News
{news}
"""
        
        prompt = f"""{STOCK_SYSTEM_PROMPT}

{data_summary}

Based on this comprehensive and validated data (which matches TradingView chart data for consistency), provide a detailed analysis of {header}.

CRITICAL: The technical indicators and price data come from the same source as the TradingView charts displayed to users, ensuring complete data consistency.

Provide your analysis following the structure outlined in the system prompt. Be specific about:
- Which indicators support your conclusions
- How valuation metrics compare to industry standards
- What the technical signals are telling you
- How fundamental strength aligns with technical outlook
- Clear risk/reward assessment

Be thorough, data-driven, and actionable."""

        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error analyzing {ticker}: {str(e)}"


def analyze_macro_with_tools(topic: str) -> str:
    """Analyze macro conditions by gathering data from tools and synthesizing with LLM."""
    try:
        # Gather data from tools
        general_indicators = get_economic_indicators.invoke("general")
        fed_policy = get_fed_policy_info.invoke("")
        bond_yields = get_bond_yields.invoke("")
        
        # Get additional specific data based on topic
        additional_data = ""
        topic_lower = topic.lower()
        if "inflation" in topic_lower:
            additional_data = "\n## Inflation Data\n" + get_economic_indicators.invoke("inflation")
        elif "employment" in topic_lower or "job" in topic_lower:
            additional_data = "\n## Employment Data\n" + get_economic_indicators.invoke("employment")
        elif "gdp" in topic_lower or "growth" in topic_lower:
            additional_data = "\n## GDP Data\n" + get_economic_indicators.invoke("gdp")
        elif "rate" in topic_lower or "yield" in topic_lower:
            additional_data = "\n## Interest Rates Data\n" + get_economic_indicators.invoke("rates")
        
        data_summary = f"""
## General Economic Indicators
{general_indicators}

## Fed Policy
{fed_policy}

## Bond Yields
{bond_yields}
{additional_data}
"""
        
        prompt = f"""{MACRO_SYSTEM_PROMPT}

Here is the current macro data:
{data_summary}

User's question/topic: {topic}

Based on this data, provide a comprehensive macro analysis including:
1. **Current Economic Conditions**: Summary of key indicators
2. **Policy Outlook**: What is the Fed likely to do?
3. **Market Implications**: How does this affect different asset classes?
4. **Investment Positioning**: Recommended allocation or strategy

Be concise but thorough."""

        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"Error analyzing macro topic: {str(e)}"


# ========== API ROUTES ==========

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')


@app.route('/api/correlation', methods=['POST'])
def correlation_matrix():
    """Compute correlation matrix for user-provided tickers."""
    try:
        data = request.get_json(force=True) or {}
        tickers_input = data.get('tickers') or data.get('symbols') or []
        period = data.get('period', '1y')

        result = build_correlation_matrix(tickers_input, period)
        return jsonify(result)
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/stock', methods=['POST'])
def analyze_stock():
    """Analyze a stock ticker."""
    try:
        data = request.get_json()
        user_query = data.get('ticker', '').strip()
        
        if not user_query:
            return jsonify({'error': 'No ticker provided'}), 400

        resolved_ticker = resolve_ticker_symbol(user_query)
        if resolved_ticker is None:
            return jsonify({'error': f"Unable to find a ticker for '{user_query}'"}), 404

        company_name = TICKER_TO_NAME.get(resolved_ticker)
        
        # Run the stock analysis with tools
        result = analyze_stock_with_tools(resolved_ticker, company_name)
        
        return jsonify({
            'ticker': resolved_ticker,
            'displayName': company_name,
            'userInput': user_query,
            'analysis': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/macro', methods=['POST'])
def analyze_macro():
    """Analyze global macro conditions."""
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        
        if not topic:
            return jsonify({'error': 'No topic provided'}), 400
        
        # Run the macro analysis with tools
        result = analyze_macro_with_tools(topic)
        
        return jsonify({
            'topic': topic,
            'analysis': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tickers/search')
def search_tickers():
    """Search tickers or company names for auto-complete."""
    try:
        query = request.args.get('q', '').strip()
        if not query or not COMPANY_TICKER_LIST:
            return jsonify([])
        matches = search_company_tickers(query)
        return jsonify(matches)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/quick/metrics/<ticker>')
def quick_metrics(ticker):
    """Quick endpoint to get financial metrics."""
    try:
        result = get_financial_metrics.invoke(ticker)
        return jsonify({'ticker': ticker, 'metrics': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/quick/indicators/<indicator_type>')
def quick_indicators(indicator_type):
    """Quick endpoint to get economic indicators."""
    try:
        result = get_economic_indicators.invoke(indicator_type)
        return jsonify({'type': indicator_type, 'indicators': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/quick/yields')
def quick_yields():
    """Quick endpoint to get bond yields."""
    try:
        result = get_bond_yields.invoke("")
        return jsonify({'yields': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chart/<ticker>')
def get_chart_data(ticker):
    """Get historical OHLCV data for TradingView Lightweight Charts."""
    try:
        ticker = ticker.strip().lstrip('$').upper()
        period = request.args.get('period', '1y')  # 1m, 3m, 6m, 1y, 2y, 5y
        
        stock = yq.Ticker(ticker)
        hist = stock.history(period=period, interval="1d")
        
        if hist.empty:
            return jsonify({'error': f'No data found for {ticker}'}), 404
        
        # Reset index to get date as a column
        hist = hist.reset_index()
        
        # Format data for TradingView Lightweight Charts
        # Candlestick data format: { time: 'YYYY-MM-DD', open, high, low, close }
        candle_data = []
        volume_data = []
        
        for _, row in hist.iterrows():
            # Handle multi-index (ticker, date) or single index
            if 'date' in hist.columns:
                date_val = row['date']
            elif 'Date' in hist.columns:
                date_val = row['Date']
            else:
                # Try to find any date-like column
                date_val = row.get('index', row.iloc[0])
            
            # Convert to string format
            if hasattr(date_val, 'strftime'):
                time_str = date_val.strftime('%Y-%m-%d')
            else:
                time_str = str(date_val)[:10]
            
            candle_data.append({
                'time': time_str,
                'open': float(row['open']) if pd.notna(row['open']) else 0,
                'high': float(row['high']) if pd.notna(row['high']) else 0,
                'low': float(row['low']) if pd.notna(row['low']) else 0,
                'close': float(row['close']) if pd.notna(row['close']) else 0,
            })
            
            volume_data.append({
                'time': time_str,
                'value': float(row['volume']) if pd.notna(row['volume']) else 0,
                'color': 'rgba(38, 166, 154, 0.5)' if row['close'] >= row['open'] else 'rgba(239, 83, 80, 0.5)'
            })
        
        # Get current price info
        current_price = candle_data[-1]['close'] if candle_data else 0
        first_price = candle_data[0]['open'] if candle_data else 0
        price_change = current_price - first_price
        price_change_pct = (price_change / first_price * 100) if first_price else 0
        
        return jsonify({
            'ticker': ticker,
            'displayName': TICKER_TO_NAME.get(ticker),
            'period': period,
            'candles': candle_data,
            'volume': volume_data,
            'currentPrice': current_price,
            'priceChange': price_change,
            'priceChangePct': price_change_pct,
            'dataPoints': len(candle_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Agentic Financial Analysis Server...")
    print("📊 Stock Analysis Agent: Ready")
    print("🌍 Macro Analysis Agent: Ready")
    print("🔗 Server running at http://localhost:5000")
    app.run(debug=True, port=5000)
