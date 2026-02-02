"""
Agentic Financial Analysis Web Application
A modern web interface for stock analysis and global macro outlook

QUAD-LLM ARCHITECTURE (4-Stage Quality Pipeline):
1. Mistral AI: Fast input validation, tool selection, and sentiment analysis
2. Groq OAI/gpt-oss-120b: Comprehensive synthesis and deep analysis (FREE)
3. MiMo-V2-Flash (OpenRouter): Output validation with reasoning chains (FREE)
4. Google AI Flash 3: Final polish and professional cleanup (virtually FREE)

Total cost per analysis: ~$0.0003 (Mistral validation only)
- Stage 1 (Mistral): $0.0003
- Stage 2 (Groq): $0.00 (free tier)
- Stage 3 (MiMo): $0.00 (free tier)
- Stage 4 (Flash 3): $0.00001 (negligible)

Benefits:
- Multi-stage quality assurance
- Each LLM optimized for its specialized task
- Transparent validation with raw output access
- Professional-grade final output
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
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai

# LangChain imports
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

# OpenRouter (for output validation with MiMo-V2-Flash)
from openai import OpenAI

# Web Search & Scraping imports
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from datetime import timedelta

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
newsapi = os.getenv("NEWS")
serpapi = os.getenv("SERP")
gnews = os.getenv("GNEWS")
brave = os.getenv("BRAVE")
openrouter_key = os.getenv("OPENROUTER")
nvidianim_key = os.getenv("NVIDIANIM") #Nvidia NIM platform 
fmp = os.getenv("FMP") #Financial Modeling Prep
gai = os.getenv("GAI") #Google AI Studio
# Google GenAI rate limiting (avoid 429s)
GOOGLE_GENAI_MIN_INTERVAL = float(os.getenv("GOOGLE_GENAI_MIN_INTERVAL", "2.0"))
GOOGLE_GENAI_MAX_RETRIES = int(os.getenv("GOOGLE_GENAI_MAX_RETRIES", "3"))
GOOGLE_GENAI_BACKOFF_BASE = float(os.getenv("GOOGLE_GENAI_BACKOFF_BASE", "1.5"))
_last_google_genai_ts = 0.0
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
        
        # === DERIVS (Derivatives - Options Market Data) ===
        try:
            options_data = stock.option_chain
            if isinstance(options_data, pd.DataFrame) and not options_data.empty:
                # yahooquery returns optionType as an index level, so slice via xs
                try:
                    calls = options_data.xs("calls", level="optionType")
                except Exception:
                    calls = pd.DataFrame()
                try:
                    puts = options_data.xs("puts", level="optionType")
                except Exception:
                    puts = pd.DataFrame()
                
                if not calls.empty and not puts.empty:
                    total_put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
                    total_call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
                    
                    if total_call_oi > 0:
                        result += "\n🔹 Derivs (Options):\n"
                        
                        # Put/Call Open Interest Ratio
                        pc_ratio = total_put_oi / total_call_oi
                        result += f"  • P/C Ratio (OI): {pc_ratio:.2f}"
                        if pc_ratio > 1.0:
                            result += " ⚠️ Bearish"
                        elif pc_ratio < 0.7:
                            result += " 📈 Bullish"
                        else:
                            result += " 📊 Neutral"
                        result += "\n"
                        
                        # Put/Call Volume Ratio
                        total_put_vol = puts['volume'].sum() if 'volume' in puts.columns else 0
                        total_call_vol = calls['volume'].sum() if 'volume' in calls.columns else 0
                        if total_call_vol > 0:
                            pc_vol_ratio = total_put_vol / total_call_vol
                            result += f"  • P/C Ratio (Vol): {pc_vol_ratio:.2f}\n"
                        
                        # Total Open Interest
                        result += f"  • Total Call OI: {int(total_call_oi):,}\n"
                        result += f"  • Total Put OI: {int(total_put_oi):,}\n"
        except Exception:
            pass
        
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
def search_brave_sentiment(query: str) -> str:
    """
    Search for recent news and sentiment using Brave Search API.
    Useful for finding recent market sentiment, breaking news, and analyst opinions.
    """
    if not brave:
        return "❌ Brave API key not configured"
    
    try:
        # Clean up query - handle ticker symbols
        search_query = query.strip().lstrip('$').upper()
        
        # Build search URL for Brave Web Search API
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": brave
        }
        params = {
            "q": f"{search_query} stock news sentiment analysis",
            "count": 8,  # Get more results for better sentiment analysis
            "freshness": "pw",  # Past week for recent news
            "text_decorations": False,
            "search_lang": "en"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"Brave Search API error: {response.status_code}")
            return f"❌ Brave Search returned error {response.status_code}"
        
        data = response.json()
        web_results = data.get("web", {}).get("results", [])
        
        if not web_results:
            return f"No recent news found for {search_query} via Brave Search"
        
        # Sentiment analysis keywords
        positive_words = ['bullish', 'growth', 'beat', 'surge', 'rally', 'gain', 'upgrade', 
                         'strong', 'profit', 'outperform', 'buy', 'positive', 'record', 'success']
        negative_words = ['bearish', 'decline', 'miss', 'drop', 'crash', 'loss', 'downgrade',
                         'weak', 'concern', 'underperform', 'sell', 'negative', 'warning', 'risk']
        
        formatted_results = []
        overall_pos = 0
        overall_neg = 0
        
        for result in web_results[:5]:  # Top 5 for display
            title = result.get("title", "")
            description = result.get("description", "")
            url = result.get("url", "")
            
            # Combine text for sentiment analysis
            text = f"{title} {description}".lower()
            
            pos_count = sum(1 for w in positive_words if w in text)
            neg_count = sum(1 for w in negative_words if w in text)
            overall_pos += pos_count
            overall_neg += neg_count
            
            if pos_count > neg_count:
                sentiment = "🟢 Positive"
            elif neg_count > pos_count:
                sentiment = "🔴 Negative"
            else:
                sentiment = "⚪ Neutral"
            
            # Truncate description if too long
            short_desc = description[:150] + "..." if len(description) > 150 else description
            formatted_results.append(f"• {title} [{sentiment}]\n  {short_desc}")
        
        # Overall sentiment summary
        if overall_pos > overall_neg * 1.5:
            overall_sentiment = "📈 **Overall Sentiment: Bullish**"
        elif overall_neg > overall_pos * 1.5:
            overall_sentiment = "📉 **Overall Sentiment: Bearish**"
        else:
            overall_sentiment = "📊 **Overall Sentiment: Mixed/Neutral**"
        
        result = f"🔍 Brave Search Sentiment for {search_query}:\n\n"
        result += f"{overall_sentiment}\n"
        result += f"Positive signals: {overall_pos} | Negative signals: {overall_neg}\n\n"
        result += "\n".join(formatted_results)
        
        return result
        
    except requests.exceptions.Timeout:
        return f"⚠️ Brave Search timed out for {query}"
    except Exception as e:
        print(f"Brave Search error: {e}")
        return f"❌ Error searching Brave: {str(e)}"


@tool
def get_economic_indicators(indicator_type: str) -> str:
    """Fetches key economic indicators from FRED API. Options: 'general', 'inflation', 'employment', 'rates', 'gdp'"""
    from datetime import datetime, timedelta
    
    if not fred_api_key:
        return "❌ FRED_API_KEY not set"
    fred = Fred(api_key=fred_api_key)
    
    # Only fetch data from the last 2 years for efficiency
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

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
                y10 = fred.get_series('GS10', observation_start=start_date)
                y2 = fred.get_series('GS2', observation_start=start_date)
                if y10.empty or y2.empty:
                    lines.append(f"• {name}: no data")
                else:
                    val = y10.iloc[-1] - y2.iloc[-1]
                    date = y10.index[-1].strftime('%Y-%m-%d')
                    lines.append(f"• {name}: {val:.2f}% (as of {date})")
                continue

            data = fred.get_series(sid, observation_start=start_date)
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
    from datetime import datetime, timedelta
    
    if not fred_api_key:
        return "❌ FRED_API_KEY not set"
    fred = Fred(api_key=fred_api_key)
    
    # Only fetch recent data (last 1 year)
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
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
            data = fred.get_series(sid, observation_start=start_date)
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


@tool
def get_market_risk(country: str = "") -> str:
    """
    Fetches current market risk premium data from Financial Modeling Prep.
    The market risk premium is the expected return from the market above the risk-free rate.
    Essential for CAPM calculations, DCF valuation, and understanding equity risk pricing.

    Args:
        country: Optional country name to get specific country data (e.g., "Japan", "Brazil", "India")

    Returns:
        Formatted market risk premium data including country-specific premiums
    """
    if not fmp:
        return "❌ FMP API key not configured"

    try:
        url = f"https://financialmodelingprep.com/api/v4/market_risk_premium?apikey={fmp}"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return f"❌ FMP API error: {response.status_code}"

        data = response.json()

        if not data or not isinstance(data, list):
            return "❌ No market risk premium data available"

        out = ["📊 Market Risk Premium Data:"]
        out.append("")

        # If specific country requested, show it first
        if country:
            country_data = next((item for item in data if item.get('country', '').lower() == country.lower()), None)
            if country_data:
                out.append(f"🎯 {country_data.get('country')} (Requested):")
                total_premium = country_data.get('totalEquityRiskPremium', 0)
                country_risk = country_data.get('countryRiskPremium', 0)
                continent = country_data.get('continent', 'N/A')

                out.append(f"  • Total Equity Risk Premium: {total_premium:.2f}%")
                out.append(f"  • Country Risk Premium: {country_risk:.2f}%")
                out.append(f"  • Continent: {continent}")
                out.append(f"  • Market Risk Premium (base): {total_premium - country_risk:.2f}%")
                out.append("")

        # Show major markets
        major_countries = ['United States', 'United Kingdom', 'Germany', 'Japan', 'China', 'France', 'Canada', 'India', 'Brazil']
        major_markets = [item for item in data if item.get('country') in major_countries]

        if major_markets:
            out.append("🌍 Major Markets:")
            for item in major_markets:
                country_name = item.get('country', 'Unknown')
                total_premium = item.get('totalEquityRiskPremium', 0)
                country_risk = item.get('countryRiskPremium', 0)

                out.append(f"  • {country_name}: {total_premium:.2f}%")
                if country_risk > 0:
                    out.append(f"    └─ Country Risk: +{country_risk:.2f}%")

        # Calculate global statistics
        out.append("")
        out.append("📈 Global Statistics:")

        total_premiums = [item['totalEquityRiskPremium'] for item in data if 'totalEquityRiskPremium' in item]

        if total_premiums:
            avg_premium = sum(total_premiums) / len(total_premiums)
            max_premium = max(total_premiums)
            min_premium = min(total_premiums)

            # Find countries with max/min
            max_country = next((item['country'] for item in data if item.get('totalEquityRiskPremium') == max_premium), '')
            min_country = next((item['country'] for item in data if item.get('totalEquityRiskPremium') == min_premium), '')

            out.append(f"  • Average Global Premium: {avg_premium:.2f}%")
            out.append(f"  • Highest: {max_premium:.2f}% ({max_country})")
            out.append(f"  • Lowest: {min_premium:.2f}% ({min_country})")
            out.append(f"  • Countries Covered: {len(data)}")

        # Highlight US market risk premium for CAPM
        us_data = next((item for item in data if item.get('country') == 'United States'), None)
        if us_data:
            us_premium = us_data.get('totalEquityRiskPremium', 0)
            out.append("")
            out.append("💼 For US CAPM Calculations:")
            out.append(f"  • US Market Risk Premium: {us_premium:.2f}%")
            out.append(f"  • Formula: Expected Return = Risk-Free Rate + Beta × {us_premium:.2f}%")

        out.append("")
        out.append("💡 Applications:")
        out.append("  • Cost of Equity: Rf + Beta × Market Risk Premium")
        out.append("  • WACC Calculation: Weighted average cost of capital")
        out.append("  • International Investments: Country risk adjustments")

        return "\n".join(out)

    except requests.Timeout:
        return "❌ FMP API request timed out"
    except requests.RequestException as e:
        return f"❌ FMP API request failed: {str(e)}"
    except json.JSONDecodeError:
        return "❌ Failed to parse FMP API response"
    except Exception as e:
        return f"❌ Error fetching market risk premium: {str(e)}"


@tool
def get_commodity_prices() -> str:
    """
    Fetch current commodity prices including gold, silver, copper, and other metals.
    Essential for analyzing mining companies like Newmont, Barrick Gold, Freeport-McMoRan, etc.
    Uses Yahoo Finance commodity futures data for real-time pricing.
    
    Returns:
        Formatted commodity prices with % changes and market context
    """
    try:
        # Commodity futures symbols on Yahoo Finance
        commodities = {
            'Gold (Troy Oz)': 'GC=F',
            'Silver (Troy Oz)': 'SI=F', 
            'Copper (Lb)': 'HG=F',
            'Platinum (Troy Oz)': 'PL=F',
            'Palladium (Troy Oz)': 'PA=F',
            'WTI Crude Oil': 'CL=F',
            'Natural Gas': 'NG=F'
        }
        
        result = ["🏗️ Current Commodity Prices:\n"]
        
        for commodity_name, symbol in commodities.items():
            try:
                stock = yq.Ticker(symbol)
                
                # Get price data
                hist = stock.history(period="5d", interval="1d")
                if hist.empty:
                    result.append(f"  • {commodity_name}: No data available")
                    continue
                
                # Get latest price and calculate change
                latest_price = None
                price_change = 0.0
                percent_change = 0.0
                
                if not hist.empty:
                    latest_price = hist['close'].iloc[-1] if 'close' in hist.columns else None
                    if len(hist) >= 2:
                        prev_price = hist['close'].iloc[-2]
                        price_change = latest_price - prev_price
                        percent_change = (price_change / prev_price) * 100 if prev_price != 0 else 0.0
                
                # Format output
                if latest_price is not None:
                    # Special formatting for different commodities
                    if 'Gold' in commodity_name or 'Silver' in commodity_name or 'Platinum' in commodity_name or 'Palladium' in commodity_name:
                        price_str = f"${latest_price:,.2f}"
                    elif 'Copper' in commodity_name:
                        price_str = f"${latest_price:.4f}"
                    elif 'Oil' in commodity_name or 'Gas' in commodity_name:
                        price_str = f"${latest_price:.2f}"
                    else:
                        price_str = f"${latest_price:.2f}"
                    
                    # Add trend indicator
                    trend = "📈" if percent_change > 1.0 else "📉" if percent_change < -1.0 else "➡️"
                    change_str = f"{price_change:+.2f} ({percent_change:+.2f}%)"
                    
                    result.append(f"  • {commodity_name}: {price_str} {change_str} {trend}")
                else:
                    result.append(f"  • {commodity_name}: Price unavailable")
                
            except Exception as e:
                result.append(f"  • {commodity_name}: Error fetching price ({str(e)[:50]})")
        
        # Add market context for gold specifically
        try:
            gold_stock = yq.Ticker('GC=F')
            gold_hist = gold_stock.history(period="1y", interval="1d")
            if not gold_hist.empty:
                gold_current = gold_hist['close'].iloc[-1]
                gold_52w_high = gold_hist['high'].max()
                gold_52w_low = gold_hist['low'].min()
                
                result.append(f"\n📊 Gold Market Context:")
                result.append(f"  • 52-Week Range: ${gold_52w_low:,.2f} - ${gold_52w_high:,.2f}")
                
                # Calculate position in range
                range_position = ((gold_current - gold_52w_low) / (gold_52w_high - gold_52w_low)) * 100
                result.append(f"  • Current Position in Range: {range_position:.1f}%")
                
                if range_position > 80:
                    result.append("  • Status: Near 52-week high ✅")
                elif range_position < 20:
                    result.append("  • Status: Near 52-week low ⚠️")
                else:
                    result.append("  • Status: Mid-range")
        except:
            pass  # Skip context if unavailable
            
        result.append(f"\n💡 Note: Prices are real-time futures contracts. Mining companies' performance correlates strongly with these commodity prices.")
        
        return "\n".join(result)
        
    except Exception as e:
        return f"❌ Error fetching commodity prices: {str(e)}"


# ========== PRIORITY 3: WEB SEARCH TOOLS ==========

@tool
def search_recent_macro_news(query: str) -> str:
    """
    Search for recent macroeconomic news and events from the last 48 hours.
    Use this when FRED/Fed data might be outdated or for breaking news.
    
    Args:
        query: Search query (e.g., "Federal Reserve interest rates", "inflation CPI")
    
    Returns:
        Formatted news results with titles, snippets, and URLs
    """
    try:
        # Search financial news sources
        search_query = f"{query} site:bloomberg.com OR site:reuters.com OR site:wsj.com"
        
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=5))
        
        if not results:
            return f"No recent news found for '{query}'"
        
        formatted = [f"📰 Recent News for '{query}':"]
        for r in results:
            title = r.get('title', 'No title')
            body = r.get('body', 'No description')[:200]
            link = r.get('href', '')
            formatted.append(f"\n• {title}\n  {body}...\n  🔗 {link}")
        
        return "\n".join(formatted)
        
    except Exception as e:
        return f"Error fetching news: {str(e)}"


@tool
def get_fed_speeches() -> str:
    """
    Get latest Federal Reserve speeches and statements from the last 7 days.
    Use this for the most recent Fed policy communications.
    
    Returns:
        Formatted list of recent Fed speeches with dates and titles
    """
    try:
        url = "https://www.federalreserve.gov/newsevents/speeches.htm"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        speeches = ["🎤 Recent Federal Reserve Speeches:"]
        
        # Parse the Fed's speeches page - look for event list items
        items = soup.select('.row.eventlist')[:5]  # Last 5 speeches
        
        if not items:
            # Fallback: try alternative selectors
            items = soup.select('.news__item')[:5]
        
        for item in items:
            try:
                date_elem = item.select_one('.news__date, .eventlist__date')
                title_elem = item.select_one('.news__headline a, .eventlist__title a')
                
                if date_elem and title_elem:
                    date = date_elem.get_text(strip=True)
                    title = title_elem.get_text(strip=True)
                    link = title_elem.get('href', '')
                    if link and not link.startswith('http'):
                        link = f"https://www.federalreserve.gov{link}"
                    
                    speeches.append(f"\n• {date}: {title}\n  🔗 {link}")
            except Exception:
                continue
        
        if len(speeches) == 1:
            return "No recent Fed speeches found. Try using search_recent_macro_news for Fed news."
        
        return "\n".join(speeches)
        
    except Exception as e:
        return f"Error fetching Fed speeches: {str(e)}. Try using search_recent_macro_news instead."


@tool
def get_latest_fomc_statement() -> str:
    """
    Get the most recent FOMC meeting statement.
    Use this for official Federal Reserve policy decisions.
    
    Returns:
        Latest FOMC statement date and key policy summary
    """
    try:
        url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        output = ["📋 FOMC Information:"]
        
        # Find FOMC calendar panels
        panels = soup.select('.panel.panel-default, .fomc-meeting')[:3]
        
        for panel in panels:
            try:
                date_elem = panel.select_one('.panel-heading, .fomc-meeting__date')
                statement_link = panel.select_one('a[href*="statement"]')
                
                if date_elem:
                    date = date_elem.get_text(strip=True)[:50]
                    output.append(f"\n• Meeting: {date}")
                    
                    if statement_link:
                        link = statement_link.get('href', '')
                        if link and not link.startswith('http'):
                            link = f"https://www.federalreserve.gov{link}"
                        output.append(f"  Statement: {link}")
            except Exception:
                continue
        
        if len(output) == 1:
            output.append("Unable to parse FOMC calendar. Visit https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error fetching FOMC statement: {str(e)}"


# ========== AGENT SETUP ==========

STOCK_SYSTEM_PROMPT = """You are an expert stock analyst with deep knowledge of financial markets and technical analysis.
You have access to comprehensive, validated data sources that match TradingView chart data for consistency.

When analyzing stocks, you MUST:
1. Fetch comprehensive financial metrics using get_financial_metrics (includes valuation, profitability, growth, financial health)
2. Get detailed technical indicators using get_technical_indicators (includes RSI, MACD, Bollinger Bands, moving averages, volume, support/resistance)
3. Check recent news and sentiment using get_market_news (Yahoo, Finnhub, Polygon)
4. Search for additional sentiment using search_brave_sentiment (web-wide sentiment from Brave Search)

IMPORTANT DATA ACCURACY NOTES:
- Technical indicators use the same data source as TradingView charts (Yahoo Finance, 1-year daily data)
- All metrics are validated and include interpretation flags (✅ ⚠️ ❌)
- Price data is consistent between technical analysis and chart visualization
- Fundamental metrics include comprehensive ratios with industry context
- Web sentiment provides additional market context from recent articles and discussions

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
   - Recent news impact from traditional sources
   - Web-wide sentiment from Brave Search
   - Overall market mood (bullish/bearish/neutral)
   
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
4. Search for recent breaking news using search_recent_macro_news
5. Get latest Fed speeches using get_fed_speeches
6. Get FOMC statements using get_latest_fomc_statement

Provide a clear, structured analysis with:
- Current economic conditions
- Policy outlook
- Market implications
- Investment positioning recommendations

Be concise but thorough. Use data to support your analysis."""

# Initialize LLMs for Dual-LLM Architecture
# Mistral: Fast, cost-effective for validation and routing tasks
llm_mistral = ChatMistralAI(
    model="mistral-large-latest",  # or "mistral-medium-latest" for cost savings
    temperature=0.1,
    api_key=mistral_key,
    max_tokens=1024,  # Shorter responses for validation/routing
    max_retries=3,
)

# Groq: Powerful analysis for final synthesis
llm_groq = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    api_key=groqK,
    max_tokens=2048,
    max_retries=3,
)

# Default LLM for backwards compatibility
llm = llm_groq

# OpenRouter: Output validation with MiMo-V2-Flash reasoning model
openrouter_client = None
if openrouter_key:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )
    print("✅ OpenRouter MiMo-V2-Flash validator initialized")
else:
    print("⚠️ OpenRouter API key not set - output validation disabled")


# ========== OUTPUT VALIDATION WITH MiMo-V2-Flash + Flash 3 CLEANUP ==========

# Initialize Google AI client for final cleanup
google_client = None
if gai:
    try:
        google_client = genai.Client(api_key=gai)
        print("✅ Google AI Flash 3 cleanup initialized")
    except Exception as e:
        print(f"⚠️ Google AI initialization failed: {e}")
else:
    print("⚠️ Google AI API key not set - final cleanup disabled")


def _should_retry_google_error(err: Exception) -> bool:
    """Return True when the error looks like a rate limit or transient failure."""
    msg = str(err).lower()
    return any(
        needle in msg
        for needle in (
            "rate limit",
            "resource_exhausted",
            "429",
            "quota",
            "temporarily unavailable",
            "timeout",
            "unavailable",
        )
    )


def _throttle_google_genai() -> None:
    """Enforce a minimum interval between Google GenAI requests."""
    global _last_google_genai_ts
    now = time.monotonic()
    wait_for = GOOGLE_GENAI_MIN_INTERVAL - (now - _last_google_genai_ts)
    if wait_for > 0:
        time.sleep(wait_for)
    _last_google_genai_ts = time.monotonic()


def polish_with_flash3(original_analysis: str, validated_analysis: str, validation_metadata: dict, context: dict) -> str:
    """
    Use Google AI Flash 3 to polish and clean up the final output.
    Combines original analysis with validation improvements for a refined result.

    Args:
        original_analysis: The initial GPT-OSS analysis
        validated_analysis: MiMo-validated/corrected version
        validation_metadata: Quality metrics from MiMo
        context: Original context data

    Returns:
        Polished, professional final analysis
    """
    if not google_client:
        return validated_analysis  # Return validated version if Flash 3 not available

    try:
        print("✨ Polishing output with Google AI Flash 3...")

        cleanup_prompt = f"""You are a professional financial writing editor. Your job is to create a final, polished analysis by combining:

1. **Original Analysis** (GPT-OSS generated)
2. **Validation Feedback** (MiMo-V2-Flash corrections)

**Original Analysis:**
{original_analysis}

**Validation Quality:** {validation_metadata.get('overall_quality', 'good')}
**Issues Found:** {', '.join(validation_metadata.get('issues', [])) if validation_metadata.get('issues') else 'None'}
**Suggestions:** {', '.join(validation_metadata.get('suggestions', [])) if validation_metadata.get('suggestions') else 'None'}

**Validated/Corrected Version:**
{validated_analysis}

**Instructions:**
1. Merge the best parts of both analyses while PRESERVING ALL important content
2. Fix any factual errors or logical inconsistencies identified
3. Apply the validation suggestions
4. Maintain professional financial writing tone
5. Keep the structure clear and scannable
6. Remove redundancy and improve clarity while MAINTAINING comprehensive coverage
7. Ensure actionable insights are prominent
8. DO NOT shorten or truncate - preserve the comprehensive nature of the analysis

**CRITICAL: The final output should be as comprehensive as the input analyses. Do not condense or summarize - polish and refine while maintaining full detail.**

**Output a clean, polished final analysis that a professional would deliver to clients.**
Do not include meta-commentary about the editing process. Just deliver the final analysis.
"""

        polished = None
        for attempt in range(1, GOOGLE_GENAI_MAX_RETRIES + 1):
            _throttle_google_genai()
            try:
                response = google_client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=cleanup_prompt
                )
                polished = response.text
                break
            except Exception as e:
                if attempt >= GOOGLE_GENAI_MAX_RETRIES or not _should_retry_google_error(e):
                    raise
                backoff = GOOGLE_GENAI_BACKOFF_BASE ** attempt
                jitter = random.uniform(0.0, 0.5)
                time.sleep(backoff + jitter)

        if polished and len(polished) > 100:
            return polished
        else:
            # Fallback if Flash 3 returns insufficient content
            return validated_analysis

    except Exception as e:
        print(f"⚠️ Flash 3 cleanup error: {e}")
        return validated_analysis  # Fallback to validated version


def validate_analysis_output(
    analysis: str,
    context: dict,
    analysis_type: str = "stock"
) -> dict:
    """
    4-LLM Quality Pipeline:
    1. Original analysis (GPT-OSS)
    2. Validation with reasoning (MiMo-V2-Flash)
    3. Corrections/enhancements (MiMo second pass)
    4. Final polish (Google AI Flash 3)

    This provides comprehensive quality control:
    - Checks for logical consistency
    - Verifies claims against provided data
    - Identifies potential errors or oversights
    - Suggests improvements
    - Delivers polished, professional output

    Args:
        analysis: The generated analysis text from GPT-OSS
        context: Dictionary with original data (ticker, metrics, etc.)
        analysis_type: "stock" or "macro"

    Returns:
        dict with validation results, raw versions, and final polished analysis
    """
    # Store original for comparison
    original_analysis = analysis

    if not openrouter_client:
        return {
            "validated": True,
            "confidence": 1.0,
            "issues": [],
            "reasoning": "OpenRouter not configured - skipping validation",
            "original_analysis": analysis,
            "validated_analysis": analysis,
            "final_analysis": analysis,
            "enhanced_analysis": analysis  # backwards compatibility
        }

    try:
        # Create validation prompt
        if analysis_type == "stock":
            ticker = context.get("ticker", "Unknown")
            validation_prompt = f"""You are a financial analysis validator. Review this stock analysis for {ticker} and check for:

1. **Logical Consistency**: Do the conclusions follow from the data?
2. **Factual Accuracy**: Are metrics interpreted correctly?
3. **Completeness**: Are important aspects missing?
4. **Clarity**: Is the analysis clear and actionable?

ANALYSIS TO VALIDATE:
{analysis}

CONTEXT DATA:
- Ticker: {ticker}
- Display Name: {context.get('displayName', 'N/A')}
- User Query: {context.get('userInput', 'N/A')}

Provide your validation in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0.0 to 1.0,
    "issues": ["list of issues found"],
    "strengths": ["what the analysis does well"],
    "suggestions": ["improvements to make"],
    "overall_quality": "excellent/good/acceptable/poor"
}}"""
        else:  # macro
            topic = context.get("topic", "Unknown")
            validation_prompt = f"""You are a macroeconomic analysis validator. Review this macro analysis on "{topic}" and check for:

1. **Economic Logic**: Do the conclusions follow sound economic reasoning?
2. **Data Interpretation**: Are indicators interpreted correctly?
3. **Policy Understanding**: Is Fed/policy analysis accurate?
4. **Market Implications**: Are investment recommendations reasonable?

ANALYSIS TO VALIDATE:
{analysis}

CONTEXT:
- Topic: {topic}

Provide your validation in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0.0 to 1.0,
    "issues": ["list of issues found"],
    "strengths": ["what the analysis does well"],
    "suggestions": ["improvements to make"],
    "overall_quality": "excellent/good/acceptable/poor"
}}"""

        # First API call with reasoning enabled
        print("🔍 Validating output with MiMo-V2-Flash reasoning...")
        response = openrouter_client.chat.completions.create(
            model="xiaomi/mimo-v2-flash:free",
            messages=[{"role": "user", "content": validation_prompt}],
            extra_body={"reasoning": {"enabled": True}}
        )

        assistant_message = response.choices[0].message
        validation_content = assistant_message.content

        # Parse the validation result
        try:
            # Extract JSON from response
            if "```json" in validation_content:
                json_start = validation_content.find("```json") + 7
                json_end = validation_content.find("```", json_start)
                validation_content = validation_content[json_start:json_end].strip()
            elif "```" in validation_content:
                json_start = validation_content.find("```") + 3
                json_end = validation_content.find("```", json_start)
                validation_content = validation_content[json_start:json_end].strip()

            validation_result = json.loads(validation_content)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            validation_result = {
                "is_valid": True,
                "confidence": 0.8,
                "issues": [],
                "strengths": ["Analysis completed"],
                "suggestions": [],
                "overall_quality": "good"
            }

        # If there are issues, do a second reasoning pass for enhancement suggestions
        enhanced_analysis = analysis
        if validation_result.get("issues") and len(validation_result["issues"]) > 0:
            print("⚠️ Validation found issues - requesting enhancement suggestions...")

            messages = [
                {"role": "user", "content": validation_prompt},
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "reasoning_details": assistant_message.reasoning_details
                },
                {
                    "role": "user",
                    "content": f"Based on these issues: {validation_result['issues']}, how should the analysis be corrected? Be specific."
                }
            ]

            response2 = openrouter_client.chat.completions.create(
                model="xiaomi/mimo-v2-flash:free",
                messages=messages,
                extra_body={"reasoning": {"enabled": True}}
            )

            enhancement_suggestions = response2.choices[0].message.content

            # Add enhancement note to analysis
            enhanced_analysis = f"{analysis}\n\n---\n**🔍 Validation Notes:**\n{enhancement_suggestions}"

        # STAGE 4: Polish with Google AI Flash 3
        validation_metadata = {
            "is_valid": validation_result.get("is_valid", True),
            "confidence": validation_result.get("confidence", 0.9),
            "issues": validation_result.get("issues", []),
            "strengths": validation_result.get("strengths", []),
            "suggestions": validation_result.get("suggestions", []),
            "overall_quality": validation_result.get("overall_quality", "good"),
        }

        final_analysis = polish_with_flash3(
            original_analysis=original_analysis,
            validated_analysis=enhanced_analysis,
            validation_metadata=validation_metadata,
            context=context
        )

        return {
            "validated": validation_result.get("is_valid", True),
            "confidence": validation_result.get("confidence", 0.9),
            "issues": validation_result.get("issues", []),
            "strengths": validation_result.get("strengths", []),
            "suggestions": validation_result.get("suggestions", []),
            "overall_quality": validation_result.get("overall_quality", "good"),
            "reasoning": assistant_message.reasoning_details if hasattr(assistant_message, 'reasoning_details') else None,
            # All 4 versions for transparency
            "original_analysis": original_analysis,  # Stage 1: GPT-OSS raw output
            "validated_analysis": enhanced_analysis,  # Stage 2-3: MiMo corrected
            "final_analysis": final_analysis,  # Stage 4: Flash 3 polished
            "enhanced_analysis": final_analysis  # backwards compatibility - use final polished version
        }

    except Exception as e:
        print(f"⚠️ Output validation error: {e}")
        return {
            "validated": True,
            "confidence": 0.7,
            "issues": [],
            "reasoning": f"Validation failed: {str(e)}",
            "original_analysis": analysis,
            "validated_analysis": analysis,
            "final_analysis": analysis,
            "enhanced_analysis": analysis
        }


# ========== MISTRAL SENTIMENT ANALYSIS HELPER ==========

def analyze_sentiment_with_mistral(text: str, context: str = "") -> dict:
    """
    Use Mistral AI for fast, accurate sentiment analysis of financial text.

    Args:
        text: The text to analyze (news headline, article, etc.)
        context: Optional context (e.g., ticker symbol, topic)

    Returns:
        dict with sentiment, confidence, and reasoning
    """
    if not text or not mistral_key:
        return {
            "sentiment": "neutral",
            "confidence": 0.5,
            "reasoning": "No text or Mistral API key not configured"
        }

    try:
        prompt = f"""Analyze the sentiment of this financial text.

Context: {context if context else 'Financial markets'}
Text: {text}

Respond ONLY with valid JSON (no markdown):
{{
    "sentiment": "bullish" or "bearish" or "neutral",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}"""

        response = llm_mistral.invoke(prompt)
        content = response.content.strip()

        # Clean up response
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        return result
    except Exception as e:
        # Fallback to simple keyword-based sentiment
        text_lower = text.lower()
        positive = sum(1 for w in ['bullish', 'growth', 'beat', 'surge', 'gain', 'profit'] if w in text_lower)
        negative = sum(1 for w in ['bearish', 'decline', 'miss', 'drop', 'loss', 'concern'] if w in text_lower)

        if positive > negative:
            sentiment = "bullish"
        elif negative > positive:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "confidence": 0.6,
            "reasoning": f"Mistral analysis failed: {str(e)}. Using keyword fallback."
        }


# ========== PRIORITY 1: QUESTION VALIDATION ==========

def validate_macro_question(question: str) -> dict:
    """
    Validates if the question is appropriate for macro analysis.
    
    Returns:
        {
            "is_valid": bool,
            "question_type": str,  # "macro_policy", "economic_data", "market_outlook", etc.
            "required_data": list, # ["fed_policy", "inflation", "yields"]
            "reasoning": str
        }
    """
    # Handle common typos like "Marco" instead of "Macro"
    clean_question = question.replace("Marco", "Macro").replace("marco", "macro")
    
    validator_prompt = f"""You are a macro economics question validator.

Analyze if this question can be answered with macro economic data:
- Central bank policy (Federal Reserve, ECB, etc.)
- Economic indicators (inflation, employment, GDP, PMI)
- Bond yields, interest rates, and currency markets
- General economic outlook for countries or regions (e.g., US, China, Venezuela, Thailand)
- Global trade and geopolitical economic impacts

Note: If the user asks for a "Marco outlook", they mean "Macro outlook".

Respond ONLY with valid JSON (no markdown, no code blocks):
{{
    "is_valid": true or false,
    "question_type": "macro_policy" or "economic_data" or "market_outlook" or "off_topic",
    "required_data": ["fed_policy", "inflation", "yields", "employment", "gdp", "news", "global_macro"],
    "reasoning": "brief explanation"
}}

Question: {clean_question}"""

    try:
        # Use Mistral for fast, cost-effective validation
        print("🤖 Using Mistral AI for question validation...")
        response = llm_mistral.invoke(validator_prompt)
        content = response.content.strip()
        
        # Try to extract JSON from response
        if content.startswith("```"):
            # Remove markdown code blocks if present
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        result = json.loads(content)
        return result
    except Exception as e:
        # Default to valid if parsing fails - allow agent to try
        return {
            "is_valid": True,
            "question_type": "general",
            "required_data": ["general"],
            "reasoning": f"Validation parsing failed: {str(e)}. Proceeding with analysis."
        }


# ========== PRIORITY 2: LLM-BASED TOOL SELECTION ==========

# Tool descriptions for intelligent selection
MACRO_TOOL_DESCRIPTIONS = {
    "get_economic_indicators": "FRED economic data (inflation, employment, GDP, rates). Pass 'general', 'inflation', 'employment', 'rates', or 'gdp' as input.",
    "get_fed_policy_info": "Federal Reserve policy indicators and current stance. No input needed.",
    "get_bond_yields": "US Treasury yield curve and bond data. No input needed.",
    "get_market_risk": "Market risk premium data for CAPM and valuation. Pass country name (e.g., 'Japan', 'Brazil') or empty for global overview.",
    "search_recent_macro_news": "Breaking macro news from last 48 hours. Pass search query as input.",
    "get_fed_speeches": "Recent Fed speeches and communications. No input needed.",
    "get_latest_fomc_statement": "Latest FOMC meeting statement. No input needed."
}

# Map tool names to functions
MACRO_TOOL_MAP = {
    "get_economic_indicators": get_economic_indicators,
    "get_fed_policy_info": get_fed_policy_info,
    "get_bond_yields": get_bond_yields,
    "get_market_risk": get_market_risk,
    "search_recent_macro_news": search_recent_macro_news,
    "get_fed_speeches": get_fed_speeches,
    "get_latest_fomc_statement": get_latest_fomc_statement
}


def select_tools_for_question(question: str, validation: dict) -> list:
    """
    Use LLM to intelligently select which tools are needed for the question.
    Returns list of (tool_name, input_value) tuples.
    """
    tool_prompt = f"""You are a tool selection expert for macro economic analysis.

Given this question, select the MINIMUM set of tools needed to answer it.

Available tools:
{json.dumps(MACRO_TOOL_DESCRIPTIONS, indent=2)}

Question type: {validation.get('question_type', 'general')}
Required data hints: {validation.get('required_data', [])}
Question: {question}

Respond ONLY with a JSON array of objects. Each object has "tool" and "input" keys.
For tools that don't need input, use empty string "".

Example response:
[
    {{"tool": "get_economic_indicators", "input": "inflation"}},
    {{"tool": "get_fed_policy_info", "input": ""}}
]

Be conservative - only select tools that are truly needed. Maximum 4 tools."""

    try:
        # Use Mistral for fast, intelligent tool selection
        print("🤖 Using Mistral AI for intelligent tool selection...")
        response = llm_mistral.invoke(tool_prompt)
        content = response.content.strip()
        
        # Extract JSON from response
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        # Find JSON array in response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            content = content[start:end]
        
        tools = json.loads(content)
        return [(t["tool"], t.get("input", "")) for t in tools if t.get("tool") in MACRO_TOOL_MAP]
    except Exception as e:
        print(f"⚠️ Tool selection failed: {e}")
        # Default fallback: basic tools
        return [
            ("get_economic_indicators", "general"),
            ("get_fed_policy_info", ""),
            ("get_bond_yields", "")
        ]


# Flag for agent availability (used for compatibility with existing code)
MACRO_AGENT_AVAILABLE = True  # Using LLM-based selection instead


def analyze_stock_with_tools(ticker: str, company_name: str | None = None) -> str:
    """Analyze a stock by gathering data from tools and synthesizing with LLM."""
    try:
        # Gather data from tools
        print(f"📊 Fetching financial metrics for {ticker}...")
        metrics = get_financial_metrics.invoke(ticker)
        
        print(f"📈 Fetching technical indicators for {ticker}...")
        technical = get_technical_indicators.invoke(ticker)
        
        print(f"📰 Fetching market news for {ticker}...")
        news = get_market_news.invoke(ticker)
        
        # Fetch Brave Search sentiment for additional context
        print(f"🔍 Searching Brave for sentiment on {ticker}...")
        brave_sentiment = ""
        if brave:  # Only if Brave API key is configured
            try:
                brave_sentiment = search_brave_sentiment.invoke(ticker)
            except Exception as e:
                print(f"⚠️ Brave Search failed: {e}")
                brave_sentiment = "Brave Search unavailable"

        # Fetch market risk premium for valuation context
        print(f"📊 Fetching market risk premium for valuation...")
        market_risk = ""
        if fmp:  # Only if FMP API key is configured
            try:
                market_risk = get_market_risk.invoke("")
            except Exception as e:
                print(f"⚠️ Market risk premium fetch failed: {e}")
                market_risk = "Market risk premium unavailable"

        # Check if this is a mining/commodity company and fetch commodity prices
        commodity_prices = ""
        mining_companies = ['NEM', 'GOLD', 'AUY', 'KGC', 'FCX', 'SCCO', 'AA', 'X', 'CLF', 'VALE', 'RIO', 'BHP', 
                           'NEWMONT', 'BARRICK', 'FREEPORT', 'ALCOA', 'NEWCREST', 'ANGLO', 'GLENCORE']
        company_lower = (company_name or "").lower()
        
        is_mining_company = (ticker.upper() in mining_companies or 
                           any(keyword in company_lower for keyword in ['mining', 'gold', 'silver', 'copper', 'metal', 'mineral']))
        
        if is_mining_company:
            print(f"🏗️ Detected mining/commodity company - fetching commodity prices...")
            try:
                commodity_prices = get_commodity_prices.invoke("")
            except Exception as e:
                print(f"⚠️ Commodity prices fetch failed: {e}")
                commodity_prices = "Commodity prices unavailable"

        # Create a comprehensive prompt for the LLM
        display_name = company_name or TICKER_TO_NAME.get(ticker)
        header = f"{display_name} ({ticker})" if display_name else ticker

        commodity_section = f"\n## Commodity Prices (Key for Mining Operations)\n{commodity_prices}" if commodity_prices else ""

        data_summary = f"""
Here is the data gathered for {header}:

## Financial Metrics
{metrics}

## Technical Indicators
{technical}

## Recent News (Yahoo/Finnhub/Polygon)
{news}

## Web Sentiment (Brave Search)
{brave_sentiment if brave_sentiment else "Not available"}

## Market Risk Premium (for Valuation)
{market_risk if market_risk else "Not available"}{commodity_section}
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
- Overall market sentiment from news and web sources
- Clear risk/reward assessment

Be thorough, data-driven, and actionable."""

        # Use Groq for comprehensive analysis synthesis
        print("🚀 Using Groq OAI/gpt-oss-120b for comprehensive stock analysis synthesis...")
        response = llm_groq.invoke(prompt)
        analysis = response.content

        # Validate output with MiMo-V2-Flash
        validation_context = {
            "ticker": ticker,
            "displayName": display_name,
            "userInput": ticker
        }
        validation_result = validate_analysis_output(analysis, validation_context, "stock")

        # Get final polished analysis
        final_analysis = validation_result.get("final_analysis", analysis)

        # Add validation metadata as a footer if there are suggestions
        if validation_result.get("suggestions") and len(validation_result["suggestions"]) > 0:
            quality_badge = {
                "excellent": "🌟",
                "good": "✅",
                "acceptable": "⚠️",
                "poor": "❌"
            }.get(validation_result.get("overall_quality", "good"), "✅")

            final_analysis += f"\n\n---\n{quality_badge} **Analysis Quality:** {validation_result.get('overall_quality', 'good').title()}"

        # Return dict with all versions for API transparency
        original = validation_result.get("original_analysis") or analysis
        validated = validation_result.get("validated_analysis") or analysis

        print(f"📊 Analysis versions prepared:")
        print(f"   - Final length: {len(final_analysis) if final_analysis else 0}")
        print(f"   - Original length: {len(original) if original else 0}")
        print(f"   - Validated length: {len(validated) if validated else 0}")

        return {
            "analysis": final_analysis,  # Main output (Flash 3 polished)
            "original_analysis": original,  # GPT-OSS raw
            "validated_analysis": validated,  # MiMo corrected
            "validation_metadata": {
                "overall_quality": validation_result.get("overall_quality"),
                "confidence": validation_result.get("confidence"),
                "issues": validation_result.get("issues", []),
                "suggestions": validation_result.get("suggestions", []),
                "strengths": validation_result.get("strengths", [])
            }
        }

    except Exception as e:
        return f"Error analyzing {ticker}: {str(e)}"


def analyze_macro_with_tools(topic: str) -> str:
    """Analyze macro conditions using ReAct agent with validation."""
    try:
        # Handle common typos like "Marco" instead of "Macro"
        clean_topic = topic.replace("Marco", "Macro").replace("marco", "macro")
        
        # ========== PRIORITY 1: Validate question first ==========
        print(f"🔍 Validating question: {clean_topic}")
        validation = validate_macro_question(clean_topic)
        
        if not validation.get("is_valid", True):
            return f"""⚠️ **Off-Topic Question Detected**

This question doesn't appear to be about macroeconomics or markets.

**Reasoning:** {validation.get('reasoning', 'Question is not related to macro analysis.')}

**What I can help with:**
- Federal Reserve policy and interest rates
- Economic indicators (inflation, employment, GDP)
- Bond yields and yield curve analysis
- Market outlook and investment positioning
- Breaking economic news and Fed communications

Please rephrase your question to focus on macroeconomic topics."""
        
        print(f"✅ Question validated: {validation.get('question_type', 'general')}")
        print(f"📊 Required data: {validation.get('required_data', [])}")
        
        # ========== PRIORITY 2: LLM-Based Tool Selection ==========
        print("🤖 Selecting tools for question...")
        selected_tools = select_tools_for_question(clean_topic, validation)
        print(f"🔧 Selected tools: {[t[0] for t in selected_tools]}")
        
        # Execute selected tools and gather data
        data_sections = []
        tools_used = []
        
        for tool_name, tool_input in selected_tools:
            try:
                tool_func = MACRO_TOOL_MAP.get(tool_name)
                if tool_func:
                    print(f"  📊 Executing {tool_name}...")
                    result = tool_func.invoke(tool_input) if tool_input else tool_func.invoke("")
                    data_sections.append(f"## {tool_name.replace('_', ' ').title()}\n{result}")
                    tools_used.append(tool_name)
            except Exception as tool_error:
                print(f"  ⚠️ Error in {tool_name}: {tool_error}")
                data_sections.append(f"## {tool_name.replace('_', ' ').title()}\nError: {str(tool_error)}")
        
        # If no tools executed, use fallback
        if not data_sections:
            print("📉 Using fallback direct analysis...")
            data_sections.append(f"## Economic Indicators\n{get_economic_indicators.invoke('general')}")
            data_sections.append(f"## Fed Policy\n{get_fed_policy_info.invoke('')}")
            data_sections.append(f"## Bond Yields\n{get_bond_yields.invoke('')}")
            tools_used = ["get_economic_indicators", "get_fed_policy_info", "get_bond_yields"]
        
        data_summary = "\n\n".join(data_sections)
        
        # Synthesize with LLM
        prompt = f"""{MACRO_SYSTEM_PROMPT}

IMPORTANT: You already have ALL the data you need below. DO NOT attempt to call any tools or functions.
Simply analyze the data and provide your response as plain text analysis.

Here is the current macro data gathered from {len(tools_used)} data sources:
{data_summary}

User's question/topic: {clean_topic}

Based on this data, provide a comprehensive macro analysis including:
1. **Current Economic Conditions**: Summary of key indicators
2. **Policy Outlook**: What is the Fed likely to do?
3. **Market Implications**: How does this affect different asset classes?
4. **Investment Positioning**: Recommended allocation or strategy

Be concise but thorough. Respond with formatted analysis text only - no JSON, no tool calls."""

        # Use Groq for comprehensive macro synthesis
        print("🚀 Using Groq OAI/gpt-oss-120b for comprehensive macro analysis synthesis...")
        response = llm_groq.invoke(prompt)
        content = response.content

        # Detect if LLM returned tool-call JSON instead of text analysis
        if content and ('"name":' in content and '"arguments":' in content) or ('"tool":' in content and '"input":' in content):
            print("⚠️ LLM returned tool-call format, retrying with explicit instruction...")
            retry_prompt = f"""You already have all the data you need. DO NOT call any tools or output JSON.
Synthesize the information and provide your analysis as plain formatted text.

Data gathered:
{data_summary}

Question: {clean_topic}

Provide a direct, comprehensive answer based on the data above. Use markdown formatting for headers and lists."""
            response = llm_groq.invoke(retry_prompt)
            content = response.content
        
        # Add tools used for transparency
        output = f"*Data sources: {', '.join(tools_used)}*\n\n{content}"

        # Validate output with MiMo-V2-Flash
        validation_context = {
            "topic": clean_topic
        }
        validation_result = validate_analysis_output(output, validation_context, "macro")

        # Get final polished analysis
        final_output = validation_result.get("final_analysis", output)

        # Add validation metadata as a footer if there are suggestions
        if validation_result.get("suggestions") and len(validation_result["suggestions"]) > 0:
            quality_badge = {
                "excellent": "🌟",
                "good": "✅",
                "acceptable": "⚠️",
                "poor": "❌"
            }.get(validation_result.get("overall_quality", "good"), "✅")

            final_output += f"\n\n---\n{quality_badge} **Analysis Quality:** {validation_result.get('overall_quality', 'good').title()}"

        # Return dict with all versions for API transparency
        original = validation_result.get("original_analysis") or output
        validated = validation_result.get("validated_analysis") or output

        print(f"📊 Macro analysis versions prepared:")
        print(f"   - Final length: {len(final_output) if final_output else 0}")
        print(f"   - Original length: {len(original) if original else 0}")
        print(f"   - Validated length: {len(validated) if validated else 0}")

        return {
            "analysis": final_output,  # Main output (Flash 3 polished)
            "original_analysis": original,  # GPT-OSS raw
            "validated_analysis": validated,  # MiMo corrected
            "validation_metadata": {
                "overall_quality": validation_result.get("overall_quality"),
                "confidence": validation_result.get("confidence"),
                "issues": validation_result.get("issues", []),
                "suggestions": validation_result.get("suggestions", []),
                "strengths": validation_result.get("strengths", [])
            }
        }

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

        # Run the stock analysis with tools (returns dict with all versions)
        result = analyze_stock_with_tools(resolved_ticker, company_name)

        # Handle both dict (new format) and string (fallback) responses
        if isinstance(result, dict):
            response_data = {
                'ticker': resolved_ticker,
                'displayName': company_name,
                'userInput': user_query,
                'analysis': result.get('analysis'),  # Flash 3 polished final version
                'originalAnalysis': result.get('original_analysis'),  # GPT-OSS raw
                'validatedAnalysis': result.get('validated_analysis'),  # MiMo corrected
                'validationMetadata': result.get('validation_metadata', {}),
                'timestamp': datetime.now().isoformat()
            }
            print(f"📤 API Response - Stock Analysis:")
            print(f"   - analysis: {len(response_data['analysis']) if response_data['analysis'] else 0} chars")
            print(f"   - originalAnalysis: {len(response_data['originalAnalysis']) if response_data['originalAnalysis'] else 0} chars")
            print(f"   - validatedAnalysis: {len(response_data['validatedAnalysis']) if response_data['validatedAnalysis'] else 0} chars")
            return jsonify(response_data)
        else:
            # Fallback for string responses (error cases)
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

        # Run the macro analysis with tools (returns dict with all versions)
        result = analyze_macro_with_tools(topic)

        # Handle both dict (new format) and string (fallback) responses
        if isinstance(result, dict):
            response_data = {
                'topic': topic,
                'analysis': result.get('analysis'),  # Flash 3 polished final version
                'originalAnalysis': result.get('original_analysis'),  # GPT-OSS raw
                'validatedAnalysis': result.get('validated_analysis'),  # MiMo corrected
                'validationMetadata': result.get('validation_metadata', {}),
                'timestamp': datetime.now().isoformat()
            }
            print(f"📤 API Response - Macro Analysis:")
            print(f"   - analysis: {len(response_data['analysis']) if response_data['analysis'] else 0} chars")
            print(f"   - originalAnalysis: {len(response_data['originalAnalysis']) if response_data['originalAnalysis'] else 0} chars")
            print(f"   - validatedAnalysis: {len(response_data['validatedAnalysis']) if response_data['validatedAnalysis'] else 0} chars")
            return jsonify(response_data)
        else:
            # Fallback for string responses (error cases)
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


@app.route('/api/calculate-return', methods=['POST'])
def calculate_return():
    """Calculate total return with FX impact."""
    try:
        data = request.get_json()

        # Extract and validate inputs
        ticker = data.get('ticker', '').strip().lstrip('$').upper()
        buy_date = data.get('buy_date')  # 'YYYY-MM-DD'
        sell_date = data.get('sell_date')  # 'YYYY-MM-DD'
        quantity = int(data.get('quantity', 1))
        buy_fx_rate = float(data.get('buy_fx_rate', 0))
        sell_fx_rate = float(data.get('sell_fx_rate', 0))
        buy_price_override = data.get('buy_price')  # None or float
        sell_price_override = data.get('sell_price')  # None or float

        # Validation
        if not ticker:
            return jsonify({'error': 'No ticker provided'}), 400
        if not buy_date or not sell_date:
            return jsonify({'error': 'Both buy and sell dates required'}), 400
        if buy_fx_rate <= 0 or sell_fx_rate <= 0:
            return jsonify({'error': 'FX rates must be positive'}), 400
        if quantity < 1 or quantity > 100:
            return jsonify({'error': 'Quantity must be 1-100'}), 400

        # Resolve ticker to company name
        resolved_ticker = resolve_ticker_symbol(ticker) or ticker
        display_name = TICKER_TO_NAME.get(resolved_ticker, ticker)

        # Fetch historical prices (if not overridden)
        stock = yq.Ticker(resolved_ticker)
        hist = stock.history(start=buy_date, end=sell_date, interval="1d")

        if hist.empty:
            return jsonify({'error': f'No historical data for {ticker}'}), 404

        hist = hist.reset_index()

        # Normalize dates for comparison - convert to date strings
        hist['date_str'] = pd.to_datetime(hist['date']).dt.strftime('%Y-%m-%d')

        # Get buy price (closest trading day to buy_date)
        if buy_price_override:
            buy_price = float(buy_price_override)
            buy_price_overridden = True
            buy_date_actual = buy_date
        else:
            # Find first trading day on/after buy_date
            hist_buy = hist[hist['date_str'] >= buy_date]
            if hist_buy.empty:
                return jsonify({'error': f'No trading data on/after buy date {buy_date}'}), 404
            buy_price = float(hist_buy.iloc[0]['close'])
            buy_price_overridden = False
            buy_date_actual = hist_buy.iloc[0]['date_str']

        # Get sell price (closest trading day to sell_date)
        if sell_price_override:
            sell_price = float(sell_price_override)
            sell_price_overridden = True
            sell_date_actual = sell_date
        else:
            # Find first trading day on/before sell_date
            hist_sell = hist[hist['date_str'] <= sell_date]
            if hist_sell.empty:
                return jsonify({'error': f'No trading data on/before sell date {sell_date}'}), 404
            sell_price = float(hist_sell.iloc[-1]['close'])
            sell_price_overridden = False
            sell_date_actual = hist_sell.iloc[-1]['date_str']

        # Calculate returns
        buy_value_usd = quantity * buy_price
        sell_value_usd = quantity * sell_price
        return_usd = sell_value_usd - buy_value_usd
        return_pct_usd = (return_usd / buy_value_usd * 100) if buy_value_usd else 0

        # THB calculations with FX impact
        buy_value_thb = buy_value_usd * buy_fx_rate
        sell_value_thb = sell_value_usd * sell_fx_rate
        return_thb = sell_value_thb - buy_value_thb
        return_pct_thb = (return_thb / buy_value_thb * 100) if buy_value_thb else 0

        # FX impact (what would THB return be if FX stayed constant?)
        sell_value_thb_no_fx = sell_value_usd * buy_fx_rate
        return_thb_no_fx = sell_value_thb_no_fx - buy_value_thb
        fx_impact = return_thb - return_thb_no_fx
        fx_impact_pct = (fx_impact / buy_value_thb * 100) if buy_value_thb else 0

        return jsonify({
            'ticker': resolved_ticker,
            'displayName': display_name,
            'buyDate': buy_date_actual,
            'sellDate': sell_date_actual,
            'quantity': quantity,
            'buyPriceUSD': round(buy_price, 2),
            'sellPriceUSD': round(sell_price, 2),
            'buyPriceOverridden': buy_price_overridden,
            'sellPriceOverridden': sell_price_overridden,
            'buyFxRate': buy_fx_rate,
            'sellFxRate': sell_fx_rate,
            'calculations': {
                'buyValueUSD': round(buy_value_usd, 2),
                'sellValueUSD': round(sell_value_usd, 2),
                'buyValueTHB': round(buy_value_thb, 2),
                'sellValueTHB': round(sell_value_thb, 2),
                'returnUSD': round(return_usd, 2),
                'returnTHB': round(return_thb, 2),
                'returnPctUSD': round(return_pct_usd, 2),
                'returnPctTHB': round(return_pct_thb, 2),
                'fxImpactTHB': round(fx_impact, 2),
                'fxImpactPct': round(fx_impact_pct, 2)
            },
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Starting Agentic Financial Analysis Server...")
    print("\n🤖 TRI-LLM ARCHITECTURE:")
    print("   ├─ Mistral AI: Input Validation & Routing")
    print("   ├─ Groq OAI/gpt-oss-120b: Analysis Synthesis")
    print("   └─ MiMo-V2-Flash: Output Validation with Reasoning")
    print("\n📊 Stock Analysis Agent: Ready")
    print("🌍 Macro Analysis Agent: Ready")
    print("\n💡 Full validation pipeline: Input → Analysis → Output")
    print("🔗 Server running at http://localhost:5092")
    app.run(debug=True, port=5092)
