# SmokeNMirror - Agentic Financial Analysis Platform

A modern web-based AI-powered financial analysis platform that combines LangChain agents with real-time market data, technical indicators, and interactive TradingView charts to provide comprehensive stock and macroeconomic analysis.

## Overview

SmokeNMirror is a full-stack financial analysis application featuring:
- **Interactive Web Interface**: Modern, responsive UI with dark/light themes
- **AI-Powered Analysis**: LangChain agents using Groq's Llama-4 for intelligent market insights
- **Real-Time Data**: Integration with Yahoo Finance, FRED, Finnhub, and Polygon APIs
- **Advanced Charts**: TradingView Lightweight Charts for professional-grade visualization
- **Dual Analysis Modes**: Stock analysis and global macro analysis

## Features

### Stock Analysis
- **Comprehensive Fundamental Metrics**
  - Valuation ratios (P/E, P/B, P/S, EV/EBITDA, EV/Revenue)
  - Profitability metrics (Profit Margin, ROE, ROA, Gross Margin, Operating Margin)
  - Growth indicators (Earnings Growth, Revenue Growth)
  - Financial health (Debt-to-Equity, Current Ratio, Quick Ratio, Cash per Share)
  - Market metrics (Beta, Dividend Yield, 52-week range, price performance)

- **Technical Analysis** (with TA-Lib)
  - Momentum Indicators: RSI, MACD, Stochastic Oscillator
  - Trend Indicators: SMA (20/50/200), EMA (12/26)
  - Volatility Indicators: Bollinger Bands, ATR
  - Volume Analysis: Volume ratios and trends
  - Support/Resistance Levels: 20-day and 50-day ranges

- **Market Intelligence**
  - Multi-source news aggregation (yahooquery, Finnhub, Polygon)
  - Sentiment analysis with positive/negative/neutral classification
  - Interactive TradingView candlestick charts with volume overlay

- **Smart Ticker Search**
  - Auto-complete with company name matching
  - Fuzzy search supporting both ticker symbols and company names
  - SEC company database integration

### Macro Analysis
- **Economic Indicators** (via FRED API)
  - General: Federal Funds Rate, Treasury Yields, VIX, Unemployment, CPI
  - Inflation: CPI, Core CPI, PCE, Core PCE, Inflation Expectations
  - Employment: Unemployment Rate, Labor Force Participation, Nonfarm Payrolls, Job Openings
  - Interest Rates: Fed Funds, 2Y/10Y/30Y Treasuries, Yield Curve Analysis
  - GDP: Real GDP, GDP Growth Rate, Personal Consumption, Business Investment

- **Federal Reserve Policy Tracking**
  - Federal Funds Rate with upper/lower targets
  - Fed balance sheet (Total Assets, Excess Reserves)
  - Policy trend analysis with visual indicators

- **Bond Market Analysis**
  - Complete Treasury yield curve (3M to 30Y)
  - Yield curve inversion detection
  - Spread analysis (10Y-2Y) with recession signals

### User Experience
- **Modern UI/UX**
  - Dark mode (default) and light mode themes
  - Gradient-enhanced design with smooth animations
  - Responsive layout for desktop and mobile
  - Real-time loading states and error handling

- **Interactive Features**
  - Auto-complete ticker search
  - Multiple chart timeframes (1M, 3M, 6M, 1Y, 2Y, 5Y)
  - Expandable analysis sections
  - Copy-to-clipboard functionality

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (index.html)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Stock Search │  │ Macro Search │  │ Chart Viewer │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└───────────────────────────┬─────────────────────────────────┘
                            │ REST API
┌───────────────────────────┴─────────────────────────────────┐
│                    Flask Backend (app.py)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Stock Agent  │  │ Macro Agent  │  │  API Routes  │       │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘       │
│         │                  │                                │
│  ┌──────┴──────────────────┴──────┐                         │
│  │      LangChain Tools            │                        │
│  │  • get_financial_metrics()      │                        │
│  │  • get_technical_indicators()   │                        │
│  │  • get_market_news()            │                        │
│  │  • get_economic_indicators()    │                        │
│  │  • get_fed_policy_info()        │                        │
│  │  • get_bond_yields()            │                        │
│  └─────────────────────────────────┘                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│                  External Data Sources                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │  Yahoo   │ │   FRED   │ │ Finnhub  │ │ Polygon  │        │
│  │ Finance  │ │   API    │ │   API    │ │   API    │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Backend
- **Framework**: Flask 3.0+ with CORS support
- **AI/ML**: LangChain + Groq (Llama-4-Scout-17B)
- **Data APIs**:
  - yahooquery (stock data - same source as TradingView)
  - FRED API (economic data)
  - Finnhub API (news)
  - Polygon API (alternative data)
- **Analysis**: TA-Lib (technical indicators), NumPy, Pandas

### Frontend
- **Charts**: TradingView Lightweight Charts
- **Styling**: Custom CSS with CSS variables for theming
- **Fonts**: Outfit (UI), JetBrains Mono (code/data)
- **JavaScript**: Vanilla ES6+ with Fetch API

### Infrastructure
- **Environment**: Python-dotenv for configuration
- **HTTP**: Requests library with retry logic
- **Data Format**: JSON for API communication

## Installation

### Prerequisites
- Python 3.8 or higher
- TA-Lib (system-level installation required)

### 1. Install TA-Lib

**macOS:**
```bash
brew install ta-lib
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ta-lib-dev
```

**Windows:**
Download pre-built binaries from [ta-lib.org](http://ta-lib.org/hdr_dw.html)

### 2. Clone Repository
```bash
git clone https://github.com/yourusername/SmokeNMirror.git
cd SmokeNMirror
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# AI/LLM API Keys
groq=your_groq_api_key_here

# Financial Data API Keys (at least FRED required)
FRED_API=your_fred_api_key_here

# Optional: Additional data sources
POLYGON=your_polygon_api_key_here
finhub=your_finnhub_api_key_here

# Legacy (not actively used but can be configured)
OPENAI_KEY=your_openai_api_key_here
MISTRAL=your_mistral_api_key_here
```

### 5. Verify Installation

Test TA-Lib installation:
```bash
python -c "import talib; print(talib.__version__)"
```

## Getting API Keys

### Required
- **Groq** (Free): https://console.groq.com/
  - Used for AI-powered analysis
  - Sign up and create an API key

- **FRED** (Free): https://fred.stlouisfed.org/docs/api/api_key.html
  - Federal Reserve Economic Data
  - Required for macro analysis
  - Request API key (instant approval)

### Optional (Enhances Features)
- **Finnhub** (Free tier): https://finnhub.io/
  - Alternative news source
  - 60 calls/minute on free tier

- **Polygon** (Free tier): https://polygon.io/
  - Alternative financial data
  - Limited calls on free tier

## Usage

### Starting the Server

```bash
python app.py
```

The server will start at `http://localhost:5000`

You should see:
```
🚀 Starting Agentic Financial Analysis Server...
📊 Stock Analysis Agent: Ready
🌍 Macro Analysis Agent: Ready
🔗 Server running at http://localhost:5000
```

### Using the Web Interface

1. **Open your browser** to `http://localhost:5000`

2. **Stock Analysis**:
   - Enter a ticker symbol (e.g., `AAPL`, `TSLA`) or company name (e.g., `Apple`, `Tesla`)
   - Auto-complete will suggest matches
   - Click "Analyze Stock" to get comprehensive analysis
   - View the interactive chart with different timeframes
   - Read AI-generated insights on valuation, technicals, and fundamentals

3. **Macro Analysis**:
   - Switch to "Macro Outlook" tab
   - Enter a topic (e.g., `inflation`, `federal reserve policy`, `yield curve`)
   - Click "Analyze Macro" for detailed economic analysis
   - Review key indicators, policy outlook, and market implications

4. **Customization**:
   - Toggle dark/light mode with the theme button
   - Expand/collapse analysis sections
   - Change chart timeframes
   - Copy analysis to clipboard

## API Endpoints

### Stock Analysis
```http
POST /api/analyze/stock
Content-Type: application/json

{
  "ticker": "AAPL"
}
```

Response:
```json
{
  "ticker": "AAPL",
  "displayName": "Apple Inc.",
  "userInput": "apple",
  "analysis": "📊 Fundamental Analysis for AAPL...",
  "timestamp": "2025-11-29T10:30:00"
}
```

### Macro Analysis
```http
POST /api/analyze/macro
Content-Type: application/json

{
  "topic": "inflation outlook"
}
```

### Chart Data
```http
GET /api/chart/{ticker}?period=1y
```

Periods: `1m`, `3m`, `6m`, `1y`, `2y`, `5y`

### Ticker Search
```http
GET /api/tickers/search?q=apple
```

### Quick Endpoints
- `GET /api/quick/metrics/{ticker}` - Financial metrics only
- `GET /api/quick/indicators/{type}` - Economic indicators (general/inflation/employment/rates/gdp)
- `GET /api/quick/yields` - Bond yields

## Project Structure

```
SmokeNMirror/
├── app.py                      # Flask backend with LangChain agents
├── index.html                  # Frontend web interface
├── company_tickers.json        # SEC company ticker database
├── requirements.txt            # Python dependencies
├── readme.md                   # This file
├── .env                        # Environment variables (not in git)
└── .gitignore                  # Git ignore rules
```

## Configuration

### LLM Settings
The application uses Groq's Llama-4 model with conservative settings for accuracy:

```python
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,        # Low temperature for consistent analysis
    max_tokens=2048,        # Sufficient for detailed analysis
    max_retries=3           # Auto-retry on failures
)
```

### Data Consistency
- Technical indicators use the **same data source** as TradingView charts (Yahoo Finance, 1-year daily data)
- All metrics include validation and interpretation flags (✅ ⚠️ ❌)
- Retry logic with exponential backoff for API rate limits

## Error Handling

The application includes robust error handling:
- **API Rate Limits**: Exponential backoff with 3 retries
- **Missing Data**: Graceful fallbacks and informative error messages
- **Network Issues**: Timeout handling and retry logic
- **Invalid Tickers**: Fuzzy matching with suggestions
- **Missing API Keys**: Clear warnings about unavailable features

If TA-Lib is not installed:
```
⚠️ TA-Lib not installed. Technical indicators will be limited.
```
The app will use fallback calculations for basic indicators.

## Performance Considerations

- **Response Times**: Initial analysis takes 5-15 seconds depending on data sources
- **API Limits**:
  - Yahoo Finance: ~2000 requests/hour (with backoff)
  - FRED: 120 requests/minute
  - Finnhub: 60 requests/minute (free tier)
  - Polygon: Limited on free tier
- **Optimization**: Consider implementing caching for production use
- **Concurrent Requests**: Flask development server handles one request at a time

## Troubleshooting

### Common Issues

**1. TA-Lib Import Error**
```
ImportError: No module named 'talib'
```
Solution: Install TA-Lib at system level (see Installation section)

**2. API Rate Limits**
```
Error fetching metrics: 429 Too Many Requests
```
Solution: Wait a few minutes, the app has built-in retry logic

**3. Missing API Keys**
```
❌ FRED_API_KEY not set
```
Solution: Add required API keys to `.env` file

**4. CORS Errors**
```
Access-Control-Allow-Origin error
```
Solution: Ensure Flask-CORS is installed and the server is running

**5. No Data for Ticker**
```
❌ No price data available for XYZ
```
Solution: Verify ticker symbol is correct and traded on major exchanges

### Debug Mode

To see detailed logs, the app runs in debug mode by default:
```python
app.run(debug=True, port=5000)
```

## Deployment Considerations

For production deployment:

1. **Disable Debug Mode**:
   ```python
   app.run(debug=False, port=5000)
   ```

2. **Use Production Server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Implement Caching**:
   - Add Redis or in-memory caching for API responses
   - Cache duration: 1-5 minutes for stock data, 1 hour for economic data

4. **Rate Limiting**:
   - Use Flask-Limiter to prevent abuse
   - Implement request queuing for API calls

5. **Environment Variables**:
   - Use proper secret management (AWS Secrets Manager, HashiCorp Vault)
   - Never commit `.env` to version control

6. **Monitoring**:
   - Add logging with rotating file handlers
   - Implement error tracking (Sentry, Rollbar)
   - Monitor API usage and quotas

## Limitations

- **Data Latency**: Market data is delayed 15-20 minutes (real-time requires paid APIs)
- **Historical Analysis**: Limited to available historical data from free APIs
- **Rate Limits**: Free API tiers have request limitations
- **AI Analysis**: Powered by LLM - should not be used as sole basis for investment decisions
- **No Financial Advice**: This tool is for informational and educational purposes only

## Future Enhancements

Potential improvements:
- [ ] User authentication and portfolio tracking
- [ ] Watchlist functionality with alerts
- [ ] Backtesting capabilities
- [ ] Options analysis (Greeks, IV, strategies)
- [ ] Cryptocurrency support
- [ ] Real-time WebSocket data feeds
- [ ] Export analysis to PDF/Excel
- [ ] Custom indicator builder
- [ ] Social sentiment analysis (Twitter, Reddit)
- [ ] Earnings calendar integration
- [ ] Comparison tools (multi-ticker analysis)

## Contributing

Contributions are welcome! Areas for improvement:
- Additional data sources and APIs
- Enhanced technical indicators
- UI/UX improvements
- Performance optimizations
- Testing and documentation
- Mobile app development

## License

This project is for educational and research purposes. Not intended as financial advice.

## Disclaimer

**IMPORTANT**: This application is provided for informational and educational purposes only. It does not constitute financial advice, investment recommendations, or an offer to buy or sell securities. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results. The creators and contributors are not responsible for any financial losses incurred from using this application.

## Acknowledgments

- **LangChain** - Agent framework and tool orchestration
- **Groq** - High-performance LLM inference
- **Yahoo Finance** - Stock price and fundamental data via yahooquery
- **Federal Reserve (FRED)** - Economic indicators and data
- **TradingView** - Professional charting library
- **TA-Lib** - Technical analysis indicators
- **Finnhub & Polygon** - Alternative financial data sources

---

**Built with** ⚡ by the SmokeNMirror team

For issues, questions, or suggestions, please open an issue on GitHub.
