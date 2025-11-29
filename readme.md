# Financial Analysis Agent

An AI-powered financial analysis system using LangChain agents to answer financial questions with comprehensive market data, technical indicators, and economic insights.

## Features

- **Multi-LLM Architecture**: Uses Groq Llama-4 for primary analysis and Mistral-7B for validation
- **Comprehensive Financial Tools**:
  - Financial metrics and ratios (PE, P/B, ROE, debt ratios, etc.)
  - Market news with sentiment analysis
  - Economic indicators from FRED API
  - Federal Reserve policy tracking
  - Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- **Date-Aware Analysis**: Extracts reference dates from queries for historical analysis
- **Intelligent Fallbacks**: Multiple data sources with automatic fallback mechanisms
- **Robust Error Handling**: Proper exception handling and retry logic

## Prerequisites

- Python 3.8+
- TA-Lib library (requires system-level installation)

### Installing TA-Lib

**macOS:**
```bash
brew install ta-lib
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ta-lib
```

**Windows:**
Download pre-built binaries from [ta-lib.org](http://ta-lib.org/hdr_dw.html)

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**

Create a `.env` file in the project root with your API keys:

```env
# LLM API Keys
OPENAI_KEY=your_openai_api_key_here
MISTRAL=your_mistral_api_key_here
groq=your_groq_api_key_here

# Financial Data API Keys
FRED_API=your_fred_api_key_here
POLYGON=your_polygon_api_key_here
finhub=your_finnhub_api_key_here
```

### Getting API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Mistral**: https://console.mistral.ai/
- **Groq**: https://console.groq.com/
- **FRED (Federal Reserve)**: https://fred.stlouisfed.org/docs/api/api_key.html (Free)
- **Polygon**: https://polygon.io/ (Free tier available)
- **Finnhub**: https://finnhub.io/ (Free tier available)

## Usage

1. **Prepare your test data:**

Ensure you have a `test.csv` file with the following structure:
```csv
id,query
1,"What is the PE ratio of AAPL?"
2,"Predict if $TSLA will rise or fall on 2024-01-15"
```

2. **Run the analysis:**
```bash
python AgenticAgentFinancialAnalysis.py
```

3. **Check outputs:**
- `submission.csv` - Predictions for all queries
- `process_log.txt` - Detailed processing log with raw and clean answers

## Project Structure

```
.
├── AgenticAgentFinancialAnalysis.py  # Main agent application
├── requirements.txt                  # Python dependencies
├── .env                             # API keys (not in git)
├── .gitignore                       # Git ignore rules
├── test.csv                         # Input queries
├── submission.csv                   # Output predictions (generated)
└── process_log.txt                  # Processing logs (generated)
```

## How It Works

1. **Query Processing**: Extracts reference dates from questions (e.g., "on 13/12/2017")
2. **Agent Analysis**: LangChain agent uses appropriate tools to gather financial data
3. **Answer Extraction**: Parses agent response to extract clean answers (A/B/C/D or Rise/Fall)
4. **Validation** (optional): Uses validator LLM to verify answer format
5. **Fallback Handling**: If agent fails, attempts direct LLM call, then defaults to 'A'

## Available Tools

### 1. `get_financial_metrics(ticker: str)`
Fetches comprehensive financial ratios and metrics:
- Valuation ratios (PE, P/B, P/S, EV/EBITDA)
- Financial health (debt-to-equity, current ratio, quick ratio)
- Profitability (profit margins, ROE, ROA)
- Growth metrics (earnings growth, revenue growth)

### 2. `get_market_news(ticker: str)`
Multi-source news aggregation with sentiment analysis:
- Sources: yahooquery, Finnhub, Polygon
- Keyword-based sentiment classification
- Date-aware filtering

### 3. `get_economic_indicators(category: str)`
FRED economic data by category:
- `general`: Fed funds, treasuries, VIX, unemployment, CPI
- `inflation`: CPI, Core CPI, PCE, inflation expectations
- `employment`: Unemployment, labor force, payrolls, job openings
- `rates`: Fed funds, treasury yields, yield curve
- `gdp`: Real GDP, growth rate, consumption, investment

### 4. `get_fed_policy_info()`
Federal Reserve policy indicators with trends:
- Federal funds rate and targets
- Total assets and excess reserves

### 5. `get_technical_indicators(ticker: str)`
Technical analysis using TA-Lib:
- 14-day RSI
- MACD (12/26)
- Bollinger Bands position
- 14-day ATR

## Configuration

### LLM Settings

**Primary Agent** (in code):
```python
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.05,
    max_tokens=2048
)
```

**Validator**:
```python
validator_llm = ChatMistralAI(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.0,
    max_tokens=32
)
```

### Agent Parameters

- **Agent Type**: `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION`
- **Max Iterations**: 3
- **Error Handling**: Enabled with parsing error handling

## Troubleshooting

### Common Issues

1. **TA-Lib Import Error**
   - Ensure TA-Lib is installed at system level (see Prerequisites)
   - Verify with: `python -c "import talib; print(talib.__version__)"`

2. **API Rate Limits**
   - yahooquery implements exponential backoff for 429 errors
   - Consider adding delays between requests if hitting limits

3. **Missing API Keys**
   - Check `.env` file exists and contains all required keys
   - Verify keys are valid by testing individual API calls

4. **Date Parsing Issues**
   - Reference dates support formats: DD/MM/YYYY, YYYY-MM-DD, "Dec 13 2017"
   - Relative dates: "yesterday", "last month"

## Performance Notes

- Uses free-tier APIs for cost efficiency
- Implements retry logic with exponential backoff
- Caches are not implemented (consider adding for production)
- Processing time depends on API response times and number of queries

## Recent Improvements

All critical bugs and issues have been fixed:

✅ **Fixed critical syntax error** with global CURRENT_REF_DATE statement
✅ **Fixed validator function** to properly use query and candidate parameters
✅ **Replaced bare exception handlers** with specific exception types and error logging
✅ **Fixed resource leak** by using context manager for log file
✅ **Added requirements.txt** with all dependencies
✅ **Created .gitignore** to protect sensitive files

## License

This project is for educational and research purposes.

## Contributing

Improvements welcome! Consider:
- Adding more financial data sources
- Implementing caching for API responses
- Modularizing code into separate files
- Adding unit tests
- Implementing async API calls for better performance

## Acknowledgments

- Built with LangChain framework
- Financial data from yahooquery, FRED, Finnhub, and Polygon
- Technical analysis powered by TA-Lib
