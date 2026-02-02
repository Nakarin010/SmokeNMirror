"""
Comprehensive Test Suite for SmokeNMirror
Tests all integrations: Mistral AI, Groq, MiMo-V2-Flash, and FMP Market Risk Premium
Version: 2.1 (Tri-LLM + Market Risk Premium)
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test results storage
TEST_RESULTS = {}


def section_header(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def record_result(category: str, test_name: str, passed: bool, message: str = ""):
    """Record a test result"""
    if category not in TEST_RESULTS:
        TEST_RESULTS[category] = {}
    TEST_RESULTS[category][test_name] = {
        "passed": passed,
        "message": message
    }
    status = "PASSED" if passed else "FAILED"
    icon = "✅" if passed else "❌"
    print(f"{icon} {test_name}: {status}")
    if message and not passed:
        print(f"   └─ {message}")


# ============================================================
# SECTION 1: ENVIRONMENT CONFIGURATION TESTS
# ============================================================

def test_environment():
    """Test that all required API keys are configured"""
    section_header("1. ENVIRONMENT CONFIGURATION")

    keys = {
        "Mistral API Key (MISTRAL)": os.getenv("MISTRAL"),
        "Groq API Key (groq)": os.getenv("groq"),
        "OpenRouter API Key (OPENROUTER)": os.getenv("OPENROUTER"),
        "FMP API Key (FMP)": os.getenv("FMP"),
        "FRED API Key (FRED)": os.getenv("FRED"),
        "Brave API Key (Brave)": os.getenv("Brave"),
    }

    required = ["Mistral API Key (MISTRAL)", "Groq API Key (groq)"]

    all_required_present = True

    for name, key in keys.items():
        if key:
            record_result("Environment", name, True)
        else:
            is_required = name in required
            if is_required:
                record_result("Environment", name, False, "REQUIRED - Add to .env file")
                all_required_present = False
            else:
                record_result("Environment", name + " (optional)", True, "Not configured but optional")

    return all_required_present


# ============================================================
# SECTION 2: LIBRARY IMPORT TESTS
# ============================================================

def test_imports():
    """Test that required libraries are installed"""
    section_header("2. LIBRARY IMPORTS")

    all_passed = True

    # Test langchain_mistralai
    try:
        from langchain_mistralai import ChatMistralAI
        record_result("Imports", "langchain_mistralai", True)
    except ImportError as e:
        record_result("Imports", "langchain_mistralai", False, "pip install langchain-mistralai")
        all_passed = False

    # Test langchain_groq
    try:
        from langchain_groq import ChatGroq
        record_result("Imports", "langchain_groq", True)
    except ImportError as e:
        record_result("Imports", "langchain_groq", False, "pip install langchain-groq")
        all_passed = False

    # Test openai
    try:
        from openai import OpenAI
        record_result("Imports", "openai", True)
    except ImportError as e:
        record_result("Imports", "openai", False, "pip install openai")
        all_passed = False

    # Test requests
    try:
        import requests
        record_result("Imports", "requests", True)
    except ImportError as e:
        record_result("Imports", "requests", False, "pip install requests")
        all_passed = False

    # Test yfinance
    try:
        import yfinance
        record_result("Imports", "yfinance", True)
    except ImportError as e:
        record_result("Imports", "yfinance", False, "pip install yfinance")
        all_passed = False

    # Test fredapi
    try:
        from fredapi import Fred
        record_result("Imports", "fredapi", True)
    except ImportError as e:
        record_result("Imports", "fredapi", False, "pip install fredapi")
        all_passed = False

    # Test langchain_core
    try:
        from langchain_core.tools import tool
        record_result("Imports", "langchain_core", True)
    except ImportError as e:
        record_result("Imports", "langchain_core", False, "pip install langchain-core")
        all_passed = False

    return all_passed


# ============================================================
# SECTION 3: LLM INITIALIZATION TESTS
# ============================================================

def test_llm_initialization():
    """Test that all LLM clients can be initialized"""
    section_header("3. LLM INITIALIZATION")

    results = []

    # Test Mistral initialization
    mistral_key = os.getenv("MISTRAL")
    if mistral_key:
        try:
            from langchain_mistralai import ChatMistralAI
            llm = ChatMistralAI(
                model="mistral-large-latest",
                temperature=0.1,
                api_key=mistral_key,
                max_tokens=100,
            )
            record_result("Initialization", "Mistral AI Client", True)
            results.append(True)
        except Exception as e:
            record_result("Initialization", "Mistral AI Client", False, str(e)[:50])
            results.append(False)
    else:
        record_result("Initialization", "Mistral AI Client", False, "API key not configured")
        results.append(False)

    # Test Groq initialization
    groq_key = os.getenv("groq")
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                api_key=groq_key,
                max_tokens=100,
            )
            record_result("Initialization", "Groq Client", True)
            results.append(True)
        except Exception as e:
            record_result("Initialization", "Groq Client", False, str(e)[:50])
            results.append(False)
    else:
        record_result("Initialization", "Groq Client", False, "API key not configured")
        results.append(False)

    # Test OpenRouter initialization
    openrouter_key = os.getenv("OPENROUTER")
    if openrouter_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )
            record_result("Initialization", "OpenRouter Client (MiMo)", True)
            results.append(True)
        except Exception as e:
            record_result("Initialization", "OpenRouter Client (MiMo)", False, str(e)[:50])
            results.append(False)
    else:
        record_result("Initialization", "OpenRouter Client (MiMo)", True, "Optional - not configured")
        results.append(True)

    return all(results)


# ============================================================
# SECTION 4: MISTRAL AI INFERENCE TESTS
# ============================================================

def test_mistral_inference():
    """Test Mistral AI inference capabilities"""
    section_header("4. MISTRAL AI INFERENCE")

    mistral_key = os.getenv("MISTRAL")

    if not mistral_key:
        record_result("Mistral", "API Inference", False, "API key not configured")
        return False

    try:
        from langchain_mistralai import ChatMistralAI
        llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.1,
            api_key=mistral_key,
            max_tokens=100,
        )

        # Test 1: Simple classification
        start = time.time()
        response = llm.invoke("Classify as bullish, bearish, or neutral: 'Stock hits all-time high'")
        latency = time.time() - start

        record_result("Mistral", f"Classification ({latency:.2f}s)", True, response.content[:50])

        # Test 2: Tool selection simulation
        start = time.time()
        response = llm.invoke("What tools are needed for: 'What is the current inflation rate?' Answer with JSON array.")
        latency = time.time() - start

        record_result("Mistral", f"Tool Selection ({latency:.2f}s)", True)

        return True

    except Exception as e:
        record_result("Mistral", "API Inference", False, str(e)[:80])
        return False


# ============================================================
# SECTION 5: GROQ INFERENCE TESTS
# ============================================================

def test_groq_inference():
    """Test Groq Llama inference capabilities"""
    section_header("5. GROQ LLAMA INFERENCE")

    groq_key = os.getenv("groq")

    if not groq_key:
        record_result("Groq", "API Inference", False, "API key not configured")
        return False

    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=groq_key,
            max_tokens=150,
        )

        # Test synthesis capability
        start = time.time()
        response = llm.invoke("Provide a one-sentence stock market analysis.")
        latency = time.time() - start

        record_result("Groq", f"Synthesis ({latency:.2f}s)", True, response.content[:50])

        return True

    except Exception as e:
        record_result("Groq", "API Inference", False, str(e)[:80])
        return False


# ============================================================
# SECTION 6: MiMo-V2-FLASH VALIDATION TESTS
# ============================================================

def test_mimo_validation():
    """Test MiMo-V2-Flash output validation"""
    section_header("6. MiMo-V2-FLASH VALIDATION")

    openrouter_key = os.getenv("OPENROUTER")

    if not openrouter_key:
        record_result("MiMo", "Output Validation", True, "Optional - not configured")
        return True

    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
        )

        # Test 1: Basic inference
        start = time.time()
        response = client.chat.completions.create(
            model="xiaomi/mimo-v2-flash:free",
            messages=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
        )
        latency = time.time() - start

        record_result("MiMo", f"Basic Inference ({latency:.2f}s)", True)

        # Test 2: Validation logic
        sample_analysis = """
**Stock Analysis: AAPL**
P/E ratio of 28 indicates fair value. RSI at 65 shows bullish momentum.
Recommendation: BUY with target $200.
"""

        validation_prompt = f"""Validate this analysis. Check for:
1. Logical consistency
2. Factual accuracy
3. Completeness

ANALYSIS:
{sample_analysis}

Respond with JSON: {{"is_valid": true/false, "issues": [], "quality": "good/poor"}}"""

        start = time.time()
        response = client.chat.completions.create(
            model="xiaomi/mimo-v2-flash:free",
            messages=[{"role": "user", "content": validation_prompt}],
        )
        latency = time.time() - start

        content = response.choices[0].message.content
        has_expected = "valid" in content.lower() or "quality" in content.lower()

        record_result("MiMo", f"Validation Logic ({latency:.2f}s)", has_expected)

        return True

    except Exception as e:
        record_result("MiMo", "Output Validation", False, str(e)[:80])
        return False


# ============================================================
# SECTION 7: FMP MARKET RISK PREMIUM TESTS
# ============================================================

def test_fmp_market_risk():
    """Test FMP Market Risk Premium API"""
    section_header("7. FMP MARKET RISK PREMIUM")

    import requests

    fmp_key = os.getenv("FMP")

    if not fmp_key:
        record_result("FMP", "Market Risk Premium", True, "Optional - not configured")
        return True

    try:
        # Test 1: API connectivity
        url = f"https://financialmodelingprep.com/api/v4/market_risk_premium?apikey={fmp_key}"

        start = time.time()
        response = requests.get(url, timeout=10)
        latency = time.time() - start

        if response.status_code != 200:
            record_result("FMP", "API Connectivity", False, f"Status code: {response.status_code}")
            return False

        record_result("FMP", f"API Connectivity ({latency:.2f}s)", True)

        # Test 2: Data parsing
        data = response.json()

        if not data or not isinstance(data, list):
            record_result("FMP", "Data Parsing", False, "Empty or invalid response")
            return False

        record_result("FMP", f"Data Parsing ({len(data)} countries)", True)

        # Test 3: Data structure validation
        sample = data[0]
        required_fields = ['country', 'countryRiskPremium', 'totalEquityRiskPremium']
        has_all_fields = all(field in sample for field in required_fields)

        record_result("FMP", "Data Structure", has_all_fields)

        # Test 4: Find US market risk premium
        us_data = next((item for item in data if item.get('country') == 'United States'), None)

        if us_data:
            us_premium = us_data.get('totalEquityRiskPremium', 0)
            record_result("FMP", f"US Market Premium: {us_premium:.2f}%", True)
        else:
            record_result("FMP", "US Market Premium", False, "US data not found")

        # Test 5: Country lookup simulation
        japan_data = next((item for item in data if item.get('country') == 'Japan'), None)

        if japan_data:
            jp_premium = japan_data.get('totalEquityRiskPremium', 0)
            record_result("FMP", f"Japan Premium: {jp_premium:.2f}%", True)

        return True

    except Exception as e:
        record_result("FMP", "Market Risk Premium", False, str(e)[:80])
        return False


# ============================================================
# SECTION 8: INTEGRATION TESTS
# ============================================================

def test_integration():
    """Test end-to-end integration by importing app functions"""
    section_header("8. INTEGRATION TESTS")

    try:
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        # Test 1: Import main app
        from app import (
            validate_analysis_output,
            get_market_risk,
            get_economic_indicators,
            openrouter_client,
            fmp
        )
        record_result("Integration", "App Import", True)

        # Test 2: Check market risk function exists
        if get_market_risk:
            record_result("Integration", "get_market_risk() Available", True)
        else:
            record_result("Integration", "get_market_risk() Available", False)

        # Test 3: Test market risk function (if FMP key available)
        if fmp:
            try:
                result = get_market_risk.invoke("")
                has_data = "Market Risk Premium" in result or "CAPM" in result
                record_result("Integration", "get_market_risk() Execution", has_data, result[:50] if has_data else "No data")
            except Exception as e:
                record_result("Integration", "get_market_risk() Execution", False, str(e)[:50])
        else:
            record_result("Integration", "get_market_risk() Execution", True, "FMP not configured")

        # Test 4: Check OpenRouter client
        if openrouter_client:
            record_result("Integration", "OpenRouter Client Ready", True)
        else:
            record_result("Integration", "OpenRouter Client Ready", True, "Not configured (optional)")

        # Test 5: Validate analysis function
        if openrouter_client:
            try:
                sample = "Test analysis: AAPL looks bullish based on P/E of 25."
                context = {"ticker": "AAPL", "displayName": "Apple Inc."}
                result = validate_analysis_output(sample, context, "stock")
                has_result = 'validated' in result or 'enhanced_analysis' in result
                record_result("Integration", "validate_analysis_output()", has_result)
            except Exception as e:
                record_result("Integration", "validate_analysis_output()", False, str(e)[:50])
        else:
            record_result("Integration", "validate_analysis_output()", True, "Skipped - no OpenRouter")

        return True

    except ImportError as e:
        record_result("Integration", "App Import", False, str(e)[:80])
        return False
    except Exception as e:
        record_result("Integration", "Integration Tests", False, str(e)[:80])
        return False


# ============================================================
# MAIN TEST RUNNER
# ============================================================

def print_summary():
    """Print comprehensive test summary"""
    section_header("TEST SUMMARY")

    total_passed = 0
    total_failed = 0

    for category, tests in TEST_RESULTS.items():
        print(f"\n{category}:")
        for test_name, result in tests.items():
            status = "✅" if result["passed"] else "❌"
            print(f"  {status} {test_name}")
            if result["passed"]:
                total_passed += 1
            else:
                total_failed += 1

    print("\n" + "=" * 70)
    print(f"  TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📊 System Status:")
        print("   • Tri-LLM Architecture: Ready")
        print("   • Input Validation (Mistral): Ready")
        print("   • Analysis Synthesis (Groq): Ready")
        print("   • Output Validation (MiMo): Ready" if os.getenv("OPENROUTER") else "   • Output Validation (MiMo): Not configured")
        print("   • Market Risk Premium (FMP): Ready" if os.getenv("FMP") else "   • Market Risk Premium (FMP): Not configured")
        print("\n🚀 Next Steps:")
        print("   1. Start server: python app.py")
        print("   2. Test stock analysis: POST /api/analyze/stock")
        print("   3. Test macro analysis: POST /api/analyze/macro")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("\n💡 Common Fixes:")
        print("   • Add missing API keys to .env file")
        print("   • Run: pip install -r requirements.txt")
        print("   • Check API key validity and quotas")
        print("   • Review error messages above")

    print("=" * 70)

    return total_failed == 0


def main():
    """Run all comprehensive tests"""
    print("=" * 70)
    print("  🧪 SMOKENMIRROR COMPREHENSIVE TEST SUITE")
    print("  Version: 2.1 (Tri-LLM + Market Risk Premium)")
    print("  " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Run all test sections
    test_environment()
    test_imports()
    test_llm_initialization()
    test_mistral_inference()
    test_groq_inference()
    test_mimo_validation()
    test_fmp_market_risk()
    test_integration()

    # Print summary
    all_passed = print_summary()

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
