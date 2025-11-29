# Global Macro Analysis Improvement Plan

## Executive Summary

This document outlines improvements for the Global Macro outlook questions feature in SmokeNMirror. The primary goals are to:
1. Add LLM-based validation and intelligent tool selection
2. Enable internet connectivity for real-time information
3. Convert to a true agentic system with iterative reasoning
4. Improve performance and user experience

---

## Current State Analysis

### Architecture Overview

**Current Flow:**
```
User Question → Keyword Matching → Fetch ALL data → LLM Synthesis → Response
```

**Key Files:**
- `app.py:999-1013` - MACRO_SYSTEM_PROMPT definition
- `app.py:1074-1126` - analyze_macro_with_tools() function
- `app.py:1166-1186` - /api/analyze/macro endpoint
- `index.html:1043-1084` - Macro analysis UI panel
- `index.html:1405-1444` - JavaScript request handler

### Current Capabilities

#### ✅ Strengths

1. **Comprehensive Data Sources**
   - FRED API for economic indicators
   - Yahoo Finance for market data
   - Finnhub for news aggregation
   - Polygon for alternative financial data

2. **Robust Error Handling**
   - Exponential backoff with 3 retries
   - HTTP 429 rate limit handling
   - 1-second base delay between retries

3. **Clean Architecture**
   - Clear frontend/backend separation
   - Well-documented system prompts
   - Tool-based design pattern

4. **Internet Connectivity**
   - Already has extensive external API integration
   - Real-time data fetching capability

#### ❌ Weaknesses

1. **Not a True Agent**
   - LLM doesn't select tools or reason iteratively
   - Tools decorated with `@tool` but never used by agent framework
   - Simple request/response pattern instead of ReAct loop

2. **Brittle Keyword Matching**
   ```python
   if "inflation" in topic_lower:
       additional_data = get_economic_indicators.invoke("inflation")
   elif "employment" in topic_lower or "job" in topic_lower:
       additional_data = get_economic_indicators.invoke("employment")
   ```
   - Misses nuanced questions
   - Can't handle multi-topic queries
   - No semantic understanding

3. **No Prompt Validation**
   - Direct interpolation: `f"User's question/topic: {topic}"`
   - Vulnerable to prompt injection
   - No content filtering or sanitization
   - No length limits

4. **Over-Fetching Data**
   - Always calls 3 tools regardless of question relevance
   - Wastes API quota
   - Slower response times
   - No caching mechanism

5. **Limited Real-Time Information**
   - FRED data can lag by weeks/months
   - No access to breaking news or Fed speeches
   - Missing real-time market events

6. **No Multi-Turn Conversation**
   - Each query is isolated
   - Can't reference previous analyses
   - No follow-up question support

---

## Recommended Improvements

### Priority 1: LLM-Based Question Validation & Tool Selection

**Problem:** Questions go directly to data fetching without validation or intelligent routing.

**Solution:** Add two-stage LLM pipeline before tool execution.

#### Implementation Details

**Stage 1: Validation LLM**
```python
def validate_macro_question(question: str) -> dict:
    """
    Validates if the question is appropriate for macro analysis.

    Returns:
        {
            "is_valid": bool,
            "question_type": str,  # "macro_policy", "economic_data", "market_outlook", etc.
            "complexity": str,     # "simple", "moderate", "complex"
            "required_data": list, # ["fed_policy", "inflation", "yields"]
            "reasoning": str
        }
    """
    validator_prompt = """You are a macro economics question validator.

Analyze if this question can be answered with macro economic data:
- Federal Reserve policy
- Economic indicators (inflation, employment, GDP)
- Bond yields and interest rates
- General economic outlook

Respond in JSON format:
{
    "is_valid": true/false,
    "question_type": "macro_policy|economic_data|market_outlook|off_topic",
    "complexity": "simple|moderate|complex",
    "required_data": ["fed_policy", "inflation", "yields", "employment", "gdp"],
    "reasoning": "explanation"
}

Question: {question}
"""

    response = validator_llm.invoke(validator_prompt.format(question=question))
    return json.loads(response.content)
```

**Stage 2: Tool Selection LLM**
```python
def select_tools_for_question(question: str, validation_result: dict) -> list:
    """
    Intelligently selects which tools to use based on the question.

    Returns:
        ["get_fed_policy_info", "get_bond_yields", ...]
    """
    tool_descriptions = {
        "get_economic_indicators": "FRED economic data (inflation, employment, GDP, rates)",
        "get_fed_policy_info": "Federal Reserve policy indicators and stance",
        "get_bond_yields": "US Treasury yield curve and bond data",
        "search_recent_macro_news": "Breaking macro news from last 48 hours",
        "get_fed_speeches": "Recent Fed speeches and FOMC statements"
    }

    planner_prompt = """You are a tool selection expert for macro analysis.

Given this validated question, select the MINIMUM set of tools needed.

Available tools:
{tools}

Question type: {question_type}
Required data: {required_data}
Question: {question}

Return ONLY a JSON array of tool names:
["tool1", "tool2"]

Be conservative - only select tools that are truly needed.
"""

    response = planner_llm.invoke(planner_prompt.format(
        tools=json.dumps(tool_descriptions, indent=2),
        question_type=validation_result["question_type"],
        required_data=validation_result["required_data"],
        question=question
    ))

    return json.loads(response.content)
```

**Integration Point:**
```python
def analyze_macro_with_tools(topic: str) -> str:
    # NEW: Validate question first
    validation = validate_macro_question(topic)

    if not validation["is_valid"]:
        return f"This question appears to be off-topic for macro analysis. {validation['reasoning']}"

    # NEW: Select tools intelligently
    selected_tools = select_tools_for_question(topic, validation)

    # Execute only selected tools
    data_summary = ""
    for tool_name in selected_tools:
        tool_result = execute_tool(tool_name, topic)
        data_summary += f"\n## {tool_name}\n{tool_result}"

    # Synthesize with macro analyst LLM
    prompt = f"{MACRO_SYSTEM_PROMPT}\n\n{data_summary}\n\nUser's question: {topic}"
    response = llm.invoke(prompt)
    return response.content
```

**Benefits:**
- ✅ Prevents nonsense/malicious questions
- ✅ Only fetches needed data (faster, cheaper)
- ✅ Can determine if question is answerable
- ✅ More flexible than keyword matching
- ✅ Better error messages for off-topic questions

**Estimated Effort:** 2-3 hours

---

### Priority 2: Convert to True LangChain Agent

**Problem:** Tools are defined but never used by an autonomous agent. LLM can't reason iteratively.

**Solution:** Implement LangChain ReAct Agent with tool calling.

#### Implementation Details

**Current (Non-Agentic):**
```python
# Tools are called directly via keyword matching
if "inflation" in topic_lower:
    data = get_economic_indicators.invoke("inflation")
```

**New (Agentic):**
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate

# Define agent prompt with ReAct format
agent_prompt = PromptTemplate.from_template("""You are a macro economic analyst with access to tools.

Answer the user's question by using the available tools to gather data.
Think step by step about what information you need.

Available tools:
{tools}

Tool names: {tool_names}

Question: {input}

Thought: Let me think about what data I need...
{agent_scratchpad}
""")

# Create agent with all tools
tools = [
    get_economic_indicators,
    get_fed_policy_info,
    get_bond_yields,
    search_recent_macro_news,  # NEW
    get_fed_speeches,          # NEW
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    max_execution_time=30,  # 30 second timeout
    handle_parsing_errors=True,
    return_intermediate_steps=True
)

# Execute agent
def analyze_macro_with_agent(topic: str) -> str:
    # Validate first (from Priority 1)
    validation = validate_macro_question(topic)
    if not validation["is_valid"]:
        return f"Invalid question: {validation['reasoning']}"

    # Let agent decide which tools to use and when
    result = agent_executor.invoke({
        "input": topic,
        "context": f"Question type: {validation['question_type']}"
    })

    return result["output"]
```

**Example Agent Reasoning:**
```
User: "What's the inflation outlook given recent Fed policy changes?"

Agent Thought: I need to understand current inflation and Fed policy.
Action: get_economic_indicators
Action Input: "inflation"
Observation: PCE inflation at 2.7%, trending down from 3.2%...

Agent Thought: Now I need to see what the Fed is saying about this.
Action: get_fed_policy_info
Action Input: None
Observation: Fed Fund Rate at 4.5%, Powell indicated holding steady...

Agent Thought: Let me check recent Fed speeches for latest guidance.
Action: get_fed_speeches
Action Input: None
Observation: Powell speech from yesterday: "inflation progress encouraging..."

Agent Thought: I now have enough information to answer.
Final Answer: Based on current data, inflation outlook is...
```

**Benefits:**
- ✅ LLM can iteratively reason and adjust strategy
- ✅ Can handle complex multi-step questions
- ✅ Can recover from errors by trying different tools
- ✅ More transparent decision-making (visible reasoning)
- ✅ Can combine multiple tools dynamically

**Estimated Effort:** 4-5 hours

---

### Priority 3: Add Web Search Tool for Real-Time Information

**Problem:** FRED data lags, missing breaking news and real-time Fed communications.

**Solution:** Add web search and Fed scraping tools.

#### Implementation Details

**Tool 1: Recent Macro News Search**
```python
from langchain_community.tools import DuckDuckGoSearchResults
from datetime import datetime, timedelta

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
        # Option A: DuckDuckGo (no API key needed)
        search = DuckDuckGoSearchResults(num_results=5)
        results = search.run(f"{query} site:bloomberg.com OR site:reuters.com OR site:wsj.com")

        # Option B: SerpAPI (requires key, more reliable)
        # from langchain_community.utilities import SerpAPIWrapper
        # search = SerpAPIWrapper()
        # results = search.run(f"{query} after:{two_days_ago}")

        return f"Recent news for '{query}':\n{results}"

    except Exception as e:
        return f"Error fetching news: {str(e)}"
```

**Tool 2: Fed Speeches Scraper**
```python
import requests
from bs4 import BeautifulSoup

@tool
def get_fed_speeches() -> str:
    """
    Get latest Federal Reserve speeches and statements from the last 7 days.
    Use this for the most recent Fed policy communications.

    Returns:
        Formatted list of recent Fed speeches with dates and key quotes
    """
    try:
        url = "https://www.federalreserve.gov/newsevents/speeches.htm"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        speeches = []
        # Parse the Fed's speeches page
        for item in soup.select('.row.eventlist')[:5]:  # Last 5 speeches
            date = item.select_one('.news-date').text.strip()
            title = item.select_one('.news-title a').text.strip()
            link = item.select_one('.news-title a')['href']

            speeches.append(f"- {date}: {title}\n  https://www.federalreserve.gov{link}")

        return "Recent Federal Reserve Speeches:\n" + "\n".join(speeches)

    except Exception as e:
        return f"Error fetching Fed speeches: {str(e)}"
```

**Tool 3: FOMC Statements**
```python
@tool
def get_latest_fomc_statement() -> str:
    """
    Get the most recent FOMC meeting statement.
    Use this for official Federal Reserve policy decisions.

    Returns:
        Latest FOMC statement with date and key policy changes
    """
    try:
        url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find latest statement
        latest = soup.select_one('.panel.panel-default')
        date = latest.select_one('.fomc-meeting__date').text.strip()
        statement_link = latest.select_one('a[href*="statement"]')['href']

        # Fetch full statement
        statement_response = requests.get(f"https://www.federalreserve.gov{statement_link}")
        statement_soup = BeautifulSoup(statement_response.content, 'html.parser')
        statement_text = statement_soup.select_one('.col-xs-12.col-sm-8.col-md-8').text.strip()

        return f"FOMC Statement ({date}):\n{statement_text[:2000]}..."  # Limit length

    except Exception as e:
        return f"Error fetching FOMC statement: {str(e)}"
```

**Integration with Agent:**
```python
tools = [
    get_economic_indicators,
    get_fed_policy_info,
    get_bond_yields,
    search_recent_macro_news,  # NEW
    get_fed_speeches,          # NEW
    get_latest_fomc_statement, # NEW
]
```

**Benefits:**
- ✅ Access to breaking news within minutes
- ✅ Real-time Fed communications
- ✅ Fills gaps in FRED data lag
- ✅ More comprehensive macro analysis
- ✅ Better answers to "recent" or "latest" questions

**Estimated Effort:** 2-3 hours

---

### Priority 4: Add Caching Layer

**Problem:** Re-fetching same data wastes API quota and slows responses.

**Solution:** Implement intelligent caching with TTL.

#### Implementation Details

**Option A: In-Memory Cache (Simple)**
```python
from functools import lru_cache
from datetime import datetime, timedelta
import hashlib

# Simple time-based cache
_cache = {}
_cache_timestamps = {}

def cached_tool(ttl_minutes: int):
    """Decorator for caching tool results with time-to-live."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hashlib.md5(str(args).encode()).hexdigest()}"

            # Check if cached and not expired
            if cache_key in _cache:
                cached_time = _cache_timestamps[cache_key]
                if datetime.now() - cached_time < timedelta(minutes=ttl_minutes):
                    print(f"Cache HIT: {cache_key}")
                    return _cache[cache_key]

            # Cache miss - execute function
            print(f"Cache MISS: {cache_key}")
            result = func(*args, **kwargs)

            # Store in cache
            _cache[cache_key] = result
            _cache_timestamps[cache_key] = datetime.now()

            return result
        return wrapper
    return decorator

# Apply to tools
@tool
@cached_tool(ttl_minutes=30)  # Cache for 30 minutes
def get_fed_policy_info() -> str:
    """Federal Reserve policy indicators (cached 30 min)"""
    # Original implementation
    pass

@tool
@cached_tool(ttl_minutes=60)  # Cache for 1 hour
def get_bond_yields() -> str:
    """US Treasury yields (cached 1 hour)"""
    # Original implementation
    pass

@tool
@cached_tool(ttl_minutes=5)  # Cache for 5 minutes
def search_recent_macro_news(query: str) -> str:
    """Recent news (cached 5 min)"""
    # Original implementation
    pass
```

**Option B: Redis Cache (Production)**
```python
import redis
import pickle
from datetime import timedelta

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=False  # We'll use pickle
)

def redis_cached_tool(ttl_minutes: int):
    """Decorator for Redis-based caching."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = f"macro_tool:{func.__name__}:{str(args)}"

            # Try to get from Redis
            cached = redis_client.get(cache_key)
            if cached:
                print(f"Redis HIT: {cache_key}")
                return pickle.loads(cached)

            # Execute function
            print(f"Redis MISS: {cache_key}")
            result = func(*args, **kwargs)

            # Store in Redis with TTL
            redis_client.setex(
                cache_key,
                timedelta(minutes=ttl_minutes),
                pickle.dumps(result)
            )

            return result
        return wrapper
    return decorator
```

**Cache Invalidation Endpoint:**
```python
@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Admin endpoint to clear cache manually."""
    _cache.clear()
    _cache_timestamps.clear()
    # Or: redis_client.flushdb()
    return jsonify({"status": "cache cleared"})
```

**Recommended TTL by Tool:**
- `get_fed_policy_info`: 30-60 minutes (updates infrequently)
- `get_bond_yields`: 15-30 minutes (updates throughout trading day)
- `get_economic_indicators`: 24 hours (FRED data updates daily)
- `search_recent_macro_news`: 5-10 minutes (news changes frequently)
- `get_fed_speeches`: 12 hours (speeches are scheduled events)

**Benefits:**
- ✅ Reduces API calls by 70-90%
- ✅ Faster response times (cached queries < 100ms)
- ✅ Saves API quota costs
- ✅ Reduces rate limit issues
- ✅ Better user experience

**Estimated Effort:** 2-3 hours

---

### Priority 5: Add Multi-Turn Conversation Support

**Problem:** Each query is isolated, can't reference previous analyses or ask follow-ups.

**Solution:** Add session-based conversation memory.

#### Implementation Details

**Backend Session Storage:**
```python
from datetime import datetime
import uuid

# In-memory session storage (use Redis in production)
sessions = {}

class MacroConversation:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

    def add_message(self, role: str, content: str):
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity = datetime.now()

    def get_context(self, max_messages: int = 5) -> str:
        """Get recent conversation context for LLM."""
        recent = self.history[-max_messages:]
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])

# Modified endpoint
@app.route('/api/analyze/macro', methods=['POST'])
def analyze_macro():
    data = request.get_json()
    topic = data.get('topic', '')
    session_id = data.get('session_id', str(uuid.uuid4()))

    # Get or create session
    if session_id not in sessions:
        sessions[session_id] = MacroConversation(session_id)

    conversation = sessions[session_id]

    # Add user message to history
    conversation.add_message("user", topic)

    # Include conversation context in agent prompt
    context = conversation.get_context()
    enhanced_topic = f"""Previous conversation:
{context}

Current question: {topic}
"""

    # Run agent with context
    result = analyze_macro_with_agent(enhanced_topic)

    # Add assistant response to history
    conversation.add_message("assistant", result)

    return jsonify({
        'analysis': result,
        'session_id': session_id,
        'message_count': len(conversation.history)
    })
```

**Frontend Updates:**
```javascript
// index.html - Add session tracking
let macroSessionId = null;

async function analyzeMacro() {
    const topic = document.getElementById('macroTopic').value.trim();

    const response = await fetch('/api/analyze/macro', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            topic: topic,
            session_id: macroSessionId  // Include session ID
        })
    });

    const data = await response.json();
    macroSessionId = data.session_id;  // Store session ID

    // Display analysis with conversation context
    displayMacroResult(data.analysis, data.message_count);
}

// Add "Clear Conversation" button
function clearMacroConversation() {
    macroSessionId = null;
    document.getElementById('macroResult').innerHTML = '';
    showStatus('Conversation cleared', 'success');
}
```

**Session Cleanup:**
```python
import threading
import time

def cleanup_old_sessions():
    """Background task to remove inactive sessions."""
    while True:
        time.sleep(3600)  # Run every hour
        now = datetime.now()
        to_remove = []

        for session_id, conversation in sessions.items():
            # Remove sessions inactive for > 2 hours
            if (now - conversation.last_activity).seconds > 7200:
                to_remove.append(session_id)

        for session_id in to_remove:
            del sessions[session_id]
            print(f"Cleaned up session: {session_id}")

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
cleanup_thread.start()
```

**Benefits:**
- ✅ Can ask follow-up questions ("What about employment?")
- ✅ Can reference previous analyses ("Compare this to what you said earlier")
- ✅ More natural conversation flow
- ✅ Better user experience
- ✅ Context-aware responses

**Estimated Effort:** 3-4 hours

---

## Implementation Roadmap

### Phase 1: LLM Validation & Tool Selection (Week 1)
**Duration:** 2-3 hours
**Priority:** HIGH

**Tasks:**
1. Create validator LLM function
2. Create tool selection LLM function
3. Modify `analyze_macro_with_tools()` to use validators
4. Add error handling for invalid questions
5. Test with edge cases (nonsense questions, off-topic queries)

**Success Criteria:**
- [ ] Validator correctly identifies off-topic questions
- [ ] Tool selection reduces average tools called from 3-4 to 1-2
- [ ] Response time improves by 30-40%
- [ ] API costs decrease by similar margin

**Testing:**
```python
# Test cases
test_cases = [
    ("What's the current inflation outlook?", True, ["get_economic_indicators"]),
    ("Tell me a joke", False, []),
    ("How do Fed policy and yields compare?", True, ["get_fed_policy_info", "get_bond_yields"]),
    ("What's the weather in NYC?", False, []),
]
```

---

### Phase 2: Convert to ReAct Agent (Week 1-2)
**Duration:** 4-5 hours
**Priority:** HIGH

**Tasks:**
1. Install LangChain agent dependencies
2. Create ReAct prompt template
3. Implement `create_react_agent` with existing tools
4. Add AgentExecutor with safeguards (max iterations, timeout)
5. Handle parsing errors gracefully
6. Test multi-step reasoning scenarios

**Success Criteria:**
- [ ] Agent successfully uses tools autonomously
- [ ] Can handle complex multi-step questions
- [ ] Reasoning is visible and logical
- [ ] Errors are handled gracefully
- [ ] No infinite loops or timeouts

**Testing:**
```python
# Test complex questions
test_questions = [
    "Given recent inflation trends, what should the Fed do next?",
    "Compare current economic conditions to pre-2008 crisis",
    "What's driving the yield curve inversion?",
]
```

---

### Phase 3: Add Web Search Tools (Week 2)
**Duration:** 2-3 hours
**Priority:** MEDIUM

**Tasks:**
1. Implement `search_recent_macro_news()` tool
2. Implement `get_fed_speeches()` scraper
3. Implement `get_latest_fomc_statement()` scraper
4. Add error handling for network failures
5. Test with real-time events
6. Add rate limiting to avoid bans

**Success Criteria:**
- [ ] Successfully fetches recent news (<48 hours old)
- [ ] Fed speeches scraper works reliably
- [ ] FOMC statements are parsed correctly
- [ ] Graceful fallback if scraping fails
- [ ] No performance degradation

**Testing:**
```python
# Test real-time scenarios
test_scenarios = [
    "What did the Fed announce today?",
    "Any breaking inflation news?",
    "What's the latest from Powell?",
]
```

---

### Phase 4: Implement Caching (Week 2-3)
**Duration:** 2-3 hours
**Priority:** MEDIUM

**Tasks:**
1. Implement in-memory cache decorator
2. Add TTL configuration per tool
3. Create cache invalidation endpoint
4. Add cache hit/miss logging
5. Test cache behavior under load
6. (Optional) Migrate to Redis for production

**Success Criteria:**
- [ ] Cache hit rate > 60% for repeated queries
- [ ] Response time for cached queries < 200ms
- [ ] API calls reduced by 70%+
- [ ] No stale data issues
- [ ] Cache invalidation works correctly

**Monitoring:**
```python
# Track cache performance
cache_stats = {
    "hits": 0,
    "misses": 0,
    "hit_rate": lambda: hits / (hits + misses)
}
```

---

### Phase 5: Multi-Turn Conversations (Week 3)
**Duration:** 3-4 hours
**Priority:** LOW

**Tasks:**
1. Implement session storage backend
2. Add conversation history tracking
3. Modify API to accept/return session IDs
4. Update frontend to maintain session state
5. Add session cleanup background task
6. Test conversation context handling

**Success Criteria:**
- [ ] Follow-up questions work correctly
- [ ] Context is maintained across queries
- [ ] Old sessions are cleaned up
- [ ] No memory leaks
- [ ] Sessions survive server restart (if using Redis)

**Testing:**
```python
# Test multi-turn scenarios
conversation = [
    "What's the current inflation rate?",
    "How does that compare to last quarter?",  # Should reference previous
    "What should the Fed do about it?",        # Should use both contexts
]
```

---

## Technical Architecture (After Implementation)

### New Data Flow

```
User Question
    ↓
Frontend (with session_id)
    ↓
POST /api/analyze/macro
    ↓
Session Management (load conversation history)
    ↓
Validation LLM
    ├─→ Invalid → Return error message
    └─→ Valid → Continue
        ↓
    ReAct Agent (with conversation context)
        ↓
    Iterative Tool Selection & Execution
        ├─→ Cache Check → Cache HIT → Return cached data
        ├─→ Cache Check → Cache MISS → Execute tool
        │       ↓
        │   get_economic_indicators (TTL: 24h)
        │   get_fed_policy_info (TTL: 30min)
        │   get_bond_yields (TTL: 15min)
        │   search_recent_macro_news (TTL: 5min) [NEW]
        │   get_fed_speeches (TTL: 12h) [NEW]
        │   get_latest_fomc_statement (TTL: 6h) [NEW]
        │       ↓
        │   Store in cache
        │       ↓
        └─→ Return tool result
            ↓
    Agent Synthesis (ReAct loop may repeat)
        ↓
    Final Answer
        ↓
    Save to conversation history
        ↓
    Return to user (with session_id)
```

### Tool Inventory (Updated)

| Tool | Data Source | Cache TTL | Use Case |
|------|-------------|-----------|----------|
| get_economic_indicators | FRED API | 24h | Historical economic data |
| get_fed_policy_info | FRED API | 30min | Fed policy stance |
| get_bond_yields | FRED API | 15min | Treasury yields |
| get_financial_metrics | Yahoo Finance | 1h | Stock fundamentals |
| get_market_news | Finnhub/Polygon | 10min | Market news |
| get_technical_indicators | Yahoo Finance | 15min | Technical analysis |
| **search_recent_macro_news** | **Web Search** | **5min** | **Breaking news** |
| **get_fed_speeches** | **Fed Website** | **12h** | **Fed communications** |
| **get_latest_fomc_statement** | **Fed Website** | **6h** | **FOMC decisions** |

---

## Monitoring & Success Metrics

### Key Performance Indicators (KPIs)

**Response Quality:**
- [ ] Accuracy: % of questions answered correctly (manual review)
- [ ] Relevance: % of questions where selected tools were appropriate
- [ ] Completeness: % of questions fully answered (vs partial)

**Performance:**
- [ ] Average response time: Target < 5 seconds (from < 10s currently)
- [ ] Cache hit rate: Target > 60%
- [ ] API calls per query: Target < 2 (from 3-4 currently)
- [ ] Error rate: Target < 2%

**Cost:**
- [ ] API costs: Target 50% reduction via caching
- [ ] LLM token usage: Track validator + agent + synthesis tokens
- [ ] Rate limit hits: Target 0 per day

### Logging & Observability

```python
import logging
from datetime import datetime

# Enhanced logging
logger = logging.getLogger('macro_analysis')

def log_query_metrics(query, validation, tools_used, response_time, cache_hits):
    logger.info({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "is_valid": validation["is_valid"],
        "question_type": validation["question_type"],
        "tools_used": tools_used,
        "num_tools": len(tools_used),
        "response_time_ms": response_time * 1000,
        "cache_hits": cache_hits,
        "cache_hit_rate": cache_hits / len(tools_used) if tools_used else 0
    })
```

---

## Risk Mitigation

### Identified Risks

1. **LLM Hallucination in Tool Selection**
   - **Risk:** Validator/planner LLM selects wrong tools or hallucinates tool names
   - **Mitigation:**
     - Strict JSON schema validation
     - Whitelist of allowed tool names
     - Fallback to keyword matching if LLM fails

2. **Agent Loops/Timeouts**
   - **Risk:** ReAct agent gets stuck in infinite reasoning loop
   - **Mitigation:**
     - `max_iterations=5` hard limit
     - `max_execution_time=30` second timeout
     - `handle_parsing_errors=True` for graceful failures

3. **Cache Staleness**
   - **Risk:** Cached data becomes outdated, misleading users
   - **Mitigation:**
     - Conservative TTL settings
     - Clear cache timestamp in responses
     - Manual cache invalidation endpoint
     - Option to force fresh data

4. **Web Scraping Failures**
   - **Risk:** Fed website changes structure, breaking scrapers
   - **Mitigation:**
     - Robust error handling
     - Fallback to alternative sources
     - Regular testing of scrapers
     - Alert on consecutive failures

5. **Rate Limits**
   - **Risk:** Too many API calls hit rate limits
   - **Mitigation:**
     - Aggressive caching
     - Exponential backoff with retries
     - Track API quota usage
     - Alert at 80% quota

6. **Session Memory Bloat**
   - **Risk:** Conversation history consumes too much memory
   - **Mitigation:**
     - Limit history to last 5 messages
     - Background cleanup of old sessions
     - Max session duration (2 hours)
     - Consider Redis for production

---

## Future Enhancements (Beyond Scope)

### Advanced Features

1. **Multi-Modal Analysis**
   - Chart generation for economic trends
   - Image-based Fed chart analysis
   - PDF parsing of Fed reports

2. **Comparative Analysis**
   - "Compare current conditions to 2008 crisis"
   - Cross-country macro comparisons
   - Historical pattern matching

3. **Predictive Modeling**
   - Recession probability models
   - Inflation forecasting
   - Rate hike prediction

4. **Custom Alerts**
   - Alert when inflation crosses threshold
   - Notify on Fed policy changes
   - Custom watchlists

5. **Export/Sharing**
   - PDF report generation
   - Email analysis summaries
   - Shareable analysis links

---

## Conclusion

This improvement plan transforms the Global Macro analysis feature from a **simple data aggregator** to an **intelligent agentic system** with:

✅ **LLM-based validation and tool selection** (replacing brittle keyword matching)
✅ **True autonomous agent behavior** (iterative reasoning with ReAct)
✅ **Real-time information access** (web search + Fed scraping)
✅ **Performance optimization** (intelligent caching)
✅ **Enhanced user experience** (multi-turn conversations)

**Total Estimated Effort:** 15-18 hours across 3 weeks

**Recommended Start:** Phase 1 (LLM Validation) - provides immediate value with minimal effort

**Questions?** Ready to begin implementation when you are.
