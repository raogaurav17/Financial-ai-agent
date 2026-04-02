# Financial AI Agent

Production-oriented FastAPI service that combines a tool-using LLM agent with quantitative finance utilities for market intelligence, portfolio analytics, and risk-aware decision support.

## Executive Summary

This project implements an AI-powered financial assistant capable of:

- Retrieving market history, key metrics, and recent news for equities.
- Estimating risk using annualized volatility and daily Value at Risk (VaR).
- Forecasting short-term trends with ARIMA time-series modeling.
- Running Monte Carlo portfolio simulations (premium tier).
- Rebalancing portfolios via Sharpe-ratio optimization (premium tier).
- Streaming intermediate tool outputs and final responses over SSE.

The architecture is designed for extensibility: tools are registered once, orchestrated by LangGraph, and exposed through clean FastAPI endpoints.

## Core Capabilities

### 1. Market Intelligence

- Data source: Yahoo Finance via `yfinance`.
- Outputs:
   - Three-month historical summary
   - Fundamental metrics (P/E, EPS, market cap, 52-week levels, dividend yield)
   - Latest headline snapshot

### 2. Risk Analytics

- Methodology:
   - Daily returns from 1-year adjusted close history
   - Annualized volatility: $\sigma_{annual} = \sigma_{daily} \times \sqrt{252}$
   - Daily VaR (95%): empirical 5th percentile of returns

### 3. Trend Prediction

- Model: ARIMA(5,1,0) from `statsmodels`.
- Horizon: 30 trading-day forecast.
- Classification: Upward/Downward based on forecast endpoint vs current price.

### 4. Portfolio Simulation (Premium)

- Technique: Monte Carlo simulation with equal-weight assumption.
- Inputs: ticker list + simulation count.
- Outputs:
   - Expected annualized return
   - Annualized risk (standard deviation)

### 5. Portfolio Rebalancing (Premium)

- Optimization objective: maximize Sharpe ratio.
- Solver: `scipy.optimize.minimize` (SLSQP).
- Constraints:
   - Fully invested ($\sum w_i = 1$)
   - Long-only bounds ($0 \le w_i \le 1$)

## Architecture

### Request Flow

1. FastAPI endpoint receives natural-language query.
2. API key dependency resolves user role (`standard` or `premium`).
3. LangGraph agent decides whether to call tools.
4. Tool node executes selected finance function.
5. Agent synthesizes the final response.
6. API returns either:
    - Single JSON response (`/query`), or
    - Incremental SSE events (`/stream/query`).

### Key Components

- API layer: FastAPI + Pydantic
- Agent orchestration: LangGraph
- Tool calling: LangChain tools
- LLM provider: Google Gemini (`langchain-google-genai`)
- Quant stack: NumPy, Pandas, SciPy, Statsmodels
- Market data provider: Yahoo Finance

## Technology Stack

- Python 3.12 (container baseline; 3.9+ generally compatible)
- FastAPI / Uvicorn
- LangChain / LangGraph
- Google Gemini API
- NumPy, Pandas, SciPy, Statsmodels
- yfinance

## Project Structure

```text
.
├── main.py           # FastAPI app, LangGraph workflow, tool definitions
├── requirements.txt  # Pinned Python dependencies (UTF-16 LE encoded)
├── Dockerfile        # Container build and runtime command
└── README.md
```

## Prerequisites

1. Python 3.9+
2. A Google API key with access to Gemini models
3. Git (for source checkout)

## Environment Variables

Create a `.env` file in the repository root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

The service fails fast at startup if `GOOGLE_API_KEY` is missing.

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/raogaurav17/Financial-ai-agent.git
cd Financial-ai-agent
```

### 2. Create and Activate Virtual Environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the API

Option A (direct script entrypoint):

```bash
python main.py
```

Option B (recommended for local dev with auto-reload):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access API Docs

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Authentication and Access Tiers

Current implementation uses static API keys in request headers:

- Standard: `my-secret-standard-key`
- Premium: `my-secret-premium-key`

Header format:

```http
x-api-key: <your-key>
```

### Feature Access Matrix

| Capability | Standard | Premium |
|---|---:|---:|
| Market data | Yes | Yes |
| Risk assessment | Yes | Yes |
| Trend prediction | Yes | Yes |
| Portfolio simulation | No | Yes |
| Portfolio rebalancing | No | Yes |

Premium-only intents (`simulate`, `rebalance`) return `403 Forbidden` for standard keys.

## API Reference

### POST `/query`

Runs the full agent cycle and returns a single consolidated response.

Request body:

```json
{
   "query": "fetch market data for AAPL"
}
```

Example cURL:

```bash
curl -X POST "http://localhost:8000/query" \
   -H "Content-Type: application/json" \
   -H "x-api-key: my-secret-standard-key" \
   -d '{"query":"assess risk for TSLA"}'
```

Successful response shape:

```json
{
   "response": "...agent response text..."
}
```

### POST `/stream/query`

Streams intermediate events and final completion via Server-Sent Events.

Example cURL:

```bash
curl -N -X POST "http://localhost:8000/stream/query" \
   -H "Content-Type: application/json" \
   -H "x-api-key: my-secret-premium-key" \
   -d '{"query":"simulate portfolio with AAPL, MSFT, NVDA"}'
```

SSE event payload types:

- `response`: agent textual output
- `tool_result`: raw tool output
- `done`: stream completion signal

## Example Prompts

- `fetch market data for AAPL`
- `assess risk for TSLA`
- `predict trends for GOOGL`
- `simulate portfolio with AAPL, TSLA, MSFT`
- `rebalance portfolio with AMZN, MSFT, NVDA`

## Docker Deployment

Build image:

```bash
docker build -t financial-ai-agent:latest .
```

Run container:

```bash
docker run --rm -p 8000:8000 \
   --env GOOGLE_API_KEY=your_google_api_key_here \
   financial-ai-agent:latest
```

## Error Handling

Common HTTP statuses:

- `401 Unauthorized`: missing or invalid API key
- `403 Forbidden`: premium-only operation attempted with standard key
- `500 Internal Server Error`: downstream/API/model/tool execution failure

Tool-level failures are returned as structured error messages from the tool wrappers.

## Performance and Operational Notes

- Market and news calls are network-bound and subject to upstream latency.
- ARIMA training and Monte Carlo simulation are compute-heavy relative to simple lookups.
- Uvicorn worker count should be tuned based on CPU and memory profile.
- For production hardening, add request-level timeouts, retries, and observability.

## Security Considerations

Current auth is intentionally minimal for prototype speed. For production rollout, prioritize:

1. JWT or OAuth2 with rotating secrets
2. Tenant-aware key management
3. Rate limiting and abuse protection
4. Structured audit logging
5. Secret management via vault/KMS

## Recommended Production Enhancements

1. Add integration and load tests for agent + tool orchestration.
2. Introduce caching for repeated market/risk requests.
3. Move long-running simulations to background workers.
4. Add response schemas for tool outputs and typed contracts.
5. Externalize role and entitlement logic to a policy layer.
