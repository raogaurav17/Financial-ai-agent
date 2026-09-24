# Financial AI Agent

Financial AI Agent is a full-stack financial assistant that combines
deterministic quantitative services with an optional Google Gemini agent.

The application is intentionally designed so common finance operations do not
consume an LLM call:

```text
Structured request
    -> finance service
    -> typed result
    -> concise response
```

The LLM is reserved for explanations, comparisons, summaries, recommendations,
and ambiguous multi-step questions.

## Features

- Market quotes and three-month historical prices
- Fundamental data and recent headlines through Yahoo Finance
- Annualized volatility and empirical 95% daily VaR
- 30-trading-day ARIMA trend forecasts
- Monte Carlo portfolio simulation
- Sharpe-ratio portfolio rebalancing
- Standard and premium API access tiers
- JSON responses and Server-Sent Events (SSE)
- Next.js dashboard branded as **Financial AI Agent**

## Project Structure

```text
.
├── backend/
│   ├── main.py             # Thin FastAPI application and route mapping
│   ├── config.py           # Environment configuration and development defaults
│   ├── models.py           # Request and response types
│   ├── auth.py             # Development API-key authentication
│   ├── finance.py          # Shared deterministic finance services and tools
│   ├── agent.py            # Optional, lazily initialized LangGraph/Gemini agent
│   ├── pyproject.toml      # Python project dependencies
│   ├── requirements.txt
│   └── .env.example        # Create locally; never commit secrets
├── frontend/
│   ├── app/
│   │   ├── page.tsx        # Dashboard and query experience
│   │   ├── layout.tsx      # Metadata and root layout
│   │   └── globals.css     # Application styling
│   ├── package.json
│   └── .env.example
└── README.md
```

## Architecture

### Deterministic routing

The backend recognizes direct requests for:

- Quotes
- Historical data
- Risk, volatility, and VaR
- Trend forecasts
- Portfolio simulation
- Portfolio rebalancing

These requests call typed finance services directly. The `/query` endpoint
returns both a concise `response` and a structured `result`.

### LLM routing

Requests containing explanation, comparison, summary, recommendation, or
multi-step reasoning language are sent through LangGraph and Gemini. The agent
can call the existing finance tools and then explain the result.

## Requirements

- Python 3.11+
- Node.js 18+
- A Google Gemini API key for LLM-backed requests (optional for deterministic requests)
- Network access to Yahoo Finance

## Backend Setup

From the repository root:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env`:

```env
GOOGLE_API_KEY=your_google_api_key_here
# Optional overrides; development defaults are used when omitted.
STANDARD_API_KEY=my-secret-standard-key
PREMIUM_API_KEY=my-secret-premium-key
```

Start the API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Available API documentation:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>

## Frontend Setup

In a second terminal, from the repository root:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open <http://localhost:3000>.

The frontend uses:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

The dashboard sends the configured API key using the `x-api-key` header. The
default key in the UI is the standard development key; premium operations
require the premium key.

## Authentication and Access Tiers

The current development authentication uses static API keys:

| Role | API key | Access |
|---|---|---|
| Standard | `my-secret-standard-key` | Market data, risk, and trends |
| Premium | `my-secret-premium-key` | All standard features plus simulation and rebalancing |

Header format:

```http
x-api-key: my-secret-standard-key
```

This authentication is suitable only for local development. Use managed
identity, rotating secrets, and tenant-aware authorization in production.

## API Reference

### `POST /query`

Request:

```json
{
  "query": "assess risk for TSLA"
}
```

Example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: my-secret-standard-key" \
  -d '{"query":"assess risk for TSLA"}'
```

Deterministic response:

```json
{
  "response": "TSLA annualized volatility is 42.10%; 95% daily VaR is -3.24%.",
  "result": {
    "ticker": "TSLA",
    "annualized_volatility": 0.421,
    "value_at_risk_95": -0.0324
  }
}
```

LLM-backed responses include `response` and omit `result`.

### `POST /stream/query`

Streams SSE events for either direct finance results or agent execution:

```bash
curl -N -X POST http://localhost:8000/stream/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: my-secret-premium-key" \
  -d '{"query":"simulate portfolio with AAPL, MSFT, NVDA"}'
```

Event types:

- `result`: structured deterministic finance result
- `response`: concise or agent-generated text
- `tool_result`: intermediate tool output from the agent
- `done`: stream completion

## Example Queries

```text
fetch market data for AAPL
get historical data for MSFT
assess risk for TSLA
predict trends for GOOGL
simulate portfolio with AAPL, TSLA, MSFT
rebalance portfolio with AMZN, MSFT, NVDA
Explain why NVDA has been volatile
Compare the risk of AAPL and MSFT
```

## Error Handling

- `401`: missing or invalid API key
- `403`: premium operation requested with a standard key
- `400`: missing, blank, or overlong query
- `502`: finance provider or known service failure
- `503`: an LLM request was made without `GOOGLE_API_KEY`

Finance-provider failures are surfaced by the API rather than silently
converted into successful-looking responses.

## Development Commands

Backend:

```bash
cd backend
python -m py_compile main.py
```

Frontend:

```bash
cd frontend
npm run build
```

## Production Considerations

- Replace static API keys with OAuth2/JWT or a managed identity provider.
- Add request timeouts, retries, caching, and structured audit logging.
- Move long-running simulations to background workers.
- Add integration tests for Yahoo Finance, deterministic routing, agent calls,
  and SSE event ordering.
- Keep `.env` files and API keys out of source control.
