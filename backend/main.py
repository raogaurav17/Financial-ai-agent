import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, TypedDict, Literal, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
import uvicorn
import yfinance as yf
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from scipy.optimize import minimize

# --- Environment and API Key Setup ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")


# A simple placeholder for API Key verification. (Will use jwt and db for authentication in future)
async def verify_api_key(request: Request):
    """A dummy dependency to simulate API key verification."""
    api_key = request.headers.get("x-api-key")
    if api_key == "my-secret-premium-key":
        return {"role": "premium"}
    if api_key == "my-secret-standard-key":
        return {"role": "standard"}
    raise HTTPException(status_code=401, detail="Invalid API Key")


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str


class QuoteResult(TypedDict):
    ticker: str
    price: float
    currency: str
    as_of: str


class HistoricalResult(TypedDict):
    ticker: str
    period: str
    rows: List[Dict[str, Any]]


class RiskResult(TypedDict):
    ticker: str
    annualized_volatility: float
    value_at_risk_95: float


class TrendResult(TypedDict):
    ticker: str
    current_price: float
    forecast_price: float
    predicted_trend: Literal["Upward", "Downward"]


class SimulationResult(TypedDict):
    portfolio: List[str]
    expected_annual_return: float
    annual_risk: float


class RebalanceResult(TypedDict):
    portfolio: List[str]
    weights: Dict[str, float]


DeterministicResult = Union[
    QuoteResult, HistoricalResult, RiskResult, TrendResult,
    SimulationResult, RebalanceResult,
]


@dataclass(frozen=True)
class DeterministicRequest:
    operation: Literal["quote", "historical", "risk", "trend", "simulate", "rebalance"]
    tickers: List[str]
    simulations: int = 10000


def _download_prices(tickers: List[str], period: str = "1y", field: str = "Adj Close") -> pd.DataFrame:
    """Download normalized price data for one or more tickers."""
    data = yf.download(tickers, period=period, progress=False, auto_adjust=False)[field]
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.dropna(how="all")


def _single_price_series(ticker: str, field: str = "Close") -> pd.Series:
    """Return a one-dimensional price series for a ticker."""
    data = _download_prices([ticker], field=field)
    series = data[ticker]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series.dropna()


def _extract_tickers(query: str) -> List[str]:
    """Extract likely ticker symbols while ignoring common request words."""
    ignored = {
        "A", "AN", "AND", "AS", "AT", "DATA", "FOR", "FETCH", "FROM", "GET",
        "HISTORICAL", "I", "IN", "IS", "LATEST", "MARKET", "OF", "ON", "OR",
        "PRICE", "QUOTE", "RISK", "SHOW", "THE", "TO", "TREND", "VAR", "WHAT",
        "WITH", "CURRENT", "LOOKUP",
    }
    candidates = re.findall(r"\b[A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?\b", query)
    return list(dict.fromkeys(
        ticker.upper() for ticker in candidates if ticker.upper() not in ignored
    ))


def parse_deterministic_request(query: str) -> DeterministicRequest | None:
    """Return a direct finance operation, or None when an LLM is appropriate."""
    normalized = query.lower()
    explanation_words = (
        "explain", "why", "compare", "comparison", "summarize", "summary",
        "recommend", "should i", "and", "then", "because",
    )
    if any(
        (f" {word} " in f" {normalized} " if " " in word else
         re.search(rf"\b{re.escape(word)}\b", normalized))
        for word in explanation_words
    ):
        return None

    if "rebalance" in normalized:
        operation = "rebalance"
    elif any(word in normalized for word in ("simulate", "monte carlo")):
        operation = "simulate"
    elif any(word in normalized for word in ("volatility", "var", "risk")):
        operation = "risk"
    elif any(word in normalized for word in ("trend", "forecast", "predict")):
        operation = "trend"
    elif any(word in normalized for word in ("historical", "history")):
        operation = "historical"
    elif any(word in normalized for word in ("quote", "price", "market data")):
        operation = "quote"
    else:
        return None

    tickers = _extract_tickers(query)
    if not tickers:
        return None
    simulations_match = re.search(r"\b(\d{2,7})\s*(?:simulations?|runs?)\b", normalized)
    simulations = int(simulations_match.group(1)) if simulations_match else 10000
    return DeterministicRequest(operation, tickers, simulations)


def execute_deterministic_request(request: DeterministicRequest) -> DeterministicResult:
    """Execute a parsed request without invoking the LLM."""
    ticker = request.tickers[0]
    if request.operation == "quote":
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        return {
            "ticker": ticker,
            "price": float(info.last_price),
            "currency": str(info.currency),
            "as_of": pd.Timestamp.now(tz="UTC").isoformat(),
        }
    if request.operation == "historical":
        data = _download_prices([ticker], period="3mo", field="Close")
        rows = [
            {"date": index.isoformat(), "close": float(value)}
            for index, value in data[ticker].dropna().items()
        ]
        return {"ticker": ticker, "period": "3mo", "rows": rows}
    if request.operation == "risk":
        data = _download_prices([ticker])[ticker].pct_change().dropna()
        if data.empty:
            raise ValueError(f"No data for {ticker}")
        return {
            "ticker": ticker,
            "annualized_volatility": float(data.std() * np.sqrt(252)),
            "value_at_risk_95": float(np.percentile(data, 5)),
        }
    if request.operation == "trend":
        data = _single_price_series(ticker).reset_index(drop=True)
        if data.empty:
            raise ValueError(f"No data for {ticker}")
        model = sm.tsa.ARIMA(data, order=(5, 1, 0)).fit()
        forecast = float(model.forecast(steps=30).iloc[-1])
        current = float(data.iloc[-1])
        return {
            "ticker": ticker,
            "current_price": current,
            "forecast_price": forecast,
            "predicted_trend": "Upward" if forecast > current else "Downward",
        }
    if request.operation == "simulate":
        if len(request.tickers) < 2:
            raise ValueError("Portfolio simulation requires at least two tickers.")
        data = _download_prices(request.tickers)[request.tickers].pct_change().dropna()
        mean_returns = data.mean().to_numpy()
        cov_matrix = data.cov().to_numpy()
        draws = np.random.multivariate_normal(
            mean_returns, cov_matrix, size=request.simulations
        )
        portfolio_returns = draws.mean(axis=1)
        return {
            "portfolio": request.tickers,
            "expected_annual_return": float(portfolio_returns.mean() * 252),
            "annual_risk": float(portfolio_returns.std() * np.sqrt(252)),
        }
    if len(request.tickers) < 2:
        raise ValueError("Portfolio rebalancing requires at least two tickers.")
    data = _download_prices(request.tickers)[request.tickers].pct_change().dropna()
    mean_returns = data.mean().to_numpy() * 252
    cov_matrix = data.cov().to_numpy() * 252
    count = len(request.tickers)

    def objective(weights: np.ndarray) -> float:
        portfolio_return = float(np.sum(mean_returns * weights))
        volatility = float(np.sqrt(weights.T @ cov_matrix @ weights))
        return -portfolio_return / volatility if volatility else 0.0

    result = minimize(
        objective, [1 / count] * count, method="SLSQP",
        bounds=tuple((0.0, 1.0) for _ in range(count)),
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
    )
    if not result.success:
        raise ValueError(f"Rebalancing failed: {result.message}")
    return {
        "portfolio": request.tickers,
        "weights": {
            ticker: float(weight)
            for ticker, weight in zip(request.tickers, result.x)
        },
    }


def format_deterministic_result(result: DeterministicResult) -> str:
    """Provide a small explanation without spending an LLM call."""
    if "ticker" in result and "price" in result:
        return f"{result['ticker']} is trading at {result['price']:.2f} {result['currency']}."
    if "annualized_volatility" in result:
        return (
            f"{result['ticker']} annualized volatility is "
            f"{result['annualized_volatility']:.2%}; 95% daily VaR is "
            f"{result['value_at_risk_95']:.2%}."
        )
    if "predicted_trend" in result:
        return (
            f"{result['ticker']} has a predicted {result['predicted_trend'].lower()} "
            f"trend over the next 30 trading days."
        )
    if "expected_annual_return" in result:
        return (
            f"Expected annual return is {result['expected_annual_return']:.2%} "
            f"with annual risk of {result['annual_risk']:.2%}."
        )
    if "weights" in result:
        weights = ", ".join(
            f"{ticker}: {weight:.2%}" for ticker, weight in result["weights"].items()
        )
        return f"Suggested Sharpe-optimized allocation: {weights}."
    return f"Retrieved {len(result['rows'])} historical observations for {result['ticker']}."


# --- Financial Tools ---
@tool
def fetch_market_data(ticker: str) -> Dict[str, Any]:
    """Fetches historical data, key financial metrics, and news from Yahoo Finance for a stock ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="3mo").reset_index().to_string()
        metrics = {
            "pe_ratio": info.get("trailingPE"), "eps": info.get("trailingEps"),
            "market_cap": info.get("marketCap"), "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"), "dividend_yield": info.get("dividendYield"),
        }
        news = "\n".join([f"- {item['title']}" for item in stock.news[:3]])
        return {"history_summary": hist, "key_metrics": metrics, "latest_news": news}
    except Exception as e:
        return {"error": f"Failed to fetch data for {ticker}: {str(e)}"}


@tool
def assess_risk(ticker: str) -> Dict[str, Any]:
    """Calculates annualized volatility and Value at Risk (VaR) for a stock."""
    try:
        data = _single_price_series(ticker, field="Adj Close").pct_change().dropna()
        if data.empty: return {"error": f"No data for {ticker}"}
        volatility = data.std() * np.sqrt(252)
        var_95 = np.percentile(data, 5)
        return {
            "annualized_volatility": f"{volatility:.2%}",
            "value_at_risk_95": f"{var_95:.2%}",
            "interpretation": "Volatility measures price fluctuation. VaR is the potential loss in a day with 95% confidence."
        }
    except Exception as e:
        return {"error": f"Risk assessment failed for {ticker}: {str(e)}"}


@tool
def predict_trends(ticker: str) -> Dict[str, Any]:
    """Predicts future stock price trends using an ARIMA model."""
    try:
        data = _single_price_series(ticker).reset_index(drop=True)
        if data.empty: return {"error": f"No data for {ticker}"}
        model = sm.tsa.ARIMA(data, order=(5, 1, 0)).fit()
        forecast = model.forecast(steps=30)
        current_price = float(data.iloc[-1])
        forecast_price = float(forecast.iloc[-1])
        trend = "Upward" if forecast_price > current_price else "Downward"
        return {
            "current_price": current_price,
            "30_day_forecast_price": forecast_price,
            "predicted_trend": trend
        }
    except Exception as e:
        return {"error": f"Prediction failed for {ticker}: {str(e)}"}


@tool
def simulate_portfolio(tickers: List[str], simulations: int = 10000) -> Dict[str, Any]:
    """Performs a Monte Carlo simulation to estimate future portfolio returns and risk."""
    try:
        if not tickers: return {"error": "Ticker list cannot be empty."}
        data = yf.download(tickers, period="1y", progress=False)['Adj Close'].pct_change().dropna()
        mean_returns = data.mean()
        cov_matrix = data.cov()

        # Assume equal weights for simulation
        weights = np.array([1 / len(tickers)] * len(tickers))

        # Simulate returns
        simulated_returns = []
        for _ in range(simulations):
            sim_return = np.sum(mean_returns + np.random.multivariate_normal(np.zeros(len(tickers)), cov_matrix))
            simulated_returns.append(sim_return)

        expected_return = np.mean(simulated_returns) * 252
        risk = np.std(simulated_returns) * np.sqrt(252)

        return {
            "portfolio": ", ".join(tickers),
            "expected_annual_return": f"{expected_return:.2%}",
            "annual_risk_(std_dev)": f"{risk:.2%}"
        }
    except Exception as e:
        return {"error": f"Simulation failed: {str(e)}"}


@tool
def rebalance_portfolio(tickers: List[str]) -> Dict[str, Any]:
    """Optimizes a portfolio for the best risk-adjusted return (Sharpe Ratio)."""
    try:
        if not tickers: return {"error": "Ticker list cannot be empty."}
        data = yf.download(tickers, period="1y", progress=False)['Adj Close']
        returns = data.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(tickers)

        def objective(weights):  # Maximize Sharpe Ratio (minimize negative Sharpe)
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_volatility

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        initial_weights = num_assets * [1. / num_assets]

        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = {ticker: f"{weight * 100:.2f}%" for ticker, weight in zip(tickers, result.x)}
        return {"sharpe_optimized_weights": optimal_weights}
    except Exception as e:
        return {"error": f"Rebalancing failed: {str(e)}"}


# --- LangGraph Agent Setup ---
class AgentState(TypedDict):
    messages: List[Any]


tools = [fetch_market_data, assess_risk, predict_trends, simulate_portfolio, rebalance_portfolio]
tool_node = ToolNode(tools)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=GOOGLE_API_KEY,
)

llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState):
    """Invokes the LLM to get the next action."""
    messages = [
        SystemMessage(
            content=(
                "You are a helpful financial assistant. "
                "Use finance tools for factual data, then provide a concise explanation. "
                "Never return an empty response."
            )
        ),
        *state["messages"],
    ]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Determines whether to continue the graph loop."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "call_tool"
    return "end"


# Define the graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges(
    "agent", should_continue, {"call_tool": "tools", "end": END}
)
graph_builder.add_edge("tools", "agent")
compiled_graph = graph_builder.compile()

# --- FastAPI Application ---
app = FastAPI(title="Financial AI Agent API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("RUNNING THE NEW, CORRECTED SERVER CODE")


@app.post("/query")
async def query_financial_agent(request: QueryRequest, user: Dict = Depends(verify_api_key)):
    """Route deterministic finance requests directly and complex requests to the agent."""
    deterministic_request = parse_deterministic_request(request.query)

    # Example of role-based access control
    if deterministic_request and deterministic_request.operation in {"rebalance", "simulate"} \
            and user["role"] != "premium":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Portfolio simulation and rebalancing requires a premium API key.")

    try:
        if deterministic_request:
            result = execute_deterministic_request(deterministic_request)
            return {"response": format_deterministic_result(result), "result": result}

        initial_state = AgentState(messages=[HumanMessage(content=request.query)])
        result = compiled_graph.invoke(initial_state)
        final_response = result["messages"][-1]
        return {"response": final_response.content}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query failed: {str(e)}")


@app.post("/stream/query")
async def stream_query(request: QueryRequest, user: Dict = Depends(verify_api_key)):
    """Stream direct finance results or the agent's intermediate steps."""
    deterministic_request = parse_deterministic_request(request.query)
    if deterministic_request and deterministic_request.operation in {"rebalance", "simulate"} \
            and user["role"] != "premium":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Portfolio simulation and rebalancing requires a premium API key.")

    async def stream_response():
        if deterministic_request:
            result = execute_deterministic_request(deterministic_request)
            yield f"data: {json.dumps({'type': 'result', 'content': result})}\n\n"
            yield f"data: {json.dumps({'type': 'response', 'content': format_deterministic_result(result)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        initial_state = AgentState(messages=[HumanMessage(content=request.query)])

        # .stream() yields the output of each node as it's executed
        for chunk in compiled_graph.stream(initial_state):
            if "agent" in chunk:
                agent_response = chunk["agent"]["messages"][-1]
                if agent_response.content:
                    yield f"data: {json.dumps({'type': 'response', 'content': agent_response.content})}\n\n"

            elif "tools" in chunk:
                tool_calls = chunk["tools"]["messages"][-1]
                yield f"data: {json.dumps({'type': 'tool_result', 'content': tool_calls.content})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

if __name__ == "__main__":
    print("Starting Financial AI Agent API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)