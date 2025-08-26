import os
import json
from typing import List, Dict, Any, TypedDict

import numpy as np
import pandas as pd
import statsmodels.api as sm
import uvicorn
import yfinance as yf
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage
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
        data = yf.download(ticker, period="1y", progress=False)['Adj Close'].pct_change().dropna()
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
        data = yf.download(ticker, period="1y", progress=False)['Close']
        if data.empty: return {"error": f"No data for {ticker}"}
        model = sm.tsa.ARIMA(data, order=(5, 1, 0)).fit()
        forecast = model.forecast(steps=30)
        trend = "Upward" if forecast.iloc[-1] > data.iloc[-1] else "Downward"
        return {
            "current_price": data.iloc[-1],
            "30_day_forecast_price": forecast.iloc[-1],
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
    model="gemini-1.5-pro",
    google_api_key=GOOGLE_API_KEY,
    model_kwargs={
        "system_instruction": (
            "You are a helpful financial assistant. "
            "Whenever you use a tool, ALWAYS return a concise summary of the result to the user. "
            "Do not leave the response empty."
        )
    }
)

llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState):
    """Invokes the LLM to get the next action."""
    response = llm_with_tools.invoke(state["messages"])
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

print("RUNNING THE NEW, CORRECTED SERVER CODE")


@app.post("/query")
async def query_financial_agent(request: QueryRequest, user: Dict = Depends(verify_api_key)):
    """Handles financial queries by routing them through the AI agent."""
    query = request.query.lower()

    # Example of role-based access control
    if any(x in query for x in ["rebalance", "simulate"]) and user["role"] != "premium":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Portfolio simulation and rebalancing requires a premium API key.")

    try:
        initial_state = AgentState(messages=[HumanMessage(content=request.query)])
        result = compiled_graph.invoke(initial_state)
        final_response = result["messages"][-1]
        return {"response": final_response.content}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query failed: {str(e)}")


@app.post("/stream/query")
async def stream_query(request: QueryRequest, user: Dict = Depends(verify_api_key)):
    """Streams the agent's intermediate steps and final response."""

    async def stream_response():
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