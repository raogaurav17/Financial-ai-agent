"""Deterministic finance services and the tools exposed to the agent."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from langchain_core.tools import tool
from scipy.optimize import minimize

from models import DeterministicResult


class FinanceServiceError(Exception):
    """A known upstream or calculation failure suitable for HTTP 502."""


@dataclass(frozen=True)
class DeterministicRequest:
    operation: Literal["quote", "historical", "risk", "trend", "simulate", "rebalance"]
    tickers: List[str]
    simulations: int = 10000


def _download_prices(tickers: List[str], period: str = "1y", field: str = "Adj Close") -> pd.DataFrame:
    try:
        data = yf.download(tickers, period=period, progress=False, auto_adjust=False)[field]
    except Exception as exc:
        raise FinanceServiceError(f"Market data provider failed: {exc}") from exc
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    return data.dropna(how="all")


def _single_price_series(ticker: str, field: str = "Close") -> pd.Series:
    data = _download_prices([ticker], field=field)
    series = data[ticker]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return series.dropna()


def _extract_tickers(query: str) -> List[str]:
    ignored = {
        "A", "AN", "AND", "AS", "AT", "DATA", "FOR", "FETCH", "FROM", "GET",
        "HISTORICAL", "I", "IN", "IS", "LATEST", "MARKET", "OF", "ON", "OR",
        "PRICE", "QUOTE", "RISK", "SHOW", "THE", "TO", "TREND", "VAR", "WHAT",
        "WITH", "CURRENT", "LOOKUP",
    }
    candidates = re.findall(r"\b[A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?\b", query)
    return list(dict.fromkeys(t.upper() for t in candidates if t.upper() not in ignored))


def parse_deterministic_request(query: str) -> DeterministicRequest | None:
    normalized = query.lower()
    explanation_words = ("explain", "why", "compare", "comparison", "summarize",
                         "summary", "recommend", "should i", "and", "then", "because")
    if any((f" {word} " in f" {normalized} " if " " in word else
            re.search(rf"\b{re.escape(word)}\b", normalized))
           for word in explanation_words):
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
    match = re.search(r"\b(\d{2,7})\s*(?:simulations?|runs?)\b", normalized)
    return DeterministicRequest(operation, tickers, int(match.group(1)) if match else 10000)


def execute_deterministic_request(request: DeterministicRequest) -> DeterministicResult:
    ticker = request.tickers[0]
    try:
        if request.operation == "quote":
            info = yf.Ticker(ticker).fast_info
            return {"ticker": ticker, "price": float(info.last_price),
                    "currency": str(info.currency),
                    "as_of": pd.Timestamp.now(tz="UTC").isoformat()}
        if request.operation == "historical":
            data = _download_prices([ticker], period="3mo", field="Close")
            return {"ticker": ticker, "period": "3mo",
                    "rows": [{"date": i.isoformat(), "close": float(v)}
                             for i, v in data[ticker].dropna().items()]}
        if request.operation == "risk":
            data = _download_prices([ticker])[ticker].pct_change().dropna()
            if data.empty:
                raise FinanceServiceError(f"No data for {ticker}")
            return {"ticker": ticker, "annualized_volatility": float(data.std() * np.sqrt(252)),
                    "value_at_risk_95": float(np.percentile(data, 5))}
        if request.operation == "trend":
            data = _single_price_series(ticker).reset_index(drop=True)
            if data.empty:
                raise FinanceServiceError(f"No data for {ticker}")
            forecast = float(sm.tsa.ARIMA(data, order=(5, 1, 0)).fit().forecast(steps=30).iloc[-1])
            current = float(data.iloc[-1])
            return {"ticker": ticker, "current_price": current, "forecast_price": forecast,
                    "predicted_trend": "Upward" if forecast > current else "Downward"}
        if len(request.tickers) < 2:
            raise FinanceServiceError(f"Portfolio {request.operation} requires at least two tickers.")
        data = _download_prices(request.tickers)[request.tickers].pct_change().dropna()
        if request.operation == "simulate":
            draws = np.random.multivariate_normal(data.mean().to_numpy(), data.cov().to_numpy(),
                                                   size=request.simulations)
            returns = draws.mean(axis=1)
            return {"portfolio": request.tickers, "expected_annual_return": float(returns.mean() * 252),
                    "annual_risk": float(returns.std() * np.sqrt(252))}
        mean_returns, cov_matrix, count = data.mean().to_numpy() * 252, data.cov().to_numpy() * 252, len(request.tickers)
        def objective(weights: np.ndarray) -> float:
            portfolio_return = float(np.sum(mean_returns * weights))
            volatility = float(np.sqrt(weights.T @ cov_matrix @ weights))
            return -portfolio_return / volatility if volatility else 0.0
        result = minimize(objective, [1 / count] * count, method="SLSQP",
                          bounds=tuple((0.0, 1.0) for _ in range(count)),
                          constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1})
        if not result.success:
            raise FinanceServiceError(f"Rebalancing failed: {result.message}")
        return {"portfolio": request.tickers,
                "weights": {t: float(w) for t, w in zip(request.tickers, result.x)}}
    except FinanceServiceError:
        raise
    except Exception as exc:
        raise FinanceServiceError(f"Finance operation failed for {ticker}: {exc}") from exc


def format_deterministic_result(result: DeterministicResult) -> str:
    if "ticker" in result and "price" in result:
        return f"{result['ticker']} is trading at {result['price']:.2f} {result['currency']}."
    if "annualized_volatility" in result:
        return f"{result['ticker']} annualized volatility is {result['annualized_volatility']:.2%}; 95% daily VaR is {result['value_at_risk_95']:.2%}."
    if "predicted_trend" in result:
        return f"{result['ticker']} has a predicted {result['predicted_trend'].lower()} trend over the next 30 trading days."
    if "expected_annual_return" in result:
        return f"Expected annual return is {result['expected_annual_return']:.2%} with annual risk of {result['annual_risk']:.2%}."
    if "weights" in result:
        return "Suggested Sharpe-optimized allocation: " + ", ".join(f"{t}: {w:.2%}" for t, w in result["weights"].items()) + "."
    return f"Retrieved {len(result['rows'])} historical observations for {result['ticker']}."


def _tool_result(operation: str, ticker: str | None = None) -> Dict[str, Any]:
    try:
        request = DeterministicRequest(operation, [ticker] if ticker else [])
        return execute_deterministic_request(request)
    except (FinanceServiceError, ValueError) as exc:
        return {"error": str(exc)}


@tool
def assess_risk(ticker: str) -> Dict[str, Any]:
    """Calculate annualized volatility and 95% daily VaR."""
    result = _tool_result("risk", ticker)
    if "error" in result:
        return result
    return {**result, "interpretation": "Volatility measures price fluctuation; VaR is the potential loss in a day with 95% confidence."}


@tool
def predict_trends(ticker: str) -> Dict[str, Any]:
    """Predict a stock trend using the shared deterministic service."""
    return _tool_result("trend", ticker)


@tool
def simulate_portfolio(tickers: List[str], simulations: int = 10000) -> Dict[str, Any]:
    """Run a Monte Carlo portfolio simulation."""
    try:
        return execute_deterministic_request(DeterministicRequest("simulate", tickers, simulations))
    except (FinanceServiceError, ValueError) as exc:
        return {"error": str(exc)}


@tool
def rebalance_portfolio(tickers: List[str]) -> Dict[str, Any]:
    """Optimize portfolio weights using the shared service."""
    try:
        return execute_deterministic_request(DeterministicRequest("rebalance", tickers))
    except (FinanceServiceError, ValueError) as exc:
        return {"error": str(exc)}


@tool
def fetch_market_data(ticker: str) -> Dict[str, Any]:
    """Fetch market history, metrics, and recent news."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {"history_summary": stock.history(period="3mo").reset_index().to_string(),
                "key_metrics": {k: info.get(k) for k in ("trailingPE", "trailingEps", "marketCap", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "dividendYield")},
                "latest_news": "\n".join(f"- {item['title']}" for item in stock.news[:3])}
    except Exception as exc:
        return {"error": f"Failed to fetch data for {ticker}: {exc}"}
