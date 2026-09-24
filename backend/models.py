from typing import Any, Dict, List, Literal, TypedDict, Union

from pydantic import BaseModel, Field, field_validator

from config import MAX_QUERY_LENGTH


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


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
