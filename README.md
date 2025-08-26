# Financial AI Agent API

A FastAPI-powered AI assistant for financial analysis.
It integrates Google Gemini with LangGraph and LangChain tools to provide insights into stocks, portfolios, and risk.

---

## Features

* **Market Data**: Fetch historical prices, financial metrics, and recent news for a stock.
* **Risk Assessment**: Calculate annualized volatility and Value at Risk (VaR).
* **Trend Prediction**: Forecast stock price trends using ARIMA models.
* **Portfolio Simulation**: Monte Carlo simulations for expected returns and risk (premium feature).
* **Portfolio Rebalancing**: Sharpe ratio optimization for better risk-adjusted returns (premium feature).
* **Streaming Responses**: Get incremental results for real-time interfaces.
* **Role-Based Access**: Standard and Premium API keys with different feature sets.

---

## Requirements

* Python 3.9+
* Google API Key for Gemini models
* Dependencies listed in `requirements.txt`

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/raogaurav17/Financial-ai-agent.git
   cd Financial-ai-agent
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add:

   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

---

## Running the Server

Start the FastAPI server with Uvicorn:

```bash
python app.py
```

The API will run at:

```
http://localhost:8000
```

Interactive docs are available at:

* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

---

## API Usage

### Authentication

Requests require an API key in headers:

* Standard key: `my-secret-standard-key`
* Premium key: `my-secret-premium-key`

Example:

```bash
-H "x-api-key: my-secret-standard-key"
```

---

### Endpoints

#### 1. Query Agent

**POST** `/query`

Request:

```json
{
  "query": "fetch market data for AAPL"
}
```

Response:

```json
{
  "response": "Apple (AAPL) key metrics: PE ratio 29, EPS 6.13..."
}
```

---

#### 2. Stream Query

**POST** `/stream/query`

Streams intermediate tool results and the final response as Server-Sent Events (SSE).

---

## Example Queries

* `"fetch market data for AAPL"`
* `"assess risk for TSLA"`
* `"predict trends for GOOGL"`
* `"simulate portfolio with AAPL, TSLA, MSFT"` *(premium only)*
* `"rebalance portfolio with AMZN, MSFT, NVDA"` *(premium only)*

---

## Roadmap

* JWT authentication with user database
* Caching for faster repeated queries
* Background task queue for heavy simulations
* Structured outputs using Pydantic models

---

## License

This project is licensed under the Apache-2.0 License.

---

Do you want me to also create a **requirements.txt** file for this project so anyone can install and run it immediately?
