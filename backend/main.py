"""FastAPI entry point. Run with ``uvicorn main:app`` from this directory."""

import json
from typing import Dict

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from agent import AgentServiceError, AgentUnavailableError, ensure_available, invoke, stream
from auth import verify_api_key
from finance import (
    FinanceServiceError,
    execute_deterministic_request,
    format_deterministic_result,
    parse_deterministic_request,
)
from models import QueryRequest

app = FastAPI(title="Financial AI Agent API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": jsonable_encoder(exc.errors())})


def _check_access(deterministic_request, user: Dict[str, str]) -> None:
    if deterministic_request and deterministic_request.operation in {"rebalance", "simulate"} and user["role"] != "premium":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Portfolio simulation and rebalancing requires a premium API key.")


@app.post("/query")
async def query_financial_agent(request: QueryRequest, user: Dict[str, str] = Depends(verify_api_key)):
    deterministic_request = parse_deterministic_request(request.query)
    _check_access(deterministic_request, user)
    try:
        if deterministic_request:
            result = await run_in_threadpool(execute_deterministic_request, deterministic_request)
            return {"response": format_deterministic_result(result), "result": result}
        result = await run_in_threadpool(invoke, request.query)
        return {"response": result["messages"][-1].content}
    except AgentUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except FinanceServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/stream/query")
async def stream_query(request: QueryRequest, user: Dict[str, str] = Depends(verify_api_key)):
    deterministic_request = parse_deterministic_request(request.query)
    _check_access(deterministic_request, user)
    if not deterministic_request:
        try:
            ensure_available()
        except AgentUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    async def stream_response():
        try:
            if deterministic_request:
                result = await run_in_threadpool(execute_deterministic_request, deterministic_request)
                yield f"data: {json.dumps({'type': 'result', 'content': result})}\n\n"
                yield f"data: {json.dumps({'type': 'response', 'content': format_deterministic_result(result)})}\n\n"
            else:
                chunks = await run_in_threadpool(lambda: list(stream(request.query)))
                for chunk in chunks:
                    if "agent" in chunk:
                        content = chunk["agent"]["messages"][-1].content
                        if content:
                            yield f"data: {json.dumps({'type': 'response', 'content': content})}\n\n"
                    elif "tools" in chunk:
                        yield f"data: {json.dumps({'type': 'tool_result', 'content': chunk['tools']['messages'][-1].content})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except AgentUnavailableError as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc), 'status': 503})}\n\n"
        except AgentServiceError as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc), 'status': 502})}\n\n"
        except FinanceServiceError as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc), 'status': 502})}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
