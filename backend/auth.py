from typing import Dict

from fastapi import HTTPException, Request

from config import PREMIUM_API_KEY, STANDARD_API_KEY


async def verify_api_key(request: Request) -> Dict[str, str]:
    api_key = request.headers.get("x-api-key")
    if api_key == PREMIUM_API_KEY:
        return {"role": "premium"}
    if api_key == STANDARD_API_KEY:
        return {"role": "standard"}
    raise HTTPException(status_code=401, detail="Invalid API Key")
