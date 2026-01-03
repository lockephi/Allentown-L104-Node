"""Simple FastAPI app (cleaned for running/debugging).

This file was restored to a minimal, runnable form so you can debug and run.
"""

import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    # Return template if available; otherwise a fallback JSON message
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception:
        return JSONResponse({"status": "ok", "message": "index template not found"})


@app.post("/api/stream")
async def sovereign_stream(request: Request):
    payload = await request.json()
    user_signal = payload.get("message", "PING")
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return JSONResponse({"error": "GEMINI_API_KEY not set"}, status_code=500)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={api_key}"
    )

    data = {"contents": [{"parts": [{"text": user_signal}]}], "generationConfig": {"temperature": 1.0}}

    async def generate():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=data) as response:
                async for chunk in response.aiter_text():
                    yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8081)
                                                                                                                                                                                                                    