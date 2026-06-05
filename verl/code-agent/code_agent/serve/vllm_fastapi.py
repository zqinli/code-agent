"""FastAPI wrapper around a vLLM OpenAI-compatible server."""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_MODEL = os.getenv("SERVED_MODEL_NAME", "code-agent-rl")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

app = FastAPI(title="Code Agent Inference API", version="0.1.0")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.2, ge=0.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stop: list[str] | str | None = None
    stream: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    prompt: str
    system: str | None = None
    model: str | None = None
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.2, ge=0.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stop: list[str] | str | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


def _model_name(model: str | None) -> str:
    return model or DEFAULT_MODEL


async def _post_vllm(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{VLLM_BASE_URL.rstrip('/')}{path}"
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail: Any
        try:
            detail = exc.response.json()
        except Exception:
            detail = exc.response.text
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"vLLM server unavailable: {exc}") from exc


@app.get("/health")
async def health() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{VLLM_BASE_URL.rstrip('/')}/v1/models")
            response.raise_for_status()
            models = response.json()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"vLLM health check failed: {exc}") from exc

    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "vllm_base_url": VLLM_BASE_URL,
        "models": models,
    }


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest) -> dict[str, Any]:
    payload = {
        "model": _model_name(request.model),
        "messages": [message.model_dump() for message in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": request.stream,
    }
    if request.stop is not None:
        payload["stop"] = request.stop
    payload.update(request.extra_body)
    return await _post_vllm("/v1/chat/completions", payload)


@app.post("/generate")
async def generate(request: GenerateRequest) -> dict[str, Any]:
    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})

    payload = {
        "model": _model_name(request.model),
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": False,
    }
    if request.stop is not None:
        payload["stop"] = request.stop
    payload.update(request.extra_body)

    raw = await _post_vllm("/v1/chat/completions", payload)
    choice = raw.get("choices", [{}])[0]
    message = choice.get("message", {})
    return {
        "text": message.get("content", ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": raw.get("usage"),
        "raw": raw,
    }
