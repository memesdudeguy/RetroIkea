"""
Tiny HTTP lobby for RetroIkea: hosts POST heartbeats; clients GET the live list.
Configure TTL via LOBBY_TTL_SEC. Bind via HOST/PORT or let the platform set PORT.
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

TTL_SEC = float(os.environ.get("LOBBY_TTL_SEC", "90"))

app = FastAPI(title="RetroIkea lobby", version="1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("LOBBY_CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# session_id -> record
_sessions: dict[str, dict] = {}


def _prune() -> None:
    now = time.time()
    dead = [k for k, v in _sessions.items() if now - v["seen"] > TTL_SEC]
    for k in dead:
        del _sessions[k]


class RegisterBody(BaseModel):
    id: str = Field(..., min_length=8, max_length=64)
    host: str = Field(..., min_length=3, max_length=63)
    port: int = Field(27341, ge=1, le=65535)
    name: str = Field("", max_length=48)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/servers")
def list_servers() -> list[dict]:
    _prune()
    out: list[dict] = []
    for sid, v in _sessions.items():
        out.append({"id": sid, "host": v["host"], "port": v["port"], "name": v.get("name", "")})
    return out


@app.post("/api/v1/servers/register")
def register_server(body: RegisterBody) -> dict[str, bool]:
    _prune()
    _sessions[body.id] = {
        "host": body.host,
        "port": body.port,
        "name": body.name or "RetroIkea",
        "seen": time.time(),
    }
    return {"ok": True}


@app.delete("/api/v1/servers/{session_id}")
def unregister(session_id: str) -> dict[str, bool]:
    _sessions.pop(session_id, None)
    return {"ok": True}
