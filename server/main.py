"""
Tiny HTTP lobby for RetroIkea: hosts POST heartbeats; clients GET the live list.
Configure TTL via LOBBY_TTL_SEC. Bind via HOST/PORT or let the platform set PORT.
"""

from __future__ import annotations

import json
import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
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


@app.get("/", response_class=HTMLResponse)
def root_landing() -> str:
    """Browser sanity check: visiting the host root is not an API error."""
    return """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>RetroIkea lobby</title>
<style>
body{font-family:system-ui,sans-serif;background:#0e1624;color:#e8f1ff;max-width:42rem;margin:2rem auto;padding:0 1rem;line-height:1.5}
a{color:#7eb8ff} code{background:#1a2838;padding:.1rem .35rem;border-radius:4px}
h1{font-weight:600;font-size:1.25rem;color:#fff;border-bottom:2px solid #0058AB;padding-bottom:.35rem}
</style></head><body>
<h1>RetroIkea lobby</h1>
<p>This service lists hosts for the game. Set <code>RETRO_IKEA_LOBBY_URL</code> to this site’s origin (no trailing slash).</p>
<ul>
<li><a href="/docs">OpenAPI docs</a></li>
<li><a href="/healthz">Health</a> — <code>GET /healthz</code></li>
<li><a href="/api/v1/servers">Live servers JSON</a> — <code>GET /api/v1/servers</code></li>
</ul>
</body></html>"""


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/servers")
def list_servers() -> Response:
    _prune()
    out: list[dict] = []
    for sid, v in _sessions.items():
        out.append({"id": sid, "host": v["host"], "port": v["port"], "name": v.get("name", "")})
    # Compact JSON (no spaces) — matches naive C++ field scanner; also smaller on the wire.
    payload = json.dumps(out, separators=(",", ":"))
    return Response(content=payload, media_type="application/json")


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
