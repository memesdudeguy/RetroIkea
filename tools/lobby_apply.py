#!/usr/bin/env python3
"""GitHub-only RetroIkea lobby: mutate ``lobby/registry.json`` from a workflow.

This script is invoked by ``.github/workflows/lobby.yml`` for three event sources:

1. ``repository_dispatch`` event_type=``lobby_register`` — upsert a host entry.
2. ``repository_dispatch`` event_type=``lobby_unregister`` — drop a host entry.
3. ``schedule`` (cron) — prune any entry whose ``expires_at`` has passed.

The registry file is committed back to the default branch by the workflow itself.
The schema is intentionally tiny so reading clients can parse it with a hand-rolled
JSON pull instead of a full library.

    {
        "version": 1,
        "ttl_sec": 90,
        "updated_at": "<ISO8601 UTC>",
        "servers": [
            {
                "id": "<session uuid>",
                "host": "1.2.3.4",
                "port": 27341,
                "name": "RetroIkea",
                "expires_at": "<ISO8601 UTC>"
            }
        ]
    }

Hosts that fail to refresh their entry within ``ttl_sec`` get pruned on the next
cron tick (~5 min on GitHub's free tier) so stale crashed hosts disappear without
operator intervention.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

REGISTRY_PATH = Path("lobby/registry.json")
DEFAULT_TTL_SEC = 90
MAX_NAME_LEN = 64
MAX_HOST_LEN = 64
MAX_ID_LEN = 64
HARD_CAP_SERVERS = 256

NOW = datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: Any) -> datetime | None:
    if not isinstance(s, str):
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except ValueError:
        return None


def _load() -> Dict[str, Any]:
    if REGISTRY_PATH.exists():
        try:
            data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("version", 1)
                data.setdefault("ttl_sec", DEFAULT_TTL_SEC)
                data.setdefault("servers", [])
                if not isinstance(data["servers"], list):
                    data["servers"] = []
                return data
        except json.JSONDecodeError as exc:
            print(f"[lobby] registry was corrupt ({exc}); recreating empty", file=sys.stderr)
    return {"version": 1, "ttl_sec": DEFAULT_TTL_SEC, "updated_at": None, "servers": []}


def _save(data: Dict[str, Any]) -> None:
    data["updated_at"] = _iso(NOW)
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _sanitize_str(value: Any, max_len: int) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = "".join(ch for ch in value if ch >= " " and ch != "\u007f")
    return cleaned[:max_len].strip()


def _sanitize_port(value: Any) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return 27341
    if port < 1 or port > 65535:
        return 27341
    return port


def _prune(servers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for s in servers:
        if not isinstance(s, dict):
            continue
        exp = _parse_iso(s.get("expires_at"))
        if exp is None or exp <= NOW:
            continue
        kept.append(s)
    return kept[:HARD_CAP_SERVERS]


def _upsert(servers: List[Dict[str, Any]], entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    new_id = entry["id"]
    out = [s for s in servers if isinstance(s, dict) and s.get("id") != new_id]
    out.append(entry)
    return out[:HARD_CAP_SERVERS]


def _drop(servers: List[Dict[str, Any]], entry_id: str) -> List[Dict[str, Any]]:
    return [s for s in servers if isinstance(s, dict) and s.get("id") != entry_id]


def _payload_to_entry(payload: Dict[str, Any], ttl_sec: int) -> Dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    sid = _sanitize_str(payload.get("id", ""), MAX_ID_LEN)
    host = _sanitize_str(payload.get("host", ""), MAX_HOST_LEN)
    if not sid or not host:
        return None
    name = _sanitize_str(payload.get("name", ""), MAX_NAME_LEN) or "RetroIkea"
    port = _sanitize_port(payload.get("port", 27341))
    expires = NOW + timedelta(seconds=max(15, min(ttl_sec * 2, 600)))
    return {
        "id": sid,
        "host": host,
        "port": port,
        "name": name,
        "expires_at": _iso(expires),
    }


def _read_event() -> Dict[str, Any]:
    path = os.environ.get("GITHUB_EVENT_PATH")
    if path and Path(path).exists():
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[lobby] GITHUB_EVENT_PATH unreadable: {exc}", file=sys.stderr)
    raw = os.environ.get("GITHUB_EVENT_PAYLOAD")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"[lobby] GITHUB_EVENT_PAYLOAD unreadable: {exc}", file=sys.stderr)
    return {}


def main() -> int:
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    event = _read_event()
    data = _load()
    ttl = int(data.get("ttl_sec", DEFAULT_TTL_SEC))

    if event_name == "repository_dispatch":
        action = event.get("action", "")
        payload = event.get("client_payload", {}) or {}
        if action == "lobby_register":
            entry = _payload_to_entry(payload, ttl)
            if entry is None:
                print("[lobby] register: invalid payload (need id+host)", file=sys.stderr)
                return 0
            data["servers"] = _upsert(_prune(data.get("servers", [])), entry)
            print(
                f"[lobby] registered {entry['name']!r} @ {entry['host']}:{entry['port']} "
                f"(id={entry['id']}, expires={entry['expires_at']})"
            )
        elif action == "lobby_unregister":
            sid = _sanitize_str(payload.get("id", ""), MAX_ID_LEN)
            if not sid:
                print("[lobby] unregister: empty id", file=sys.stderr)
                return 0
            data["servers"] = _drop(_prune(data.get("servers", [])), sid)
            print(f"[lobby] unregistered id={sid}")
        else:
            print(f"[lobby] ignoring unknown action {action!r}", file=sys.stderr)
            return 0
    elif event_name in ("schedule", "workflow_dispatch"):
        before = data.get("servers", [])
        kept = _prune(before)
        data["servers"] = kept
        print(f"[lobby] cron prune: {len(before)} -> {len(kept)}")
    else:
        print(f"[lobby] unsupported event {event_name!r}", file=sys.stderr)
        return 0

    _save(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
