# RetroIkea lobby server (legacy FastAPI; optional)

Beta 15+ builds default to a **GitHub-only lobby** — see the project README's [Online lobby](../README.md#online-lobby-browser-server-list) section. No external server is required for that flow.

This `server/` directory is the legacy FastAPI lobby and is still supported as an alternative for operators who want sub-second host registration without distributing GitHub PATs. To use it, run the server somewhere (Render, Fly.io, Railway, your own VPS, or locally for LAN tests) and set `RETRO_IKEA_LOBBY_URL=https://your-host.example` per-PC, or rebuild with `-DRETRO_IKEA_DEFAULT_LOBBY_URL=https://your-host.example`. Origin only, **no trailing slash**.

## Run locally

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8765
```

Point the game at `http://127.0.0.1:8765`.

## API

- `GET /` — tiny HTML landing page (open in a browser to verify deploy)
- `GET /docs` — Swagger UI (FastAPI)
- `GET /api/v1/servers` — JSON array: `{ id, host, port, name }[]`
- `POST /api/v1/servers/register` — JSON `{ id, host, port, name }` (repeat to refresh TTL)
- `DELETE /api/v1/servers/{id}` — optional unregister when host stops

## Deploy (GitHub holds code; run the process elsewhere)

GitHub Pages only serves static files — run this FastAPI app on a small host:

- **Render (recommended with this repo):** In the Render dashboard use **New → Blueprint**, connect the RetroIkea GitHub repo, and apply the root [`render.yaml`](../render.yaml). It runs from the repo root with `uvicorn server.main:app`.
- **Render (manual):** Web Service from the repo root, build `pip install -r server/requirements.txt`, start `uvicorn server.main:app --host 0.0.0.0 --port $PORT --workers 1`.
- If the public URL returns plain text `Not Found` with header `x-render-routing: no-server`, Render is not routing that hostname to any live service. Reapply the blueprint or recreate the service named `retro-ikea-lobby`, then redeploy.
- **Railway / Fly.io**: same idea — set `PORT`, enable HTTPS at the edge.

Set `LOBBY_TTL_SEC` (default `90`) so stale sessions disappear if a host crashes without unregistering.

Enable `LOBBY_CORS_ORIGINS` if you need a restricted browser origin list (game native client ignores CORS).
