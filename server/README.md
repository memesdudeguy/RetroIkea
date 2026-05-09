# RetroIkea lobby server

In-memory session list with TTL. Game clients set `RETRO_IKEA_LOBBY_URL` to this service’s public origin (no trailing slash), e.g. `https://retro-ikea-lobby.onrender.com`.

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

- **Render (recommended with this repo):** In the Render dashboard use **New → Blueprint**, connect the RetroIkea GitHub repo, and apply the root [`render.yaml`](../render.yaml). That defines a Python web service with `rootDir: server`.
- **Render (manual):** Web Service, root directory `server`, build `pip install -r requirements.txt`, start `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- **Railway / Fly.io**: same idea — set `PORT`, enable HTTPS at the edge.

Set `LOBBY_TTL_SEC` (default `90`) so stale sessions disappear if a host crashes without unregistering.

Enable `LOBBY_CORS_ORIGINS` if you need a restricted browser origin list (game native client ignores CORS).
