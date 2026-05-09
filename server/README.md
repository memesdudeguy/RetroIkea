# RetroIkea lobby server

In-memory session list with TTL. Game clients use `RETRO_IKEA_LOBBY_URL`; you can also bake a default at build time with CMake (`-DRETRO_IKEA_DEFAULT_LOBBY_URL=https://your-service.example`). Point both at this service’s public origin with **no trailing slash**.

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
