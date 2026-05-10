# RetroIkea

A first-person Vulkan + SDL2 game: explore a big-box store, avoid staff at night, and survive. C++17, instanced rendering, skinned characters, and a full audio mix.

**Repository:** [github.com/memesdudeguy/RetroIkea](https://github.com/memesdudeguy/RetroIkea)

## Demo

Demo video source file: **`/home/memesdudeguy/2026-03-30 21-50-04.mp4`** (same recording committed as [`media/demo.mp4`](media/demo.mp4) for the repo).

GitHub **does not show** `<video>` embeds in READMEs (they are stripped). Use the poster (click) or the link below to watch the MP4.

[![Watch demo — opens MP4](media/demo-poster.jpg)](https://github.com/memesdudeguy/RetroIkea/raw/main/media/demo.mp4)

**[Open demo video (MP4)](https://github.com/memesdudeguy/RetroIkea/raw/main/media/demo.mp4)**

## Requirements

- **CMake** 3.16+
- **Vulkan** 1.1+ (driver + [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) or `glslc` on `PATH`)
- **SDL2** and **SDL2_image**
- **Python** 3.6+ (embeds some assets at configure time)

On Arch Linux, typical packages: `vulkan-devel`, `sdl2`, `sdl2_image`, `cmake`, `python`.

## Build (Linux)

```bash
cmake -S . -B build
cmake --build build -j
./build/vulkan_game
```

The Linux binary target name is `vulkan_game`. Run it from the repo root so `assets/` resolves correctly, or set paths as described in `CMakeLists.txt` / compile definitions.

## Build (Windows)

Native MinGW/MSVC: see [`packaging/windows_build_and_setup.txt`](packaging/windows_build_and_setup.txt). The shipped Windows executable name is **`RetroIkea.exe`**.

Cross-compile from Linux (MinGW-w64) uses the same CMake project with `-DCMAKE_SYSTEM_NAME=Windows` and a Windows Vulkan import library; details are in that packaging doc.

## Installer (Windows)

With [Inno Setup](https://jrsoftware.org/isinfo.php) 6 installed:

```bash
iscc packaging/windows_setup.iss
```

Produces `packaging/RetroIkea-Beta-Setup.exe` (installer) that installs `RetroIkea.exe` and the `assets` folder. That file is not committed here (it is large); upload it as a [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) asset instead.

## Online lobby (browser server list)

Beta builds ship with **`RETRO_IKEA_DEFAULT_LOBBY_URL=github://memesdudeguy/RetroIkea`** baked in via CMake. The lobby is hosted **entirely on GitHub** — no external server, no Render, no Fly.io:

- A small JSON file ([`lobby/registry.json`](lobby/registry.json)) tracked in this repo is the lobby state.
- Clients read it via `https://raw.githubusercontent.com/memesdudeguy/RetroIkea/main/lobby/registry.json` (anonymous, free, CDN-cached).
- Hosts publish heartbeats by triggering the [`lobby` workflow](.github/workflows/lobby.yml) through GitHub's `repository_dispatch` API (`event_type=lobby_register` / `lobby_unregister`).
- A scheduled run of the same workflow prunes any entry whose `expires_at` has passed (~5 min granularity on the free GitHub Actions tier).

Trade-off: GitHub Actions runs take 10–30s to spin up a runner, so a freshly launched host appears in the browser within ~30–60s — slower than a dedicated server but free, durable, and stateless. Cold-start tolerant timeouts (8s connect / 30s read) are baked into the title-menu fetch.

### Hosts: how to publish to the public lobby

The dispatch API requires authentication, so each host needs a personal access token:

1. Open <https://github.com/settings/personal-access-tokens/new>.
2. Create a **fine-grained token** with **only** these scopes on `memesdudeguy/RetroIkea`:
   - **Contents: Read & Write** (so the lobby workflow can commit registry updates the token triggered).
3. Set `RETRO_IKEA_GH_TOKEN=<token>` in the launching shell / environment before `RetroIkea.exe`. Without the token, the host can still browse and join lobbies — they just can't be listed in the public registry.
4. The pause-menu status line shows `LISTED AS …` once the dispatch succeeds and the workflow commit has propagated to the raw CDN.

Skip everything above if you only want to **join** existing servers — the lobby browser works without any token.

### Self-hosted alternative (legacy FastAPI server)

If you'd rather run your own always-on lobby (faster registration than GitHub Actions, no PAT requirement), the FastAPI app in [`server/`](server/) is still here:

- **Run locally:** see [`server/README.md`](server/README.md).
- **Deploy on Render / Fly.io / Railway:** apply [`render.yaml`](render.yaml) or run `uvicorn server.main:app` from the repo root. Then point the game at it with `RETRO_IKEA_LOBBY_URL=https://your-host.example` or rebuild with `-DRETRO_IKEA_DEFAULT_LOBBY_URL=https://your-host.example`.

## Push this repo to GitHub

If the remote is empty ([RetroIkea](https://github.com/memesdudeguy/RetroIkea.git)):

```bash
git init
git add .
git commit -m "Initial import: RetroIkea game"
git branch -M main
git remote add origin https://github.com/memesdudeguy/RetroIkea.git
git push -u origin main
```

**Authentication:** GitHub does **not** accept your account password for `git push` over HTTPS. Use one of these:

1. **HTTPS + Personal Access Token (PAT)** — [Create a token](https://github.com/settings/tokens) (classic: enable `repo`). When `git push` asks for a password, paste the **token**, not your GitHub password.
2. **SSH** — [Add an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) to your GitHub account, then:
   ```bash
   git remote set-url origin git@github.com:memesdudeguy/RetroIkea.git
   git push -u origin main
   ```
3. **GitHub CLI** — `pacman -S github-cli` then `gh auth login` and push as usual.

## Inno Setup under Wine (Linux)

There is no `iscc` in Arch repos; install Inno Setup with Wine (download [is.exe](https://jrsoftware.org/isdl.php)), then compile from the **project root** (this folder is named `retro ikea`):

```bash
wine "$HOME/.wine/drive_c/Program Files (x86)/Inno Setup 6/ISCC.exe" \
  "Z:\\home\\$(whoami)\\Downloads\\retro ikea\\packaging\\windows_setup.iss"
```

Adjust the `Z:\\...` path if your clone lives elsewhere. Output: `packaging/RetroIkea-Beta-Setup.exe` (see `OutputBaseFilename` in `packaging/windows_setup.iss`).

## License

No license file is included yet; add one if you want to clarify redistribution.
