# RetroIkea

A first-person Vulkan + SDL2 game: explore a big-box store, avoid staff at night, and survive. C++17, instanced rendering, skinned characters, and a full audio mix.

**Repository:** [github.com/memesdudeguy/RetroIkea](https://github.com/memesdudeguy/RetroIkea)

## Demo

Screen recording (2026-03-30): [`media/demo.mp4`](media/demo.mp4)

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

Produces `packaging/RetroIkea.exe` (installer) that installs `RetroIkea.exe` and the `assets` folder. That file is not committed here (it is large); upload it as a [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) asset instead.

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

## License

No license file is included yet; add one if you want to clarify redistribution.
