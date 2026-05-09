#include "portable_path.hpp"

#include <SDL2/SDL_filesystem.h>

#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

#ifdef _WIN32
std::filesystem::path utf8Path(const char* s) {
  return std::filesystem::u8path(s ? s : "");
}
std::string pathToUtf8(const std::filesystem::path& p) {
  return p.u8string();
}
#else
std::filesystem::path utf8Path(const char* s) {
  return std::filesystem::path(s ? s : "");
}
std::string pathToUtf8(const std::filesystem::path& p) {
  return p.string();
}
#endif

// Logical CMake/game paths look like "assets/textures/foo.png". When using VULKAN_GAME_ASSETS_ROOT,
// that env points at the real assets directory on disk, so strip one leading "assets/" prefix.
std::string stripAssetsPrefix(const char* path) {
  std::string rel(path ? path : "");
  static const char kFwd[] = "assets/";
  static const char kBack[] = "assets\\";
  if (rel.size() >= sizeof(kFwd) - 1 && rel.compare(0, sizeof(kFwd) - 1, kFwd) == 0)
    rel.erase(0, sizeof(kFwd) - 1);
  else if (rel.size() >= sizeof(kBack) - 1 && rel.compare(0, sizeof(kBack) - 1, kBack) == 0)
    rel.erase(0, sizeof(kBack) - 1);
  return rel;
}

} // namespace

std::string resolvePortableAssetPath(const char* path) {
  if (!path || !path[0])
    return {};
#ifdef _WIN32
  const bool absolute = (path[0] != '\0' && path[1] == ':') ||
                        (path[0] == '\\' && path[1] == '\\');
#else
  const bool absolute = path[0] == '/';
#endif
  if (absolute)
    return std::string(path);

  namespace fs = std::filesystem;
  std::error_code ec;

#ifdef _WIN32
  const fs::path relPath = fs::u8path(path);
#else
  const fs::path relPath(path);
#endif

  if (const char* rootEnv = std::getenv("VULKAN_GAME_ASSETS_ROOT")) {
    if (rootEnv[0]) {
      const fs::path combined =
          (utf8Path(rootEnv) / utf8Path(stripAssetsPrefix(path).c_str())).lexically_normal();
      return pathToUtf8(combined);
    }
  }

  const auto existsFile = [&ec](const fs::path& p) -> bool {
    ec.clear();
    return fs::is_regular_file(p, ec);
  };

  auto utf8Canon = [&](const fs::path& p) -> std::string {
    ec.clear();
    return pathToUtf8(fs::weakly_canonical(p, ec));
  };

  const auto tryResolved = [&](const fs::path& baseDir) -> std::optional<std::string> {
    if (baseDir.empty())
      return {};
    fs::path direct = (baseDir / relPath).lexically_normal();
    if (existsFile(direct))
      return utf8Canon(direct);
    // Incomplete build-dir tree: build-*/assets/ often has only shaders/; full art lives in repo assets/.
    fs::path alt = (baseDir.parent_path() / relPath).lexically_normal();
    ec.clear();
    const fs::path assetsDir = baseDir / "assets";
    if (!fs::is_directory(assetsDir, ec))
      return {};
    if (existsFile(alt))
      return utf8Canon(alt);
    return {};
  };

  ec.clear();
  if (existsFile(relPath))
    return utf8Canon(relPath);

  std::vector<fs::path> bases;
  auto addUniqueBase = [&](fs::path b) {
    if (b.empty())
      return;
    b = b.lexically_normal();
    for (const fs::path& o : bases) {
      if (o == b)
        return;
    }
    bases.push_back(b);
  };

  if (char* bp = SDL_GetBasePath()) {
    addUniqueBase(utf8Path(bp));
    SDL_free(bp);
  }

#ifdef _WIN32
  wchar_t wbuf[MAX_PATH];
  const DWORD n = GetModuleFileNameW(nullptr, wbuf, MAX_PATH);
  if (n > 0 && n < MAX_PATH)
    addUniqueBase(fs::path(std::wstring(wbuf, wbuf + n)).parent_path());
#endif

  ec.clear();
  addUniqueBase(fs::current_path(ec));

  fs::path primaryForErrors;
  for (const fs::path& base : bases) {
    if (primaryForErrors.empty())
      primaryForErrors = (base / relPath).lexically_normal();
    if (std::optional<std::string> found = tryResolved(base))
      return *found;
  }

  // Match historical behaviour: deterministic path string for loaders / IMG_GetError diagnostics.
  if (!primaryForErrors.empty())
    return pathToUtf8(primaryForErrors);
  return pathToUtf8(relPath.lexically_normal());
}
