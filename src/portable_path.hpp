#pragma once

#include <string>

// CMake embeds "assets/..." for Windows installs. Assimp/SDL open paths literally — resolve against
// SDL_GetBasePath() / the executable directory first. Cross-build dirs often contain only assets/shaders
// (compiled .spv); when an assets/ folder exists beside the exe but the requested file is missing, we try
// ../<same relative path> (repo-root assets/). Shortcuts leaving cwd wrong are still covered.
//
// Optional override (dev / tooling): set VULKAN_GAME_ASSETS_ROOT to the absolute path of the
// assets directory (the folder that contains audio/, textures/, models/). Relative paths like
// "assets/textures/foo.png" are rooted there (the leading "assets/" segment is stripped).
std::string resolvePortableAssetPath(const char* path);
