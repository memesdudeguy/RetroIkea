#pragma once

#include <cstdint>
#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace emp_mesh {

// Must match game `Vertex` (pos, normal, color, uv).
struct LoadedVertex {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec4 color; // rgb = employee tag; a = part: 0.02 jeans, 0.04 shirt, 0.06 skin, 0.08 unknown
  glm::vec2 uv;    // mesh UV0
};

// Loads all meshes from a file (FBX, GLB/glTF, etc. via Assimp); centers XZ, feet at y=0, scales to height.
// maxVertsForDecimation: 0 = defaults (20k if targetHeightMeters >= 1, else 8k for props); else hard cap.
// Optional: first PBR base-color / diffuse texture as RGBA8 (embedded or next to file) for in-engine sampling.
bool loadFbx(const char* path, float targetHeightMeters, std::vector<LoadedVertex>& out,
             std::string& errOut, std::vector<uint8_t>* outDiffuseRgba = nullptr,
             uint32_t* outDiffuseW = nullptr, uint32_t* outDiffuseH = nullptr,
             size_t maxVertsForDecimation = 0);

} // namespace emp_mesh
