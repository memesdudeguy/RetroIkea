#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace staff_skin {

// Must match shader_staff.vert STAFF_MAX_BONES and main.cpp allocation.
constexpr int kMaxPaletteBones = 64;

// Interleaved staff VBO: matches pipeline vertex attributes (stride padded).
struct SkinnedVertex {
  glm::vec3 pos;
  float _pad0;
  glm::vec3 normal;
  float _pad1;
  glm::vec4 color;
  glm::vec2 uv;
  glm::vec2 _pad2;
  glm::ivec4 boneIds{0, 0, 0, 0};
  glm::vec4 boneWts{1.f, 0.f, 0.f, 0.f};
};

struct Rig {
  glm::mat4 meshNorm{1.f};
  int boneCount = 0;
  std::vector<std::string> boneNames;
  std::vector<glm::mat4> invBindTweaked;
  std::unordered_map<std::string, int> boneNameToIndex;

  struct NodeRec {
    std::string parent;
    std::vector<std::string> children;
    glm::mat4 bindLocal{1.f};
  };
  std::unordered_map<std::string, NodeRec> nodes;
  std::string rootName;

  struct AnimChannel {
    std::string nodeName;
    std::vector<std::pair<double, glm::vec3>> posKeys;
    std::vector<std::pair<double, glm::quat>> rotKeys;
    std::vector<std::pair<double, glm::vec3>> sclKeys;
  };
  struct AnimClip {
    double duration = 1.0;
    double ticksPerSecond = 25.0;
    // When true (default), may strip baked root/hips translation (see flags below).
    bool lockLocomotionRoot = true;
    // If lockLocomotionRoot: freeze armature root translation to bind (stops lateral root motion).
    bool lockRootTranslationToBind = true;
    // If lockLocomotionRoot: freeze hips/pelvis translation to bind. Jump sets false — full track
    // fixes broken legs; Y-only was still wrong when XZ keys exist.
    bool lockHipsTranslationToBind = true;
    // Optional slerp torso toward bind (unused for jump).
    bool blendTorsoTowardBind = false;
    // Jump: spine/chest/neck/head/clavicle/shoulder use bind pose (upright); legs/arms/root/hips animate.
    bool lockJumpTorsoToBind = false;
    std::vector<AnimChannel> channels;
    std::unordered_map<std::string, size_t> channelByNode;
  };
  std::vector<AnimClip> clips;
};

// Load skinned idle GLB (mesh + skeleton + first animation as clips[0]). Applies same
// normalization as employee_mesh (feet at y=0, XZ centered, target height).
bool loadSkinnedIdleGlb(const char* path, float targetHeightMeters, std::vector<SkinnedVertex>& outVerts,
                        Rig& outRig, std::string& errOut, std::vector<uint8_t>* outDiffuseRgba = nullptr,
                        uint32_t* outDiffuseW = nullptr, uint32_t* outDiffuseH = nullptr,
                        bool loadAllAnimationsFromFile = false);

// Append animation from another GLB (walk/run) — channels matched by node name. Returns false on failure.
bool appendAnimationFromGlb(const char* path, Rig& rig, std::string& errOut);

// Pick the longest clip in the file that retargets to rig; append once. freeRootMotion: false = lock root/hips
// like locomotion clips; true = keep authored motion (dance synced to another mesh using the same rig).
bool appendLongestRetargetedClipFromGlb(const char* path, Rig& rig, bool freeRootMotion, int& outClipIndex,
                                        std::string& errOut);

// clipIndex: 0 idle, 1 walk (if loaded), 2 run (if loaded). Falls back to clip 0 if missing.
void computePalette(const Rig& rig, int clipIndex, double phaseSec, glm::mat4* outPalette,
                    bool loopPhase = true);

// Smooth transition between clips (t=0 → A, t=1 → B). Per-bone matrix lerp + smoothstep(t).
void computePaletteLerp(const Rig& rig, int clipA, double phaseA, bool loopA, int clipB, double phaseB,
                        bool loopB, float t, glm::mat4* outPalette);

// Wall-clock length in seconds (Assimp stores mDuration in ticks; sim uses seconds).
inline double clipDuration(const Rig& rig, int clipIndex) {
  if (clipIndex < 0 || clipIndex >= static_cast<int>(rig.clips.size()) || rig.clips.empty())
    return 1.0;
  const Rig::AnimClip& c = rig.clips[static_cast<size_t>(clipIndex)];
  const double dTicks = c.duration > 1e-6 ? c.duration : 1.0;
  const double tps = c.ticksPerSecond > 1e-6 ? c.ticksPerSecond : 25.0;
  const double sec = dTicks / tps;
  return sec > 1e-9 ? sec : 1.0;
}

} // namespace staff_skin
