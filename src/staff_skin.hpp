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
constexpr int kMaxPaletteBones = 32;

// Interleaved staff VBO: matches pipeline vertex attributes (tightly packed).
struct SkinnedVertex {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec4 color;
  glm::vec2 uv;
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

// Same as computePalette, then applies extra local rotations (radians, XYZ then Z*Y*X) per palette bone
// on top of the sampled clip. extraLocalEulerPerBone[i] is for boneNames[i]; nullptr = no extras.
void computePaletteWithRagdollExtras(const Rig& rig, int clipIndex, double phaseSec, bool loopPhase,
                                     const glm::vec3* extraLocalEulerPerBone, glm::mat4* outPalette);

// Bone globals before meshNorm * invBind (same space as internal eval). optional euler extras per palette bone.
void sampleClipBoneGlobalMatrices(const Rig& rig, int clipIndex, double phaseSec, bool loopPhase,
                                  const glm::vec3* extraLocalEulerPerBone, glm::mat4* outGlobalBone);

// Rest pose only (no clip sampling): meshNorm * bind hierarchy * invBind per bone.
void computeBindPosePalette(const Rig& rig, glm::mat4* outPalette);

// Bind pose + same extra local euler convention as computePaletteWithRagdollExtras (no clip keys).
void computeBindPosePaletteWithRagdollExtras(const Rig& rig, const glm::vec3* extraLocalEulerPerBone,
                                             glm::mat4* outPalette);

// Rigid-body ragdoll: per-sim-bone world transforms (must match characterModel * meshNorm * bindGlobal for that
// bone at spawn). Other bones follow bind pose relative to the nearest simulated ancestor.
void computePaletteFromRagdollSimWorldMatrices(const Rig& rig, const glm::mat4& characterModel,
                                               const glm::mat4* bindGlobalArmature, int nSim,
                                               const int* simRigBoneIdx, const glm::mat4* worldBoneSim,
                                               glm::mat4* outPalette);

// Bind hierarchy globals (before meshNorm * invBind), optional per-bone euler extras.
void sampleBindBoneGlobalMatricesWithExtras(const Rig& rig, const glm::vec3* extraLocalEulerPerBone,
                                            glm::mat4* outGlobalBone);

// Bind pose with extra local rotations on limbs (ragdollAngVel = pitch/yaw/roll rates, rad/s).
void computeLooseBindPosePalette(const Rig& rig, glm::mat4* outPalette, const glm::vec3& ragdollAngVelRadPerSec,
                                 float simTimeSec, uint32_t hashSeed);

// Smooth transition between clips (t=0 → A, t=1 → B). Per-bone matrix lerp + smoothstep(t).
void computePaletteLerp(const Rig& rig, int clipA, double phaseA, bool loopA, int clipB, double phaseB,
                        bool loopB, float t, glm::mat4* outPalette);

// Shrink all clips in-place: collapse constant tracks to 1 key, drop identity scale,
// remove duplicate consecutive keyframes. Returns total keys removed.
size_t optimizeRigClips(Rig& rig);

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
