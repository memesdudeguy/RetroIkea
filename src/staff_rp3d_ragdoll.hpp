#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstdint>

namespace staff_skin {
struct Rig;
}

namespace staff_rp3d {

bool init(float gravityY);
void shutdown();

bool active(uint64_t residentKey);
// boneGlobalsArmature: per-bone globals in armature space at death (from sampleClipBoneGlobalMatrices). When
// null, uses bind pose — must match RigidBody W0 = M*G or skinning stretches (invM*W*invW0*M0*G0).
// impactLinearScale / impactAngularScale: death velocity & tumble (hit / fall). Defaults 1 = moderate knock.
bool spawnCorpse(uint64_t residentKey, glm::vec2 posXZ, float feetWorldY, float yaw, glm::vec3 bodyScale,
                 glm::vec2 knockXZ, float gravityY, float employeeHeightMeters, const staff_skin::Rig& rig,
                 const glm::mat4* boneGlobalsArmature, float impactLinearScale = 1.f,
                 float impactAngularScale = 1.f);
void destroyCorpse(uint64_t residentKey);

void step(float dt);

// Updates feet/ground from hip body only. Does not change yaw — root yaw must stay at the death-pose
// value or inv(M) in fillBonePalette fights the tumbling hip and the mesh goes nuts.
void syncNpcFromPhysics(uint64_t residentKey, glm::vec2& posXZ, float& feetWorldY, glm::vec3 bodyScale);

bool fillBonePalette(uint64_t residentKey, const staff_skin::Rig& rig, const glm::mat4& npcModelMatrix,
                     glm::mat4* outPalette);

} // namespace staff_rp3d
