#include "staff_rp3d_ragdoll.hpp"
#include "staff_skin.hpp"

#include <reactphysics3d/reactphysics3d.h>
#include <reactphysics3d/constraint/BallAndSocketJoint.h>
#include <reactphysics3d/constraint/FixedJoint.h>
#include <reactphysics3d/constraint/HingeJoint.h>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/geometric.hpp>

#include <cmath>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <vector>

namespace staff_rp3d {

namespace {

using reactphysics3d::decimal;

constexpr int kParts = 12;
constexpr int kMaxCorpses = 6;

reactphysics3d::PhysicsCommon* gPc = nullptr;
reactphysics3d::PhysicsWorld* gWorld = nullptr;

reactphysics3d::BoxShape* gFloorShape = nullptr;

enum Part : int {
  P_HEAD = 0,
  P_CHEST,
  P_WAIST,
  P_HIP,
  P_LUA,
  P_LLA,
  P_RUA,
  P_RLA,
  P_LUL,
  P_LLL,
  P_RUL,
  P_RLL
};

struct Corpse {
  uint64_t key = 0;
  glm::mat4 M0{1.f};
  glm::mat4 invW0[kParts]{};
  glm::mat4 G0_bind[staff_skin::kMaxPaletteBones]{};
  int boneOfPart[kParts];
  reactphysics3d::RigidBody* rb[kParts]{};
  reactphysics3d::RigidBody* floorBody = nullptr;
  reactphysics3d::Collider* floorCollider = nullptr;
  std::vector<reactphysics3d::Joint*> joints;
  int hipPartIdx = P_HIP;
  reactphysics3d::SphereShape* shapeHead = nullptr;
  reactphysics3d::CapsuleShape* shapeChest = nullptr;
  reactphysics3d::CapsuleShape* shapeWaist = nullptr;
  reactphysics3d::CapsuleShape* shapeHip = nullptr;
  reactphysics3d::CapsuleShape* shapeArm = nullptr;
  reactphysics3d::CapsuleShape* shapeForeArm = nullptr;
  reactphysics3d::CapsuleShape* shapeThigh = nullptr;
  reactphysics3d::CapsuleShape* shapeShin = nullptr;
};

std::unordered_map<uint64_t, Corpse> gCorpses;

static glm::vec3 rotYawXZ(const glm::vec3& v, float yaw) {
  const float c = std::cos(yaw);
  const float s = std::sin(yaw);
  return glm::vec3(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

static glm::mat4 matFromRp3d(const reactphysics3d::Transform& tr) {
  const reactphysics3d::Vector3& p = tr.getPosition();
  const reactphysics3d::Quaternion& q = tr.getOrientation();
  const glm::vec3 gp(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
  glm::quat gq(static_cast<float>(q.w), static_cast<float>(q.x), static_cast<float>(q.y),
               static_cast<float>(q.z));
  gq = glm::normalize(gq);
  return glm::translate(glm::mat4(1.f), gp) * glm::mat4_cast(gq);
}

// World-space bone matrix (M * armature global) -> RP3D transform; orthonormalizes rotation.
static reactphysics3d::Transform mat4WorldToRp3d(const glm::mat4& m) {
  const glm::vec3 p(m[3][0], m[3][1], m[3][2]);
  glm::vec3 c0(m[0][0], m[1][0], m[2][0]);
  glm::vec3 c1(m[0][1], m[1][1], m[2][1]);
  c0 = glm::normalize(c0);
  c1 = glm::normalize(c1 - c0 * glm::dot(c0, c1));
  glm::vec3 c2 = glm::normalize(glm::cross(c0, c1));
  c1 = glm::normalize(glm::cross(c2, c0));
  const glm::mat3 Ro(c0, c1, c2);
  glm::quat gq = glm::normalize(glm::quat_cast(Ro));
  const reactphysics3d::Quaternion rq(decimal(gq.x), decimal(gq.y), decimal(gq.z), decimal(gq.w));
  return reactphysics3d::Transform(
      reactphysics3d::Vector3(decimal(p.x), decimal(p.y), decimal(p.z)), rq);
}

static glm::vec3 mat4Trans(const glm::mat4& m) { return glm::vec3(m[3][0], m[3][1], m[3][2]); }

static reactphysics3d::Vector3 rv3(float x, float y, float z) {
  return reactphysics3d::Vector3(decimal(x), decimal(y), decimal(z));
}

static reactphysics3d::Transform trPosEuler(float x, float y, float z, float ex, float ey, float ez) {
  reactphysics3d::Quaternion rq =
      reactphysics3d::Quaternion::fromEulerAngles(decimal(ex), decimal(ey), decimal(ez));
  return reactphysics3d::Transform(rv3(x, y, z), rq);
}

static int findBoneSubstr(const staff_skin::Rig& rig, const char* sub, const char* exclude = nullptr) {
  for (int i = 0; i < rig.boneCount; ++i) {
    std::string lower = rig.boneNames[static_cast<size_t>(i)];
    for (char& c : lower)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower.find(sub) == std::string::npos)
      continue;
    if (exclude && lower.find(exclude) != std::string::npos)
      continue;
    return i;
  }
  return -1;
}

static int findArmBone(const staff_skin::Rig& rig, bool wantLeft, bool wantUpper) {
  for (int i = 0; i < rig.boneCount; ++i) {
    std::string lower = rig.boneNames[static_cast<size_t>(i)];
    for (char& c : lower)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower.find("arm") == std::string::npos)
      continue;
    // "shoulder" / "clavicle" contain "arm" as a substring — exclude from upper/lower arm picks.
    if (lower.find("shoulder") != std::string::npos || lower.find("clavicle") != std::string::npos)
      continue;
    const bool isLeft =
        lower.find("left") != std::string::npos || lower.find(".l") != std::string::npos ||
        lower.find("_l") != std::string::npos;
    const bool isRight =
        lower.find("right") != std::string::npos || lower.find(".r") != std::string::npos ||
        lower.find("_r") != std::string::npos;
    if (wantLeft && (!isLeft || isRight))
      continue;
    if (!wantLeft && (!isRight || isLeft))
      continue;
    const bool looksLower =
        lower.find("lower") != std::string::npos || lower.find("fore") != std::string::npos;
    if (wantUpper && looksLower)
      continue;
    if (!wantUpper && !looksLower)
      continue;
    return i;
  }
  return -1;
}

static bool legNameLooksUpper(const std::string& lower) {
  if (lower.find("thigh") != std::string::npos)
    return true;
  // Mixamo: LeftUpLeg / RightUpLeg
  if (lower.find("upleg") != std::string::npos)
    return true;
  if (lower.find("upperleg") != std::string::npos)
    return true;
  if (lower.find("upper") != std::string::npos && lower.find("leg") != std::string::npos)
    return true;
  return false;
}

static bool legNameLooksLower(const std::string& lower) {
  if (lower.find("calf") != std::string::npos || lower.find("shin") != std::string::npos)
    return true;
  if (lower.find("lowleg") != std::string::npos || lower.find("lowerleg") != std::string::npos)
    return true;
  if (lower.find("foot") != std::string::npos || lower.find("toe") != std::string::npos)
    return false;
  // Mixamo: LeftLeg / RightLeg (between up leg and foot), distinct from upper segment.
  if (lower.find("leg") != std::string::npos && !legNameLooksUpper(lower))
    return true;
  return false;
}

static int findLegBone(const staff_skin::Rig& rig, bool wantLeft, bool wantUpperLeg) {
  for (int i = 0; i < rig.boneCount; ++i) {
    std::string lower = rig.boneNames[static_cast<size_t>(i)];
    for (char& c : lower)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const bool isLeft =
        lower.find("left") != std::string::npos || lower.find(".l") != std::string::npos;
    const bool isRight =
        lower.find("right") != std::string::npos || lower.find(".r") != std::string::npos;
    if (wantLeft && (!isLeft || isRight))
      continue;
    if (!wantLeft && (!isRight || isLeft))
      continue;

    const bool looksUpper = legNameLooksUpper(lower);
    const bool looksLower = legNameLooksLower(lower);

    if (wantUpperLeg && looksUpper && !looksLower)
      return i;
    if (!wantUpperLeg && looksLower)
      return i;
  }
  return -1;
}

static void dfsBoneOrder(const staff_skin::Rig& rig, const std::string& nm, std::vector<int>& out) {
  auto it = rig.boneNameToIndex.find(nm);
  if (it != rig.boneNameToIndex.end())
    out.push_back(it->second);
  auto nit = rig.nodes.find(nm);
  if (nit == rig.nodes.end())
    return;
  for (const auto& ch : nit->second.children)
    dfsBoneOrder(rig, ch, out);
}

static void destroyCorpseInternal(Corpse& c) {
  if (!gWorld)
    return;
  for (reactphysics3d::Joint* j : c.joints) {
    if (j)
      gWorld->destroyJoint(j);
  }
  c.joints.clear();
  for (int i = 0; i < kParts; ++i) {
    if (c.rb[i]) {
      gWorld->destroyRigidBody(c.rb[i]);
      c.rb[i] = nullptr;
    }
  }
  if (c.floorBody) {
    gWorld->destroyRigidBody(c.floorBody);
    c.floorBody = nullptr;
  }
  c.floorCollider = nullptr;
  if (gPc) {
    if (c.shapeHead) {
      gPc->destroySphereShape(c.shapeHead);
      c.shapeHead = nullptr;
    }
    if (c.shapeChest) {
      gPc->destroyCapsuleShape(c.shapeChest);
      c.shapeChest = nullptr;
    }
    if (c.shapeWaist) {
      gPc->destroyCapsuleShape(c.shapeWaist);
      c.shapeWaist = nullptr;
    }
    if (c.shapeHip) {
      gPc->destroyCapsuleShape(c.shapeHip);
      c.shapeHip = nullptr;
    }
    if (c.shapeArm) {
      gPc->destroyCapsuleShape(c.shapeArm);
      c.shapeArm = nullptr;
    }
    if (c.shapeForeArm) {
      gPc->destroyCapsuleShape(c.shapeForeArm);
      c.shapeForeArm = nullptr;
    }
    if (c.shapeThigh) {
      gPc->destroyCapsuleShape(c.shapeThigh);
      c.shapeThigh = nullptr;
    }
    if (c.shapeShin) {
      gPc->destroyCapsuleShape(c.shapeShin);
      c.shapeShin = nullptr;
    }
  }
}

} // namespace

bool init(float gravityY) {
  if (gPc)
    return true;
  gPc = new reactphysics3d::PhysicsCommon();
  reactphysics3d::PhysicsWorld::WorldSettings ws;
  ws.worldName = "RetroIkea_staff_ragdoll";
  ws.gravity = rv3(0.f, gravityY, 0.f);
  gWorld = gPc->createPhysicsWorld(ws);
  if (gWorld && !gFloorShape)
    gFloorShape = gPc->createBoxShape(rv3(52.f, 0.35f, 52.f));
  return gWorld != nullptr;
}

void shutdown() {
  if (!gPc)
    return;
  for (auto& kv : gCorpses)
    destroyCorpseInternal(kv.second);
  gCorpses.clear();
  if (gWorld) {
    gPc->destroyPhysicsWorld(gWorld);
    gWorld = nullptr;
  }
  if (gFloorShape) {
    gPc->destroyBoxShape(gFloorShape);
    gFloorShape = nullptr;
  }
  delete gPc;
  gPc = nullptr;
}

bool active(uint64_t residentKey) { return gCorpses.find(residentKey) != gCorpses.end(); }

bool spawnCorpse(uint64_t residentKey, glm::vec2 posXZ, float feetWorldY, float yaw, glm::vec3 bodyScale,
                 glm::vec2 knockXZ, float gravityY, float employeeHeightMeters,
                 const staff_skin::Rig& rig, const glm::mat4* boneGlobalsArmature, float impactLinearScale,
                 float impactAngularScale) {
  if (!init(gravityY) || !gWorld || static_cast<int>(gCorpses.size()) >= kMaxCorpses)
    return false;
  if (gCorpses.count(residentKey))
    destroyCorpse(residentKey);

  const float H = employeeHeightMeters * bodyScale.y;
  const float layoutScale = H / 10.25f;

  const float px = posXZ.x;
  const float pz = posXZ.y;

  const glm::vec3 kHead = rotYawXZ(glm::vec3(0.f, 9.15f * layoutScale, 0.f), yaw);
  const glm::vec3 headPos(px + kHead.x, feetWorldY + kHead.y, pz + kHead.z);

  glm::vec3 kChest = rotYawXZ(glm::vec3(0.f, -1.75f * layoutScale, 0.f), yaw);
  const glm::vec3 chestPos = headPos + kChest;

  glm::vec3 kWaist = rotYawXZ(glm::vec3(0.f, -2.f * layoutScale, 0.f), yaw);
  const glm::vec3 waistPos = chestPos + kWaist;

  glm::vec3 kHip = rotYawXZ(glm::vec3(0.f, -2.f * layoutScale, 0.f), yaw);
  const glm::vec3 hipPos = waistPos + kHip;

  const glm::vec3 luaOff = rotYawXZ(glm::vec3(2.25f, 0.f, 0.f) * layoutScale, yaw);
  const glm::vec3 luaPos = chestPos + luaOff;

  const glm::vec3 llaOff = rotYawXZ(glm::vec3(2.5f, 0.f, 0.f) * layoutScale, yaw);
  const glm::vec3 llaPos = luaPos + llaOff;

  const glm::vec3 ruaOff = rotYawXZ(glm::vec3(-2.25f, 0.f, 0.f) * layoutScale, yaw);
  const glm::vec3 ruaPos = chestPos + ruaOff;

  const glm::vec3 rlaOff = rotYawXZ(glm::vec3(-2.5f, 0.f, 0.f) * layoutScale, yaw);
  const glm::vec3 rlaPos = ruaPos + rlaOff;

  const glm::vec3 lulOff = rotYawXZ(glm::vec3(0.8f, -1.5f, 0.f) * layoutScale, yaw);
  const glm::vec3 lulPos = hipPos + lulOff;

  const glm::vec3 lllOff = rotYawXZ(glm::vec3(0.f, -3.f, 0.f) * layoutScale, yaw);
  const glm::vec3 lllPos = lulPos + lllOff;

  const glm::vec3 rulOff = rotYawXZ(glm::vec3(-0.8f, -1.5f, 0.f) * layoutScale, yaw);
  const glm::vec3 rulPos = hipPos + rulOff;

  const glm::vec3 rllOff = rotYawXZ(glm::vec3(0.f, -3.f, 0.f) * layoutScale, yaw);
  const glm::vec3 rllPos = rulPos + rllOff;

  Corpse c{};
  c.key = residentKey;
  c.shapeHead = gPc->createSphereShape(decimal(0.75 * layoutScale));
  c.shapeChest = gPc->createCapsuleShape(decimal(1.0 * layoutScale), decimal(1.5 * layoutScale));
  c.shapeWaist = gPc->createCapsuleShape(decimal(1.0 * layoutScale), decimal(1.5 * layoutScale));
  c.shapeHip = gPc->createCapsuleShape(decimal(1.0 * layoutScale), decimal(1.0 * layoutScale));
  c.shapeArm = gPc->createCapsuleShape(decimal(0.5 * layoutScale), decimal(2.0 * layoutScale));
  c.shapeForeArm = gPc->createCapsuleShape(decimal(0.5 * layoutScale), decimal(2.0 * layoutScale));
  c.shapeThigh = gPc->createCapsuleShape(decimal(0.75 * layoutScale), decimal(2.0 * layoutScale));
  c.shapeShin = gPc->createCapsuleShape(decimal(0.5 * layoutScale), decimal(3.0 * layoutScale));
  for (int i = 0; i < kParts; ++i)
    c.boneOfPart[i] = -1;
  for (int i = 0; i < staff_skin::kMaxPaletteBones; ++i)
    c.G0_bind[i] = glm::mat4(1.f);

  c.boneOfPart[P_HEAD] = findBoneSubstr(rig, "head", "top");
  // Mixamo: Spine -> Spine01 -> Spine02; UE: spine_01, spine_02, chest. Prefer highest spine for chest.
  c.boneOfPart[P_CHEST] = findBoneSubstr(rig, "spine02");
  if (c.boneOfPart[P_CHEST] < 0)
    c.boneOfPart[P_CHEST] = findBoneSubstr(rig, "spine_02");
  if (c.boneOfPart[P_CHEST] < 0)
    c.boneOfPart[P_CHEST] = findBoneSubstr(rig, "spine2");
  if (c.boneOfPart[P_CHEST] < 0)
    c.boneOfPart[P_CHEST] = findBoneSubstr(rig, "chest");
  if (c.boneOfPart[P_CHEST] < 0)
    c.boneOfPart[P_CHEST] = findBoneSubstr(rig, "spine");
  c.boneOfPart[P_WAIST] = findBoneSubstr(rig, "spine01");
  if (c.boneOfPart[P_WAIST] < 0)
    c.boneOfPart[P_WAIST] = findBoneSubstr(rig, "spine_01");
  if (c.boneOfPart[P_WAIST] < 0)
    c.boneOfPart[P_WAIST] = findBoneSubstr(rig, "spine1");
  if (c.boneOfPart[P_WAIST] < 0 && c.boneOfPart[P_CHEST] >= 0) {
    // Single-segment spine rigs: pick another "spine" bone index when possible.
    for (int i = 0; i < rig.boneCount; ++i) {
      if (i == c.boneOfPart[P_CHEST])
        continue;
      std::string lower = rig.boneNames[static_cast<size_t>(i)];
      for (char& ch : lower)
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      if (lower.find("spine") != std::string::npos) {
        c.boneOfPart[P_WAIST] = i;
        break;
      }
    }
  }
  c.boneOfPart[P_HIP] = findBoneSubstr(rig, "pelvis");
  if (c.boneOfPart[P_HIP] < 0)
    c.boneOfPart[P_HIP] = findBoneSubstr(rig, "hips");
  c.boneOfPart[P_LUA] = findArmBone(rig, true, true);
  c.boneOfPart[P_LLA] = findArmBone(rig, true, false);
  c.boneOfPart[P_RUA] = findArmBone(rig, false, true);
  c.boneOfPart[P_RLA] = findArmBone(rig, false, false);

  c.boneOfPart[P_LUL] = findLegBone(rig, true, true);
  c.boneOfPart[P_LLL] = findLegBone(rig, true, false);
  c.boneOfPart[P_RUL] = findLegBone(rig, false, true);
  c.boneOfPart[P_RLL] = findLegBone(rig, false, false);

  if (boneGlobalsArmature) {
    for (int i = 0; i < rig.boneCount; ++i)
      c.G0_bind[static_cast<size_t>(i)] = boneGlobalsArmature[i];
  } else {
    glm::mat4 Gtmp[staff_skin::kMaxPaletteBones];
    staff_skin::sampleBindBoneGlobalMatricesWithExtras(rig, nullptr, Gtmp);
    for (int i = 0; i < rig.boneCount; ++i)
      c.G0_bind[static_cast<size_t>(i)] = Gtmp[i];
  }

  const glm::mat4 rotM = glm::rotate(glm::mat4(1.f), yaw, glm::vec3(0.f, 1.f, 0.f));
  const glm::mat4 M = glm::translate(glm::mat4(1.f), glm::vec3(px, feetWorldY, pz)) * rotM *
                      glm::scale(glm::mat4(1.f), bodyScale);
  c.M0 = M;

  auto trFromBoneOr = [&](int part, reactphysics3d::Transform layoutTr) -> reactphysics3d::Transform {
    const int bi = c.boneOfPart[part];
    if (bi >= 0 && bi < rig.boneCount)
      return mat4WorldToRp3d(M * c.G0_bind[static_cast<size_t>(bi)]);
    return layoutTr;
  };

  // Dynamic ragdoll parts share category 0x0002; they only collide with default scene geometry (0x0001),
  // not with each other — overlapping capsules were generating huge impulses (spinning / spaghetti mesh).
  constexpr unsigned short kCatRagdollPart = 0x0002;
  constexpr unsigned short kMaskWorldGeometry = 0x0001;

  auto makeDynamic = [&](reactphysics3d::CollisionShape* shape, const reactphysics3d::Transform& tr,
                         int partIdx) -> reactphysics3d::RigidBody* {
    reactphysics3d::RigidBody* body = gWorld->createRigidBody(tr);
    body->setType(reactphysics3d::BodyType::DYNAMIC);
    reactphysics3d::Collider* col = body->addCollider(shape, reactphysics3d::Transform::identity());
    col->setCollisionCategoryBits(kCatRagdollPart);
    col->setCollideWithMaskBits(kMaskWorldGeometry);
    col->getMaterial().setMassDensity(decimal(8));
    col->getMaterial().setFrictionCoefficient(decimal(0.55));
    body->updateMassPropertiesFromColliders();
    body->setLinearDamping(decimal(0.12));
    body->setAngularDamping(decimal(0.25));
    (void)partIdx;
    return body;
  };

  c.rb[P_HEAD] =
      makeDynamic(c.shapeHead, trFromBoneOr(P_HEAD, trPosEuler(headPos.x, headPos.y, headPos.z, 0, 0, 0)), P_HEAD);
  c.rb[P_CHEST] =
      makeDynamic(c.shapeChest,
                  trFromBoneOr(P_CHEST, trPosEuler(chestPos.x, chestPos.y, chestPos.z, 0, 0,
                                                   static_cast<float>(3.14159265 * 0.5))),
                  P_CHEST);
  c.rb[P_WAIST] = makeDynamic(c.shapeWaist,
                              trFromBoneOr(P_WAIST, trPosEuler(waistPos.x, waistPos.y, waistPos.z, 0, 0, 0)),
                              P_WAIST);
  c.rb[P_HIP] =
      makeDynamic(c.shapeHip,
                  trFromBoneOr(P_HIP, trPosEuler(hipPos.x, hipPos.y, hipPos.z, 0, 0,
                                                 static_cast<float>(3.14159265 * 0.5))),
                  P_HIP);
  c.rb[P_LUA] =
      makeDynamic(c.shapeArm,
                  trFromBoneOr(P_LUA, trPosEuler(luaPos.x, luaPos.y, luaPos.z, 0, 0,
                                                 static_cast<float>(3.14159265 * 0.5))),
                  P_LUA);
  c.rb[P_LLA] =
      makeDynamic(c.shapeForeArm,
                  trFromBoneOr(P_LLA, trPosEuler(llaPos.x, llaPos.y, llaPos.z, 0, 0,
                                                 static_cast<float>(3.14159265 * 0.5))),
                  P_LLA);
  c.rb[P_RUA] =
      makeDynamic(c.shapeArm,
                  trFromBoneOr(P_RUA, trPosEuler(ruaPos.x, ruaPos.y, ruaPos.z, 0, 0,
                                                 static_cast<float>(3.14159265 * 0.5))),
                  P_RUA);
  c.rb[P_RLA] =
      makeDynamic(c.shapeForeArm,
                  trFromBoneOr(P_RLA, trPosEuler(rlaPos.x, rlaPos.y, rlaPos.z, 0, 0,
                                                 static_cast<float>(3.14159265 * 0.5))),
                  P_RLA);
  c.rb[P_LUL] =
      makeDynamic(c.shapeThigh,
                  trFromBoneOr(P_LUL, trPosEuler(lulPos.x, lulPos.y, lulPos.z, 0, 0, 0)), P_LUL);
  c.rb[P_LLL] =
      makeDynamic(c.shapeShin,
                  trFromBoneOr(P_LLL, trPosEuler(lllPos.x, lllPos.y, lllPos.z, 0, 0, 0)), P_LLL);
  c.rb[P_RUL] =
      makeDynamic(c.shapeThigh,
                  trFromBoneOr(P_RUL, trPosEuler(rulPos.x, rulPos.y, rulPos.z, 0, 0, 0)), P_RUL);
  c.rb[P_RLL] =
      makeDynamic(c.shapeShin,
                  trFromBoneOr(P_RLL, trPosEuler(rllPos.x, rllPos.y, rllPos.z, 0, 0, 0)), P_RLL);

  for (int i = 0; i < kParts; ++i)
    c.invW0[i] = glm::inverse(matFromRp3d(c.rb[i]->getTransform()));

  auto addBall = [&](reactphysics3d::RigidBody* a, reactphysics3d::RigidBody* b, const glm::vec3& anchor,
                     float coneDeg) {
    reactphysics3d::BallAndSocketJointInfo ji(a, b, rv3(anchor.x, anchor.y, anchor.z));
    ji.isCollisionEnabled = false;
    reactphysics3d::Joint* j = gWorld->createJoint(ji);
    if (auto* bs = dynamic_cast<reactphysics3d::BallAndSocketJoint*>(j)) {
      bs->enableConeLimit(true);
      bs->setConeLimitHalfAngle(decimal(coneDeg * 3.14159265 / 180.0));
    }
    c.joints.push_back(j);
  };
  auto addHinge = [&](reactphysics3d::RigidBody* a, reactphysics3d::RigidBody* b, const glm::vec3& anchor,
                      const glm::vec3& axis, float minDeg, float maxDeg) {
    reactphysics3d::HingeJointInfo ji(a, b, rv3(anchor.x, anchor.y, anchor.z), rv3(axis.x, axis.y, axis.z),
                                      decimal(minDeg * 3.14159265 / 180.0),
                                      decimal(maxDeg * 3.14159265 / 180.0));
    ji.isCollisionEnabled = false;
    c.joints.push_back(gWorld->createJoint(ji));
  };
  auto addFixed = [&](reactphysics3d::RigidBody* a, reactphysics3d::RigidBody* b, const glm::vec3& anchor) {
    reactphysics3d::FixedJointInfo ji(a, b, rv3(anchor.x, anchor.y, anchor.z));
    ji.isCollisionEnabled = false;
    c.joints.push_back(gWorld->createJoint(ji));
  };

  auto jointMid = [&](int pa, int pb, const glm::vec3& fallback) -> glm::vec3 {
    const int ba = c.boneOfPart[pa];
    const int bb = c.boneOfPart[pb];
    if (ba >= 0 && bb >= 0 && ba < rig.boneCount && bb < rig.boneCount)
      return 0.5f * (mat4Trans(M * c.G0_bind[ba]) + mat4Trans(M * c.G0_bind[bb]));
    return fallback;
  };

  glm::vec3 mid;

  mid = jointMid(P_HEAD, P_CHEST, glm::mix(glm::vec3(headPos), glm::vec3(chestPos), 0.5f));
  addBall(c.rb[P_HEAD], c.rb[P_CHEST], mid, 40.f);

  mid = jointMid(P_LUA, P_LLA, glm::mix(glm::vec3(luaPos), glm::vec3(llaPos), 0.5f));
  addHinge(c.rb[P_LUA], c.rb[P_LLA], mid, glm::vec3(0.f, 0.f, 1.f), 0.f, 340.f);

  mid = jointMid(P_CHEST, P_WAIST, glm::mix(glm::vec3(chestPos), glm::vec3(waistPos), 0.5f));
  addFixed(c.rb[P_CHEST], c.rb[P_WAIST], mid);

  mid = jointMid(P_WAIST, P_HIP, glm::mix(glm::vec3(waistPos), glm::vec3(hipPos), 0.5f));
  addFixed(c.rb[P_WAIST], c.rb[P_HIP], mid);

  mid = jointMid(P_HIP, P_LUL, hipPos + rotYawXZ(glm::vec3(0.8f, 0.f, 0.f) * layoutScale, yaw));
  addBall(c.rb[P_HIP], c.rb[P_LUL], mid, 80.f);

  mid = jointMid(P_LUL, P_LLL, glm::mix(glm::vec3(lulPos), glm::vec3(lllPos), 0.5f));
  addHinge(c.rb[P_LUL], c.rb[P_LLL], mid, glm::vec3(1.f, 0.f, 0.f), 0.f, 140.f);

  mid = jointMid(P_CHEST, P_RUA, chestPos + rotYawXZ(glm::vec3(-2.25f, 0.f, 0.f) * layoutScale, yaw) +
                                       rotYawXZ(glm::vec3(1.f, 0.f, 0.f) * layoutScale, yaw));
  addBall(c.rb[P_CHEST], c.rb[P_RUA], mid, 180.f);

  mid = jointMid(P_RUA, P_RLA, glm::mix(glm::vec3(ruaPos), glm::vec3(rlaPos), 0.5f));
  addHinge(c.rb[P_RUA], c.rb[P_RLA], mid, glm::vec3(0.f, 0.f, 1.f), 0.f, 340.f);

  mid = jointMid(P_HIP, P_RUL, hipPos + rotYawXZ(glm::vec3(-0.8f, 0.f, 0.f) * layoutScale, yaw));
  addBall(c.rb[P_HIP], c.rb[P_RUL], mid, 80.f);

  mid = jointMid(P_RUL, P_RLL, glm::mix(glm::vec3(rulPos), glm::vec3(rllPos), 0.5f));
  addHinge(c.rb[P_RUL], c.rb[P_RLL], mid, glm::vec3(1.f, 0.f, 0.f), 0.f, 140.f);

  mid = jointMid(P_CHEST, P_LUA, chestPos + rotYawXZ(glm::vec3(2.25f, 0.f, 0.f) * layoutScale, yaw));
  addBall(c.rb[P_CHEST], c.rb[P_LUA], mid, 180.f);

  {
    reactphysics3d::RigidBody* floorB = gWorld->createRigidBody(
        reactphysics3d::Transform(rv3(px, feetWorldY - 0.18f, pz), reactphysics3d::Quaternion::identity()));
    floorB->setType(reactphysics3d::BodyType::STATIC);
    c.floorCollider = floorB->addCollider(gFloorShape, reactphysics3d::Transform::identity());
    // Default category 0x0001 — ragdoll parts use collide mask 0x0001 so they still hit this floor.
    c.floorCollider->getMaterial().setFrictionCoefficient(decimal(0.85));
    c.floorBody = floorB;
  }

  const reactphysics3d::Vector3 zeroV(0, 0, 0);
  for (int pi = 0; pi < kParts; ++pi) {
    if (c.rb[pi]) {
      c.rb[pi]->setAngularVelocity(zeroV);
      c.rb[pi]->setLinearVelocity(zeroV);
    }
  }

  glm::vec2 kn = knockXZ;
  if (glm::dot(kn, kn) > 1e-10f)
    kn = glm::normalize(kn);
  else
    kn = glm::vec2(std::sin(yaw), std::cos(yaw));

  const float linS = glm::clamp(impactLinearScale, 0.2f, 4.f);
  const float angS = glm::clamp(impactAngularScale, 0.f, 4.f);

  const glm::vec3 kickVel(kn.x * (5.2f * linS), 2.7f * linS, kn.y * (5.2f * linS));
  const glm::vec3 chestBoost(kn.x * (3.5f * linS), 2.0f * linS, kn.y * (3.5f * linS));

  glm::vec3 kd(kn.x, 0.f, kn.y);
  glm::vec3 wHip(0.f), wChest(0.f), wHead(0.f);
  if (angS > 1e-4f && glm::dot(kd, kd) > 1e-10f) {
    kd = glm::normalize(kd);
    glm::vec3 tumbleAxis = glm::cross(glm::vec3(0.f, 1.f, 0.f), kd);
    const float tal = glm::length(tumbleAxis);
    if (tal > 1e-4f)
      tumbleAxis /= tal;
    const float sp = 9.f * angS;
    wHip = tumbleAxis * sp + glm::vec3(-kd.z * 5.f * angS, kd.x * 3.5f * angS, kd.x * 5.f * angS);
    wChest = tumbleAxis * (sp * 0.62f) + glm::vec3(kd.x * 4.f * angS, -3.2f * angS, kd.z * 4.f * angS);
    wHead = tumbleAxis * (sp * 0.42f);
  }

  if (c.rb[P_HIP]) {
    c.rb[P_HIP]->setLinearVelocity(rv3(kickVel.x, kickVel.y, kickVel.z));
    c.rb[P_HIP]->setAngularVelocity(rv3(wHip.x, wHip.y, wHip.z));
  }
  if (c.rb[P_CHEST]) {
    c.rb[P_CHEST]->setLinearVelocity(rv3(chestBoost.x, chestBoost.y, chestBoost.z));
    c.rb[P_CHEST]->setAngularVelocity(rv3(wChest.x, wChest.y, wChest.z));
  }
  if (c.rb[P_HEAD])
    c.rb[P_HEAD]->setAngularVelocity(rv3(wHead.x, wHead.y, wHead.z));

  const glm::vec3 armLin(kn.x * (3.4f * linS), 1.1f * linS, kn.y * (3.4f * linS));
  const glm::vec3 shinLin(kn.x * (2.6f * linS), -0.6f * linS, kn.y * (2.6f * linS));
  if (c.rb[P_LUA])
    c.rb[P_LUA]->setLinearVelocity(rv3(armLin.x, armLin.y, armLin.z));
  if (c.rb[P_RUA])
    c.rb[P_RUA]->setLinearVelocity(rv3(armLin.x, armLin.y, armLin.z));
  if (c.rb[P_LLL])
    c.rb[P_LLL]->setLinearVelocity(rv3(shinLin.x, shinLin.y, shinLin.z));
  if (c.rb[P_RLL])
    c.rb[P_RLL]->setLinearVelocity(rv3(shinLin.x, shinLin.y, shinLin.z));

  gCorpses[residentKey] = std::move(c);
  return true;
}

void destroyCorpse(uint64_t residentKey) {
  auto it = gCorpses.find(residentKey);
  if (it == gCorpses.end())
    return;
  destroyCorpseInternal(it->second);
  gCorpses.erase(it);
}

void step(float dt) {
  if (!gWorld || dt <= 0.f)
    return;
  constexpr int kSubsteps = 3;
  const decimal h = decimal(dt) / decimal(kSubsteps);
  for (int s = 0; s < kSubsteps; ++s)
    gWorld->update(h);
}

void syncNpcFromPhysics(uint64_t residentKey, glm::vec2& posXZ, float& feetWorldY, glm::vec3 bodyScale) {
  auto it = gCorpses.find(residentKey);
  if (it == gCorpses.end())
    return;
  Corpse& c = it->second;
  reactphysics3d::RigidBody* hip = c.rb[c.hipPartIdx];
  if (!hip)
    return;
  const glm::mat4 Wh = matFromRp3d(hip->getTransform());
  const glm::vec3 pxz(Wh[3][0], Wh[3][1], Wh[3][2]);
  posXZ = glm::vec2(pxz.x, pxz.z);
  feetWorldY = pxz.y - 0.92f * bodyScale.y;
}

bool fillBonePalette(uint64_t residentKey, const staff_skin::Rig& rig, const glm::mat4& npcModelMatrix,
                     glm::mat4* outPalette) {
  auto it = gCorpses.find(residentKey);
  if (it == gCorpses.end())
    return false;
  Corpse& c = it->second;
  const glm::mat4 invM = glm::inverse(npcModelMatrix);

  glm::mat4 Gbind[staff_skin::kMaxPaletteBones];
  staff_skin::sampleBindBoneGlobalMatricesWithExtras(rig, nullptr, Gbind);

  glm::mat4 G[staff_skin::kMaxPaletteBones];
  for (int i = 0; i < staff_skin::kMaxPaletteBones; ++i)
    G[i] = glm::mat4(1.f);

  std::vector<uint8_t> filled(static_cast<size_t>(rig.boneCount), 0);

  std::vector<int> order;
  dfsBoneOrder(rig, rig.rootName, order);

  for (int bi : order) {
    int rbIdx = -1;
    for (int p = 0; p < kParts; ++p) {
      if (c.boneOfPart[p] == bi) {
        rbIdx = p;
        break;
      }
    }
    const std::string& nm = rig.boneNames[static_cast<size_t>(bi)];
    auto nit = rig.nodes.find(nm);
    if (nit == rig.nodes.end()) {
      G[bi] = Gbind[bi];
      filled[static_cast<size_t>(bi)] = 1;
      continue;
    }

    if (rbIdx >= 0 && c.rb[rbIdx]) {
      const glm::mat4 W = matFromRp3d(c.rb[rbIdx]->getTransform());
      G[bi] = invM * W * c.invW0[rbIdx] * c.M0 * c.G0_bind[bi];
    } else if (nit->second.parent.empty()) {
      G[bi] = nit->second.bindLocal;
    } else {
      auto pit = rig.boneNameToIndex.find(nit->second.parent);
      if (pit == rig.boneNameToIndex.end())
        G[bi] = nit->second.bindLocal;
      else
        G[bi] = G[pit->second] * nit->second.bindLocal;
    }
    filled[static_cast<size_t>(bi)] = 1;
  }

  for (int i = 0; i < rig.boneCount; ++i) {
    if (!filled[static_cast<size_t>(i)])
      G[i] = Gbind[i];
  }

  staff_skin::computePaletteFromBoneGlobals(rig, G, outPalette);
  return true;
}

} // namespace staff_rp3d
