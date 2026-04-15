#include "staff_skin.hpp"

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <SDL2/SDL_image.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace staff_skin {

namespace {

constexpr glm::vec3 kEmployeeTag{0.52f, 0.88f, 0.91f};

enum class EmpTexPart : int { Jeans = 0, Shirt = 1, Skin = 2 };

static float partAlpha(int part) {
  return 0.02f * static_cast<float>(part + 1);
}

static std::string toLowerCopy(aiString s) {
  std::string o(s.C_Str());
  for (char& c : o)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return o;
}

static bool contains(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
}

static bool hasWord(const std::string& s, const char* w) {
  const size_t n = std::strlen(w);
  if (n == 0 || s.size() < n)
    return false;
  size_t pos = 0;
  while (true) {
    pos = s.find(w, pos);
    if (pos == std::string::npos)
      return false;
    const bool leftOk = pos == 0 || !std::isalnum(static_cast<unsigned char>(s[pos - 1]));
    const bool rightOk =
        pos + n >= s.size() || !std::isalnum(static_cast<unsigned char>(s[pos + n]));
    if (leftOk && rightOk)
      return true;
    ++pos;
  }
}

static int classifyEmployeePart(const aiMesh* mesh, const aiNode* node, const aiScene* scene) {
  std::string blob = toLowerCopy(mesh->mName);
  blob.push_back(' ');
  blob += toLowerCopy(node->mName);
  if (mesh->mMaterialIndex < scene->mNumMaterials) {
    const aiMaterial* mat = scene->mMaterials[mesh->mMaterialIndex];
    aiString mn;
    if (mat->Get(AI_MATKEY_NAME, mn) == AI_SUCCESS) {
      blob.push_back(' ');
      blob += toLowerCopy(mn);
    }
  }
  auto sub = [&](const char* w) { return contains(blob, w); };
  auto word = [&](const char* w) { return hasWord(blob, w); };
  if (word("head") || word("face") || sub("hair") || sub("scalp") || word("neck"))
    return static_cast<int>(EmpTexPart::Skin);
  if (word("hand") || sub("finger") || sub("thumb") || word("wrist") || sub("palm"))
    return static_cast<int>(EmpTexPart::Skin);
  if (word("arm") || sub("elbow") || sub("shoulder") || sub("clavicle"))
    return static_cast<int>(EmpTexPart::Skin);
  if (word("foot") || sub("feet") || word("toe") || sub("ankle"))
    return static_cast<int>(EmpTexPart::Skin);
  if (sub("pants") || sub("jeans") || sub("trouser") || sub("shorts") || sub("glute") || sub("butt") ||
      sub("hips") || sub("pelvis"))
    return static_cast<int>(EmpTexPart::Jeans);
  if (sub("thigh") || sub("calf") || sub("knee"))
    return static_cast<int>(EmpTexPart::Jeans);
  if (word("leg") || sub("leg_") || sub("_leg"))
    return static_cast<int>(EmpTexPart::Jeans);
  if (word("boot") || sub("boots") || sub("shoe") || sub("shoes"))
    return static_cast<int>(EmpTexPart::Jeans);
  if (sub("shirt") || sub("torso") || sub("chest") || sub("jacket") || sub("coat") || sub("vest") ||
      sub("sweater") || sub("hoodie") || sub("polo") || sub("uniform"))
    return static_cast<int>(EmpTexPart::Shirt);
  if (sub("tshirt") || sub("t-shirt") || sub("tee"))
    return static_cast<int>(EmpTexPart::Shirt);
  return -1;
}

glm::mat4 aiMatToGlm(const aiMatrix4x4& from) {
  glm::mat4 to;
  to[0][0] = from.a1;
  to[1][0] = from.a2;
  to[2][0] = from.a3;
  to[3][0] = from.a4;
  to[0][1] = from.b1;
  to[1][1] = from.b2;
  to[2][1] = from.b3;
  to[3][1] = from.b4;
  to[0][2] = from.c1;
  to[1][2] = from.c2;
  to[2][2] = from.c3;
  to[3][2] = from.c4;
  to[0][3] = from.d1;
  to[1][3] = from.d2;
  to[2][3] = from.d3;
  to[3][3] = from.d4;
  return to;
}

glm::quat aiQuatToGlm(const aiQuaternion& q) {
  return glm::quat(q.w, q.x, q.y, q.z);
}

static bool decodeAiTextureToRgba(const aiTexture* tex, std::vector<uint8_t>& out, uint32_t& tw,
                                  uint32_t& th) {
  if (!tex || !tex->pcData)
    return false;
  if (tex->mHeight == 0) {
    SDL_RWops* rw = SDL_RWFromConstMem(tex->pcData, static_cast<int>(tex->mWidth));
    if (!rw)
      return false;
    SDL_Surface* loaded = IMG_Load_RW(rw, 1);
    if (!loaded)
      return false;
    SDL_Surface* rgba = SDL_ConvertSurfaceFormat(loaded, SDL_PIXELFORMAT_RGBA32, 0);
    SDL_FreeSurface(loaded);
    if (!rgba)
      return false;
    tw = static_cast<uint32_t>(rgba->w);
    th = static_cast<uint32_t>(rgba->h);
    out.resize(static_cast<size_t>(tw) * th * 4);
    uint8_t* d = out.data();
    if (SDL_MUSTLOCK(rgba))
      SDL_LockSurface(rgba);
    const auto* src = static_cast<const uint8_t*>(rgba->pixels);
    for (uint32_t y = 0; y < th; ++y)
      std::memcpy(d + static_cast<size_t>(y) * tw * 4,
                  src + static_cast<size_t>(y) * static_cast<size_t>(rgba->pitch),
                  static_cast<size_t>(tw) * 4);
    if (SDL_MUSTLOCK(rgba))
      SDL_UnlockSurface(rgba);
    SDL_FreeSurface(rgba);
    return tw > 0 && th > 0;
  }
  tw = tex->mWidth;
  th = tex->mHeight;
  if (tw == 0 || th == 0 || tw > 16384u || th > 16384u)
    return false;
  const size_t n = static_cast<size_t>(tw) * th;
  out.resize(n * 4);
  for (size_t i = 0; i < n; ++i) {
    const aiTexel& p = tex->pcData[i];
    out[i * 4 + 0] = p.r;
    out[i * 4 + 1] = p.g;
    out[i * 4 + 2] = p.b;
    out[i * 4 + 3] = p.a;
  }
  return true;
}

static bool extractDiffuseFromMaterial(const aiScene* scene, unsigned matIndex, const std::string& modelDir,
                                       std::vector<uint8_t>& rgba, uint32_t& w, uint32_t& h) {
  if (matIndex >= scene->mNumMaterials)
    return false;
  const aiMaterial* mat = scene->mMaterials[matIndex];
  static const aiTextureType kTry[] = {aiTextureType_BASE_COLOR, aiTextureType_DIFFUSE};
  for (aiTextureType tt : kTry) {
    const unsigned tc = mat->GetTextureCount(tt);
    for (unsigned ti = 0; ti < tc; ++ti) {
      aiString p;
      if (mat->GetTexture(tt, ti, &p) != AI_SUCCESS)
        continue;
      const aiTexture* emb = scene->GetEmbeddedTexture(p.C_Str());
      if (emb) {
        if (decodeAiTextureToRgba(emb, rgba, w, h))
          return true;
      } else {
        const std::string full = modelDir.empty() ? std::string(p.C_Str()) : modelDir + p.C_Str();
        SDL_Surface* loaded = IMG_Load(full.c_str());
        if (!loaded)
          continue;
        SDL_Surface* conv = SDL_ConvertSurfaceFormat(loaded, SDL_PIXELFORMAT_RGBA32, 0);
        SDL_FreeSurface(loaded);
        if (!conv)
          continue;
        w = static_cast<uint32_t>(conv->w);
        h = static_cast<uint32_t>(conv->h);
        rgba.resize(static_cast<size_t>(w) * h * 4);
        uint8_t* d = rgba.data();
        if (SDL_MUSTLOCK(conv))
          SDL_LockSurface(conv);
        const auto* src = static_cast<const uint8_t*>(conv->pixels);
        for (uint32_t y = 0; y < h; ++y)
          std::memcpy(d + static_cast<size_t>(y) * w * 4,
                      src + static_cast<size_t>(y) * static_cast<size_t>(conv->pitch),
                      static_cast<size_t>(w) * 4);
        if (SDL_MUSTLOCK(conv))
          SDL_UnlockSurface(conv);
        SDL_FreeSurface(conv);
        return w > 0 && h > 0;
      }
    }
  }
  return false;
}

// Prefer the base-color map on the material that covers most skinned vertices (matches mesh UVs), not
// the first material slot in the file (often a tiny auxiliary map).
static bool extractDiffuseFromDominantSkinnedMaterial(const aiScene* scene, const std::string& modelDir,
                                                      std::vector<uint8_t>& rgba, uint32_t& w, uint32_t& h) {
  std::unordered_map<unsigned, size_t> votes;
  for (unsigned mi = 0; mi < scene->mNumMeshes; ++mi) {
    const aiMesh* mesh = scene->mMeshes[mi];
    if (!mesh->HasBones())
      continue;
    votes[mesh->mMaterialIndex] += mesh->mNumVertices;
  }
  if (votes.empty())
    return false;
  std::vector<std::pair<unsigned, size_t>> order;
  order.reserve(votes.size());
  for (const auto& kv : votes)
    order.push_back(kv);
  std::sort(order.begin(), order.end(),
            [](const std::pair<unsigned, size_t>& a, const std::pair<unsigned, size_t>& b) {
              return a.second > b.second;
            });
  for (const auto& pr : order) {
    if (extractDiffuseFromMaterial(scene, pr.first, modelDir, rgba, w, h))
      return true;
  }
  return false;
}

static bool extractFirstStaffDiffuse(const aiScene* scene, const std::string& modelDir,
                                     std::vector<uint8_t>& rgba, uint32_t& w, uint32_t& h) {
  for (unsigned mi = 0; mi < scene->mNumMaterials; ++mi) {
    if (extractDiffuseFromMaterial(scene, mi, modelDir, rgba, w, h))
      return true;
  }
  return false;
}

void buildNodeHierarchy(const aiNode* node, const std::string& parentName, Rig& rig) {
  const std::string name = node->mName.C_Str();
  if (!parentName.empty()) {
    rig.nodes[name].parent = parentName;
    rig.nodes[parentName].children.push_back(name);
  } else {
    rig.rootName = name;
  }
  rig.nodes[name].bindLocal = aiMatToGlm(node->mTransformation);
  for (unsigned i = 0; i < node->mNumChildren; ++i)
    buildNodeHierarchy(node->mChildren[i], name, rig);
}

static glm::mat4 composeTrs(const glm::vec3& t, const glm::quat& r, const glm::vec3& s) {
  glm::mat4 M = glm::translate(glm::mat4(1.f), t) * glm::mat4_cast(r) * glm::scale(glm::mat4(1.f), s);
  return M;
}

static glm::vec3 interpVec(const std::vector<std::pair<double, glm::vec3>>& keys, double t,
                           const glm::vec3& defV) {
  if (keys.empty())
    return defV;
  if (t <= keys.front().first)
    return keys.front().second;
  if (t >= keys.back().first)
    return keys.back().second;
  for (size_t i = 0; i + 1 < keys.size(); ++i) {
    if (keys[i + 1].first >= t) {
      const double t0 = keys[i].first;
      const double t1 = keys[i + 1].first;
      const float a = static_cast<float>((t - t0) / std::max(1e-8, t1 - t0));
      const float s = a * a * (3.f - 2.f * a);
      return glm::mix(keys[i].second, keys[i + 1].second, s);
    }
  }
  return keys.back().second;
}

static glm::quat interpQuat(const std::vector<std::pair<double, glm::quat>>& keys, double t,
                            const glm::quat& defQ) {
  if (keys.empty())
    return defQ;
  if (t <= keys.front().first)
    return keys.front().second;
  if (t >= keys.back().first)
    return keys.back().second;
  for (size_t i = 0; i + 1 < keys.size(); ++i) {
    if (keys[i + 1].first >= t) {
      const double t0 = keys[i].first;
      const double t1 = keys[i + 1].first;
      const float a = static_cast<float>((t - t0) / std::max(1e-8, t1 - t0));
      const float s = a * a * (3.f - 2.f * a);
      return glm::normalize(glm::slerp(keys[i].second, keys[i + 1].second, s));
    }
  }
  return keys.back().second;
}

static glm::vec3 matScale(const glm::mat4& m) {
  return {glm::length(glm::vec3(m[0])), glm::length(glm::vec3(m[1])), glm::length(glm::vec3(m[2]))};
}

static glm::quat matToQuat(const glm::mat4& m) {
  glm::vec3 s = matScale(m);
  glm::vec3 c0 = glm::vec3(m[0]) / std::max(1e-8f, s.x);
  glm::vec3 c1 = glm::vec3(m[1]) / std::max(1e-8f, s.y);
  glm::vec3 c2 = glm::vec3(m[2]) / std::max(1e-8f, s.z);
  glm::mat3 R(c0, c1, c2);
  return glm::normalize(glm::quat_cast(R));
}

static glm::mat4 localFromChannel(const Rig::AnimChannel* ch, double tickT, const glm::mat4& bindLocal) {
  const glm::vec3 defT(bindLocal[3]);
  const glm::quat defR = matToQuat(bindLocal);
  const glm::vec3 defS = matScale(bindLocal);
  if (!ch)
    return bindLocal;
  const glm::vec3 t = interpVec(ch->posKeys, tickT, defT);
  const glm::quat r = interpQuat(ch->rotKeys, tickT, defR);
  const glm::vec3 s = interpVec(ch->sclKeys, tickT, defS);
  return composeTrs(t, r, s);
}

// Strip baked root motion from locomotion clips: keep feet under the avatar by using bind
// translation on the armature root and typical hip bones while preserving animated rotation.
static bool shouldLockLocomotionTranslationToBind(const std::string& nm) {
  std::string lower = nm;
  for (char& c : lower)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (lower.find("hips") != std::string::npos)
    return true;
  if (lower.find("pelvis") != std::string::npos)
    return true;
  return false;
}

static void copyAnimClip(const aiAnimation* a, Rig::AnimClip& out) {
  out.duration = a->mDuration;
  out.ticksPerSecond = a->mTicksPerSecond > 1e-6 ? a->mTicksPerSecond : 25.0;
  out.channels.clear();
  out.channelByNode.clear();
  for (unsigned i = 0; i < a->mNumChannels; ++i) {
    const aiNodeAnim* na = a->mChannels[i];
    Rig::AnimChannel ch;
    ch.nodeName = na->mNodeName.C_Str();
    for (unsigned k = 0; k < na->mNumPositionKeys; ++k) {
      const aiVector3D& v = na->mPositionKeys[k].mValue;
      ch.posKeys.push_back({na->mPositionKeys[k].mTime, glm::vec3(v.x, v.y, v.z)});
    }
    for (unsigned k = 0; k < na->mNumRotationKeys; ++k) {
      ch.rotKeys.push_back({na->mRotationKeys[k].mTime, aiQuatToGlm(na->mRotationKeys[k].mValue)});
    }
    for (unsigned k = 0; k < na->mNumScalingKeys; ++k) {
      const aiVector3D& v = na->mScalingKeys[k].mValue;
      ch.sclKeys.push_back({na->mScalingKeys[k].mTime, glm::vec3(v.x, v.y, v.z)});
    }
    out.channelByNode[ch.nodeName] = out.channels.size();
    out.channels.push_back(std::move(ch));
  }
}

static std::string boneMatchKey(std::string s) {
  for (char& c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  const char* mix = "mixamorig:";
  if (s.size() >= 10 && s.compare(0, 10, mix) == 0)
    s = s.substr(10);
  const auto pipe = s.rfind('|');
  if (pipe != std::string::npos)
    s = s.substr(pipe + 1);
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
    s.erase(0, 1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
    s.pop_back();
  // Blender duplicate names: "Hips.001" vs "Hips" on appended clips.
  for (;;) {
    const auto dot = s.rfind('.');
    if (dot == std::string::npos || dot + 1 >= s.size())
      break;
    size_t j = dot + 1;
    bool allDigit = true;
    for (; j < s.size(); ++j) {
      if (!std::isdigit(static_cast<unsigned char>(s[j]))) {
        allDigit = false;
        break;
      }
    }
    if (!allDigit || j <= dot + 1)
      break;
    s.resize(dot);
  }
  return s;
}

static bool pathLowerContains(const char* path, const char* needle) {
  if (!path || !needle || !needle[0])
    return false;
  std::string p(path);
  for (char& c : p)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  std::string n(needle);
  for (char& c : n)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return p.find(n) != std::string::npos;
}

// True for the main pelvis / hips node (Mixamo "Hips"), not thighs.
static bool isPelvisHipsBone(const std::string& lower) {
  if (lower.find("pelvis") != std::string::npos)
    return true;
  if (lower.find("thigh") != std::string::npos || lower.find("upleg") != std::string::npos)
    return false;
  if (lower == "hips" || lower == "hip" || lower.find("hips") != std::string::npos)
    return true;
  return false;
}

// Bones whose rotation is blended toward bind for jump clips (reduces hunched / bent upper body).
static bool isTorsoUprightBone(const std::string& nm) {
  std::string lower = nm;
  for (char& c : lower)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (isPelvisHipsBone(lower))
    return false;
  if (lower.find("hip") != std::string::npos)
    return false;
  if (lower.find("pelvis") != std::string::npos)
    return false;
  if (lower.find("leg") != std::string::npos || lower.find("thigh") != std::string::npos ||
      lower.find("calf") != std::string::npos || lower.find("knee") != std::string::npos)
    return false;
  if (lower.find("foot") != std::string::npos || lower.find("toe") != std::string::npos)
    return false;
  if (lower.find("arm") != std::string::npos || lower.find("hand") != std::string::npos ||
      lower.find("finger") != std::string::npos || lower.find("thumb") != std::string::npos)
    return false;
  if (lower.find("spine") != std::string::npos || lower.find("chest") != std::string::npos ||
      lower.find("neck") != std::string::npos || lower.find("clavicle") != std::string::npos ||
      lower.find("shoulder") != std::string::npos)
    return true;
  if (lower.find("head") != std::string::npos)
    return true;
  return false;
}

// Retarget appended clips (e.g. Mixamo FBX) onto the rig: channel node names often differ from GLB.
static void remapAnimClipToRig(Rig::AnimClip& clip, const Rig& rig) {
  Rig::AnimClip out;
  out.duration = clip.duration;
  out.ticksPerSecond = clip.ticksPerSecond;
  out.lockLocomotionRoot = clip.lockLocomotionRoot;
  out.lockRootTranslationToBind = clip.lockRootTranslationToBind;
  out.lockHipsTranslationToBind = clip.lockHipsTranslationToBind;
  out.blendTorsoTowardBind = clip.blendTorsoTowardBind;
  out.lockJumpTorsoToBind = clip.lockJumpTorsoToBind;
  out.channels.clear();
  out.channelByNode.clear();
  for (const auto& kv : rig.nodes) {
    const std::string& rigName = kv.first;
    const Rig::AnimChannel* src = nullptr;
    auto it = clip.channelByNode.find(rigName);
    if (it != clip.channelByNode.end())
      src = &clip.channels[it->second];
    else {
      const std::string want = boneMatchKey(rigName);
      for (const auto& kv2 : clip.channelByNode) {
        if (boneMatchKey(kv2.first) == want) {
          src = &clip.channels[kv2.second];
          break;
        }
      }
    }
    if (!src)
      continue;
    Rig::AnimChannel ch = *src;
    ch.nodeName = rigName;
    out.channelByNode[rigName] = out.channels.size();
    out.channels.push_back(std::move(ch));
  }
  clip = std::move(out);
}

} // namespace

bool loadSkinnedIdleGlb(const char* path, float targetHeightMeters, std::vector<SkinnedVertex>& outVerts,
                        Rig& outRig, std::string& errOut, std::vector<uint8_t>* outDiffuseRgba,
                        uint32_t* outDiffuseW, uint32_t* outDiffuseH, bool loadAllAnimationsFromFile) {
  outVerts.clear();
  outRig = Rig{};
  errOut.clear();

  Assimp::Importer importer;
  const unsigned flags = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_LimitBoneWeights |
                         aiProcess_ImproveCacheLocality | aiProcess_RemoveRedundantMaterials |
                         aiProcess_ValidateDataStructure;
  const aiScene* scene = importer.ReadFile(path, flags);
  if (!scene || !scene->mRootNode || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) != 0) {
    errOut = importer.GetErrorString();
    return false;
  }

  std::unordered_map<std::string, int> boneMap;
  std::unordered_map<std::string, glm::mat4> boneOffsets;

  for (unsigned mi = 0; mi < scene->mNumMeshes; ++mi) {
    const aiMesh* mesh = scene->mMeshes[mi];
    if (!mesh->HasBones())
      continue;
    for (unsigned bi = 0; bi < mesh->mNumBones; ++bi) {
      const aiBone* bone = mesh->mBones[bi];
      const std::string bname = bone->mName.C_Str();
      if (boneMap.count(bname))
        continue;
      const int idx = static_cast<int>(boneMap.size());
      if (idx >= kMaxPaletteBones) {
        errOut = "too many bones (> kMaxPaletteBones)";
        return false;
      }
      boneMap[bname] = idx;
      boneOffsets[bname] = aiMatToGlm(bone->mOffsetMatrix);
    }
  }

  if (boneMap.empty()) {
    errOut = "no skinned bones in glb";
    return false;
  }

  buildNodeHierarchy(scene->mRootNode, "", outRig);
  outRig.boneNameToIndex = boneMap;
  outRig.boneCount = static_cast<int>(boneMap.size());
  outRig.boneNames.resize(static_cast<size_t>(outRig.boneCount));
  for (const auto& kv : boneMap)
    outRig.boneNames[static_cast<size_t>(kv.second)] = kv.first;
  outRig.invBindTweaked.resize(static_cast<size_t>(outRig.boneCount));
  for (int i = 0; i < outRig.boneCount; ++i)
    outRig.invBindTweaked[static_cast<size_t>(i)] = boneOffsets[outRig.boneNames[static_cast<size_t>(i)]];

  struct Wv {
    int id[4]{0, 0, 0, 0};
    float w[4]{0, 0, 0, 0};
    int n = 0;
  };

  glm::vec3 bmin(std::numeric_limits<float>::max());
  glm::vec3 bmax(-std::numeric_limits<float>::max());

  for (unsigned mi = 0; mi < scene->mNumMeshes; ++mi) {
    const aiMesh* mesh = scene->mMeshes[mi];
    if (!mesh->HasBones())
      continue;
    const aiNode* meshNode = nullptr;
    std::function<bool(const aiNode*)> findNode = [&](const aiNode* n) -> bool {
      for (unsigned i = 0; i < n->mNumMeshes; ++i) {
        if (n->mMeshes[i] == mi) {
          meshNode = n;
          return true;
        }
      }
      for (unsigned c = 0; c < n->mNumChildren; ++c) {
        if (findNode(n->mChildren[c]))
          return true;
      }
      return false;
    };
    findNode(scene->mRootNode);
    // Vertices and bone offsets are in mesh bind space; do not bake mesh node world transform here.

    std::vector<Wv> wv(mesh->mNumVertices);
    for (unsigned bi = 0; bi < mesh->mNumBones; ++bi) {
      const aiBone* bone = mesh->mBones[bi];
      const int gidx = boneMap[bone->mName.C_Str()];
      for (unsigned wj = 0; wj < bone->mNumWeights; ++wj) {
        const unsigned vi = bone->mWeights[wj].mVertexId;
        const float ww = bone->mWeights[wj].mWeight;
        Wv& acc = wv[vi];
        if (acc.n < 4) {
          acc.id[acc.n] = gidx;
          acc.w[acc.n] = ww;
          ++acc.n;
        } else {
          int m = 0;
          for (int k = 1; k < 4; ++k)
            if (acc.w[k] < acc.w[m])
              m = k;
          if (ww > acc.w[m]) {
            acc.id[m] = gidx;
            acc.w[m] = ww;
          }
        }
      }
    }

    const int part = classifyEmployeePart(mesh, meshNode ? meshNode : scene->mRootNode, scene);

    for (unsigned fi = 0; fi < mesh->mNumFaces; ++fi) {
      const aiFace& face = mesh->mFaces[fi];
      if (face.mNumIndices != 3)
        continue;
      for (unsigned k = 0; k < 3; ++k) {
        const unsigned idx = face.mIndices[k];
        glm::vec4 pw = glm::vec4(mesh->mVertices[idx].x, mesh->mVertices[idx].y, mesh->mVertices[idx].z, 1.f);
        glm::vec3 n{0.f, 1.f, 0.f};
        if (mesh->HasNormals()) {
          glm::vec3 rawN(mesh->mNormals[idx].x, mesh->mNormals[idx].y, mesh->mNormals[idx].z);
          n = glm::normalize(rawN);
          if (!std::isfinite(n.x) || glm::length(n) < 0.1f)
            n = glm::vec3(0.f, 1.f, 0.f);
        }
        glm::vec2 uvo{0.f, 0.f};
        if (mesh->HasTextureCoords(0)) {
          const aiVector3D& tc = mesh->mTextureCoords[0][idx];
          uvo = glm::vec2(tc.x, tc.y);
        } else if (part >= 0) {
          uvo = glm::vec2(pw.x * 0.37f + 0.5f, pw.z * 0.37f + 0.5f);
        }
        const float pa = part >= 0 ? partAlpha(part) : 0.08f;
        SkinnedVertex sv{};
        sv.pos = glm::vec3(pw);
        sv.normal = n;
        sv.color = glm::vec4(kEmployeeTag, pa);
        sv.uv = uvo;
        const Wv& ww = wv[idx];
        if (ww.n <= 0) {
          sv.boneIds = glm::ivec4(0);
          sv.boneWts = glm::vec4(1.f, 0.f, 0.f, 0.f);
        } else {
          float ws = 0.f;
          for (int t = 0; t < 4; ++t) {
            if (t < ww.n) {
              sv.boneIds[t] = ww.id[t];
              sv.boneWts[t] = ww.w[t];
              ws += ww.w[t];
            } else {
              sv.boneIds[t] = 0;
              sv.boneWts[t] = 0.f;
            }
          }
          if (ws > 1e-6f)
            sv.boneWts /= ws;
          else {
            sv.boneIds = glm::ivec4(0);
            sv.boneWts = glm::vec4(1.f, 0.f, 0.f, 0.f);
          }
        }
        bmin = glm::min(bmin, sv.pos);
        bmax = glm::max(bmax, sv.pos);
        outVerts.push_back(sv);
      }
    }
  }

  if (outVerts.empty()) {
    errOut = "no skinned triangles";
    return false;
  }

  const float minY = bmin.y;
  const glm::vec3 cxz((bmin.x + bmax.x) * 0.5f, 0.f, (bmin.z + bmax.z) * 0.5f);
  for (auto& v : outVerts) {
    v.pos.x -= cxz.x;
    v.pos.z -= cxz.z;
    v.pos.y -= minY;
  }
  bmin -= glm::vec3(cxz.x, minY, cxz.z);
  bmax -= glm::vec3(cxz.x, minY, cxz.z);
  const float h = std::max(bmax.y - bmin.y, 0.001f);
  const float s = targetHeightMeters / h;
  for (auto& v : outVerts) {
    v.pos *= s;
    v.normal = glm::normalize(v.normal);
  }

  const glm::mat4 N = glm::scale(glm::mat4(1.f), glm::vec3(s, s, s)) *
                      glm::translate(glm::mat4(1.f), glm::vec3(-cxz.x, -minY, -cxz.z));
  outRig.meshNorm = N;
  const glm::mat4 invN = glm::inverse(N);
  for (int i = 0; i < outRig.boneCount; ++i)
    outRig.invBindTweaked[static_cast<size_t>(i)] = outRig.invBindTweaked[static_cast<size_t>(i)] * invN;

  Rig::AnimClip idleClip;
  if (scene->mNumAnimations > 0)
    copyAnimClip(scene->mAnimations[0], idleClip);
  else {
    idleClip.duration = 1.0;
    idleClip.ticksPerSecond = 25.0;
  }
  outRig.clips.push_back(std::move(idleClip));

  if (loadAllAnimationsFromFile && scene->mNumAnimations > 1u) {
    for (unsigned ai = 1u; ai < scene->mNumAnimations; ++ai) {
      Rig::AnimClip c;
      copyAnimClip(scene->mAnimations[ai], c);
      remapAnimClipToRig(c, outRig);
      if (!c.channels.empty())
        outRig.clips.push_back(std::move(c));
    }
  }

  if (outDiffuseRgba && outDiffuseW && outDiffuseH) {
    std::string dir;
    const char* sl = std::strrchr(path, '/');
    if (sl)
      dir.assign(path, sl - path + 1);
    *outDiffuseW = 0;
    *outDiffuseH = 0;
    outDiffuseRgba->clear();
    if (!extractDiffuseFromDominantSkinnedMaterial(scene, dir, *outDiffuseRgba, *outDiffuseW, *outDiffuseH) &&
        !extractFirstStaffDiffuse(scene, dir, *outDiffuseRgba, *outDiffuseW, *outDiffuseH)) {
      outDiffuseRgba->clear();
      *outDiffuseW = 0;
      *outDiffuseH = 0;
    }
  }

  return true;
}

bool appendLongestRetargetedClipFromGlb(const char* path, Rig& rig, bool freeRootMotion, int& outClipIndex,
                                        std::string& errOut) {
  errOut.clear();
  outClipIndex = -1;
  Assimp::Importer importer;
  const unsigned flags =
      aiProcess_Triangulate | aiProcess_RemoveRedundantMaterials | aiProcess_ValidateDataStructure;
  const aiScene* scene = importer.ReadFile(path, flags);
  if (!scene) {
    errOut = importer.GetErrorString();
    return false;
  }
  if (scene->mNumAnimations == 0) {
    errOut = "no animations in file";
    return false;
  }
  double bestDurSec = -1.0;
  Rig::AnimClip best{};
  bool have = false;
  for (unsigned ai = 0; ai < scene->mNumAnimations; ++ai) {
    Rig::AnimClip test;
    copyAnimClip(scene->mAnimations[ai], test);
    if (freeRootMotion) {
      test.lockLocomotionRoot = false;
      test.lockRootTranslationToBind = false;
      test.lockHipsTranslationToBind = false;
    } else {
      test.lockLocomotionRoot = true;
      test.lockRootTranslationToBind = true;
      test.lockHipsTranslationToBind = true;
    }
    test.blendTorsoTowardBind = false;
    remapAnimClipToRig(test, rig);
    if (test.channels.empty())
      continue;
    const double tps = test.ticksPerSecond > 1e-6 ? test.ticksPerSecond : 25.0;
    const double dTicks = test.duration > 1e-6 ? test.duration : 1.0;
    const double sec = dTicks / tps;
    if (!have || sec > bestDurSec) {
      bestDurSec = sec;
      best = std::move(test);
      have = true;
    }
  }
  if (!have || best.channels.empty()) {
    errOut = "no animations matched rig bone names (retarget failed)";
    return false;
  }
  outClipIndex = static_cast<int>(rig.clips.size());
  rig.clips.push_back(std::move(best));
  return true;
}

bool appendAnimationFromGlb(const char* path, Rig& rig, std::string& errOut) {
  errOut.clear();
  Assimp::Importer importer;
  const unsigned flags =
      aiProcess_Triangulate | aiProcess_RemoveRedundantMaterials | aiProcess_ValidateDataStructure;
  const aiScene* scene = importer.ReadFile(path, flags);
  if (!scene) {
    errOut = importer.GetErrorString();
    return false;
  }
  if (scene->mNumAnimations == 0) {
    errOut = "no animations in file";
    return false;
  }
  const bool isJumpLike =
      pathLowerContains(path, "jump") || pathLowerContains(path, "leap") ||
      pathLowerContains(path, "vault");

  Rig::AnimClip clip{};
  bool haveClip = false;
  size_t bestChannelCount = 0;

  for (unsigned ai = 0; ai < scene->mNumAnimations; ++ai) {
    Rig::AnimClip test;
    copyAnimClip(scene->mAnimations[ai], test);
    test.lockLocomotionRoot = true;
    test.blendTorsoTowardBind = false;
    if (isJumpLike) {
      test.lockRootTranslationToBind = false;
      test.lockHipsTranslationToBind = false;
    }
    remapAnimClipToRig(test, rig);
    if (test.channels.empty())
      continue;

    if (!isJumpLike) {
      clip = std::move(test);
      haveClip = true;
      break;
    }
    const size_t nMap = test.channelByNode.size();
    if (!haveClip || nMap > bestChannelCount) {
      bestChannelCount = nMap;
      clip = std::move(test);
      haveClip = true;
    }
  }
  if (!haveClip || clip.channels.empty()) {
    errOut = "animation channels did not match rig bone names (retarget failed)";
    return false;
  }
  // Mixamo/other mismatched rigs: lock torso to bind. Meshy jump on Meshy body: play clip as authored.
  if (isJumpLike && !pathLowerContains(path, "meshy"))
    clip.lockJumpTorsoToBind = true;
  rig.clips.push_back(std::move(clip));
  return true;
}

// Reused across evalClipToBoneGlobals calls to avoid per-NPC map allocations (hot path).
static thread_local std::unordered_map<std::string, glm::mat4> gTlsBoneLocals;
static thread_local std::unordered_map<std::string, glm::mat4> gTlsBoneGlobals;

static void dfsAccumBoneGlobals(const Rig& rig, const std::string& name, const glm::mat4& parent,
                                const std::unordered_map<std::string, glm::mat4>& locals,
                                std::unordered_map<std::string, glm::mat4>& globals) {
  glm::mat4 G = parent * locals.at(name);
  globals[name] = G;
  for (const std::string& ch : rig.nodes.at(name).children)
    dfsAccumBoneGlobals(rig, ch, G, locals, globals);
}

// Per-bone global transforms (before meshNorm * invBind) for correct cross-blending.
static void evalClipToBoneGlobals(const Rig& rig, int clipIndex, double phaseSec, bool loopPhase,
                                  glm::mat4* outG, const glm::vec3* extraLocalEulerPerBone = nullptr) {
  for (int i = 0; i < kMaxPaletteBones; ++i)
    outG[i] = glm::mat4(1.f);
  if (rig.boneCount <= 0 || rig.clips.empty())
    return;

  const int ci = std::clamp(clipIndex, 0, static_cast<int>(rig.clips.size()) - 1);
  const Rig::AnimClip& clip = rig.clips[static_cast<size_t>(ci)];
  double tickT = phaseSec * clip.ticksPerSecond;
  if (clip.duration > 1e-6) {
    if (loopPhase) {
      tickT = std::fmod(tickT, clip.duration);
      if (tickT < 0)
        tickT += clip.duration;
    } else {
      tickT = std::clamp(tickT, 0.0, clip.duration - 1e-6);
    }
  }

  const size_t nNodes = rig.nodes.size();
  gTlsBoneLocals.clear();
  gTlsBoneGlobals.clear();
  if (gTlsBoneLocals.bucket_count() < nNodes * 2)
    gTlsBoneLocals.reserve(nNodes);
  if (gTlsBoneGlobals.bucket_count() < nNodes * 2)
    gTlsBoneGlobals.reserve(nNodes);

  for (const auto& kv : rig.nodes) {
    const std::string& nm = kv.first;
    const Rig::AnimChannel* ch = nullptr;
    auto it = clip.channelByNode.find(nm);
    if (it != clip.channelByNode.end())
      ch = &clip.channels[it->second];
    glm::mat4 L = localFromChannel(ch, tickT, kv.second.bindLocal);
    if (clip.lockLocomotionRoot) {
      if (nm == rig.rootName) {
        if (clip.lockRootTranslationToBind) {
          const glm::mat4& bindRoot = rig.nodes.at(rig.rootName).bindLocal;
          L[3][0] = bindRoot[3][0];
          L[3][1] = bindRoot[3][1];
          L[3][2] = bindRoot[3][2];
        }
      } else if (shouldLockLocomotionTranslationToBind(nm) && clip.lockHipsTranslationToBind) {
        const glm::mat4& bindLocal = kv.second.bindLocal;
        const glm::vec3 bindT(bindLocal[3]);
        L = composeTrs(bindT, matToQuat(L), matScale(L));
      }
    }
    if (clip.lockJumpTorsoToBind && isTorsoUprightBone(nm))
      L = kv.second.bindLocal;
    // Optional partial slerp toward bind (non-jump clips if enabled).
    if (clip.blendTorsoTowardBind) {
      std::string lower = nm;
      for (char& c : lower)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      float u = 0.f;
      if (nm == rig.rootName)
        u = 0.26f;
      else if (isPelvisHipsBone(lower))
        u = 0.34f;
      else if (isTorsoUprightBone(nm)) {
        const bool isHead = lower.find("head") != std::string::npos;
        u = isHead ? 0.48f : 0.58f;
      }
      if (u > 0.f) {
        const glm::mat4& bindLocal = kv.second.bindLocal;
        const glm::vec3 t(L[3]);
        glm::quat animR = matToQuat(L);
        glm::quat bindR = matToQuat(bindLocal);
        if (glm::dot(animR, bindR) < 0.f)
          bindR = -bindR;
        const glm::quat r = glm::normalize(glm::slerp(animR, bindR, u));
        L = composeTrs(t, r, matScale(L));
      }
    }
    gTlsBoneLocals[nm] = L;
  }

  if (extraLocalEulerPerBone) {
    for (int bi = 0; bi < rig.boneCount; ++bi) {
      const glm::vec3& ex = extraLocalEulerPerBone[bi];
      if (glm::dot(ex, ex) < 1e-14f)
        continue;
      const std::string& bnm = rig.boneNames[static_cast<size_t>(bi)];
      auto lit = gTlsBoneLocals.find(bnm);
      if (lit == gTlsBoneLocals.end())
        continue;
      glm::mat4& Lm = lit->second;
      const glm::quat qe = glm::normalize(
          glm::angleAxis(ex.z, glm::vec3(0.f, 0.f, 1.f)) * glm::angleAxis(ex.y, glm::vec3(0.f, 1.f, 0.f)) *
          glm::angleAxis(ex.x, glm::vec3(1.f, 0.f, 0.f)));
      glm::quat r = glm::normalize(matToQuat(Lm) * qe);
      const glm::quat r0 = matToQuat(Lm);
      if (glm::dot(r, r0) < 0.f)
        r = -r;
      Lm = composeTrs(glm::vec3(Lm[3]), r, matScale(Lm));
    }
  }

  dfsAccumBoneGlobals(rig, rig.rootName, glm::mat4(1.f), gTlsBoneLocals, gTlsBoneGlobals);

  for (int i = 0; i < rig.boneCount; ++i) {
    const std::string& nm = rig.boneNames[static_cast<size_t>(i)];
    auto git = gTlsBoneGlobals.find(nm);
    outG[i] = git != gTlsBoneGlobals.end() ? git->second : glm::mat4(1.f);
  }
}

void computePalette(const Rig& rig, int clipIndex, double phaseSec, glm::mat4* outPalette, bool loopPhase) {
  glm::mat4 g[kMaxPaletteBones];
  evalClipToBoneGlobals(rig, clipIndex, phaseSec, loopPhase, g, nullptr);
  for (int i = 0; i < rig.boneCount; ++i)
    outPalette[i] = rig.meshNorm * g[i] * rig.invBindTweaked[static_cast<size_t>(i)];
  for (int i = rig.boneCount; i < kMaxPaletteBones; ++i)
    outPalette[i] = glm::mat4(1.f);
}

void computePaletteWithRagdollExtras(const Rig& rig, int clipIndex, double phaseSec, bool loopPhase,
                                     const glm::vec3* extraLocalEulerPerBone, glm::mat4* outPalette) {
  glm::mat4 g[kMaxPaletteBones];
  evalClipToBoneGlobals(rig, clipIndex, phaseSec, loopPhase, g, extraLocalEulerPerBone);
  for (int i = 0; i < rig.boneCount; ++i)
    outPalette[i] = rig.meshNorm * g[i] * rig.invBindTweaked[static_cast<size_t>(i)];
  for (int i = rig.boneCount; i < kMaxPaletteBones; ++i)
    outPalette[i] = glm::mat4(1.f);
}

void sampleClipBoneGlobalMatrices(const Rig& rig, int clipIndex, double phaseSec, bool loopPhase,
                                  const glm::vec3* extraLocalEulerPerBone, glm::mat4* outGlobalBone) {
  glm::mat4 g[kMaxPaletteBones];
  evalClipToBoneGlobals(rig, clipIndex, phaseSec, loopPhase, g, extraLocalEulerPerBone);
  for (int i = 0; i < rig.boneCount; ++i)
    outGlobalBone[i] = g[i];
  for (int i = rig.boneCount; i < kMaxPaletteBones; ++i)
    outGlobalBone[i] = glm::mat4(1.f);
}

// Local-space euler (radians), applied as Z * Y * X after bind rotation for a prone / collapsed silhouette.
static glm::quat ragdollQuatEulerXYZ(float ex, float ey, float ez) {
  return glm::normalize(glm::angleAxis(ez, glm::vec3(0.f, 0.f, 1.f)) *
                        glm::angleAxis(ey, glm::vec3(0.f, 1.f, 0.f)) *
                        glm::angleAxis(ex, glm::vec3(1.f, 0.f, 0.f)));
}

static int ragdollSideSign(const std::string& lower) {
  if (lower.find("right") != std::string::npos)
    return -1;
  if (lower.find("left") != std::string::npos)
    return 1;
  return 0;
}

// Baked prone pose: keep angles modest — large per-spine flex stacks into a fetal curl; big local-Z on
// upper arms reads as “hands overhead” on Mixamo-style rest poses.
static glm::vec3 ragdollProneCollapseEuler(const std::string& nm, const std::string& rootName) {
  if (nm == rootName)
    return {0.08f, 0.f, 0.f};
  std::string lower = nm;
  for (char& c : lower)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  const int side = ragdollSideSign(lower);

  if (isPelvisHipsBone(lower))
    return {0.12f, 0.f, 0.f};

  // Less flex per spine bone avoids a sharp “cardboard hinge” at the waist; chest gets a bit more.
  if (lower.find("chest") != std::string::npos)
    return {0.1f, 0.f, side * 0.02f};
  if (lower.find("spine") != std::string::npos) {
    if (lower.find("spine2") != std::string::npos || lower.find("spine_02") != std::string::npos ||
        lower.find("spine03") != std::string::npos || lower.find("spine_03") != std::string::npos)
      return {0.07f, 0.f, side * 0.015f};
    if (lower.find("spine1") != std::string::npos || lower.find("spine_01") != std::string::npos ||
        lower.find("spine02") != std::string::npos)
      return {0.08f, 0.f, side * 0.015f};
    return {0.09f, 0.f, side * 0.02f};
  }
  if (lower.find("neck") != std::string::npos)
    return {0.14f, 0.f, 0.f};
  if (lower.find("head") != std::string::npos)
    return {0.12f, side * 0.05f, 0.f};

  // Roll shoulders forward / inward — negative Y pulls typical Mixamo arms off a wide T-pose.
  if (lower.find("clavicle") != std::string::npos || lower.find("collar") != std::string::npos)
    return {0.04f, side * -0.2f, side * -0.06f};
  if (lower.find("shoulder") != std::string::npos)
    return {0.06f, side * -0.22f, side * -0.08f};

  if (lower.find("forearm") != std::string::npos || lower.find("lowerarm") != std::string::npos ||
      lower.find("elbow") != std::string::npos)
    return {0.68f, side * 0.03f, 0.f};
  if (lower.find("hand") != std::string::npos || lower.find("wrist") != std::string::npos ||
      lower.find("finger") != std::string::npos || lower.find("thumb") != std::string::npos)
    return {0.1f, side * -0.08f, 0.f};
  // Upper arm: +X forward flex, -Y adduct (swing toward torso), -Z slight inward roll.
  if (lower.find("upperarm") != std::string::npos || lower.find("uparm") != std::string::npos)
    return {0.4f, side * -0.32f, side * -0.38f};
  if (lower.find("arm") != std::string::npos)
    return {0.36f, side * -0.3f, side * -0.34f};

  if (lower.find("thigh") != std::string::npos || lower.find("upleg") != std::string::npos)
    return {-0.12f, side * 0.1f, side * 0.04f};
  if (lower.find("calf") != std::string::npos || lower.find("shin") != std::string::npos ||
      lower.find("knee") != std::string::npos)
    return {0.42f, 0.f, 0.f};
  if (lower.find("foot") != std::string::npos || lower.find("toe") != std::string::npos ||
      lower.find("ankle") != std::string::npos)
    return {-0.22f, side * 0.1f, side * 0.08f};
  if (lower.find("leg") != std::string::npos)
    return {0.18f, 0.f, 0.f};

  return {0.f, 0.f, 0.f};
}

static float ragdollJointSlack(const std::string& nm, const std::string& rootName) {
  if (nm == rootName)
    return 0.f;
  std::string lower = nm;
  for (char& c : lower)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (isPelvisHipsBone(lower))
    return 0.1f;
  if (lower.find("spine") != std::string::npos || lower.find("chest") != std::string::npos)
    return 0.4f;
  if (lower.find("neck") != std::string::npos)
    return 0.42f;
  if (lower.find("head") != std::string::npos)
    return 0.52f;
  if (lower.find("clavicle") != std::string::npos || lower.find("shoulder") != std::string::npos)
    return 0.58f;
  if (lower.find("upperarm") != std::string::npos || lower.find("lowerarm") != std::string::npos ||
      lower.find("forearm") != std::string::npos || lower.find("elbow") != std::string::npos)
    return 0.9f;
  if (lower.find("hand") != std::string::npos || lower.find("wrist") != std::string::npos ||
      lower.find("finger") != std::string::npos || lower.find("thumb") != std::string::npos)
    return 0.95f;
  if (lower.find("arm") != std::string::npos)
    return 0.88f;
  if (lower.find("thigh") != std::string::npos || lower.find("upleg") != std::string::npos)
    return 0.85f;
  if (lower.find("calf") != std::string::npos || lower.find("shin") != std::string::npos)
    return 0.9f;
  if (lower.find("knee") != std::string::npos)
    return 0.84f;
  if (lower.find("foot") != std::string::npos || lower.find("toe") != std::string::npos ||
      lower.find("ankle") != std::string::npos)
    return 0.93f;
  if (lower.find("leg") != std::string::npos)
    return 0.78f;
  return 0.16f;
}

static glm::quat quatSmallAxisAngleVec(const glm::vec3& v) {
  const float m = glm::length(v);
  if (m < 1e-7f)
    return glm::quat(1.f, 0.f, 0.f, 0.f);
  const float a = glm::min(m, 1.25f);
  const glm::vec3 axis = v * (1.f / m);
  const float h = 0.5f * a;
  return glm::normalize(glm::quat(std::cos(h), axis * std::sin(h)));
}

static uint32_t hashBoneWobble(uint32_t seed, const std::string& nm) {
  uint32_t h = seed ^ (static_cast<uint32_t>(nm.size()) * 2166136261u);
  for (char c : nm) {
    h ^= static_cast<uint8_t>(c);
    h *= 16777619u;
  }
  return h;
}

static glm::vec3 ragdollProneCollapseEulerScaled(const std::string& nm, const std::string& rootName,
                                                 uint32_t hashSeed) {
  glm::vec3 e = ragdollProneCollapseEuler(nm, rootName);
  if (glm::length(e) < 1e-5f)
    return e;
  const uint32_t h = hashBoneWobble(hashSeed, nm + "|rag");
  const float v = 0.82f + static_cast<float>(h & 255u) * (0.34f / 255.f);
  return e * v;
}

void computeBindPosePalette(const Rig& rig, glm::mat4* outPalette) {
  if (rig.boneCount <= 0) {
    for (int i = 0; i < kMaxPaletteBones; ++i)
      outPalette[i] = glm::mat4(1.f);
    return;
  }
  const size_t nNodes = rig.nodes.size();
  gTlsBoneLocals.clear();
  gTlsBoneGlobals.clear();
  if (gTlsBoneLocals.bucket_count() < nNodes * 2)
    gTlsBoneLocals.reserve(nNodes);
  if (gTlsBoneGlobals.bucket_count() < nNodes * 2)
    gTlsBoneGlobals.reserve(nNodes);
  for (const auto& kv : rig.nodes)
    gTlsBoneLocals[kv.first] = kv.second.bindLocal;
  dfsAccumBoneGlobals(rig, rig.rootName, glm::mat4(1.f), gTlsBoneLocals, gTlsBoneGlobals);
  for (int i = 0; i < rig.boneCount; ++i) {
    const std::string& nm = rig.boneNames[static_cast<size_t>(i)];
    auto git = gTlsBoneGlobals.find(nm);
    const glm::mat4 g = git != gTlsBoneGlobals.end() ? git->second : glm::mat4(1.f);
    outPalette[i] = rig.meshNorm * g * rig.invBindTweaked[static_cast<size_t>(i)];
  }
  for (int i = rig.boneCount; i < kMaxPaletteBones; ++i)
    outPalette[i] = glm::mat4(1.f);
}

static void accumBindLocalsWithOptionalExtras(const Rig& rig, const glm::vec3* extraLocalEulerPerBone) {
  const size_t nNodes = rig.nodes.size();
  gTlsBoneLocals.clear();
  gTlsBoneGlobals.clear();
  if (gTlsBoneLocals.bucket_count() < nNodes * 2)
    gTlsBoneLocals.reserve(nNodes);
  if (gTlsBoneGlobals.bucket_count() < nNodes * 2)
    gTlsBoneGlobals.reserve(nNodes);
  for (const auto& kv : rig.nodes)
    gTlsBoneLocals[kv.first] = kv.second.bindLocal;
  if (extraLocalEulerPerBone) {
    for (int bi = 0; bi < rig.boneCount; ++bi) {
      const glm::vec3& ex = extraLocalEulerPerBone[bi];
      if (glm::dot(ex, ex) < 1e-14f)
        continue;
      const std::string& bnm = rig.boneNames[static_cast<size_t>(bi)];
      auto lit = gTlsBoneLocals.find(bnm);
      if (lit == gTlsBoneLocals.end())
        continue;
      glm::mat4& Lm = lit->second;
      const glm::quat qe = glm::normalize(
          glm::angleAxis(ex.z, glm::vec3(0.f, 0.f, 1.f)) * glm::angleAxis(ex.y, glm::vec3(0.f, 1.f, 0.f)) *
          glm::angleAxis(ex.x, glm::vec3(1.f, 0.f, 0.f)));
      glm::quat r = glm::normalize(matToQuat(Lm) * qe);
      const glm::quat r0 = matToQuat(Lm);
      if (glm::dot(r, r0) < 0.f)
        r = -r;
      Lm = composeTrs(glm::vec3(Lm[3]), r, matScale(Lm));
    }
  }
  dfsAccumBoneGlobals(rig, rig.rootName, glm::mat4(1.f), gTlsBoneLocals, gTlsBoneGlobals);
}

void computeBindPosePaletteWithRagdollExtras(const Rig& rig, const glm::vec3* extraLocalEulerPerBone,
                                             glm::mat4* outPalette) {
  if (rig.boneCount <= 0) {
    for (int i = 0; i < kMaxPaletteBones; ++i)
      outPalette[i] = glm::mat4(1.f);
    return;
  }
  accumBindLocalsWithOptionalExtras(rig, extraLocalEulerPerBone);
  for (int i = 0; i < rig.boneCount; ++i) {
    const std::string& nm = rig.boneNames[static_cast<size_t>(i)];
    auto git = gTlsBoneGlobals.find(nm);
    const glm::mat4 g = git != gTlsBoneGlobals.end() ? git->second : glm::mat4(1.f);
    outPalette[i] = rig.meshNorm * g * rig.invBindTweaked[static_cast<size_t>(i)];
  }
  for (int i = rig.boneCount; i < kMaxPaletteBones; ++i)
    outPalette[i] = glm::mat4(1.f);
}

static int boneParentIndexSkin(const Rig& rig, int boneIdx) {
  if (boneIdx < 0 || boneIdx >= rig.boneCount)
    return -1;
  const std::string& nm = rig.boneNames[static_cast<size_t>(boneIdx)];
  auto it = rig.nodes.find(nm);
  if (it == rig.nodes.end())
    return -1;
  const std::string& pn = it->second.parent;
  if (pn.empty())
    return -1;
  auto pi = rig.boneNameToIndex.find(pn);
  if (pi == rig.boneNameToIndex.end())
    return -1;
  return pi->second;
}

void computePaletteFromRagdollSimWorldMatrices(const Rig& rig, const glm::mat4& characterModel,
                                               const glm::mat4* bindGlobalArmature, int nSim,
                                               const int* simRigBoneIdx, const glm::mat4* worldBoneSim,
                                               glm::mat4* outPalette) {
  if (rig.boneCount <= 0) {
    for (int i = 0; i < kMaxPaletteBones; ++i)
      outPalette[i] = glm::mat4(1.f);
    return;
  }
  const glm::mat4 invChar = glm::inverse(characterModel);
  std::vector<int> parent(static_cast<size_t>(rig.boneCount));
  std::vector<int> depth(static_cast<size_t>(rig.boneCount), 0);
  for (int i = 0; i < rig.boneCount; ++i) {
    parent[static_cast<size_t>(i)] = boneParentIndexSkin(rig, i);
  }
  for (int i = 0; i < rig.boneCount; ++i) {
    int d = 0;
    for (int x = i; x >= 0;) {
      const int p = parent[static_cast<size_t>(x)];
      if (p < 0)
        break;
      d++;
      x = p;
    }
    depth[static_cast<size_t>(i)] = d;
  }
  std::vector<int> order(static_cast<size_t>(rig.boneCount));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return depth[static_cast<size_t>(a)] < depth[static_cast<size_t>(b)]; });
  std::vector<uint8_t> isSim(static_cast<size_t>(rig.boneCount), 0);
  std::vector<int> simSlotForBone(static_cast<size_t>(rig.boneCount), -1);
  for (int j = 0; j < nSim; ++j) {
    const int bi = simRigBoneIdx[j];
    if (bi >= 0 && bi < rig.boneCount) {
      isSim[static_cast<size_t>(bi)] = 1;
      simSlotForBone[static_cast<size_t>(bi)] = j;
    }
  }
  static thread_local std::vector<glm::mat4> W;
  W.assign(static_cast<size_t>(rig.boneCount), glm::mat4(1.f));
  for (int idx : order) {
    const size_t i = static_cast<size_t>(idx);
    if (isSim[i]) {
      const int sj = simSlotForBone[i];
      W[i] = worldBoneSim[sj];
      continue;
    }
    const int p = parent[i];
    if (p < 0)
      W[i] = characterModel * rig.meshNorm * bindGlobalArmature[i];
    else
      W[i] = W[static_cast<size_t>(p)] * glm::inverse(bindGlobalArmature[static_cast<size_t>(p)]) *
             bindGlobalArmature[i];
  }
  for (int i = 0; i < rig.boneCount; ++i)
    outPalette[i] = invChar * W[static_cast<size_t>(i)] * rig.invBindTweaked[static_cast<size_t>(i)];
  for (int i = rig.boneCount; i < kMaxPaletteBones; ++i)
    outPalette[i] = glm::mat4(1.f);
}

void sampleBindBoneGlobalMatricesWithExtras(const Rig& rig, const glm::vec3* extraLocalEulerPerBone,
                                            glm::mat4* outGlobalBone) {
  for (int i = 0; i < kMaxPaletteBones; ++i)
    outGlobalBone[i] = glm::mat4(1.f);
  if (rig.boneCount <= 0)
    return;
  accumBindLocalsWithOptionalExtras(rig, extraLocalEulerPerBone);
  for (int i = 0; i < rig.boneCount; ++i) {
    const std::string& nm = rig.boneNames[static_cast<size_t>(i)];
    auto git = gTlsBoneGlobals.find(nm);
    outGlobalBone[i] = git != gTlsBoneGlobals.end() ? git->second : glm::mat4(1.f);
  }
}

void computeLooseBindPosePalette(const Rig& rig, glm::mat4* outPalette, const glm::vec3& ragdollAngVelRadPerSec,
                                 float simTimeSec, uint32_t hashSeed) {
  if (rig.boneCount <= 0) {
    for (int i = 0; i < kMaxPaletteBones; ++i)
      outPalette[i] = glm::mat4(1.f);
    return;
  }
  constexpr float kVelToAngle = 0.11f;
  constexpr float kWobbleRad = 0.17f;
  const glm::vec3 velBasis = ragdollAngVelRadPerSec * kVelToAngle;

  const size_t nNodes = rig.nodes.size();
  gTlsBoneLocals.clear();
  gTlsBoneGlobals.clear();
  if (gTlsBoneLocals.bucket_count() < nNodes * 2)
    gTlsBoneLocals.reserve(nNodes);
  if (gTlsBoneGlobals.bucket_count() < nNodes * 2)
    gTlsBoneGlobals.reserve(nNodes);

  for (const auto& kv : rig.nodes) {
    const std::string& nm = kv.first;
    const glm::mat4& bindLocal = kv.second.bindLocal;
    const glm::vec3 bindT(bindLocal[3]);
    const glm::quat bindR = matToQuat(bindLocal);
    const glm::vec3 bindS = matScale(bindLocal);

    const glm::vec3 colE = ragdollProneCollapseEulerScaled(nm, rig.rootName, hashSeed);
    const glm::quat qCollapse =
        glm::length(colE) > 1e-5f ? ragdollQuatEulerXYZ(colE.x, colE.y, colE.z) : glm::quat(1.f, 0.f, 0.f, 0.f);
    glm::quat rBase = glm::normalize(bindR * qCollapse);
    if (glm::dot(rBase, bindR) < 0.f)
      rBase = -rBase;

    const float slack = ragdollJointSlack(nm, rig.rootName);
    glm::quat rOut = rBase;
    if (slack > 1e-4f) {
      const uint32_t bh = hashBoneWobble(hashSeed, nm);
      const float ph0 = static_cast<float>(bh & 1023u) * 0.00613592315f;
      const float ph1 = static_cast<float>((bh >> 10) & 1023u) * 0.00613592315f;
      const float ph2 = static_cast<float>((bh >> 20) & 1023u) * 0.00613592315f;
      const float t = simTimeSec;
      const glm::vec3 wobble(
          std::sin(t * 4.7f + ph0) * kWobbleRad,
          std::sin(t * 3.9f + ph1) * kWobbleRad * 0.85f,
          std::sin(t * 5.4f + ph2) * kWobbleRad * 0.9f);

      const float ax = slack * (0.35f + static_cast<float>((bh >> 3) & 63u) * 0.01f);
      const float ay = slack * (0.28f + static_cast<float>((bh >> 9) & 63u) * 0.01f);
      const float az = slack * (0.32f + static_cast<float>((bh >> 15) & 63u) * 0.01f);
      glm::vec3 swing = glm::vec3(velBasis.x * ax, velBasis.y * ay, velBasis.z * az);
      swing += wobble * slack * 0.75f;

      const glm::quat qExtra = quatSmallAxisAngleVec(swing);
      glm::quat rW = glm::normalize(rBase * qExtra);
      if (glm::dot(rW, rBase) < 0.f)
        rW = -rW;
      const float looseBlend = glm::clamp(slack * 1.08f, 0.f, 0.97f);
      rOut = glm::normalize(glm::slerp(rBase, rW, looseBlend));
    }
    gTlsBoneLocals[nm] = composeTrs(bindT, rOut, bindS);
  }

  dfsAccumBoneGlobals(rig, rig.rootName, glm::mat4(1.f), gTlsBoneLocals, gTlsBoneGlobals);
  for (int i = 0; i < rig.boneCount; ++i) {
    const std::string& nm = rig.boneNames[static_cast<size_t>(i)];
    auto git = gTlsBoneGlobals.find(nm);
    const glm::mat4 g = git != gTlsBoneGlobals.end() ? git->second : glm::mat4(1.f);
    outPalette[i] = rig.meshNorm * g * rig.invBindTweaked[static_cast<size_t>(i)];
  }
  for (int i = rig.boneCount; i < kMaxPaletteBones; ++i)
    outPalette[i] = glm::mat4(1.f);
}

void computePaletteLerp(const Rig& rig, int clipA, double phaseA, bool loopA, int clipB, double phaseB,
                        bool loopB, float t, glm::mat4* outPalette) {
  glm::mat4 gA[kMaxPaletteBones];
  glm::mat4 gB[kMaxPaletteBones];
  evalClipToBoneGlobals(rig, clipA, phaseA, loopA, gA, nullptr);
  evalClipToBoneGlobals(rig, clipB, phaseB, loopB, gB, nullptr);
  const float u = glm::clamp(t, 0.f, 1.f);
  const float s = u * u * u * (u * (u * 6.f - 15.f) + 10.f);
  for (int i = 0; i < rig.boneCount; ++i) {
    const glm::vec3 tA(gA[i][3]);
    const glm::vec3 tB(gB[i][3]);
    glm::quat rA = matToQuat(gA[i]);
    glm::quat rB = matToQuat(gB[i]);
    if (glm::dot(rA, rB) < 0.f)
      rB = -rB;
    const glm::vec3 sA = matScale(gA[i]);
    const glm::vec3 sB = matScale(gB[i]);
    const glm::vec3 tt = glm::mix(tA, tB, s);
    const glm::quat rr = glm::normalize(glm::slerp(rA, rB, s));
    const glm::vec3 ss = glm::mix(sA, sB, s);
    const glm::mat4 G = composeTrs(tt, rr, ss);
    outPalette[i] = rig.meshNorm * G * rig.invBindTweaked[static_cast<size_t>(i)];
  }
  for (int i = rig.boneCount; i < kMaxPaletteBones; ++i)
    outPalette[i] = glm::mat4(1.f);
}

// ---------------------------------------------------------------------------
// optimizeRigClips – shrink animation data in place
// ---------------------------------------------------------------------------

static bool vec3Near(const glm::vec3& a, const glm::vec3& b, float eps) {
  return std::abs(a.x - b.x) < eps && std::abs(a.y - b.y) < eps && std::abs(a.z - b.z) < eps;
}

static bool quatNear(const glm::quat& a, const glm::quat& b, float eps) {
  const float d = std::abs(glm::dot(a, b));
  return d > 1.f - eps;
}

template <typename V, typename Near>
static size_t collapseConstantTrack(std::vector<std::pair<double, V>>& keys, Near nearFn, float eps) {
  if (keys.size() <= 1)
    return 0;
  const V& ref = keys.front().second;
  bool allSame = true;
  for (size_t i = 1; i < keys.size(); ++i) {
    if (!nearFn(ref, keys[i].second, eps)) {
      allSame = false;
      break;
    }
  }
  if (!allSame)
    return 0;
  size_t removed = keys.size() - 1;
  auto first = keys.front();
  keys.clear();
  keys.push_back(first);
  keys.shrink_to_fit();
  return removed;
}

template <typename V, typename Near>
static size_t removeDuplicateConsecutive(std::vector<std::pair<double, V>>& keys, Near nearFn, float eps) {
  if (keys.size() <= 2)
    return 0;
  size_t write = 1;
  for (size_t read = 1; read < keys.size(); ++read) {
    bool keep = (read == keys.size() - 1) || !nearFn(keys[write - 1].second, keys[read].second, eps);
    if (keep)
      keys[write++] = keys[read];
  }
  size_t removed = keys.size() - write;
  keys.resize(write);
  if (removed > 0)
    keys.shrink_to_fit();
  return removed;
}

size_t optimizeRigClips(Rig& rig) {
  constexpr float kVecEps = 1e-5f;
  constexpr float kQuatEps = 1e-6f;
  const glm::vec3 identityScale(1.f);

  size_t totalRemoved = 0;
  size_t totalBefore = 0;

  for (auto& clip : rig.clips) {
    for (auto& ch : clip.channels) {
      totalBefore += ch.posKeys.size() + ch.rotKeys.size() + ch.sclKeys.size();

      totalRemoved += collapseConstantTrack(ch.posKeys, vec3Near, kVecEps);
      totalRemoved += collapseConstantTrack(ch.rotKeys, quatNear, kQuatEps);

      bool isIdentityScale = true;
      for (const auto& sk : ch.sclKeys) {
        if (!vec3Near(sk.second, identityScale, kVecEps)) {
          isIdentityScale = false;
          break;
        }
      }
      if (isIdentityScale && !ch.sclKeys.empty()) {
        totalRemoved += ch.sclKeys.size();
        ch.sclKeys.clear();
        ch.sclKeys.shrink_to_fit();
      } else {
        totalRemoved += collapseConstantTrack(ch.sclKeys, vec3Near, kVecEps);
      }

      totalRemoved += removeDuplicateConsecutive(ch.posKeys, vec3Near, kVecEps);
      totalRemoved += removeDuplicateConsecutive(ch.rotKeys, quatNear, kQuatEps);
      totalRemoved += removeDuplicateConsecutive(ch.sclKeys, vec3Near, kVecEps);
    }
  }

  size_t totalAfter = 0;
  for (const auto& clip : rig.clips)
    for (const auto& ch : clip.channels)
      totalAfter += ch.posKeys.size() + ch.rotKeys.size() + ch.sclKeys.size();

  std::fprintf(stderr, "[anim opt] %zu clips: %zu -> %zu keys (removed %zu, %.1f%%)\n",
               rig.clips.size(), totalBefore, totalAfter, totalRemoved,
               totalBefore > 0 ? 100.0 * static_cast<double>(totalRemoved) / static_cast<double>(totalBefore) : 0.0);
  return totalRemoved;
}

} // namespace staff_skin
