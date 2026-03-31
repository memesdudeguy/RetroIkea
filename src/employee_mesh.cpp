#include "employee_mesh.hpp"

#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <SDL2/SDL_image.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

namespace emp_mesh {

namespace {

// Shader tag (must match shader.vert / shader.frag employee branch).
constexpr glm::vec3 kEmployeeTag{0.52f, 0.88f, 0.91f};

// Fragment shader reads part from color.a: 0.02*(p+1) for p=0,1,2; unknown = 0.08.
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

// Whole-token match so "face" does not fire on "polySurface"/"surface", "arm" not on "farm", etc.
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

// Classify submesh from Assimp mesh name, parent node name, and material name.
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

  // Skin / exposed — use word() for short tokens that appear inside DCC names (e.g. "face" in "surface").
  if (word("head") || word("face") || sub("hair") || sub("scalp") || word("neck"))
    return static_cast<int>(EmpTexPart::Skin);
  if (word("hand") || sub("finger") || sub("thumb") || word("wrist") || sub("palm"))
    return static_cast<int>(EmpTexPart::Skin);
  if (word("arm") || sub("elbow") || sub("shoulder") || sub("clavicle"))
    return static_cast<int>(EmpTexPart::Skin);
  if (word("foot") || sub("feet") || word("toe") || sub("ankle"))
    return static_cast<int>(EmpTexPart::Skin);

  // Pants / legs (substring OK for multi-char keywords).
  if (sub("pants") || sub("jeans") || sub("trouser") || sub("shorts") || sub("glute") || sub("butt") ||
      sub("hips") || sub("pelvis"))
    return static_cast<int>(EmpTexPart::Jeans);
  if (sub("thigh") || sub("calf") || sub("knee"))
    return static_cast<int>(EmpTexPart::Jeans);
  if (word("leg") || sub("leg_") || sub("_leg"))
    return static_cast<int>(EmpTexPart::Jeans);
  if (word("boot") || sub("boots") || sub("shoe") || sub("shoes"))
    return static_cast<int>(EmpTexPart::Jeans);

  // Torso clothing (avoid generic "body"/"top" — combined meshes would go all-shirt).
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

void appendMeshTriangles(const aiMesh* mesh, const aiNode* node, const aiScene* scene, const glm::mat4& T,
                         std::vector<LoadedVertex>& out, glm::vec3& bmin, glm::vec3& bmax) {
  const glm::mat3 Nmat = glm::mat3(glm::transpose(glm::inverse(T)));
  const int part = classifyEmployeePart(mesh, node, scene);

  for (unsigned fi = 0; fi < mesh->mNumFaces; ++fi) {
    const aiFace& face = mesh->mFaces[fi];
    if (face.mNumIndices != 3)
      continue;
    for (unsigned k = 0; k < 3; ++k) {
      const unsigned idx = face.mIndices[k];
      glm::vec4 pw = T * glm::vec4(mesh->mVertices[idx].x, mesh->mVertices[idx].y, mesh->mVertices[idx].z, 1.f);
      glm::vec3 n{0.f, 1.f, 0.f};
      if (mesh->HasNormals()) {
        glm::vec3 rawN(mesh->mNormals[idx].x, mesh->mNormals[idx].y, mesh->mNormals[idx].z);
        n = glm::normalize(Nmat * rawN);
        if (!std::isfinite(n.x) || glm::length(n) < 0.1f)
          n = glm::vec3(0.f, 1.f, 0.f);
      }
      glm::vec2 meshUv{0.f, 0.f};
      if (mesh->HasTextureCoords(0)) {
        const aiVector3D& tc = mesh->mTextureCoords[0][idx];
        meshUv = glm::vec2(tc.x, tc.y);
      } else if (part >= 0) {
        glm::vec3 ppre(pw);
        meshUv = glm::vec2(ppre.x * 0.37f + 0.5f, ppre.z * 0.37f + 0.5f);
      }
      glm::vec3 p(pw);
      const float pa = part >= 0 ? partAlpha(part) : 0.08f;
      glm::vec4 col(kEmployeeTag, pa);
      bmin = glm::min(bmin, p);
      bmax = glm::max(bmax, p);
      out.push_back({p, n, col, meshUv});
    }
  }
}

void processNode(const aiNode* node, const aiScene* scene, const glm::mat4& parent,
                 std::vector<LoadedVertex>& out, glm::vec3& bmin, glm::vec3& bmax) {
  const glm::mat4 T = parent * aiMatToGlm(node->mTransformation);
  for (unsigned i = 0; i < node->mNumMeshes; ++i) {
    const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
    appendMeshTriangles(mesh, node, scene, T, out, bmin, bmax);
  }
  for (unsigned i = 0; i < node->mNumChildren; ++i)
    processNode(node->mChildren[i], scene, T, out, bmin, bmax);
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

static bool extractFirstStaffDiffuse(const aiScene* scene, const std::string& modelDir,
                                   std::vector<uint8_t>& rgba, uint32_t& w, uint32_t& h) {
  static const aiTextureType kTry[] = {aiTextureType_BASE_COLOR, aiTextureType_DIFFUSE};
  for (unsigned mi = 0; mi < scene->mNumMaterials; ++mi) {
    const aiMaterial* mat = scene->mMaterials[mi];
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
  }
  return false;
}

} // namespace

static bool pathLooksLikeGltf(const char* path) {
  const char* dot = std::strrchr(path, '.');
  if (!dot)
    return false;
  std::string ext;
  for (const char* p = dot; *p; ++p)
    ext.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
  return ext == ".glb" || ext == ".gltf";
}

bool loadFbx(const char* path, float targetHeightMeters, std::vector<LoadedVertex>& out,
             std::string& errOut, std::vector<uint8_t>* outDiffuseRgba, uint32_t* outDiffuseW,
             uint32_t* outDiffuseH) {
  out.clear();
  errOut.clear();
  if (!path || !*path) {
    errOut = "empty path";
    return false;
  }

  unsigned aiFlags = aiProcess_Triangulate | aiProcess_GenSmoothNormals |
                     aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality |
                     aiProcess_RemoveRedundantMaterials | aiProcess_ValidateDataStructure;
  // Skinned glTF/GLB: bake scene graph into mesh vertices for static instancing (no GPU skinning yet).
  if (pathLooksLikeGltf(path))
    aiFlags |= aiProcess_PreTransformVertices;

  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(path, aiFlags);
  if (!scene || !scene->mRootNode ||
      (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) != 0) {
    errOut = importer.GetErrorString();
    return false;
  }

  glm::vec3 bmin(std::numeric_limits<float>::max());
  glm::vec3 bmax(-std::numeric_limits<float>::max());
  processNode(scene->mRootNode, scene, glm::mat4(1.f), out, bmin, bmax);

  if (out.empty()) {
    errOut = "no triangles in scene";
    return false;
  }

  const float minY = bmin.y;
  const glm::vec3 cxz((bmin.x + bmax.x) * 0.5f, 0.f, (bmin.z + bmax.z) * 0.5f);
  for (auto& v : out) {
    v.pos.x -= cxz.x;
    v.pos.z -= cxz.z;
    v.pos.y -= minY;
  }
  bmin -= glm::vec3(cxz.x, minY, cxz.z);
  bmax -= glm::vec3(cxz.x, minY, cxz.z);

  const float h = std::max(bmax.y - bmin.y, 0.001f);
  const float s = targetHeightMeters / h;
  for (auto& v : out) {
    v.pos *= s;
    v.normal = glm::normalize(v.normal);
  }

  // High-poly FBX characters × instancing tank the GPU; keep a hard cap on triangle count.
  constexpr size_t kMaxVerts = 20'000;  // ~6.7k tris — lighter instanced draws
  if (out.size() > kMaxVerts) {
    const size_t nTri = out.size() / 3;
    const size_t keepTri = kMaxVerts / 3;
    const size_t step = std::max(size_t{1}, (nTri + keepTri - 1) / keepTri);
    std::vector<LoadedVertex> decim;
    decim.reserve((nTri / step + 1) * 3);
    for (size_t t = 0; t < nTri; t += step) {
      const size_t b = t * 3;
      decim.push_back(out[b]);
      decim.push_back(out[b + 1]);
      decim.push_back(out[b + 2]);
    }
    out.swap(decim);
  }

  if (outDiffuseRgba && outDiffuseW && outDiffuseH) {
    std::string dir;
    const char* sl = std::strrchr(path, '/');
    if (sl)
      dir.assign(path, sl - path + 1);
    *outDiffuseW = 0;
    *outDiffuseH = 0;
    outDiffuseRgba->clear();
    if (!extractFirstStaffDiffuse(scene, dir, *outDiffuseRgba, *outDiffuseW, *outDiffuseH)) {
      outDiffuseRgba->clear();
      *outDiffuseW = 0;
      *outDiffuseH = 0;
    }
  }

  return true;
}

} // namespace emp_mesh
