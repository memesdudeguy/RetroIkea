#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec4 instCol0;
layout(location = 4) in vec4 instCol1;
layout(location = 5) in vec4 instCol2;
layout(location = 6) in vec4 instCol3;
layout(location = 7) in vec2 inUv;
layout(location = 8) in ivec4 boneIds;
layout(location = 9) in vec4 boneWts;

layout(binding = 0) uniform UniformBufferObject {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 fogParams;
    vec4 shadowParams;
    vec4 employeeFadeH;
    vec4 employeeBounds;
    ivec4 extraTexInfo;
    vec4 staffAnim;
} ubo;

layout(push_constant) uniform PushModel {
    mat4 model;
    vec4 staffShade;
} push;

// Must match staff_skin::kMaxPaletteBones and C++ SSBO stride.
const int STAFF_MAX_BONES = 32;

layout(std430, binding = 7) readonly buffer StaffBoneSSBO {
    mat4 bones[];
} staffBones;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragWorldPos;
layout(location = 2) flat out vec4 fragColor;
layout(location = 3) out vec3 fragLocalPos;
layout(location = 4) out vec2 fragTexCoord;
layout(location = 5) out vec3 fragLocalNormal;

mat4 fetchBoneMat(int idx) {
    int base = gl_InstanceIndex * STAFF_MAX_BONES;
    return staffBones.bones[base + clamp(idx, 0, STAFF_MAX_BONES - 1)];
}

void main() {
    fragColor = inColor;
    bool isUiBackdrop = inColor.r < 0.055 && inColor.g > 0.96 && inColor.b > 0.96;
    bool isUiText =
        inColor.r > 0.992 && inColor.g > 0.35 && inColor.g < 0.65 && inColor.b < 0.09;
    if (isUiBackdrop || isUiText) {
        gl_Position = vec4(inPosition.x, -inPosition.y, 0.0, 1.0);
        fragWorldPos = ubo.cameraPos.xyz;
        fragNormal = vec3(0.0, 0.0, 1.0);
        fragLocalPos = inPosition;
        fragTexCoord = vec2(0.0);
        fragLocalNormal = inNormal;
        return;
    }
    bool isCrosshair = inColor.r > 0.97 && inColor.b > 0.97 && inColor.g < 0.06;
    if (isCrosshair) {
        const float ch = 0.036;
        gl_Position = vec4(inPosition.x * ch, -inPosition.y * ch, 0.0, 1.0);
        fragWorldPos = ubo.cameraPos.xyz;
        fragNormal = vec3(0.0, 0.0, 1.0);
        fragLocalPos = inPosition;
        fragTexCoord = vec2(0.0);
        fragLocalNormal = inNormal;
        return;
    }

    vec3 tagE = inColor.rgb;
    bool isEmployeeV = tagE.r > 0.49 && tagE.r < 0.56 && tagE.g > 0.84 && tagE.g < 0.93 &&
                       tagE.b > 0.86 && tagE.b < 0.94;

    mat4 instMat = mat4(instCol0, instCol1, instCol2, instCol3);
    mat4 model = push.model * instMat;

    mat4 skin = mat4(1.0);
    if (isEmployeeV) {
        skin = boneWts.x * fetchBoneMat(boneIds.x) + boneWts.y * fetchBoneMat(boneIds.y) +
               boneWts.z * fetchBoneMat(boneIds.z) + boneWts.w * fetchBoneMat(boneIds.w);
    }

    mat4 M = model * skin;
    vec4 worldPos = M * vec4(inPosition, 1.0);
    fragWorldPos = worldPos.xyz;
    mat3 m3 = mat3(M);
    mat3 cofactor = mat3(cross(m3[1], m3[2]), cross(m3[2], m3[0]), cross(m3[0], m3[1]));
    fragNormal = normalize(cofactor * inNormal);
    fragLocalPos = inPosition;
    fragTexCoord = inUv;
    fragLocalNormal = inNormal;
    vec4 clip = ubo.viewProj * worldPos;
    float parkourPs1Pk = clamp(ubo.staffAnim.z, 0.0, 1.0);
    vec3 tag = inColor.rgb;
    bool isSign = tag.b > 0.95 && tag.r < 0.1 && tag.g < 0.1;
    bool isString = tag.r > 0.95 && tag.g > 0.95 && tag.b < 0.1;
    bool isWarehouseMetal = tag.r > 0.88 && tag.g < 0.14 && tag.b < 0.14;
    bool isWarehouseWood = tag.g > 0.90 && tag.r < 0.14 && tag.b < 0.14;
    bool isWarehouseCrate = tag.r > 0.82 && tag.r < 0.92 && tag.g > 0.38 && tag.g < 0.50 &&
                            tag.b > 0.08 && tag.b < 0.22;
    bool isFluoro = tag.r > 0.92 && tag.g > 0.88 && tag.b > 0.40 && tag.b < 0.52;
    bool isEmployee = isEmployeeV;
    bool isWarehouse = isWarehouseMetal || isWarehouseWood || isWarehouseCrate || isFluoro;
    bool fpMode = ubo.employeeFadeH.w > 0.5;
    float clipSnap = 120.0;
    if (fpMode) clipSnap = 180.0;
    if (!isSign && !isString && !isEmployee) {
        clip.xy = floor(clip.xy * clipSnap) / clipSnap;
    }
    if (isEmployee) {
        float snap = mix(140.0, 80.0, parkourPs1Pk * 0.92);
        if (push.staffShade.y > 0.5) {
            snap = mix(180.0, 140.0, parkourPs1Pk * 0.86);
        }
        if (fpMode) snap *= 1.6;
        clip.xy = floor(clip.xy * snap) / snap;
        float wv = fract(sin(dot(worldPos.xyz * vec3(2.71, 6.11, 3.83) + vec3(ubo.staffAnim.x * 6.2),
                               vec3(12.9898, 78.233, 45.164))) *
                        43758.5453) -
                 0.5;
        float zJit = 0.0008 * parkourPs1Pk;
        if (push.staffShade.y > 0.5)
            zJit *= 0.42;
        if (fpMode) zJit *= 0.3;
        clip.z += wv * zJit * clip.w;
    }
    gl_Position = clip;
}
