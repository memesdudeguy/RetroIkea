#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor; // rgb = material tags; a = staff baked part (see shader.frag)
layout(location = 3) in vec4 instCol0;
layout(location = 4) in vec4 instCol1;
layout(location = 5) in vec4 instCol2;
layout(location = 6) in vec4 instCol3;
layout(location = 7) in vec2 inUv; // mesh UV (staff FBX); unused for procedural geometry

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

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragWorldPos;
layout(location = 2) flat out vec4 fragColor;
layout(location = 3) out vec3 fragLocalPos;
layout(location = 4) out vec2 fragTexCoord;
layout(location = 5) out vec3 fragLocalNormal;

void main() {
    fragColor = inColor;
    // Fullscreen / HUD text: NDC inPosition.xy (tags must match shader.frag).
    bool isUiBackdrop = inColor.r < 0.055 && inColor.g > 0.96 && inColor.b > 0.96;
    bool isUiText =
        inColor.r > 0.992 && inColor.g > 0.35 && inColor.g < 0.65 && inColor.b < 0.09;
    bool isUiHealthTrack =
        inColor.r > 0.015 && inColor.r < 0.028 && inColor.g > 0.32 && inColor.g < 0.39 && inColor.b > 0.992;
    bool isUiHealthFill = inColor.r > 0.022 && inColor.r < 0.042 && inColor.g > 0.87 && inColor.g < 0.94 &&
                          inColor.b > 0.04 && inColor.b < 0.095;
    bool isUiHealthFillCrit = inColor.r > 0.054 && inColor.r < 0.063 && inColor.g > 0.20 && inColor.g < 0.28 &&
                              inColor.b > 0.075 && inColor.b < 0.098;
    bool isUiHealthFrame =
        inColor.r > 0.065 && inColor.r < 0.076 && inColor.g > 0.10 && inColor.g < 0.13 && inColor.b > 0.992;
    bool isUiPipHud = inColor.r > 0.064 && inColor.r < 0.072 && inColor.g > 0.985 && inColor.b > 0.01 &&
                      inColor.b < 0.115;
    bool isUiHudVignette =
        inColor.r > 0.016 && inColor.r < 0.022 && inColor.g > 0.85 && inColor.g < 0.91 && inColor.b > 0.988;
    bool isUiDeathVignette =
        inColor.r > 0.0138 && inColor.r < 0.0152 && inColor.g > 0.785 && inColor.g < 0.812 &&
        inColor.b > 0.985 && inColor.b < 0.996;
    bool isUiHudFont = inColor.r > 0.076 && inColor.r < 0.082 && inColor.g > 0.897 && inColor.g < 0.906 &&
                       inColor.b > 0.068 && inColor.b < 0.075;
    bool isUiIkeaPanel = inColor.r > 0.008 && inColor.r < 0.016 && inColor.g > 0.308 && inColor.g < 0.328 &&
                         inColor.b > 0.702 && inColor.b < 0.728;
    bool isUiMenuFrame = inColor.r > 0.0095 && inColor.r < 0.0135 && inColor.g > 0.310 && inColor.g < 0.326 &&
                         inColor.b > 0.498 && inColor.b < 0.512;
    bool isUiIkeaLogo = inColor.r > 0.0105 && inColor.r < 0.0135 && inColor.g > 0.315 && inColor.g < 0.324 &&
                        inColor.b > 0.510 && inColor.b < 0.522;
    bool isUiIkeaFont = inColor.r > 0.0832 && inColor.r < 0.0885 && inColor.g > 0.9015 && inColor.g < 0.9075 &&
                        inColor.b > 0.104 && inColor.b < 0.114;
    bool isUiDeathTitleFont = inColor.r > 0.089 && inColor.r < 0.0925 && inColor.g > 0.9005 && inColor.g < 0.9065 &&
                              inColor.b > 0.104 && inColor.b < 0.114;
    if (isUiBackdrop || isUiText || isUiHealthTrack || isUiHealthFill || isUiHealthFillCrit ||
        isUiHealthFrame || isUiPipHud || isUiHudVignette || isUiDeathVignette || isUiHudFont ||
        isUiIkeaPanel || isUiMenuFrame || isUiIkeaLogo || isUiIkeaFont || isUiDeathTitleFont) {
        // Vulkan clip space is Y-down; our UI verts use Y-up — flip Y so text reads upright.
        gl_Position = vec4(inPosition.x, -inPosition.y, 0.0, 1.0);
        fragWorldPos = ubo.cameraPos.xyz;
        fragNormal = vec3(0.0, 0.0, 1.0);
        fragLocalPos = inPosition;
        // Pip HUD: per-vert UV for fragment AA. TrueType HUD: glyph UVs. Menu vignettes: NDC→UV radial.
        fragTexCoord = (isUiPipHud || isUiHudFont || isUiIkeaFont || isUiIkeaLogo || isUiDeathTitleFont) ? inUv
                       : (isUiHudVignette || isUiDeathVignette) ? inPosition.xy * 0.5 + 0.5
                                                               : vec2(0.5);
        fragLocalNormal = inNormal;
        return;
    }
    // Screen-space crosshair (NDC quad, drawn last).
    // push.staffShade: .x = scale (1 = default), .y = radians rotation (shove feedback).
    bool isCrosshair = inColor.r > 0.97 && inColor.b > 0.97 && inColor.g < 0.06;
    if (isCrosshair) {
        float chMul = (push.staffShade.x > 1e-5) ? push.staffShade.x : 1.0;
        const float ch = 0.036 * chMul;
        float ang = push.staffShade.y;
        vec2 p = inPosition.xy;
        float c = cos(ang);
        float s = sin(ang);
        vec2 pr = vec2(c * p.x - s * p.y, s * p.x + c * p.y);
        gl_Position = vec4(pr.x * ch, -pr.y * ch, 0.0, 1.0);
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
    vec3 posL = inPosition;
    // Cheap motion read: not real skeletal clips — vertex wobble from sim time (day vs night gait).
    if (isEmployeeV) {
        float t = ubo.staffAnim.x;
        float gait = max(ubo.staffAnim.y, 0.4);
        // Per-instance phase so a crowd does not move in lockstep (translation is column 3 of instance mat).
        float instPh = instMat[3].x * 0.173 + instMat[3].z * 0.131;
        float legW = clamp(inPosition.y * 1.65, 0.0, 1.0);
        float bob = sin(t * (2.85 * gait) + instPh + inPosition.x * 3.17 + inPosition.z * 2.71);
        posL.x += bob * 0.026 * legW;
        posL.z += cos(t * (2.55 * gait) + instPh * 1.3 + inPosition.y * 5.5) * 0.021 * legW;
        posL.y += sin(t * (3.15 * gait) + instPh * 0.7 + inPosition.z * 4.0) * 0.014 * (0.35 + legW);
    }
    vec4 worldPos = model * vec4(posL, 1.0);
    fragWorldPos = worldPos.xyz;
    // Normal transform: upper 3×3 inverse-transpose (cheaper than full mat4 inverse).
    mat3 m3 = mat3(model);
    mat3 normalMat = transpose(inverse(m3));
    fragNormal = normalize(normalMat * inNormal);
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
    // PS1-style geometry wobble (kept moderate to avoid giant artifacts).
    const float clipSnap = 120.0;
    if (!isSign && !isString && !isWarehouse && !isEmployee) {
        clip.xy = floor(clip.xy * clipSnap) / clipSnap;
    }
    // Staff: chunky clip-space verts (separate from shelves); tighter under parkour (PS1 × DL motion).
    if (isEmployee) {
        float snap = mix(68.0, 38.0, parkourPs1Pk * 0.92);
        clip.xy = floor(clip.xy * snap) / snap;
        float wv = fract(sin(dot(worldPos.xyz * vec3(2.71, 6.11, 3.83) + vec3(ubo.staffAnim.x * 6.2),
                               vec3(12.9898, 78.233, 45.164))) *
                        43758.5453) -
                 0.5;
        clip.z += wv * 0.0021 * parkourPs1Pk * clip.w;
    }
    gl_Position = clip;
}
