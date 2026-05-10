#version 450

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragWorldPos;
layout(location = 2) flat in vec4 fragColor;
layout(location = 3) in vec3 fragLocalPos;
layout(location = 4) in vec2 fragTexCoord;
layout(location = 5) in vec3 fragLocalNormal;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 fogParams;
    vec4 shadowParams;
    vec4 employeeFadeH;
    vec4 employeeBounds;
    // .x = loaded extra texture count; .y = world triplanar blend extraTex[0]; .z = staff multi-tex blend (0..255).
    ivec4 extraTexInfo;
    vec4 staffAnim;
} ubo;

layout(push_constant) uniform PushModel {
    mat4 model;
    vec4 staffShade;
} push;

layout(binding = 1) uniform sampler2D sceneTex;
layout(binding = 2) uniform sampler2D signTex;
layout(binding = 3) uniform sampler2D shelfTex;
layout(binding = 4) uniform sampler2D crateTex;
// Disk-loaded images (VULKAN_GAME_EXTRA_TEXTURES). Length must match kMaxExtraTextures in main.cpp.
layout(binding = 5) uniform sampler2D extraTex[16];
// Embedded staff mesh diffuse (GLB/glTF) sampled with mesh UVs when extraTexInfo.w != 0.
layout(binding = 6) uniform sampler2D staffGlbTex;
// Shrek egg mesh diffuse only (push.staffShade.w in [2,3)); binding 6 stays staff for all other draws.
layout(binding = 8) uniform sampler2D shrekEggTex;
layout(binding = 9) uniform sampler2D hudFontTex;
layout(binding = 10) uniform sampler2D titleIkeaLogoTex;
layout(binding = 11) uniform sampler2D boxCutterTex;
layout(binding = 12) uniform sampler2D rustyPipeTex;
layout(binding = 13) uniform sampler2D palletTex;

// Staff: extraTex[0]=pants, [1]=shirt, [2]=skin. Unrolled for Vulkan sampler arrays.
vec2 staffUvToTexelCenter(vec2 uv, vec2 texDims, float blockMul) {
    vec2 d = max(texDims * blockMul, vec2(1.0));
    vec2 q = uv * d;
    return (floor(q) + vec2(0.5)) / d;
}

vec3 sampleStaffExtraTex(int slot, vec2 uv) {
    const float kStaffPartUvBlocks = 0.34;
    if (slot == 0) {
        vec2 uvc = staffUvToTexelCenter(uv, vec2(textureSize(extraTex[0], 0)), kStaffPartUvBlocks);
        return texture(extraTex[0], uvc).rgb;
    }
    if (slot == 1) {
        vec2 uvc = staffUvToTexelCenter(uv, vec2(textureSize(extraTex[1], 0)), kStaffPartUvBlocks);
        return texture(extraTex[1], uvc).rgb;
    }
    if (slot == 2) {
        vec2 uvc = staffUvToTexelCenter(uv, vec2(textureSize(extraTex[2], 0)), kStaffPartUvBlocks);
        return texture(extraTex[2], uvc).rgb;
    }
    return vec3(0.0);
}

vec3 applyOptionalExtraLayer0(vec3 base) {
    if (ubo.extraTexInfo.x < 1)
        return base;
    float t = clamp(float(ubo.extraTexInfo.y) * (1.0 / 255.0), 0.0, 1.0);
    if (t <= 0.0)
        return base;
    vec2 uve = fract(fragWorldPos.xz * 0.08);
    return mix(base, texture(extraTex[0], uve).rgb, t);
}

vec3 applyStoreLightMul(vec3 rgb) {
    return rgb * max(ubo.fogParams.z, 0.0);
}

float atmosphereFogAmt(float dist) {
    float t = smoothstep(ubo.fogParams.x, ubo.fogParams.y, dist);
    if (ubo.fogParams.w > 0.5)
        t = 1.0 - pow(max(1.0 - t, 0.0), 1.22);
    else
        t = pow(clamp(t, 0.0, 1.0), 1.28);
    return clamp(t, 0.0, 1.0);
}

vec3 atmosphereFogColor() {
    if (ubo.fogParams.w > 0.5)
        return vec3(0.11, 0.118, 0.14);
    return vec3(0.82, 0.83, 0.86);
}

vec3 fogMixLit(vec3 lit, float dist) {
    float a = atmosphereFogAmt(dist);
    if (ubo.fogParams.w > 0.5)
        a *= 0.88;
    else
        a = min(1.0, a * 1.22);
    return mix(lit, atmosphereFogColor(), a);
}

// One textureSize per pass; avoids three lookups in triplanar.
vec2 crunchyUvDims(vec2 uv, vec2 texDims) {
    return fract(uv);
}

vec3 retroWorldSamplePos(vec3 worldPos) {
    const float k = 14.0;
    return floor(worldPos * k) / k;
}

vec3 empTriSample(sampler2D tex, vec3 localPos, vec3 localN, float tile) {
    vec2 dim = vec2(textureSize(tex, 0));
    vec3 an = abs(localN);
    vec3 blend = max(an, vec3(0.0001));
    blend /= (blend.x + blend.y + blend.z);
    vec3 sx = texture(tex, crunchyUvDims(localPos.yz * tile, dim)).rgb;
    vec3 sy = texture(tex, crunchyUvDims(localPos.xz * tile, dim)).rgb;
    vec3 sz = texture(tex, crunchyUvDims(localPos.xy * tile, dim)).rgb;
    return sx * blend.x + sy * blend.y + sz * blend.z;
}

// Procedural staff uniform (no photo textures): yellow polo + dark vertical pinstripes, name tag on front.
vec3 empEmployeeShirt(vec3 lp, vec3 LN) {
    vec3 lpQ = floor(lp * 36.0) / 36.0;
    float yLo = ubo.employeeBounds.x;
    float yHi = ubo.employeeBounds.y;
    float relY = clamp((lp.y - yLo) / max(yHi - yLo, 0.01), 0.0, 1.0);
    float a = atan(lpQ.x, lpQ.z);
    const float kStripe = 15.0;
    float ph = fract(a * (kStripe / 6.28318530718));
    float str = smoothstep(0.44, 0.49, ph) * (1.0 - smoothstep(0.51, 0.56, ph));
    vec3 yel = vec3(0.98, 0.86, 0.12);
    vec3 dk = vec3(0.04, 0.06, 0.11);
    vec3 c = mix(yel, dk, str * 0.84);
    bool frontish = LN.z > 0.34;
    bool tag = frontish && lp.x > 0.048 && lp.x < 0.185 && relY > 0.34 && relY < 0.58 && lp.z > -0.03 && lp.z < 0.13;
    if (tag)
        c = mix(c, vec3(0.03, 0.09, 0.22), 0.93);
    float hem = smoothstep(0.0, 0.07, relY) * (1.0 - smoothstep(0.90, 1.0, relY));
    c *= 0.86 + 0.14 * hem;
    float facing = clamp(0.52 + 0.48 * LN.z, 0.0, 1.0);
    c *= mix(0.74, 1.0, facing);
    return c;
}

vec3 empEmployeePants(vec3 lp) {
    float g = fract(sin(dot(floor(lp * 16.0), vec3(127.1, 311.7, 74.7))) * 43758.5453);
    vec3 blue = vec3(0.09, 0.36, 0.86);
    return blue * (0.91 + 0.09 * g);
}

vec3 empEmployeeSkin(vec3 lp) {
    float g = fract(sin(dot(floor(lp * 20.0), vec3(19.1, 47.3, 91.7))) * 17834.413);
    vec3 pale = vec3(0.82, 0.81, 0.78);
    return mix(pale * 0.92, pale * 1.06, g);
}

vec3 triplanarTexture(vec3 worldPos, vec3 normal) {
    vec2 sz = vec2(textureSize(sceneTex, 0));
    vec3 an = abs(normal);
    vec3 blend = max(an, vec3(0.0001));
    blend /= (blend.x + blend.y + blend.z);
    const float tile = 0.06;
    vec3 sx = texture(sceneTex, crunchyUvDims(worldPos.yz * tile, sz)).rgb;
    vec3 sy = texture(sceneTex, crunchyUvDims(worldPos.xz * tile, sz)).rgb;
    vec3 szm = texture(sceneTex, crunchyUvDims(worldPos.xy * tile, sz)).rgb;
    return sx * blend.x + sy * blend.y + szm * blend.z;
}

float bayer4x4(vec2 p) {
    ivec2 i = ivec2(floor(p)) & 3;
    int idx = i.x + i.y * 4;
    const float mat[16] = float[16](
        0.0,  8.0,  2.0, 10.0,
       12.0,  4.0, 14.0,  6.0,
        3.0, 11.0,  1.0,  9.0,
       15.0,  7.0, 13.0,  5.0
    );
    return mat[idx] / 16.0;
}

// N must be unit; avoids redundant normalize in callers.
vec3 applyPlayerShadow(vec3 lit, vec3 N, vec3 worldPos) {
    vec4 sp = ubo.shadowParams;
    float ndUp = dot(N, vec3(0.0, 1.0, 0.0));
    if (ndUp < 0.48)
        return lit;
    float h = abs(worldPos.y - sp.z);
    if (h > 0.16)
        return lit;
    float d = length(worldPos.xz - sp.xy);
    float r = max(sp.w, 0.08);
    float falloff = 1.0 - smoothstep(0.0, r, d);
    float shadowAmt = falloff * falloff * 0.36;
    return lit * (1.0 - shadowAmt);
}

vec3 ditherQuantize(vec3 rgb, float pixelSize, float levels, float ditherScale) {
    levels = max(levels, 36.0);
    ditherScale *= 0.40;
    vec2 blockCoord = floor(gl_FragCoord.xy / pixelSize) * pixelSize;
    float di = (bayer4x4(blockCoord) - 0.5) / levels;
    rgb = clamp(rgb + di * ditherScale, 0.0, 1.0);
    return floor(rgb * levels) / levels;
}

// Pip HUD: anti-alias quad edges (thin NDC lines alias badly at 1x; fwidth smoothsteps ~1–2 px blend).
float pipHudQuadFeather(vec2 uv) {
    vec2 fw = fwidth(uv);
    float a = max(fw.x + fw.y, 1e-4);
    float m = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    float span = clamp(a * 3.2, 0.006, 0.5);
    // Negative lower bound: corners (m=0) stay opaque; outer silhouette still softens (~1–2 px).
    return smoothstep(-span * 0.38, span, m);
}

float pipHudTriFeather(vec2 uv) {
    vec2 fw = fwidth(uv);
    float w = max(max(fw.x, fw.y), 1e-4);
    float d = min(min(uv.x, uv.y), max(0.0, 1.0 - uv.x - uv.y));
    float span = clamp(w * 3.0, 0.006, 0.5);
    return smoothstep(-span * 0.42, span, d);
}

float pipHudFeatherMask() {
    bool tri = fragLocalNormal.z < 0.25;
    return tri ? pipHudTriFeather(fragTexCoord) : pipHudQuadFeather(fragTexCoord);
}

void main() {
    vec3 tag = fragColor.rgb;
    bool uiBackdrop = tag.r < 0.055 && tag.g > 0.96 && tag.b > 0.96;
    bool uiText = tag.r > 0.992 && tag.g > 0.35 && tag.g < 0.65 && tag.b < 0.09;
    bool uiHealthTrack =
        tag.r > 0.015 && tag.r < 0.028 && tag.g > 0.32 && tag.g < 0.39 && tag.b > 0.992;
    bool uiHealthFill = tag.r > 0.022 && tag.r < 0.042 && tag.g > 0.87 && tag.g < 0.94 && tag.b > 0.04 &&
                        tag.b < 0.095;
    bool uiHealthFillCrit =
        tag.r > 0.054 && tag.r < 0.063 && tag.g > 0.20 && tag.g < 0.28 && tag.b > 0.075 && tag.b < 0.098;
    bool uiHealthFrame =
        tag.r > 0.065 && tag.r < 0.076 && tag.g > 0.10 && tag.g < 0.13 && tag.b > 0.992;
    bool pipBg =
        tag.r > 0.064 && tag.r < 0.072 && tag.g > 0.985 && tag.b > 0.012 && tag.b < 0.032;
    bool pipBright =
        tag.r > 0.064 && tag.r < 0.072 && tag.g > 0.985 && tag.b > 0.034 && tag.b < 0.052;
    bool pipDim =
        tag.r > 0.064 && tag.r < 0.072 && tag.g > 0.985 && tag.b > 0.054 && tag.b < 0.072;
    bool pipCrit =
        tag.r > 0.064 && tag.r < 0.072 && tag.g > 0.985 && tag.b > 0.074 && tag.b < 0.092;
    bool pipText =
        tag.r > 0.064 && tag.r < 0.072 && tag.g > 0.985 && tag.b > 0.092 && tag.b < 0.112;
    bool uiHudVignette =
        tag.r > 0.016 && tag.r < 0.022 && tag.g > 0.85 && tag.g < 0.91 && tag.b > 0.988;
    bool uiDeathVignette =
        tag.r > 0.0138 && tag.r < 0.0152 && tag.g > 0.785 && tag.g < 0.812 && tag.b > 0.985 && tag.b < 0.996;
    bool uiHudFont = tag.r > 0.076 && tag.r < 0.082 && tag.g > 0.897 && tag.g < 0.906 && tag.b > 0.068 &&
                     tag.b < 0.075;
    bool uiIkeaPanel = tag.r > 0.008 && tag.r < 0.016 && tag.g > 0.308 && tag.g < 0.328 && tag.b > 0.702 &&
                       tag.b < 0.728;
    bool uiOptionBtn = tag.r > 0.011 && tag.r < 0.014 && tag.g > 0.310 && tag.g < 0.326 &&
                       tag.b > 0.625 && tag.b < 0.645;
    bool uiMenuFrame =
        tag.r > 0.0095 && tag.r < 0.0135 && tag.g > 0.310 && tag.g < 0.326 && tag.b > 0.498 && tag.b < 0.512;
    bool uiIkeaLogo = tag.r > 0.0105 && tag.r < 0.0135 && tag.g > 0.315 && tag.g < 0.324 && tag.b > 0.510 &&
                      tag.b < 0.522;
    bool uiDeathTitleFont = tag.r > 0.089 && tag.r < 0.0925 && tag.g > 0.9005 && tag.g < 0.9065 && tag.b > 0.104 &&
                            tag.b < 0.114;
    bool uiIkeaFont = tag.r > 0.0832 && tag.r < 0.0885 && tag.g > 0.9015 && tag.g < 0.9075 && tag.b > 0.104 &&
                      tag.b < 0.114;
    if (uiIkeaLogo) {
        // Title wordmark: sample assets/ui/title_ikea_logo.png; ubo.employeeFadeH.z = title time (see main.cpp).
        float t = ubo.employeeFadeH.z;
        vec2 ctr = fragTexCoord - 0.5;
        float breathe = 1.0 + 0.02 * sin(t * 2.35);
        vec2 uv = ctr / breathe + 0.5;
        vec2 wob = vec2(sin(t * 5.9 + uv.y * 28.0), cos(t * 4.7 + uv.x * 26.0)) * 0.0011;
        uv = clamp(uv + wob, vec2(0.002), vec2(0.998));
        float roll = 0.00065 * sin(t * 7.3);
        vec4 base = texture(titleIkeaLogoTex, uv);
        float r = texture(titleIkeaLogoTex, uv + vec2(roll, 0.0)).r;
        float g = base.g;
        float b = texture(titleIkeaLogoTex, uv - vec2(roll, 0.0)).b;
        vec3 c = vec3(r, g, b);
        float scan = sin(fragLocalPos.y * 480.0 + t * 2.1) * 0.5 + 0.5;
        c *= mix(0.9, 1.06, scan);
        c *= vec3(1.04, 0.97, 0.93);
        float vig = 1.0 - dot(ctr, ctr) * 0.28;
        c *= vig;
        outColor = vec4(c, base.a);
        return;
    }
    if (uiDeathVignette) {
        float uiFade = fragLocalPos.z > 0.001 ? fragLocalPos.z : 1.0;
        float uiTime = fragLocalNormal.z;
        vec2 uv = fragTexCoord - 0.5;
        float d = length(uv) * 1.22;
        float falloff = smoothstep(0.28, 1.0, d);
        bool isLoadScreen = uiTime > 0.01;
        if (isLoadScreen) {
            float pulse = 0.5 + 0.5 * sin(uiTime * 0.8);
            float pulse2 = 0.5 + 0.5 * sin(uiTime * 0.5 + 1.2);
            vec3 cen = vec3(0.38, 0.40, 0.46) + vec3(0.06, 0.05, 0.08) * pulse;
            vec3 rim = vec3(0.18, 0.19, 0.22) + vec3(0.04, 0.03, 0.05) * pulse2;
            float softVig = smoothstep(0.15, 1.1, d);
            vec3 tint = mix(cen, rim, softVig);
            float glow = 0.08 * exp(-d * d * 3.0) * (0.7 + 0.3 * pulse);
            tint += vec3(glow * 0.9, glow * 0.92, glow);
            outColor = vec4(tint, uiFade);
        } else {
            vec3 rim = vec3(0.38, 0.03, 0.05);
            vec3 cen = vec3(0.04, 0.01, 0.02);
            vec3 tint = mix(cen, rim, falloff);
            float a = mix(0.44, 0.82, falloff) * fragColor.a;
            outColor = vec4(tint, a);
        }
        return;
    }
    if (uiHudVignette) {
        vec2 uv = fragTexCoord - 0.5;
        float d = length(uv) * 1.18;
        float falloff = smoothstep(0.22, 0.98, d);
        // Cool store-floor blue in center → warm rim (pause / menu vignette).
        vec3 rim = vec3(0.10, 0.09, 0.12);
        vec3 cen = vec3(0.015, 0.028, 0.055);
        vec3 tint = mix(cen, rim, falloff * 0.88);
        float a = mix(0.46, 0.84, falloff) * fragColor.a;
        outColor = vec4(tint, a);
        return;
    }
    if (uiBackdrop) {
        float uiFade = fragLocalPos.z > 0.001 ? fragLocalPos.z : 1.0;
        // Deep blue-black scrim instead of flat black.
        outColor = vec4(0.018, 0.035, 0.085, uiFade);
        return;
    }
    if (uiMenuFrame) {
        vec2 uv = fragTexCoord;
        vec2 pxSz = fwidth(uv);
        vec2 panelPx = 1.0 / max(pxSz, vec2(0.0001));
        vec2 posPx = (uv - 0.5) * panelPx;
        vec2 halfPx = panelPx * 0.5;
        float r = clamp(min(halfPx.x, halfPx.y) * 0.40, 22.0, 64.0);
        vec2 d = abs(posPx) - halfPx + vec2(r);
        float sdf = min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0))) - r;
        float aa = 1.35;
        const float strokePx = 3.2;
        float outer = 1.0 - smoothstep(-aa, aa, sdf);
        float inner = 1.0 - smoothstep(-aa, aa, sdf + strokePx);
        float stroke = clamp(outer - inner, 0.0, 1.0);
        if (stroke < 0.02)
            discard;
        vec3 rim = mix(vec3(0.52, 0.78, 1.0), vec3(1.0, 0.88, 0.45), 0.38);
        float pulse = 0.5 + 0.5 * sin(ubo.staffAnim.x * 2.6 + (uv.x + uv.y) * 6.28);
        rim += vec3(0.04, 0.03, 0.02) * pulse;
        outColor = vec4(rim, stroke * fragColor.a);
        return;
    }
    if (uiIkeaPanel) {
        vec2 uv = fragTexCoord;
        vec2 pxSz = fwidth(uv);
        vec2 panelPx = 1.0 / max(pxSz, vec2(0.0001));
        vec2 posPx = (uv - 0.5) * panelPx;
        vec2 halfPx = panelPx * 0.5;
        float r = clamp(min(halfPx.x, halfPx.y) * 0.40, 22.0, 62.0);
        vec2 d = abs(posPx) - halfPx + vec2(r);
        float sdf = min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0))) - r;
        float aa = 1.25;
        float mask = 1.0 - smoothstep(-aa, aa, sdf);
        if (mask < 0.01) discard;
        float inner = 1.0 - smoothstep(-aa, aa, sdf + 2.8);
        // Vertical cool gradient + soft top sheen (store signage feel).
        float gy = smoothstep(0.08, 0.92, uv.y);
        vec3 fillHi = vec3(0.14, 0.18, 0.26);
        vec3 fillLo = vec3(0.07, 0.08, 0.11);
        vec3 edgeHi = vec3(0.32, 0.38, 0.48);
        vec3 edgeLo = vec3(0.18, 0.19, 0.22);
        vec3 fill = mix(fillHi, fillLo, gy);
        vec3 edge = mix(edgeHi, edgeLo, gy);
        vec3 base = mix(edge, fill, inner);
        float inward = max(-sdf, 0.0);
        const float outlinePx = 2.75;
        float outlineBand = 1.0 - smoothstep(outlinePx - 1.35, outlinePx + 1.15, inward);
        vec3 outlineBlue = vec3(0.38, 0.62, 0.95);
        vec3 outlineGold = vec3(0.95, 0.82, 0.38);
        float goldMix = 0.22 + 0.12 * sin(atan(uv.x - 0.5, uv.y - 0.5) * 3.0);
        vec3 outlineRgb = mix(outlineBlue, outlineGold, clamp(goldMix, 0.0, 1.0));
        vec3 col = mix(base, outlineRgb, outlineBand * 0.82);
        float sheen = smoothstep(0.2, 0.48, uv.y) * smoothstep(0.62, 0.38, uv.y) * inner;
        col += sheen * vec3(0.12, 0.14, 0.16);
        outColor = vec4(col, mask);
        return;
    }
    if (uiOptionBtn) {
        vec2 uv = fragTexCoord;
        vec2 pxSz = fwidth(uv);
        vec2 panelPx = 1.0 / max(pxSz, vec2(0.0001));
        vec2 posPx = (uv - 0.5) * panelPx;
        vec2 halfPx = panelPx * 0.5;
        float r = clamp(min(halfPx.x, halfPx.y) * 0.58, 16.0, 44.0);
        vec2 d = abs(posPx) - halfPx + vec2(r);
        float sdf = min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0))) - r;
        float aa = 1.25;
        float mask = 1.0 - smoothstep(-aa, aa, sdf);
        if (mask < 0.01) discard;
        float inner = 1.0 - smoothstep(-aa, aa, sdf + 2.55);
        float gx = smoothstep(0.0, 1.0, uv.x);
        vec3 fillL = vec3(0.22, 0.30, 0.44);
        vec3 fillR = vec3(0.12, 0.16, 0.26);
        vec3 edgeL = vec3(0.38, 0.48, 0.62);
        vec3 edgeR = vec3(0.24, 0.32, 0.44);
        vec3 fill = mix(fillL, fillR, gx);
        vec3 edge = mix(edgeL, edgeR, gx);
        vec3 base = mix(edge, fill, inner);
        float inward = max(-sdf, 0.0);
        const float outlinePx = 2.55;
        float outlineBand = 1.0 - smoothstep(outlinePx - 1.15, outlinePx + 1.1, inward);
        vec3 outlineRgb = mix(vec3(0.55, 0.78, 1.0), vec3(0.98, 0.86, 0.42), 0.32);
        vec3 col = mix(base, outlineRgb, outlineBand * 0.92);
        float cap = smoothstep(0.32, 0.48, abs(uv.x - 0.5) * 2.0);
        col += (1.0 - cap) * smoothstep(0.58, 0.22, uv.y) * inner * vec3(0.10, 0.12, 0.16);
        outColor = vec4(col, mask);
        return;
    }
    if (pipBg) {
        float f = pipHudFeatherMask();
        vec2 u = fragTexCoord;
        vec3 c = mix(vec3(0.04, 0.06, 0.10), vec3(0.07, 0.09, 0.14), u.y);
        outColor = vec4(c, 0.84 * f);
        return;
    }
    if (pipDim) {
        float f = pipHudFeatherMask();
        outColor = vec4(0.28, 0.32, 0.40, f);
        return;
    }
    if (pipBright) {
        float f = pipHudFeatherMask();
        outColor = vec4(0.78, 0.82, 0.92, f);
        return;
    }
    if (pipCrit) {
        float f = pipHudFeatherMask();
        outColor = vec4(1.0, 0.42, 0.32, f);
        return;
    }
    if (pipText) {
        // stb_easy_font quads are already blocky; edge feather uses fwidth(uv) and can wipe thin glyphs when
        // coverage is only a few pixels. Solid fill keeps labels readable; HP/compass text is built larger in C++.
        outColor = vec4(0.88, 0.91, 0.98, 1.0);
        return;
    }
    if (uiHealthFrame) {
        float gx = fragTexCoord.x;
        vec3 hi = vec3(0.55, 0.58, 0.65);
        vec3 lo = vec3(0.35, 0.36, 0.40);
        outColor = vec4(mix(lo, hi, gx), 1.0);
        return;
    }
    if (uiHealthTrack) {
        outColor = vec4(0.06, 0.08, 0.12, 1.0);
        return;
    }
    if (uiHealthFill) {
        float gx = fragTexCoord.x;
        vec3 a = vec3(0.15, 0.72, 0.42);
        vec3 b = vec3(0.45, 0.98, 0.62);
        outColor = vec4(mix(a, b, smoothstep(0.0, 1.0, gx)), 1.0);
        return;
    }
    if (uiHealthFillCrit) {
        float gx = fragTexCoord.x;
        vec3 a = vec3(0.75, 0.12, 0.08);
        vec3 b = vec3(1.0, 0.55, 0.28);
        outColor = vec4(mix(a, b, smoothstep(0.0, 1.0, gx)), 1.0);
        return;
    }
    if (uiText) {
        outColor = vec4(0.92, 0.94, 1.0, 1.0);
        return;
    }
    if (uiDeathTitleFont) {
        float cov = texture(hudFontTex, fragTexCoord).a;
        float a = smoothstep(0.12, 0.92, cov);
        vec3 hot = vec3(1.0, 0.22, 0.12);
        vec3 deep = vec3(0.55, 0.05, 0.04);
        float gy = fragTexCoord.y;
        vec3 col = mix(deep, hot, smoothstep(0.15, 0.85, gy));
        col += cov * cov * vec3(0.35, 0.12, 0.08);
        outColor = vec4(col, a);
        return;
    }
    if (uiIkeaFont) {
        float cov = texture(hudFontTex, fragTexCoord).a;
        float a = smoothstep(0.12, 0.92, cov);
        float gy = fragTexCoord.y;
        // Titles: warm gold; body: cool white; options: icy blue-white.
        vec3 pri = mix(vec3(0.88, 0.90, 0.96), vec3(0.72, 0.78, 0.92), gy * 0.5);
        vec3 acc = mix(vec3(0.98, 0.90, 0.48), vec3(0.92, 0.72, 0.28), gy);
        vec3 opt = mix(vec3(0.78, 0.88, 1.0), vec3(0.55, 0.72, 0.95), gy * 0.65);
        vec3 col = tag.r > 0.0865 ? opt : (fragColor.g > 0.9048 ? acc : pri);
        col *= 1.0 + 0.18 * cov * cov;
        outColor = vec4(col, a);
        return;
    }
    if (uiHudFont) {
        float cov = texture(hudFontTex, fragTexCoord).a;
        float a = smoothstep(0.04, 0.88, cov);
        float uiFade = fragLocalPos.z > 0.001 ? fragLocalPos.z : 1.0;
        float gy = fragTexCoord.y;
        vec3 pri = mix(vec3(0.96, 0.97, 1.0), vec3(0.75, 0.82, 0.95), gy * 0.35);
        vec3 acc = mix(vec3(0.85, 0.88, 0.95), vec3(0.55, 0.65, 0.88), gy);
        vec3 col = fragColor.g > 0.902 ? acc : pri;
        col *= 1.0 + 0.1 * cov;
        outColor = vec4(col, a * uiFade);
        return;
    }
    if (tag.r > 0.97 && tag.b > 0.97 && tag.g < 0.06) {
        vec2 p = fragLocalPos.xy;
        float ax = abs(p.x);
        float ay = abs(p.y);
        float arm = 0.92;
        float t = 0.085;
        float gap = 0.16;
        float h = step(gap, ax) * step(ax, arm) * step(ay, t);
        float v = step(gap, ay) * step(ay, arm) * step(ax, t);
        if (max(h, v) < 0.5)
            discard;
        float br = (push.staffShade.z > 1e-5) ? push.staffShade.z : 1.0;
        br = min(br, 1.65);
        outColor = vec4(vec3(0.92, 0.94, 0.98) * br, 1.0);
        return;
    }

    bool isShelfMetal = tag.r > 0.88 && tag.g < 0.14 && tag.b < 0.14;
    bool isShelfWood = tag.g > 0.90 && tag.r < 0.14 && tag.b < 0.14;
    bool isShelfLadder = tag.r > 0.10 && tag.r < 0.20 && tag.g > 0.11 && tag.g < 0.22 &&
                          tag.b > 0.08 && tag.b < 0.19;
    bool isShelfBoxCutter = tag.r > 0.21 && tag.r < 0.30 && tag.g > 0.04 && tag.g < 0.11 &&
                             tag.b > 0.48 && tag.b < 0.55;
    bool isShelfRustyPipe = tag.r > 0.47 && tag.r < 0.59 && tag.g > 0.22 && tag.g < 0.36 &&
                             tag.b > 0.05 && tag.b < 0.13;
    bool isShelfPallet = tag.r > 0.62 && tag.r < 0.70 && tag.g > 0.46 && tag.g < 0.54 &&
                          tag.b > 0.24 && tag.b < 0.32;
    bool isShelfCrate = tag.r > 0.82 && tag.r < 0.92 && tag.g > 0.38 && tag.g < 0.50 &&
                         tag.b > 0.08 && tag.b < 0.22;
    bool isEmployee = tag.r > 0.49 && tag.r < 0.56 && tag.g > 0.84 && tag.g < 0.93 &&
                      tag.b > 0.86 && tag.b < 0.94;
    float parkourPs1Pk = clamp(ubo.staffAnim.z, 0.0, 1.0);

    float employeePopA = 1.0;
    if (isEmployee && push.staffShade.w < 0.5) {
        float dH = length(fragWorldPos.xz - ubo.cameraPos.xz);
        // FP near-camera discard: hide NPC fragments inside/near the player capsule.
        // Arms/weapons during melee can extend past the NPC center, so the radius
        // must cover the full body extent, not just the torso center.
        // First-person only: cut fragments that actually intersect the eye (prevents z-fight / inside-mesh
        // flashes). Older 1.2–2.2 m smoothstep faded whole NPCs at normal conversation range — mix with fog
        // looked like they "vanished" when you stepped close.
        if (ubo.employeeFadeH.w > 0.5) {
            float d3 = length(fragWorldPos - ubo.cameraPos.xyz);
            const float kEyeHard = 0.36;
            const float kEyeSoft = 0.52;
            if (d3 < kEyeHard)
                discard;
            if (d3 < kEyeSoft)
                employeePopA *= smoothstep(kEyeHard, kEyeSoft, d3);
        }
        float inR = max(ubo.employeeFadeH.x, 0.5);
        float outR = max(ubo.employeeFadeH.y, inR + 0.5);
        employeePopA *= clamp(1.0 - smoothstep(inR, outR, dH), 0.0, 1.0);
        if (employeePopA < 0.02)
            discard;
    }

    // Local player skinned body: flat grey (no GLB texture). w in (0.5, 1.5). staffShade.x: FP-only mesh clip.
    // w >= 2: textured skinned draw (e.g. Shrek egg) — skip grey.
    if (isEmployee && push.staffShade.w > 0.5 && push.staffShade.w < 1.5) {
        // Defence-in-depth: in first-person mode (employeeFadeH.w == 1), discard ALL
        // local-player body fragments so a draw-gate regression cannot leak the mesh.
        if (ubo.employeeFadeH.w > 0.5)
            discard;
        vec3 Nf = normalize(fragNormal);
        vec3 toCam = ubo.cameraPos.xyz - fragWorldPos;
        float distCam = length(toCam);
        vec3 Vdir = distCam > 1e-4 ? toCam * (1.0 / distCam) : vec3(0.0, 0.0, 1.0);
        vec3 base = vec3(0.56, 0.56, 0.58);
        vec3 Lk = normalize(vec3(0.33, 0.90, 0.28));
        vec3 Lfill = normalize(vec3(0.46, 0.36, 0.48));
        float wrap = 0.12;
        float ndkRaw = max(dot(Nf, Lk), 0.0);
        float ndkW = clamp((ndkRaw + wrap) / (1.0 + wrap), 0.0, 1.0);
        float shade = pow(ndkW * 0.5 + 0.5, 1.08);
        float ndf = max(dot(Nf, Lfill), 0.0) * 0.54;
        float amb = 0.24 + 0.078 * max(Nf.y, 0.0);
        vec3 lit = base * amb + base * (0.44 * shade + ndf);
        float ndV = max(dot(Nf, Vdir), 0.0);
        float rimAmt = pow(1.0 - ndV, 2.2) * 0.08;
        vec3 rimCol = vec3(0.72, 0.72, 0.74) * rimAmt;
        lit += rimCol;
        float spec = pow(max(dot(reflect(-Lk, Nf), Vdir), 0.0), 18.0) * 0.028;
        lit += vec3(spec);
        lit = applyPlayerShadow(lit, Nf, fragWorldPos);
        vec3 rgb = fogMixLit(lit, distCam);
        outColor = vec4(applyStoreLightMul(rgb), 1.0);
        return;
    }

    vec3 N = normalize(fragNormal);
    vec3 toCam = ubo.cameraPos.xyz - fragWorldPos;
    float distCam = length(toCam);
    vec3 V = distCam > 1e-5 ? toCam * (1.0 / distCam) : vec3(0.0, 0.0, 1.0);

    if (isEmployee) {
        vec3 Nfac = normalize(fragNormal);
        float nStep = mix(4.5, 3.05, parkourPs1Pk);
        vec3 Nq = vec3(ivec3(round(Nfac * nStep)));
        vec3 N = length(Nq) > 0.01 ? normalize(Nq) : Nfac;
        // Staff: binding 6 + optional UV block quantize. Shrek egg: binding 8 + direct mesh UV (never staff atlas).
        if (ubo.extraTexInfo.w != 0 || (push.staffShade.w >= 2.0 && push.staffShade.w < 3.0)) {
            vec2 uvg = vec2(fragTexCoord.x, 1.0 - fragTexCoord.y);
            vec3 base;
            if (push.staffShade.w >= 2.0 && push.staffShade.w < 3.0)
                base = texture(shrekEggTex, uvg).rgb;
            else {
                vec2 dimG = vec2(textureSize(staffGlbTex, 0));
                const float kStaffGlbUvBlocks = 0.26;
                vec2 uvp = staffUvToTexelCenter(uvg, dimG, kStaffGlbUvBlocks);
                base = texture(staffGlbTex, uvp).rgb;
            }
            vec3 Lk = normalize(vec3(0.35, 0.88, 0.32));
            vec3 Lfill = normalize(vec3(0.5, 0.35, 0.45));
            float ndk = max(dot(N, Lk), 0.0);
            float ndf = max(dot(N, Lfill), 0.0) * 0.48;
            float shade = pow(ndk * 0.5 + 0.5, 1.08);
            float amb = 0.24 + 0.07 * max(N.y, 0.0);
            vec3 lit = base * amb + base * (0.48 * shade + ndf);
            lit = applyPlayerShadow(lit, N, fragWorldPos);
            vec3 rgb = fogMixLit(lit, distCam);
            vec3 fogCol = atmosphereFogColor();
            rgb = mix(fogCol, rgb, employeePopA);
            rgb = ditherQuantize(rgb, mix(8.0, 5.2, parkourPs1Pk), mix(9.0, 6.2, parkourPs1Pk),
                                 mix(0.78, 1.08, parkourPs1Pk));
            outColor = vec4(applyStoreLightMul(rgb), 1.0);
            return;
        }
        // Baked part id in fragColor.a: 0.02 jeans, 0.04 shirt, 0.06 skin, 0.08 unknown (matches C++ loader).
        vec3 LN = normalize(fragLocalNormal);
        vec3 lp = fragLocalPos;
        float y = lp.y;
        float mapLo = ubo.employeeBounds.x;
        float mapHi = ubo.employeeBounds.y;
        float pantsTop = ubo.employeeBounds.z;
        float torsoR = ubo.employeeBounds.w;

        float pa = fragColor.a;
        bool bakedMode = pa >= 0.015 && pa <= 0.085;
        int bakedPart = -1;
        if (bakedMode)
            bakedPart = int(round(pa / 0.02)) - 1;

        float rxz = length(lp.xz);
        float azimuth = atan(lp.x, lp.z);
        bool neckLike = (rxz < 0.15) && (y > 0.93) && (y < 1.10);
        // Central belly only — do not treat as hand (keeps thumbs near torso on skin, not jeans).
        bool nearNavelOnly = (rxz < 0.095) && (y > 0.87) && (y < 1.00);
        // Limbs out to the sides (T-pose / arms-down): |sin(az)| high — not torso / legs column.
        bool sideArmLike = (abs(sin(azimuth)) > 0.56) && (y > 0.38) && (y < 1.08) && (rxz > 0.052) &&
                           (rxz < 0.44) && (abs(LN.y) < 0.63);
        // Arms forward / A-pose (low |sin(az)|): not belly — require rxz past narrow torso column.
        bool forwardArmLike = (abs(sin(azimuth)) < 0.42) && (y > 0.52) && (y < 1.18) && (rxz > 0.128) &&
                              (rxz < 0.40) && (abs(LN.y) < 0.60);
        // Wrists / lower hands (often miss sideArm when facing camera).
        bool lowHandLike = (y > 0.52) && (y < 0.78) && (rxz > 0.055) && (rxz < 0.34) && (abs(LN.y) < 0.54);
        // Hands / forearms / digits: must win over y<pantsTop jeans (do NOT use a broad y+rxz "slim" rule — it matched calves).
        bool empHandSkin =
            lowHandLike ||
            forwardArmLike ||
            sideArmLike ||
            neckLike ||
            (!nearNavelOnly && (y > 0.40) && (y < 1.30) && (abs(LN.y) < 0.62) &&
             ((rxz > 0.31) || (rxz > 0.072 && rxz < 0.42 && y > 0.74) ||
              (rxz > 0.24 && y < 0.94 && y > 0.46)));
        // Narrow XZ at belt → was classified as "torso" → yellow shirt on groin / inseam slivers.
        bool groinPants =
            (rxz < 0.108) && (y > pantsTop - 0.11) && (y < pantsTop + 0.12) && (abs(LN.y) < 0.72);
        bool inseamPants = (rxz < 0.112) && (y > 0.70) && (y < pantsTop + 0.16) && (abs(sin(azimuth)) > 0.46) &&
                           (abs(LN.y) < 0.70);
        float shirtR = min(torsoR + 0.10, 0.39);
        vec3 base;
        if (bakedPart == 0) {
            if (empHandSkin)
                base = empEmployeeSkin(lp);
            else
                base = empEmployeePants(lp);
        } else if (bakedPart == 1) {
            if (empHandSkin)
                base = empEmployeeSkin(lp);
            else if (groinPants || inseamPants)
                base = empEmployeePants(lp);
            else
                base = empEmployeeShirt(lp, LN);
        } else if (bakedPart == 2) {
            base = empEmployeeSkin(lp);
        } else {
            if (empHandSkin)
                base = empEmployeeSkin(lp);
            else if (y < pantsTop)
                base = empEmployeePants(lp);
            else if (groinPants || inseamPants)
                base = empEmployeePants(lp);
            else if (y > mapHi)
                base = empEmployeeSkin(lp);
            else if (rxz < shirtR)
                base = empEmployeeShirt(lp, LN);
            else
                // Outside torso column in shirt band: upper arms / sides — not jeans (was painting arms blue).
                base = empEmployeeSkin(lp);
        }
        // Multiple disk textures on staff: map part → extraTex slot when that slot loaded (.x > slot).
        int staffSlot = -1;
        if (bakedMode) {
            if (empHandSkin || bakedPart == 2) {
                if (ubo.extraTexInfo.x > 2)
                    staffSlot = 2;
            } else if (bakedPart == 0) {
                if (ubo.extraTexInfo.x > 0)
                    staffSlot = 0;
            } else if (bakedPart == 1) {
                if (groinPants || inseamPants) {
                    if (ubo.extraTexInfo.x > 0)
                        staffSlot = 0;
                } else if (ubo.extraTexInfo.x > 1)
                    staffSlot = 1;
            }
        }
        float stBlend = clamp(float(ubo.extraTexInfo.z) * (1.0 / 255.0), 0.0, 1.0);
        if (staffSlot >= 0 && stBlend > 0.0) {
            vec3 tRgb = sampleStaffExtraTex(staffSlot, fragTexCoord);
            base = mix(base, tRgb, stBlend);
        }
        vec3 Lk = normalize(vec3(0.35, 0.88, 0.32));
        vec3 Lfill = normalize(vec3(0.5, 0.35, 0.45));
        float ndk = max(dot(N, Lk), 0.0);
        float ndf = max(dot(N, Lfill), 0.0) * 0.48;
        float shade = pow(ndk * 0.5 + 0.5, 1.08);
        float amb = 0.24 + 0.07 * max(N.y, 0.0);
        vec3 lit = base * amb + base * (0.48 * shade + ndf);
        lit = applyPlayerShadow(lit, N, fragWorldPos);
        vec3 rgb = fogMixLit(lit, distCam);
        vec3 fogCol = atmosphereFogColor();
        rgb = mix(fogCol, rgb, employeePopA);
        rgb = ditherQuantize(rgb, mix(8.0, 5.2, parkourPs1Pk), mix(9.0, 6.2, parkourPs1Pk),
                             mix(0.78, 1.08, parkourPs1Pk));
        outColor = vec4(applyStoreLightMul(rgb), 1.0);
        return;
    }

    if (isShelfMetal) {
        vec3 wp = fragWorldPos;
        vec3 an = abs(N);
        vec3 blend = pow(max(an, vec3(0.001)), vec3(3.2));
        blend /= (blend.x + blend.y + blend.z);
        vec2 dimS = vec2(textureSize(shelfTex, 0));
        const float kTex = 0.38;
        vec3 tX = texture(shelfTex, crunchyUvDims(wp.yz * kTex, dimS)).rgb;
        vec3 tY = texture(shelfTex, crunchyUvDims(wp.xz * kTex, dimS)).rgb;
        vec3 tZ = texture(shelfTex, crunchyUvDims(wp.xy * kTex, dimS)).rgb;
        vec3 texD = tX * blend.x + tY * blend.y + tZ * blend.z;
        float texM = dot(texD, vec3(0.299, 0.587, 0.114));
        vec3 base = vec3(0.145, 0.15, 0.158);
        base *= 0.92 + 0.24 * texM;
        vec3 Lk = normalize(vec3(0.12, 0.97, 0.22));
        vec3 Lfill = normalize(vec3(0.55, 0.35, 0.45));
        vec3 Lceil = normalize(vec3(0.08, 0.92, 0.18));
        float ndk = max(dot(N, Lk), 0.0);
        float ndf = max(dot(N, Lfill), 0.0) * 0.52;
        float ndc = max(dot(N, Lceil), 0.0) * 0.44;
        float shade = pow(ndk * 0.5 + 0.5, 1.05);
        float amb = 0.175 + 0.065 * max(N.y, 0.0);
        float spec = pow(max(dot(reflect(-Lk, N), V), 0.0), 40.0) * (0.22 + 0.16 * texM);
        float fres = pow(1.0 - max(dot(N, V), 0.0), 3.2) * 0.035;
        vec3 lit = base * amb + base * (0.40 * shade + ndf + ndc) + vec3(spec) + base * fres;
        lit = applyPlayerShadow(lit, N, fragWorldPos);
        vec3 rgb = fogMixLit(lit, distCam);
        rgb = ditherQuantize(rgb, 5.0, 20.0, 1.0);
        outColor = vec4(applyStoreLightMul(rgb), 1.0);
        return;
    }

    if (isShelfLadder) {
        vec3 wp = fragWorldPos;
        vec3 an = abs(N);
        vec3 blend = pow(max(an, vec3(0.001)), vec3(2.5));
        blend /= (blend.x + blend.y + blend.z);
        vec2 dimS = vec2(textureSize(shelfTex, 0));
        const float kT = 0.55;
        vec3 tX = texture(shelfTex, crunchyUvDims(wp.yz * kT, dimS)).rgb;
        vec3 tY = texture(shelfTex, crunchyUvDims(wp.xz * kT, dimS)).rgb;
        vec3 tZ = texture(shelfTex, crunchyUvDims(wp.xy * kT, dimS)).rgb;
        vec3 grain = tX * blend.x + tY * blend.y + tZ * blend.z;
        float g = dot(grain, vec3(0.299, 0.587, 0.114));
        vec3 base = vec3(0.22, 0.23, 0.25) * (0.82 + 0.28 * g);
        vec3 Lk = normalize(vec3(0.28, 0.88, 0.26));
        vec3 Lfill = normalize(vec3(0.5, 0.35, 0.45));
        float ndk = max(dot(N, Lk), 0.0);
        float ndf = max(dot(N, Lfill), 0.0) * 0.42;
        float shade = pow(ndk * 0.5 + 0.5, 1.06);
        float amb = 0.18 + 0.06 * max(N.y, 0.0);
        float spec = pow(max(dot(reflect(-Lk, N), V), 0.0), 28.0) * 0.12;
        vec3 lit = base * amb + base * (0.42 * shade + ndf) + vec3(spec);
        lit = applyPlayerShadow(lit, N, fragWorldPos);
        vec3 rgb = fogMixLit(lit, distCam);
        rgb = ditherQuantize(rgb, 5.0, 20.0, 1.0);
        outColor = vec4(applyStoreLightMul(rgb), 1.0);
        return;
    }

    if (isShelfBoxCutter) {
        vec2 uvg = vec2(fragTexCoord.x, 1.0 - fragTexCoord.y);
        vec3 base = texture(boxCutterTex, uvg).rgb;
        vec3 Lk = normalize(vec3(0.32, 0.88, 0.28));
        vec3 Lfill = normalize(vec3(0.5, 0.35, 0.45));
        float ndk = max(dot(N, Lk), 0.0);
        float ndf = max(dot(N, Lfill), 0.0) * 0.48;
        float shade = pow(ndk * 0.5 + 0.5, 1.06);
        float amb = 0.20 + 0.075 * max(N.y, 0.0);
        float spec = pow(max(dot(reflect(-Lk, N), V), 0.0), 22.0) * 0.10;
        vec3 lit = base * amb + base * (0.44 * shade + ndf) + vec3(spec);
        vec3 rgb = fogMixLit(lit * 1.20 + vec3(0.02), distCam);
        rgb = ditherQuantize(rgb, 5.0, 20.0, 1.0);
        outColor = vec4(applyStoreLightMul(rgb), 1.0);
        return;
    }

    if (isShelfRustyPipe) {
        vec2 uvg = vec2(fragTexCoord.x, 1.0 - fragTexCoord.y);
        vec3 base = texture(rustyPipeTex, uvg).rgb;
        vec3 Lk = normalize(vec3(0.30, 0.86, 0.26));
        vec3 Lfill = normalize(vec3(0.5, 0.35, 0.45));
        float ndk = max(dot(N, Lk), 0.0);
        float ndf = max(dot(N, Lfill), 0.0) * 0.48;
        float shade = pow(ndk * 0.5 + 0.5, 1.05);
        float amb = 0.19 + 0.072 * max(N.y, 0.0);
        float spec = pow(max(dot(reflect(-Lk, N), V), 0.0), 18.0) * 0.085;
        vec3 lit = base * amb + base * (0.43 * shade + ndf) + vec3(spec);
        vec3 rgb = fogMixLit(lit * 1.20 + vec3(0.02), distCam);
        rgb = ditherQuantize(rgb, 5.0, 20.0, 1.0);
        outColor = vec4(applyStoreLightMul(rgb), 1.0);
        return;
    }

    if (isShelfPallet) {
        vec3 wp = fragWorldPos;
        vec3 an = abs(N);
        vec3 blend = pow(max(an, vec3(0.001)), vec3(2.65));
        blend /= (blend.x + blend.y + blend.z);
        vec2 dimP = vec2(textureSize(palletTex, 0));
        const float kP = 0.095;
        vec3 sx = texture(palletTex, crunchyUvDims(wp.yz * kP, dimP)).rgb;
        vec3 sy = texture(palletTex, crunchyUvDims(wp.xz * kP, dimP)).rgb;
        vec3 sz = texture(palletTex, crunchyUvDims(wp.xy * kP, dimP)).rgb;
        vec3 base = sx * blend.x + sy * blend.y + sz * blend.z;
        vec3 Lk = normalize(vec3(0.32, 0.86, 0.28));
        vec3 Lceil = normalize(vec3(0.1, 0.92, 0.12));
        float ndk = max(dot(N, Lk), 0.0);
        float ndc = max(dot(N, Lceil), 0.0) * 0.46;
        float shade = pow(ndk * 0.5 + 0.5, 1.05);
        float amb = 0.198 + 0.085 * max(N.y, 0.0);
        float spec = pow(max(dot(reflect(-Lk, N), V), 0.0), 22.0) * 0.065;
        vec3 lit = base * amb + base * (0.46 * shade + ndc) + vec3(spec);
        lit = applyPlayerShadow(lit, N, fragWorldPos);
        vec3 rgb = fogMixLit(lit, distCam);
        rgb = ditherQuantize(rgb, 5.0, 20.0, 1.0);
        outColor = vec4(applyStoreLightMul(rgb), 1.0);
        return;
    }

    if (isShelfCrate) {
        vec3 wp = fragWorldPos;
        vec3 an = abs(N);
        vec3 blend = pow(max(an, vec3(0.001)), vec3(2.7));
        blend /= (blend.x + blend.y + blend.z);
        vec2 dimC = vec2(textureSize(crateTex, 0));
        const float kC = 0.11;
        vec3 sx = texture(crateTex, crunchyUvDims(wp.yz * kC, dimC)).rgb;
        vec3 sy = texture(crateTex, crunchyUvDims(wp.xz * kC, dimC)).rgb;
        vec3 sz = texture(crateTex, crunchyUvDims(wp.xy * kC, dimC)).rgb;
        vec3 base = sx * blend.x + sy * blend.y + sz * blend.z;
        vec3 Lk = normalize(vec3(0.32, 0.86, 0.28));
        vec3 Lceil = normalize(vec3(0.1, 0.92, 0.12));
        float ndk = max(dot(N, Lk), 0.0);
        float ndc = max(dot(N, Lceil), 0.0) * 0.46;
        float shade = pow(ndk * 0.5 + 0.5, 1.05);
        float amb = 0.205 + 0.08 * max(N.y, 0.0);
        float spec = pow(max(dot(reflect(-Lk, N), V), 0.0), 24.0) * 0.08;
        vec3 lit = base * amb + base * (0.44 * shade + ndc) + vec3(spec);
        lit = applyPlayerShadow(lit, N, fragWorldPos);
        vec3 rgb = fogMixLit(lit, distCam);
        rgb = ditherQuantize(rgb, 5.0, 20.0, 1.0);
        outColor = vec4(applyStoreLightMul(rgb), 1.0);
        return;
    }

    if (isShelfWood) {
        vec3 wp = fragWorldPos;
        vec3 an = abs(N);
        vec3 blend = pow(max(an, vec3(0.001)), vec3(2.8));
        blend /= (blend.x + blend.y + blend.z);
        vec2 dimS = vec2(textureSize(shelfTex, 0));
        const float kDeckTex = 0.55;
        vec3 gX = texture(shelfTex, crunchyUvDims(wp.yz * kDeckTex, dimS)).rgb;
        vec3 gY = texture(shelfTex, crunchyUvDims(wp.xz * kDeckTex, dimS)).rgb;
        vec3 gZ = texture(shelfTex, crunchyUvDims(wp.xy * kDeckTex, dimS)).rgb;
        vec3 grain = gX * blend.x + gY * blend.y + gZ * blend.z;
        vec3 base = grain * vec3(0.92, 0.86, 0.78);
        float chip = fract(dot(wp.xz, vec2(5.1, 3.7)) + wp.y * 0.08);
        base *= 0.96 + 0.04 * chip;
        vec3 Lk = normalize(vec3(0.35, 0.88, 0.32));
        vec3 Lceil = normalize(vec3(0.1, 0.92, 0.12));
        float ndk = max(dot(N, Lk), 0.0);
        float ndc = max(dot(N, Lceil), 0.0) * 0.48;
        float shade = pow(ndk * 0.5 + 0.5, 1.06);
        float amb = 0.215 + 0.085 * max(N.y, 0.0);
        vec3 lit = base * amb + base * (0.44 * shade + ndc);
        lit = applyPlayerShadow(lit, N, fragWorldPos);
        vec3 rgb = fogMixLit(lit, distCam);
        rgb = ditherQuantize(rgb, 5.0, 20.0, 1.0);
        outColor = vec4(applyStoreLightMul(rgb), 1.0);
        return;
    }

    if (tag.r > 0.95 && tag.g > 0.95 && tag.b < 0.1) {
        vec3 base = vec3(0.03, 0.03, 0.03);
        vec3 L = normalize(vec3(0.35, 0.88, 0.32));
        float ndl = max(dot(N, L), 0.0);
        float shade = pow(ndl * 0.5 + 0.5, 1.2);
        vec3 lit = base * (0.50 + 0.50 * shade);
        vec3 rgb = fogMixLit(lit, distCam);
        rgb = ditherQuantize(rgb, 5.0, 20.0, 1.0);
        outColor = vec4(applyStoreLightMul(rgb), 1.0);
        return;
    }

    if (tag.b > 0.95) {
        vec2 uv = vec2(fragLocalPos.x * 0.5 + 0.5, 1.0 - (fragLocalPos.y * 0.5 + 0.5));
        if (dot(N, V) < 0.0)
            uv.x = 1.0 - uv.x;
        const float kSignUvCells = 56.0;
        vec2 uvP = floor(uv * kSignUvCells) / kSignUvCells;
        vec2 dimN = vec2(textureSize(signTex, 0));
        vec4 sc = texture(signTex, crunchyUvDims(uvP, dimN));
        if (sc.a < 0.1)
            discard;
        vec3 L = normalize(vec3(0.35, 0.88, 0.32));
        float ndl = max(dot(N, L), 0.0);
        float shade = pow(ndl * 0.5 + 0.5, 1.2);
        float rim = pow(1.0 - max(dot(N, V), 0.0), 2.0) * 0.06;
        vec3 lit = sc.rgb * (0.50 + 0.50 * shade);
        lit += sc.rgb * rim;
        vec3 rgb = fogMixLit(lit, distCam);
        rgb = ditherQuantize(rgb, 5.0, 20.0, 1.0);
        outColor = vec4(applyStoreLightMul(rgb), 1.0);
        return;
    }

    if (tag.r > 0.92 && tag.g > 0.88 && tag.b > 0.40 && tag.b < 0.52) {
        vec3 em = vec3(0.94, 0.95, 0.93);
        float fogAmt = atmosphereFogAmt(distCam);
        vec3 fogCol = atmosphereFogColor();
        float fogMix = ubo.fogParams.w > 0.5 ? 0.30 : 0.34;
        vec3 rgb = mix(em, fogCol, fogAmt * fogMix);
        rgb = ditherQuantize(rgb, 5.0, 20.0, 1.0);
        outColor = vec4(applyStoreLightMul(rgb), 1.0);
        return;
    }

    vec3 samplePos = retroWorldSamplePos(fragWorldPos);
    vec3 base = triplanarTexture(samplePos, N);
    base = mix(base, vec3(0.62, 0.63, 0.66), 0.30);
    base = applyOptionalExtraLayer0(base);
    vec3 L = normalize(vec3(0.35, 0.88, 0.32));
    float ndl = max(dot(N, L), 0.0);
    float shade = pow(ndl * 0.5 + 0.5, 1.2);
    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.0) * 0.06;
    vec3 lit = base * (0.50 + 0.50 * shade);
    lit += base * rim;
    lit = applyPlayerShadow(lit, N, fragWorldPos);
    vec3 rgb = fogMixLit(lit, distCam);
    rgb = ditherQuantize(rgb, 5.0, 20.0, 1.0);
    outColor = vec4(applyStoreLightMul(rgb), 1.0);
}
