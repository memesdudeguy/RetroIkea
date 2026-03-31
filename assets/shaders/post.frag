#version 450

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D sceneTex;

layout(push_constant) uniform PushPost {
    vec4 g; // x = time (s), y = night horror weight 0..1, z/w = viewport size (px) for UV from gl_FragCoord
    vec4 v; // x = damage hit pulse 0..1, y = low-HP rim (hp<=35) 0..1, z = parkour PS1 mix, w = blackout pursuit 0..1
} pc;

float hash12(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    // Vulkan NDC Y vs OpenGL breaks oversized-triangle UV interpolation; derive coverage from frag coord.
    vec2 vp = max(vec2(pc.g.z, pc.g.w), vec2(1.0));
    vec2 st = (gl_FragCoord.xy + vec2(0.5)) / vp;

    vec2 texSz = vec2(textureSize(sceneTex, 0));
    vec2 uv = (floor(st * texSz) + vec2(0.5)) / texSz;
    vec3 col = texture(sceneTex, uv).rgb;

    // No colour grade / grey film — pc.g.y only ramps scan, grain, crush when lights are out.
    float filmW = clamp(pc.g.y, 0.0, 1.0);
    float ps1Pk = clamp(pc.v.z, 0.0, 1.0);
    float pursuit = clamp(pc.v.w, 0.0, 1.0);

    const float scanPeriodPx = 10.0;
    float scan = sin(gl_FragCoord.y * 6.283185307 / scanPeriodPx);
    float scanAmt = 0.012 + filmW * 0.085 + pursuit * filmW * 0.055;
    col *= 1.0 - scanAmt * (0.5 + 0.5 * scan);
    float scan2 = sin(gl_FragCoord.y * 6.283185307 / (scanPeriodPx * 3.17) + 1.1);
    col *= 1.0 - (0.005 + filmW * 0.032) * (0.5 + 0.5 * scan2);

    float gn = hash12(gl_FragCoord.xy + pc.g.x * vec2(113.0, 197.0)) - 0.5;
    col += gn * (0.009 + filmW * 0.048 + ps1Pk * 0.034 + pursuit * filmW * 0.028);

    float levels = mix(56.0, 18.0 + filmW * 4.0, filmW);
    levels = mix(levels, levels * (0.72 + filmW * 0.18), ps1Pk * 0.62);
    col = floor(col * levels + vec3(0.5)) / levels;

    col = clamp(col * (1.0 + 0.04 * (1.0 - filmW)) + vec3(0.006) * (1.0 - filmW * 0.65), 0.0, 1.0);
    // Night (high filmW): small lift so blackout stays readable without washing out day.
    col = clamp(col + vec3(0.014) * filmW, 0.0, 1.0);

    // Red screen edges: recent damage (pulse) + persistent rim while health is critically low.
    float dx = min(st.x, 1.0 - st.x);
    float dy = min(st.y, 1.0 - st.y);
    float dNear = min(dx, dy);
    float edgeT = pow(clamp(1.0 - dNear / 0.26, 0.0, 1.0), 1.2);
    float pulse = clamp(pc.v.x, 0.0, 1.0);
    float crit = clamp(pc.v.y, 0.0, 1.0);
    float rim = edgeT * clamp(pulse + crit, 0.0, 1.0);
    vec3 rimRgb = vec3(0.96, 0.04, 0.09);
    col = mix(col, rimRgb, rim * 0.74);
    // Blackout pursuit: slight edge crush + warm pressure (not damage red — reads as tunnel / adrenaline).
    float pursEdge = edgeT * pursuit * filmW;
    col *= 1.0 - pursEdge * 0.11;
    vec3 pursTint = vec3(1.04, 0.94, 0.88);
    col = mix(col, col * pursTint, pursEdge * 0.22);

    outColor = vec4(col, 1.0);
}
