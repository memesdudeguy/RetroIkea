// Vulkan 3D mini-game: flat infinite ground (SDL2 + GLM)
#ifndef VULKAN_GAME_VERSION_STRING
#define VULKAN_GAME_VERSION_STRING "1.1.0"
#endif
#include <SDL2/SDL.h>
#include <SDL2/SDL_filesystem.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "audio.hpp"
#include "employee_mesh.hpp"
#include "staff_skin.hpp"

// MinGW may define isfinite as a macro, which breaks qualified std::isfinite calls.
#ifdef isfinite
#undef isfinite
#endif
#if defined(_WIN32) && defined(__MINGW32__)
// MinGW can rewrite std::isfinite(...) to std::__builtin_isfinite(...).
// Provide shim overloads so existing std::isfinite call sites compile unchanged.
namespace std {
inline bool __builtin_isfinite(float x) { return ::isfinite(x); }
inline bool __builtin_isfinite(double x) { return ::isfinite(x); }
inline bool __builtin_isfinite(long double x) { return ::isfinite(x); }
} // namespace std
#endif

// Runtime-loaded textures (VULKAN_GAME_EXTRA_TEXTURES); must match shader.frag extraTex[] length.
constexpr uint32_t kMaxExtraTextures = 16;

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4505)
#endif
#include "stb_easy_font.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

// Used when wall-clock delta is missing/invalid (same order of magnitude as typical refresh).
constexpr int kTargetFps = 60;

namespace {

template <typename DownFn>
static bool inputSprintHeld(DownFn down) {
  return down(SDL_SCANCODE_LSHIFT) || down(SDL_SCANCODE_RSHIFT);
}
template <typename DownFn>
static bool inputForward(DownFn down) {
  return down(SDL_SCANCODE_W) || down(SDL_SCANCODE_UP) || down(SDL_SCANCODE_Z);
}
template <typename DownFn>
static bool inputBack(DownFn down) {
  return down(SDL_SCANCODE_S) || down(SDL_SCANCODE_DOWN);
}
template <typename DownFn>
static bool inputStrafeLeft(DownFn down) {
  return down(SDL_SCANCODE_A) || down(SDL_SCANCODE_LEFT);
}
template <typename DownFn>
static bool inputStrafeRight(DownFn down) {
  return down(SDL_SCANCODE_D) || down(SDL_SCANCODE_RIGHT);
}

// SDL surfaces may pad rows (pitch > width*4); a single memcpy corrupts GPU uploads.
static std::vector<std::string> parseExtraTexturePathsFromEnv() {
  std::vector<std::string> out;
  const char* e = std::getenv("VULKAN_GAME_EXTRA_TEXTURES");
#if defined(VULKAN_GAME_DEFAULT_EXTRA_TEXTURES)
  if (!e || !*e)
    e = VULKAN_GAME_DEFAULT_EXTRA_TEXTURES;
#endif
  if (!e || !*e)
    return out;
  std::string s(e);
  size_t start = 0;
  while (start < s.size()) {
    const size_t sep = s.find_first_of(":;", start);
    std::string token =
        sep == std::string::npos ? s.substr(start) : s.substr(start, sep - start);
    while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front())))
      token.erase(0, 1);
    while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back())))
      token.pop_back();
    if (!token.empty())
      out.push_back(std::move(token));
    if (sep == std::string::npos)
      break;
    start = sep + 1;
  }
  if (out.size() > kMaxExtraTextures) {
    std::cerr << "[textures] VULKAN_GAME_EXTRA_TEXTURES: keeping first " << kMaxExtraTextures
              << " paths\n";
    out.resize(kMaxExtraTextures);
  }
  return out;
}

static void copySdlRgbaSurfaceRowsToBuffer(const SDL_Surface* rgba, void* dst, uint32_t texW,
                                          uint32_t texH) {
  const size_t rowBytes = static_cast<size_t>(texW) * 4u;
  auto* d = static_cast<uint8_t*>(dst);
  const auto* src = static_cast<const uint8_t*>(rgba->pixels);
  for (uint32_t y = 0; y < texH; ++y)
    std::memcpy(d + y * rowBytes, src + static_cast<size_t>(y) * static_cast<size_t>(rgba->pitch),
                rowBytes);
}

constexpr int kWidth = 1280;
constexpr int kHeight = 720;
constexpr int kMaxFramesInFlight = 2;

constexpr float kGroundY = 0.0f;
constexpr float kCeilingY = 38.0f;
constexpr float kEyeHeight = 1.62f;
// First-person: don’t allow looking through own chest; max downward angle (~-28°).
constexpr float kPitchMaxLookDown = -0.49f;
constexpr float kPitchMaxLookUp = glm::half_pi<float>() - 0.06f;
// First-person climbing look clamp: keep view aligned with climb direction but still allow free look.
constexpr float kClimbLookYawClamp = glm::radians(85.0f);
constexpr float kClimbLookPitchDown = -1.05f;
constexpr float kClimbLookPitchUp = 0.95f;
constexpr float kCrouchEyeHeight = 1.0f;
constexpr float kCrouchTransitionRate = 7.2f;
// Grounding vs vertical snap must agree: isGrounded allows feet this far above support; snap must cover at least that
// or the player hovers slightly while still "grounded" (noticeable when crouched / sliding).
constexpr float kGroundedFeetAboveSupport = 0.11f;
// Slightly above grounded tolerance so snap always pulls small gaps closed; velY noise can otherwise block snap.
constexpr float kFeetSnapDownSlop = 0.22f;
// Skinned crouch / slide clips only (standing walk/idle: no sink — feetY = camPos - eyeHeight).
constexpr float kAvatarCrouchWalkFeetVisualDown = 0.24f;
constexpr float kAvatarCrouchIdleBowFeetVisualDown = 0.40f;
constexpr float kAvatarSlideFeetVisualDown = 0.45f;
// Jump / ledge: Meshy clips leave soles above the nominal foot pivot while falling or in land pose (like crouch).
constexpr float kAvatarJumpFallFeetVisualDown = 0.38f;
// Compensates Meshy jump land pose (soles above pivot); scaled down + faded over post-land so touchdown doesn’t read as sinking.
constexpr float kAvatarJumpLandFeetVisualDown = 0.05f;
constexpr float kAvatarLedgeClimbFeetVisualDown = 0.25f;
constexpr float kAvatarLedgeHangFeetVisualDown = 0.12f;
// After grab: vertical delta on feet pivot so hands sit on the deck lip (negative = lower whole mesh).
// Meshy biped hang/shimmy GLBs place wrists well above the shelf at the default camera/foot pivot.
// Shimmy clips also use kAvatarLedgeClimbFeetVisualDown via clipFeetSink (same as ledge-climb hang loop).
constexpr float kAvatarLedgeHangHandsDeckAdjustY = -0.24f;
// Shimmy clip root sits lower than idle ledge-hang; lift while shimmy pose is active.
constexpr float kAvatarLedgeHangShimmyStrafeDeckAdjustExtraY = 0.20f;
constexpr float kCrouchSpeedMult = 0.52f;
// Horizontal locomotion only — bob / sway follow these caps. Tuned for agile runner / parkour flow (Dying Light–like
// motion: strong air steer, kept momentum, fast sprint ramp — not animation-driven).
constexpr float kMovementSpeedScale = 0.8f;
constexpr float kSlideDuration = 4.65f;
constexpr float kSlideCooldown = 0.26f;
constexpr float kSlideStartSpeed = 26.0f * kMovementSpeedScale;
constexpr float kSlideStopSpeed = 0.22f;
constexpr float kSlideDecel = 4.35f;
constexpr float kSlideBobFreq = 0.36f;
// Slide bob phase only — do not use kViewBobPhaseBoost here so walk/run bob tweaks leave slide alone.
constexpr float kSlideViewBobPhaseBoost = 0.14f;
constexpr float kSlideBobYScale = 0.12f;
constexpr float kSlideBobSideScale = 1.35f;
constexpr float kSlidePitchScale = 0.22f;
// First-person slide camera damping (3P keeps stronger authored motion).
constexpr float kFpSlideBobYMul = 0.55f;
constexpr float kFpSlideBobSideMul = 0.35f;
constexpr float kFpSlidePitchMul = 0.45f;
constexpr float kGravity = 25.0f;
// Tuned down vs old parkour defaults — jumps are for clearance, not long horizontal travel.
constexpr float kJumpVel = 9.65f;
// <1 bleeds some run speed on takeoff so sprint-jump chaining covers less ground.
constexpr float kJumpHorizCarry = 0.92f;
// Default air-time estimate for jump clip sync (overridden per jump from actual takeoff speed).
constexpr float kJumpPredictedAirTimeSec = 2.f * kJumpVel / kGravity;
constexpr float kJumpAirTimeTargetMinSec = 0.4f;
constexpr float kJumpAirTimeTargetMaxSec = 2.15f;
// Parkour / free-runner: one jump asset is split (lip / mid-air / pre-land / landing tail). Slightly faster
// scrub than before so air phases feel athletic (Dying Light–style composition from the same clips).
constexpr float kJumpAnimPlaybackScale = 0.64f;
// Hold last jump frame briefly so contact lines up with landing footstep (clip may end slightly off).
constexpr float kJumpLandPoseHoldSec = 0.14f;
// Longer hold after sprint jump clip — its end reads rushed at the standing jump hold length.
constexpr float kJumpLandPoseHoldSecRunJump = 0.24f;
// Landing: max wall seconds when scrubbing jump second half (snappy recovery, still readable).
constexpr float kPlayerLandClipMaxWallSec = 0.34f;
// Skip the landing clip on very soft impacts (tiny hop / almost flat landing).
constexpr float kPlayerLandClipMinDownVel = 0.80f;
constexpr float kJumpRunTailRateScale = 0.82f;
constexpr float kWalkAccel = 248.0f;
constexpr float kAirAccel = 54.0f;
constexpr float kFriction = 54.0f;
// Tuned with kWalkAccel, accel headroom curve, and ~60 Hz dt to sit near kMaxSpeed / kSprintSpeed.
constexpr float kGroundFrictionScaleWalk = 0.23f;
constexpr float kGroundFrictionScaleSprint = 0.245f;
// Low drag keeps run/jump momentum in the air (strong steer via kAirAccel above).
constexpr float kAirDrag = 0.31f;
// Ground walk/sprint (scaled by kMovementSpeedScale; animation coeffs use same kSprintSpeed/kMaxSpeed).
constexpr float kMaxSpeed = 16.0f * kMovementSpeedScale;
constexpr float kSprintSpeed = 39.0f * kMovementSpeedScale;
constexpr float kSprintAccelMult = 2.08f;
constexpr float kAirSpeedCap = 46.0f * kMovementSpeedScale;
constexpr float kAccelHeadroomFracGround = 0.27f;
constexpr float kAccelHeadroomFracAir = 0.38f;
constexpr float kAccelMinScaleGround = 0.14f;
constexpr float kAccelMinScaleAir = 0.22f;
constexpr float kCoyoteTime = 0.135f;
// Slightly longer coyote on shelf/crate tops so space-jump can chain ledge-to-ledge more reliably.
constexpr float kCoyoteTimeLedge = 0.19f;
// Airborne shelf pull-up (Dying Light–style parkour). Jump/sprint toward a deck + Space.
constexpr bool kPlayerLedgeMantleEnabled = true;
// Walk/step off a ledge (no space jump): air = normal jump clip; landing tail still uses modest-drop gate below.
constexpr float kPlayerWalkOffLandMinDropM = 0.05f;
// Minimum horizontal speed or WASD intent to treat as “walking off” (not standing drop / physics glitch).
constexpr float kPlayerWalkOffLedgeAnimMinHorizSp = 0.022f;
constexpr float kPlayerWalkOffLedgeMaxVelYForAnim = 0.78f;
// Drop to next support at this or below: walk/run in air over the ledge; above this use fall / pre-fall jump clips.
constexpr float kPlayerWalkOffSmallGapMaxDropM = 0.78f;
// Ignore samples closer than this when probing “ahead” so feet / current deck don’t zero the drop.
constexpr float kWalkOffGapProbeForwardMinM = 0.42f;
// Same-tier “next deck” probe: ignore hits this close (~past deck depth + gap) so current plank isn’t “ahead”.
// (Was 2.85m — too strict for short generic ledges; next plank never counted as “far” same-tier.)
constexpr float kWalkOffGapSameTierMinForwardM = 1.22f;
// Non–same-surface samples ignore the gap lip (void→floor) until this far ahead — otherwise floor wins
// and every narrow hop reads as an infinite drop (no air-walk on most ledges).
constexpr float kWalkOffGapNonSameMinForwardM = 0.88f;
// When classifying big fall vs small step, ignore support within this height of lastSupportY (current surface).
constexpr float kWalkOffGapExcludeSameSurfaceEpsM = 0.055f;
// Big-gap walk-off: wall-time on the lip while we scrub the jump clip from t=0 through the first-half boundary
// (see kJumpClipLedgeFirstHalfFrac); then mid-air holds that boundary pose; pre-land + touchdown scrub the second half.
constexpr float kPlayerPreFallBeforeFallSec = 0.3f;
constexpr float kJumpClipLedgeFirstHalfFrac = 0.5f;
constexpr float kJumpLedgePreLandLeadSec = 0.32f;
// First-person view: dip nose slightly mid-charge (squat) then level for push-off.
constexpr float kPreFallChargeViewPitchAmp = 0.028f;
constexpr float kJumpBufferTime = 0.12f;
constexpr float kJumpCutMult = 0.64f;
constexpr float kJumpMinIntervalSec = 0.16f;
constexpr float kJumpRepeatNerfWindowSec = 0.50f;
constexpr float kJumpRepeatCarryMul = 0.45f;
constexpr float kJumpRepeatForwardMul = 0.62f;
// Ground jump-squat when a shelf pull-up is out of reach (far target); tiny gaps stay tap-to-hop. Works with mantle on.
constexpr float kJumpSquatCloseMaxDeltaY = 0.48f;
constexpr float kJumpSquatCloseMaxHoriz = 0.62f;
constexpr float kJumpSquatFullChargeSec = 0.5f;
constexpr float kJumpSquatVelMinMul = 0.74f;
constexpr float kJumpSquatVelMaxMul = 1.12f;
constexpr float kJumpSquatViewPitchAmp = 0.021f;
// Depth jump: after dropping onto feet, brief window for a stronger reactive jump.
constexpr float kDepthJumpWindowSec = 0.28f;
constexpr float kDepthJumpMinDropM = 0.36f;
constexpr float kDepthJumpBonusVelY = 1.35f;
// Jump from shelf/crate top: multiply horizontal vel after kJumpHorizCarry (nearly vertical hop; no run launch).
constexpr float kJumpFromLedgeHorizMul = 0.05f;
// Walk/sprint off a shelf into a large drop: still bleed forward, but keep enough speed to feel like a runner leap.
constexpr float kWalkOffLedgeHorizMul = 0.125f;
// Ground vault over shelf crates / box-sized props: extra impulse + slightly faster jump clip (same jump arch / fall / land).
constexpr float kVaultMinCrateHalfY = 0.14f;
constexpr float kVaultMaxCrateHalfY = 0.92f;
constexpr float kVaultMinCrateHalfXZ = 0.17f;
constexpr float kVaultMaxCrateHalfXZ = 1.22f;
constexpr float kVaultMinClearanceAboveFeet = 0.28f;
constexpr float kVaultMaxClearanceAboveFeet = 1.32f;
constexpr float kVaultFeetToCrateBaseMaxD = 0.15f;
constexpr float kVaultMinForwardM = 0.035f;
constexpr float kVaultMaxForwardM = 0.86f;
constexpr float kVaultMinDepthAlongRayM = 0.2f;
constexpr float kVaultMoveAlignDotMin = 0.36f;
constexpr float kVaultForwardImpulseMin = 1.92f;
constexpr float kVaultForwardImpulseExtra = 2.35f;
constexpr float kVaultVertImpulseBonus = 0.45f;
constexpr float kVaultJumpAnimRateScale = 1.28f;
// First tick when large-drop pre-fall starts: bleed run-in so you don’t slide the whole ledge in one frame.
constexpr float kPlayerPreFallStartHorizMul = 0.06f;
// While pre-fall timer is active, strong horizontal decay (no sprinting along the lip).
constexpr float kPreFallHorizBrakePerSec = 38.f;
// When pre-fall ends: step forward so feet leave the deck footprint (support probe reads void / fall).
constexpr float kWalkOffLedgeCommitPushM = 0.26f;
// Narrow bay “air walk”: allow real walk/run steering across short hops (still weaker than grounded).
constexpr float kAirWalkSmallGapAccelMul = 0.5f;
constexpr float kAirWalkSmallGapMaxSpMul = 0.74f;
// Below this downward vel (m/s), airborne jump clips use the mid-clip pose (hang / fall) instead of the outro.
constexpr float kAvatarJumpFallVelYThr = -0.12f;
// Normalized phase in jump / run-jump clips for that mid-air pose (0 = start, 1 = end).
constexpr float kJumpClipMidPhaseFrac = 0.52f;
constexpr float kJumpClipRunMidPhaseFrac = 0.50f;
// Falling: scrub forward from mid-pose toward pre-landing at this rate (frac/sec).
constexpr float kFallAnimScrubRate = 1.1f;
// Falling: don't advance past this fraction (avoid reaching the landing squat).
constexpr float kFallAnimMaxFrac = 0.88f;
// Run-jump variant.
constexpr float kFallAnimMaxFracRun = 0.85f;
constexpr float kBobAmp = 0.058f;
constexpr float kBobSideAmp = 0.031f;
constexpr float kWalkPitchAmp = 0.052f;
constexpr float kSwayRollStr = 0.152f;
// Slide view-bob: meters per 2π phase while sliding (smaller = snappier animation).
constexpr float kBobSlideStrideM = 15.0f;
// Walk/run grounded bob phase vs stride (footsteps use strideM alone). Slide uses kSlideViewBobPhaseBoost.
constexpr float kViewBobPhaseBoost = 0.4f;
// Grounded bob only: extra phase speed while sprinting (footstep cadence unchanged).
constexpr float kRunBobPhaseMult = 1.18f;
constexpr float kRunRollScale = 0.48f;
constexpr float kRunBobAmpScale = 0.17f;
constexpr float kRunBobSideScale = 0.82f;
constexpr float kRunPitchOscScale = 0.55f;
constexpr float kRunSideSwayAmp = 0.027f;
// In-air: faint bob/sway tied to horizontal motion only (no speed cap change).
constexpr float kAirBobPhaseScale = 0.42f;
constexpr float kAirBobAmpMul = 0.19f;
constexpr float kAirBobSideMul = 0.18f;
constexpr float kAirBobPitchMul = 0.12f;
constexpr float kRunAnimBlendInRate = 4.95f;
constexpr float kRunAnimBlendOutRate = 3.25f;
constexpr float kIdleAnimBlendRate = 7.8f;
constexpr float kJumpTakeoffPitch = 0.041f;
constexpr float kLandingPitchMin = 0.010f;
constexpr float kLandingPitchMax = 0.038f;
constexpr float kLandingPitchImpactRef = 18.0f;
constexpr float kLandingPitchDecay = 8.2f;
constexpr float kLandingPitchDecayGroundBoost = 5.4f;
// Hard landings + fall-damage punch used to stack into a deep negative offset that decayed slowly — feels
// “camera locked” looking at the floor after big jumps / tall drops.
constexpr float kLandingPitchOfsClampRad = 0.095f;
constexpr float kFallDamageViewPitchPerSqrtDmg = 0.022f;
constexpr float kGroundEaseRate = 7.5f;
constexpr float kSwayPitchDamp = 4.2f;
constexpr float kSwayPitchDampAir = 2.4f;
constexpr float kIdleSpeed = 0.82f;
constexpr float kIdlePitchAmp = 0.026f;
constexpr float kIdleRollAmp = 0.034f;
constexpr float kIdleBobAmp = 0.016f;
constexpr float kIdleSideAmp = 0.019f;
constexpr float kRandomSwaySpeed = 0.78f;
constexpr float kRandomSwayPitchAmp = 0.008f;
constexpr float kRandomSwayRollAmp = 0.011f;
constexpr float kRandomSwayBobAmp = 0.0065f;
constexpr float kRandomSwaySideAmp = 0.008f;
// Floor/ceiling halo: mesh edge must stay past kViewFogEnd or you see new quads “spawn” in clear air.
// With chunkWorld = kChunkCellCount * kCellSize, min distance to mesh edge at stream boundary is
// kTerrainStreamMarginChunks * chunkWorld (see rebuildTerrainIfNeeded). Need that >= fog end + slack.
constexpr int kChunkRadius = 6;
constexpr int kTerrainStreamMarginChunks = 5; // R-1 → recenters early; edge ~5*chunkWorld ≈ 390m
constexpr int kChunkCellCount = 24;
constexpr float kCellSize = 3.25f;
static_assert(kTerrainStreamMarginChunks > 0 && kTerrainStreamMarginChunks < kChunkRadius,
              "terrain stream margin must leave a comfort band inside the halo");

// Auto step onto thin horizontal surfaces (shelf decks, etc.).
constexpr float kMaxStepHeight = 0.24f;
// Ledge grab: aim from view basis; tilt & cone scale with ledge height (low = easier look-down, high = look-up).
constexpr float kLedgeGrabMinRise = 0.65f;
constexpr float kLedgeGrabMaxRise = 2.48f;
constexpr float kLedgeGrabMaxReachXZ = 1.38f;
constexpr float kLedgeGrabMaxVelY = 13.4f;
constexpr float kLedgeGrabMaxFallVelY = -27.f;
constexpr float kLedgeGrabReachLeniency = 1.34f;
constexpr float kLedgeGrabFwdPull = 0.58f;
constexpr float kLedgeGrabDuration = 0.27f;
constexpr float kLedgeGrabMinEyeHeight = 0.76f;
// Ledge mantle uses only the first half of the slow ladder climb clip (pull-up phase).
constexpr float kLedgeClimbAnimClipFrac = 0.5f;
// Ground ladder: interact radius / facing; climb duration follows full ladder clip (see advanceLadderClimb).
constexpr float kLadderClimbInteractRadiusXZ = 0.62f;
constexpr float kLadderClimbMinLookDot = 0.20f;
// Feet end a bit above the shelf deck so the pull-up clears the ladder top / rungs (was 0.04f inline).
constexpr float kLadderClimbFeetAboveDeck = 0.04f;
constexpr float kLedgeGrabEaseOutPow = 2.45f;
constexpr float kLedgeGrabAnimLiftPhaseEnd = 0.66f;
constexpr float kLedgeGrabAnimLiftHeightFrac = 0.89f;
constexpr float kLedgeGrabForwardReachPerSpeed = 0.095f;
constexpr float kLedgeGrabForwardReachBonusMax = 0.52f;
constexpr float kLedgeGrabExitSpeed = 1.82f * kMovementSpeedScale;
// After mantle: keep run/sprint momentum along the ledge (DL-style flow).
constexpr float kLedgeMantleExitFwdCarry = 0.58f;
constexpr float kLedgeMantleExitSideCarry = 0.76f;
constexpr float kLedgeMantleExitMinForward = 0.88f;
// Presentation: smooth motion clips + PS1-style vertex crunch / stepped playback when parkouring.
constexpr float kPs1ParkourPresentSmoothHz = 9.f;
// Higher baseline = more vertex snap / staff dither everywhere (retro PS1 store look).
constexpr float kPs1ParkourBaselineMix = 0.34f;
// Extra PS1 crunch during blackout / low light (stacked on top of parkour spikes).
constexpr float kPs1HorrorNightBoost = 0.44f;
constexpr float kPs1PlayerPhaseFromParkourMul = 0.82f;
constexpr float kPs1NpcClipPhaseStrength = 0.34f;
constexpr float kWallProbeMaxDist = 1.05f;
constexpr float kWallProbeChestHeightFrac = 0.56f;
constexpr float kWallRunMinSpeed = 12.0f * kMovementSpeedScale;
constexpr float kWallRunMaxDuration = 1.4f;
constexpr float kWallRunUpForce = 3.8f;
constexpr float kWallRunGravityScale = 0.18f;
constexpr float kWallRunDetachSpeed = 6.5f;
constexpr float kWallRunCameraTiltRad = 0.18f;
constexpr float kWallRunMinHeight = 0.5f;
constexpr float kWallRunWallDist = 0.65f;
constexpr float kWallRunCooldown = 0.35f;
constexpr float kWallClimbMaxDuration = 0.9f;
constexpr float kWallClimbSpeed = 7.2f;
constexpr float kWallClimbMinApproachSpeed = 8.0f * kMovementSpeedScale;
constexpr float kWallClimbMinDotForward = 0.7f;
constexpr float kWallClimbCooldown = 0.4f;
constexpr float kWallJumpVelY = 10.2f;
constexpr float kWallJumpAwaySpeed = 8.5f;
constexpr float kWallJumpUpMul = 0.85f;
constexpr float kWallJumpCooldown = 0.25f;
constexpr float kSpeedVaultMaxHeight = 1.15f;
constexpr float kSpeedVaultMinSpeed = 14.0f * kMovementSpeedScale;
constexpr float kSpeedVaultDuration = 0.32f;
constexpr float kSpeedVaultSpeedRetain = 0.88f;
constexpr float kSpeedVaultUpBoost = 4.2f;
constexpr float kSpeedVaultArcHeight = 0.7f;
constexpr float kLedgeHangGrabDuration = 0.22f;
constexpr float kLedgeHangShimmySpeed = 1.8f;
constexpr float kLedgeHangPullBack = 0.85f;
// While A/D shimmy: pull camera/feet this much closer to the ledge (m) than kLedgeHangPullBack.
constexpr float kLedgeHangShimmyPullBackTowardLedge = -0.08f;
// After grab, no strafe: slow phase advance for idle hang (clips that sample ledgeHangShimmyPhase).
constexpr float kLedgeHangIdleShimmyPhaseRate = 0.40f;
constexpr float kLedgeHangOutsideClearance = 0.06f;
constexpr float kLedgeHangDropVelY = -2.0f;
constexpr float kLedgeHangJumpAwayVelY = 8.2f;
constexpr float kLedgeHangJumpAwaySpeed = 6.8f;
constexpr float kLedgeHangEyeDropFrac = 0.82f;
constexpr float kLedgeHangMinRise = 1.0f;
constexpr float kLedgeHangMinLookToward = 0.4f;
constexpr float kLedgeCrosshairRayMax = 10.5f;
constexpr float kLedgeCrosshairEdgeBand = 1.65f;
constexpr float kLedgeCrosshairTopSlop = 0.38f;
constexpr float kLedgeGrabRayMinYForTopPlane = 0.0045f;
// Cone half-angle (deg) blended by ledge rise between low/high; mid-rise uses ~average.
constexpr float kLedgeGrabConeHalfLowRiseDeg = 24.5f;
constexpr float kLedgeGrabConeHalfHighRiseDeg = 19.5f;
// Feet this close to shelf edge (XZ): any deck-top hit in slop box counts (no lip aim).
constexpr float kLedgeGrabRelaxLipFootM = 1.38f;
// Camera-space tilt along rolled camUp: low ledges bias slightly down, high ledges up (radians-ish scale).
constexpr float kLedgeGrabAimTiltLowDown = 0.078f;
constexpr float kLedgeGrabAimTiltHighUp = 0.182f;
constexpr float kLedgeGrabReachBonusAtSprint = 1.05f;
constexpr float kLedgeGrabMaxVelYBonusRun = 6.0f;
// No-hands assist: center view hits deck top inside expanded rect (only if cone+lip test missed).
constexpr float kLedgeGrabAssistXZPad = 1.12f;
constexpr float kLedgeGrabAssistMinLookYLowRise = 0.001f;
constexpr float kLedgeGrabAssistMinLookYHighRise = 0.0042f;
// Feet near shelf lip (XZ): allow almost-flat assist ray if main aim missed.
constexpr float kLedgeGrabNearLipForAssistM = 0.92f;
constexpr float kLedgeGrabAssistMinLookYNearLip = 0.0002f;
constexpr float kLedgeGrabMegaAssistFootLipM = 1.2f;
constexpr float kLedgeGrabMegaAssistXZPad = 1.48f;
// Snap aim to deck-top center; hug = accept with no ray hit (very close to lip only).
constexpr float kLedgeGrabSnapRayFootLipM = 1.15f;
constexpr float kLedgeGrabHugAutoFootLipM = 0.72f;
// Feet on aisle-facing side of rack: (feetXZ - deckCenterXZ) · normalize(aisleX - shelfColumnX, 0) — aisles run along +Z, racks face across X.
constexpr float kLedgeGrabMinTowardAisleDotXZ = 0.02f;
// Deck must sit in front of the camera in XZ (stops mantling shelves behind you along the aisle or when turned away).
constexpr float kLedgeGrabMinLookTowardDeckDotXZ = 0.14f;
// Winning grab ray cannot aim backward relative to view (catches snap/hug edge cases).
constexpr float kLedgeGrabMinMantleRayForwardDotXZ = 0.07f;
static float mantleRunT(const glm::vec2& horizVel) {
  return glm::clamp(glm::length(horizVel) / std::max(kSprintSpeed, 0.01f), 0.f, 1.f);
}

// Extra reach / vertical tolerance in air and when moving fast (ledge grab while jumping or sprinting).
static void mantleLedgeMovementAid(const glm::vec2& horizVel, bool grounded, float& outRunTBoost,
                                   float& outExtraFallAllow) {
  const float sp = glm::length(horizVel);
  outRunTBoost = 0.f;
  outExtraFallAllow = 0.f;
  if (!grounded) {
    outRunTBoost += 0.52f;
    outExtraFallAllow += 5.4f;
  }
  if (sp > 4.2f) {
    const float sprintish = glm::clamp((sp - 4.2f) / (kSprintSpeed - 4.2f), 0.f, 1.f);
    outRunTBoost += 0.32f * sprintish;
    outExtraFallAllow += 2.85f * sprintish;
  }
  outRunTBoost = glm::min(outRunTBoost, 0.78f);
}

static float mantleReachXZFromView(const glm::vec3& rdCenter, const glm::vec2& horizVel,
                                   float runTBoostExtra = 0.f) {
  const float runT = glm::min(1.f, mantleRunT(horizVel) + runTBoostExtra);
  float reachXZ = kLedgeGrabMaxReachXZ + runT * kLedgeGrabReachBonusAtSprint;
  glm::vec2 lookH(rdCenter.x, rdCenter.z);
  const float lh = glm::length(lookH);
  if (lh > 1e-4f) {
    lookH *= 1.f / lh;
    const float towardSp = std::max(0.f, glm::dot(horizVel, lookH));
    reachXZ += std::min(towardSp * kLedgeGrabForwardReachPerSpeed, kLedgeGrabForwardReachBonusMax);
  }
  return reachXZ;
}

static float mantleMaxVelUp(float runT) {
  return kLedgeGrabMaxVelY + runT * kLedgeGrabMaxVelYBonusRun;
}

// Blend toward discrete animation time (smooth DL-style keys, PS1-style ~10–13 Hz hold).
static double ps1QuantizeClipPhase(double phaseSec, float strength01) {
  if (strength01 < 1e-5f)
    return phaseSec;
  constexpr double kAnimHoldHz = 12.5;
  const double step = 1.0 / kAnimHoldHz;
  const double snapped = std::floor(phaseSec / step + 1e-9) * step;
  const double w = glm::clamp(static_cast<double>(strength01), 0.0, 1.0);
  return glm::mix(phaseSec, snapped, w);
}

static float effectivePs1HorrorMix(float parkourMix, float nightHorrorWeight) {
  return glm::clamp(parkourMix + nightHorrorWeight * kPs1HorrorNightBoost, 0.f, 1.f);
}

// Shallow upward look needs a long ray to reach the deck top (t = dy/rd.y); fixed kLedgeCrosshairRayMax
// was forcing players to pitch up hard. Cap stays bounded — only nearby decks are tested.
static float mantleLedgeTopPlaneTMax(const glm::vec3& ro, float deckTopY, const glm::vec3& rd) {
  constexpr float kHardCap = 55.f;
  const float base = kLedgeCrosshairRayMax;
  if (rd.y < 0.01f)
    return base;
  const float dy = deckTopY - ro.y;
  if (dy <= 0.f)
    return base;
  const float tNeed = dy / std::max(rd.y, 0.01f);
  return std::min(std::max(base, tNeed + 1.5f), kHardCap);
}

// Flat floor/ceiling: one quad per streamed chunk (same plane + color as the old per-cell grid).
constexpr size_t kMaxTerrainVerts() {
  const int side = kChunkRadius * 2 + 1;
  return static_cast<size_t>(side * side * 6);
}

constexpr float kPillarHalfW = 2.8f;
constexpr float kPillarHalfD = 2.8f;
constexpr float kPillarHeight = 34.0f;
constexpr float kPillarSpacing = 105.0f;
constexpr int kPillarGridRadius = 4;
// Draw/sign loops only — collision still uses kPillarGridRadius (wider).
constexpr int kPillarDrawGridRadius = 4;
// SCP-3008–style infinite big-box: long aisles along +Z, racks on each side (kShelfAisleModulePitch).
constexpr float kStoreAisleWidth = 5.65f;
// Legacy fixed radii (replaced by shelfGridWindowForRange for each query). Kept for comments /
// “infinite store” feel: visible racks use kShelfCullHardDist-sized windows around the camera.
constexpr int kShelfAlongLineRadius = 110;
constexpr int kShelfAislesRadius = 100;
// Bays are rolled per N×N cluster: 75% empty / 25% shelf biome.
constexpr int kShelfBiomeClusterSpan = 8;
// Draw distance / atmosphere — heavier fog reads as endless fluorescent hall.
constexpr float kViewFogStart = 44.0f;
constexpr float kViewFogEnd = 228.0f;
// Blackout: closer, thicker fog + dimmer lighting; fogParams.w flags dark fog in shader.
constexpr float kViewFogStartBlackout = 20.0f;
constexpr float kViewFogEndBlackout = 86.0f;
constexpr float kStoreLightMulBlackout = 0.14f;
constexpr float kProjFarPlane = 295.0f;
// Tighter than fog end — fewer shelf/crate instances + less CPU in recordCommandBuffer near dense aisles.
constexpr float kShelfCullHardDist = 156.0f;
constexpr float kShelfCullNominalDist = 128.0f;
constexpr float kShelfCullRimJitterM = 5.0f;
constexpr uint32_t kMaxShelfInstances = 12288;
// Deck props (boxes + pallets) share one instanced draw.
constexpr uint32_t kMaxShelfCrates = 12288;
constexpr uint32_t kMaxShelfLadders = 2048;
constexpr uint32_t kMaxShelfBoxCutters = 1024;
constexpr uint32_t kMaxShelfRustyPipes = 1024;
// Set true in createVertexBuffers after the box cutter GLB uploads successfully.
static bool gShelfBoxCutterMeshLoaded = false;
static bool gShelfRustyPipeMeshLoaded = false;
static bool gDeliPizzaMeshLoaded = false;
static bool gDeliMeatballMeshLoaded = false;
constexpr uint32_t kMaxDeliPizzaInstances = 2048;
constexpr uint32_t kMaxDeliMeatballInstances = 2048;
constexpr int kDeliPizzaSlicesPerCounter = 6;
constexpr int kDeliMeatballsPerCounter = 6;
constexpr float kDeliFoodRenderDist = 10.0f;
constexpr uint32_t kMaxMarketInstances = 512;
constexpr int kDeliBiomeClusterSpan = 10;
constexpr int kDeliClearPad = 1;
constexpr float kDeliCounterHalfL = 2.0f;
constexpr float kDeliCounterHalfD = 0.5f;
constexpr float kDeliCounterH = 0.90f;
constexpr float kDeliCounterTopH = 0.08f;
constexpr float kDeliCounterTopOverhang = 0.05f;
constexpr float kDeliGlassHeight = 0.55f;
constexpr float kDeliGlassThick = 0.03f;
constexpr float kPlayerHalfXZ = 0.34f;
constexpr float kCameraClipRadius = 0.24f;
// Third-person orbit: keep lens above walkable tops at camera XZ (stops clipping through floor mesh).
constexpr float kThirdPersonCamMinAboveSupportM = 0.36f;
constexpr float kMaxHorizMoveStep = 0.72f;

struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec4 color; // .a = 1 for world geometry; staff uses .a for baked part tag
  glm::vec2 uv{};
};

static inline glm::vec4 vrgb(const glm::vec3& c) {
  return glm::vec4(c, 1.f);
}

static_assert(sizeof(emp_mesh::LoadedVertex) == sizeof(Vertex));
// Staff mesh is instanced; without a tight draw radius, hundreds of FBX figures land in one
// view (see screencast) and tank the GPU. Shelves already cull by distance — match that idea.
constexpr uint32_t kMaxEmployees = 18;
// Skinned staff SSBO / instance buffer: staff + player avatar + optional Shrek easter egg (own VBO, extra slot).
constexpr uint32_t kShrekEggStaffSlotIndex = kMaxEmployees + 1u;
constexpr uint32_t kStaffSkinnedInstanceSlots = kMaxEmployees + 2u;
// Eligible shelf cells: staff spawns when (hash % modulus) == 0. Larger modulus → fewer NPCs.
constexpr uint32_t kShelfEmpSpawnModulus = 83u;
constexpr float kEmployeeVisualHeight = 1.72f;
// Horizontal XZ: shader fades alpha between inner (opaque) and outer (gone); CPU lists through outer.
// Horizontal XZ distance (m): shader smooth-fades; keep CPU list radius larger than outer to avoid hard pop.
constexpr float kEmployeeFadeInnerH = 58.f;
constexpr float kEmployeeFadeOuterH = 122.f;
constexpr float kStaffCpuListDist = 130.f;
// Beyond this distance to the player, skip multi-probe nav (staffNpcAABBTouchesWorld per angle).
constexpr float kStaffNavLookaheadSkipDist = 135.f;
// Feet probe for terrainSupportY — match player auto-step; extra slack so kFeetBand (0.12) still snaps ledge tops.
constexpr float kStaffTerrainStepProbe = kMaxStepHeight + 0.18f;
// Staff collision: local XZ half-extents, full height; Y rotation = NPC yaw (same as draw matrix).
constexpr float kStaffHitHalfW = 0.30f;
constexpr float kStaffHitHalfD = 0.28f;
// XZ broadphase for player vs staff: beyond (player half-extent + staff footprint radius + pad) feet
// cannot overlap; pad covers sprint/chase closing speed per frame (tighter than a huge fixed radius).
constexpr float kStaffPlayerCollisionPadM = 2.85f;
constexpr uint32_t kStaffPaletteBoneCount = static_cast<uint32_t>(staff_skin::kMaxPaletteBones);
// Lightweight “ragdoll”: euler extras + optional PBD world particles (no external physics lib).
constexpr int kStaffRagdollSimMaxBones = 24;

static int gStaffDeadRagdollSimBoneCount = 0;
static int gStaffDeadRagdollSimBoneRigIdx[kStaffRagdollSimMaxBones];
static int gStaffDeadRagdollSimParentIdx[kStaffRagdollSimMaxBones];
static float gStaffDeadRagdollSimRestLen[kStaffRagdollSimMaxBones];
// 0=default (knee, elbow, etc.), 1=neck, 2=hand/foot/wrist/ankle, 3=forearm/calf,
// 4=spine/chest/pelvis/hips — tight limits; 5=upper arm/clavicle/shoulder; 6=thigh/upleg (long tip reach);
// 8=head (RP3D capsule + euler/PBD clamps).
static uint8_t gStaffDeadRagdollSimBoneKind[kStaffRagdollSimMaxBones];

static int staffRigBoneDepth(const staff_skin::Rig& rig, int bi, int guard = 0) {
  if (guard > 96 || bi < 0 || bi >= rig.boneCount)
    return 999;
  const std::string& nm = rig.boneNames[static_cast<size_t>(bi)];
  auto it = rig.nodes.find(nm);
  if (it == rig.nodes.end())
    return 0;
  const std::string& pnm = it->second.parent;
  if (pnm.empty() || pnm == rig.rootName)
    return 1;
  auto pi = rig.boneNameToIndex.find(pnm);
  if (pi == rig.boneNameToIndex.end())
    return 1;
  return 1 + staffRigBoneDepth(rig, pi->second, guard + 1);
}

static int staffRigFindParentSimIndex(const staff_skin::Rig& rig, int bi,
                                      const std::unordered_map<int, int>& rigToSim) {
  std::string cur = rig.boneNames[static_cast<size_t>(bi)];
  for (int guard = 0; guard < 96; ++guard) {
    auto it = rig.nodes.find(cur);
    if (it == rig.nodes.end())
      return -1;
    const std::string& pnm = it->second.parent;
    if (pnm.empty() || pnm == rig.rootName)
      return -1;
    auto pi = rig.boneNameToIndex.find(pnm);
    if (pi == rig.boneNameToIndex.end())
      return -1;
    auto si = rigToSim.find(pi->second);
    if (si != rigToSim.end())
      return si->second;
    cur = pnm;
  }
  return -1;
}

static void staffRebuildDeadRagdollSimBoneMap(const staff_skin::Rig& rig) {
  gStaffDeadRagdollSimBoneCount = 0;
  for (int j = 0; j < kStaffRagdollSimMaxBones; ++j) {
    gStaffDeadRagdollSimBoneRigIdx[j] = 0;
    gStaffDeadRagdollSimParentIdx[j] = -1;
    gStaffDeadRagdollSimRestLen[j] = 0.f;
    gStaffDeadRagdollSimBoneKind[j] = 0;
  }
  if (rig.boneCount <= 0)
    return;
  std::vector<int> candidates;
  candidates.reserve(static_cast<size_t>(rig.boneCount));
  for (int i = 0; i < rig.boneCount; ++i) {
    const std::string& nm = rig.boneNames[static_cast<size_t>(i)];
    if (nm == rig.rootName)
      continue;
    std::string lower = nm;
    for (char& c : lower)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower.find("thumb") != std::string::npos || lower.find("finger") != std::string::npos ||
        lower.find("toe") != std::string::npos || lower.find("twist") != std::string::npos ||
        lower.find("ik") != std::string::npos || lower.find("eye") != std::string::npos ||
        lower.find("hair") != std::string::npos ||
        lower.find("weapon") != std::string::npos || lower.find("prop") != std::string::npos ||
        lower.find("jiggle") != std::string::npos || lower.find("breast") != std::string::npos ||
        lower.find("boob") != std::string::npos)
      continue;
    const bool limb =
        lower.find("upperarm") != std::string::npos || lower.find("lowerarm") != std::string::npos ||
        lower.find("forearm") != std::string::npos || lower.find("elbow") != std::string::npos ||
        lower.find("hand") != std::string::npos || lower.find("wrist") != std::string::npos ||
        lower.find("thigh") != std::string::npos || lower.find("upleg") != std::string::npos ||
        lower.find("calf") != std::string::npos || lower.find("shin") != std::string::npos ||
        lower.find("knee") != std::string::npos || lower.find("foot") != std::string::npos ||
        lower.find("ankle") != std::string::npos || lower.find("clavicle") != std::string::npos ||
        lower.find("neck") != std::string::npos || lower.find("spine") != std::string::npos ||
        lower.find("chest") != std::string::npos || lower.find("pelvis") != std::string::npos ||
        lower.find("hips") != std::string::npos || lower.find("shoulder") != std::string::npos ||
        (lower.find("head") != std::string::npos && lower.find("forehead") == std::string::npos);
    if (!limb)
      continue;
    candidates.push_back(i);
  }
  std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
    const int da = staffRigBoneDepth(rig, a);
    const int db = staffRigBoneDepth(rig, b);
    if (da != db)
      return da < db;
    return a < b;
  });
  const int nTake = std::min(static_cast<int>(candidates.size()), kStaffRagdollSimMaxBones);
  std::unordered_map<int, int> rigToSim;
  rigToSim.reserve(static_cast<size_t>(nTake) * 2u);
  for (int t = 0; t < nTake; ++t) {
    const int bi = candidates[static_cast<size_t>(t)];
    gStaffDeadRagdollSimBoneRigIdx[t] = bi;
    rigToSim[bi] = t;
    gStaffDeadRagdollSimBoneCount = t + 1;
  }
  for (int j = 0; j < gStaffDeadRagdollSimBoneCount; ++j) {
    const int bi = gStaffDeadRagdollSimBoneRigIdx[j];
    gStaffDeadRagdollSimParentIdx[j] = staffRigFindParentSimIndex(rig, bi, rigToSim);
    std::string lower = rig.boneNames[static_cast<size_t>(bi)];
    for (char& c : lower)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    uint8_t kind = 0;
    if (lower.find("neck") != std::string::npos)
      kind = 1;
    else if (lower.find("head") != std::string::npos)
      kind = 8;
    else if (lower.find("spine") != std::string::npos || lower.find("chest") != std::string::npos ||
             lower.find("pelvis") != std::string::npos || lower.find("hips") != std::string::npos)
      kind = 4;
    else if (lower.find("hand") != std::string::npos || lower.find("wrist") != std::string::npos ||
             lower.find("foot") != std::string::npos || lower.find("ankle") != std::string::npos)
      kind = 2;
    else if (lower.find("forearm") != std::string::npos || lower.find("lowerarm") != std::string::npos ||
             lower.find("calf") != std::string::npos || lower.find("shin") != std::string::npos)
      kind = 3;
    else if (lower.find("upperarm") != std::string::npos || lower.find("shoulder") != std::string::npos ||
             lower.find("clavicle") != std::string::npos)
      kind = 5;
    else if (lower.find("thigh") != std::string::npos || lower.find("upleg") != std::string::npos)
      kind = 6;
    gStaffDeadRagdollSimBoneKind[j] = kind;
  }
  glm::mat4 gGlob[staff_skin::kMaxPaletteBones];
  if (!rig.clips.empty())
    staff_skin::sampleClipBoneGlobalMatrices(rig, 0, 0.0, true, nullptr, gGlob);
  else
    staff_skin::sampleBindBoneGlobalMatricesWithExtras(rig, nullptr, gGlob);
  for (int j = 0; j < gStaffDeadRagdollSimBoneCount; ++j) {
    const int bi = gStaffDeadRagdollSimBoneRigIdx[j];
    const int pj = gStaffDeadRagdollSimParentIdx[j];
    const glm::vec3 ppos =
        glm::vec3(rig.meshNorm * gGlob[bi] * glm::vec4(0.f, 0.f, 0.f, 1.f));
    if (pj >= 0) {
      const int pbi = gStaffDeadRagdollSimBoneRigIdx[pj];
      const glm::vec3 ppar =
          glm::vec3(rig.meshNorm * gGlob[pbi] * glm::vec4(0.f, 0.f, 0.f, 1.f));
      gStaffDeadRagdollSimRestLen[j] = glm::length(ppos - ppar);
    } else
      gStaffDeadRagdollSimRestLen[j] = 0.f;
  }
}

// Single pooled struct per visible NPC: one vector growth pattern, contiguous fields (cache-friendly vs
// eight parallel std::vectors). Cleared each frame; capacity reserved once at init — no heap allocs on steady state.
struct StaffNpcDrawSlot {
  glm::mat4 model{};
  int clipIdx = 0;
  double phase = 0.0;
  uint8_t animLoop = 1u;
  float meleeBlend = 1.f;
  int meleeFromClip = 0;
  double meleeFromPhase = 0.0;
  uint8_t meleeFromLoop = 1u;
  // Dead ragdoll: palette from static fall-clip frame (or loose bind if no fall clip).
  uint8_t bindPoseOnly = 0u;
  glm::vec3 ragdollAngVelForSkin{0.f};
  uint32_t ragdollLooseSeed = 0u;
  uint8_t deadRagdollJointCount = 0u;
  glm::vec3 deadRagdollJointEuler[kStaffRagdollSimMaxBones]{};
};

struct UniformBufferObject {
  glm::mat4 viewProj{1.0f};
  glm::vec4 cameraPos{0, 0, 0, 0};
  glm::vec4 fogParams{kViewFogStart, kViewFogEnd, 1.f, 0.f};
  // xy = feet XZ, z = ground height under feet, w = shadow radius (m).
  glm::vec4 shadowParams{0, 0, 0, 0.58f};
  // Staff pop-in softening: .x = full opacity inside this XZ radius (m), .y = invisible beyond (m).
  glm::vec4 employeeFadeH{kEmployeeFadeInnerH, kEmployeeFadeOuterH, 0.f, 0.f};
  // Staff materials (from FBX scan): .xy = shirt vertical band Y lo/hi, .z = pants top Y, .w = torso XZ radius.
  glm::vec4 employeeBounds{0.88f, 1.39f, 0.90f, 0.24f};
  // .x = number of extra textures successfully loaded from disk (0..kMaxExtraTextures).
  // .y = optional blend (0..255) of extraTex[0] onto default world triplanar (0 = off).
  // .z = staff multi-texture strength (0..255): pants/shirt/skin from extraTex[0..2] in shader (255 = full).
  // .w = 1: staff uses binding 6 staffGlbTex; Shrek egg .w = 2 uses binding 8 shrekEggTex (not staff atlas).
  glm::ivec4 extraTexInfo{0, 0, 0, 0};
  // .x = sim time (s) for vertex wobble; .y = gait scale (~1.2 day wander, ~2.8 night chase).
  glm::vec4 staffAnim{0.f};
};

namespace {

glm::vec4 computeEmployeeBoundsFromMesh(const std::vector<emp_mesh::LoadedVertex>& verts) {
  constexpr float kFallbackLo = 0.88f;
  constexpr float kFallbackHi = 1.38f;
  auto bakedPart = [](const emp_mesh::LoadedVertex& v) -> int {
    const float a = v.color.a;
    if (a < 0.015f || a > 0.07f)
      return -1;
    return static_cast<int>(std::lround(a / 0.02f)) - 1;
  };
  float shirtY0 = std::numeric_limits<float>::infinity();
  float shirtY1 = -std::numeric_limits<float>::infinity();
  float jeansY1 = -std::numeric_limits<float>::infinity();
  float shirtRxzMax = 0.f;
  bool anyShirt = false;
  bool anyJeans = false;
  for (const auto& v : verts) {
    const int p = bakedPart(v);
    if (p < 0)
      continue;
    if (p == 1) {
      anyShirt = true;
      shirtY0 = std::min(shirtY0, v.pos.y);
      shirtY1 = std::max(shirtY1, v.pos.y);
      shirtRxzMax = std::max(shirtRxzMax, glm::length(glm::vec2(v.pos.x, v.pos.z)));
    } else if (p == 0) {
      anyJeans = true;
      jeansY1 = std::max(jeansY1, v.pos.y);
    }
  }
  // Pad shirt Y span; also clamp so a collar-only shirt mesh does not shrink the shirt zone to a strip.
  float mapLo = anyShirt ? (shirtY0 - 0.22f) : kFallbackLo;
  float mapHi = anyShirt ? (shirtY1 + 0.22f) : kFallbackHi;
  mapLo = std::max(0.f, std::min(mapLo, 0.93f));
  mapHi = std::max({mapLo + 0.42f, mapHi, 1.39f});
  float pantsTop;
  if (anyJeans && jeansY1 > mapLo - 0.28f && jeansY1 > 0.52f) {
    pantsTop = std::min(jeansY1 + 0.04f, mapLo - 0.03f);
  } else {
    pantsTop = std::min(mapLo - 0.04f, 0.93f);
  }
  // Never pull pants up to the chest (old min(pantsTop, mapLo+0.02) did that when mapLo was high).
  pantsTop = std::clamp(pantsTop, 0.76f, 0.94f);
  const float torsoR =
      anyShirt ? std::clamp(shirtRxzMax * 1.30f, 0.22f, 0.36f) : 0.26f;
  return glm::vec4(mapLo, mapHi, pantsTop, torsoR);
}

glm::vec4 computeEmployeeBoundsFromSkinnedMesh(const std::vector<staff_skin::SkinnedVertex>& verts) {
  constexpr float kFallbackLo = 0.88f;
  constexpr float kFallbackHi = 1.38f;
  auto bakedPart = [](const staff_skin::SkinnedVertex& v) -> int {
    const float a = v.color.a;
    if (a < 0.015f || a > 0.07f)
      return -1;
    return static_cast<int>(std::lround(a / 0.02f)) - 1;
  };
  float shirtY0 = std::numeric_limits<float>::infinity();
  float shirtY1 = -std::numeric_limits<float>::infinity();
  float jeansY1 = -std::numeric_limits<float>::infinity();
  float shirtRxzMax = 0.f;
  bool anyShirt = false;
  bool anyJeans = false;
  for (const auto& v : verts) {
    const int p = bakedPart(v);
    if (p < 0)
      continue;
    if (p == 1) {
      anyShirt = true;
      shirtY0 = std::min(shirtY0, v.pos.y);
      shirtY1 = std::max(shirtY1, v.pos.y);
      shirtRxzMax = std::max(shirtRxzMax, glm::length(glm::vec2(v.pos.x, v.pos.z)));
    } else if (p == 0) {
      anyJeans = true;
      jeansY1 = std::max(jeansY1, v.pos.y);
    }
  }
  float mapLo = anyShirt ? (shirtY0 - 0.22f) : kFallbackLo;
  float mapHi = anyShirt ? (shirtY1 + 0.22f) : kFallbackHi;
  mapLo = std::max(0.f, std::min(mapLo, 0.93f));
  mapHi = std::max({mapLo + 0.42f, mapHi, 1.39f});
  float pantsTop;
  if (anyJeans && jeansY1 > mapLo - 0.28f && jeansY1 > 0.52f) {
    pantsTop = std::min(jeansY1 + 0.04f, mapLo - 0.03f);
  } else {
    pantsTop = std::min(mapLo - 0.04f, 0.93f);
  }
  pantsTop = std::clamp(pantsTop, 0.76f, 0.94f);
  const float torsoR =
      anyShirt ? std::clamp(shirtRxzMax * 1.30f, 0.22f, 0.36f) : 0.26f;
  return glm::vec4(mapLo, mapHi, pantsTop, torsoR);
}

} // namespace

struct PushModel {
  glm::mat4 model{1.0f};
  // .w = 1: fragment uses flat grey for skinned local avatar (no GLB / procedural uniform).
  glm::vec4 staffShade{0.f, 0.f, 0.f, 0.f};
};

struct PushPost {
  glm::vec4 g{0.f}; // x=time, y=horror night weight, z=viewport W, h=viewport H (float, post pass)
  glm::vec4 v{0.f}; // x=damage hit pulse 0..1, y=critical HP edge (hp<=35) 0..1, z=parkour PS1 mix, w=night pursuit 0..1
};

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;
  bool complete() const { return graphicsFamily && presentFamily; }
};

struct SwapchainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities{};
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

#define VK_CHECK(r, msg)                                                                           \
  do {                                                                                             \
    if ((r) != VK_SUCCESS)                                                                         \
      throw std::runtime_error(std::string(msg) + " (" + std::to_string(r) + ")");                 \
  } while (0)

std::vector<char> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file)
    throw std::runtime_error("failed to open file: " + path);
  const size_t size = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(size);
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(size));
  return buffer;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code) {
  VkShaderModuleCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  ci.codeSize = code.size();
  ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
  VkShaderModule module{};
  VK_CHECK(vkCreateShaderModule(device, &ci, nullptr, &module), "vkCreateShaderModule");
  return module;
}

uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                        VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((typeFilter & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
      return i;
  }
  throw std::runtime_error("findMemoryType");
}

void createBuffer(VkPhysicalDevice phys, VkDevice dev, VkDeviceSize size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags memProps, VkBuffer& outBuf, VkDeviceMemory& outMem) {
  VkBufferCreateInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &outBuf), "vkCreateBuffer");

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(dev, outBuf, &req);
  VkMemoryAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = findMemoryType(phys, req.memoryTypeBits, memProps);
  VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &outMem), "vkAllocateMemory");
  vkBindBufferMemory(dev, outBuf, outMem, 0);
}

void copyBuffer(VkDevice dev, VkCommandPool pool, VkQueue queue, VkBuffer src, VkBuffer dst,
                VkDeviceSize size) {
  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd{};
  vkAllocateCommandBuffers(dev, &ai, &cmd);

  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);
  VkBufferCopy region{};
  region.size = size;
  vkCmdCopyBuffer(cmd, src, dst, 1, &region);
  vkEndCommandBuffer(cmd);

  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(dev, pool, 1, &cmd);
}

VkCommandBuffer beginOneShotCommands(VkDevice dev, VkCommandPool pool) {
  VkCommandBufferAllocateInfo ai{};
  ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  ai.commandPool = pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd{};
  vkAllocateCommandBuffers(dev, &ai, &cmd);
  VkCommandBufferBeginInfo bi{};
  bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);
  return cmd;
}

void endOneShotCommands(VkDevice dev, VkCommandPool pool, VkQueue queue, VkCommandBuffer cmd) {
  vkEndCommandBuffer(cmd);
  VkSubmitInfo si{};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(dev, pool, 1, &cmd);
}

void transitionImageLayout(VkDevice dev, VkCommandPool pool, VkQueue queue, VkImage image,
                           VkImageLayout oldLayout, VkImageLayout newLayout) {
  VkCommandBuffer cmd = beginOneShotCommands(dev, pool);
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else {
    throw std::runtime_error("unsupported image layout transition");
  }

  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  endOneShotCommands(dev, pool, queue, cmd);
}

void copyBufferToImage(VkDevice dev, VkCommandPool pool, VkQueue queue, VkBuffer buffer, VkImage image,
                       uint32_t width, uint32_t height) {
  VkCommandBuffer cmd = beginOneShotCommands(dev, pool);
  VkBufferImageCopy region{};
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {width, height, 1};
  vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  endOneShotCommands(dev, pool, queue, cmd);
}

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
  QueueFamilyIndices idx;
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
  std::vector<VkQueueFamilyProperties> props(count);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &count, props.data());

  int i = 0;
  for (const auto& p : props) {
    if (p.queueFlags & VK_QUEUE_GRAPHICS_BIT)
      idx.graphicsFamily = static_cast<uint32_t>(i);
    VkBool32 presentSupport = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, static_cast<uint32_t>(i), surface, &presentSupport);
    if (presentSupport)
      idx.presentFamily = static_cast<uint32_t>(i);
    if (idx.complete())
      break;
    ++i;
  }
  return idx;
}

bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
  const char* ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> available(count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &count, available.data());
  for (const auto& e : available) {
    if (strcmp(e.extensionName, ext) == 0)
      return true;
  }
  return false;
}

SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device, VkSurfaceKHR surface) {
  SwapchainSupportDetails d;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &d.capabilities);
  uint32_t n = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &n, nullptr);
  if (n) {
    d.formats.resize(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &n, d.formats.data());
  }
  n = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &n, nullptr);
  if (n) {
    d.presentModes.resize(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &n, d.presentModes.data());
  }
  return d;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
  for (const auto& f : formats) {
    if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
        f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
      return f;
  }
  return formats[0];
}

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& modes) {
  // FIFO = vsync; steadier cadence on many compositors (less micro-stutter than uncapped present).
  if (const char* v = std::getenv("VULKAN_GAME_VSYNC")) {
    if (v[0] == '1' && v[1] == '\0') {
      for (auto m : modes) {
        if (m == VK_PRESENT_MODE_FIFO_KHR)
          return m;
      }
    }
  }
  for (auto m : modes) {
    if (m == VK_PRESENT_MODE_MAILBOX_KHR)
      return m;
  }
  for (auto m : modes) {
    if (m == VK_PRESENT_MODE_IMMEDIATE_KHR)
      return m;
  }
  for (auto m : modes) {
    if (m == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
      return m;
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& caps, int w, int h) {
  if (caps.currentExtent.width != UINT32_MAX)
    return caps.currentExtent;
  VkExtent2D e{static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
  e.width = std::clamp(e.width, caps.minImageExtent.width, caps.maxImageExtent.width);
  e.height = std::clamp(e.height, caps.minImageExtent.height, caps.maxImageExtent.height);
  return e;
}

VkFormat findSupportedFormat(VkPhysicalDevice phys, const std::vector<VkFormat>& candidates,
                             VkImageTiling tiling, VkFormatFeatureFlags features) {
  for (VkFormat f : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(phys, f, &props);
    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features)
      return f;
    if (tiling == VK_IMAGE_TILING_OPTIMAL &&
        (props.optimalTilingFeatures & features) == features)
      return f;
  }
  throw std::runtime_error("findSupportedFormat");
}

VkFormat findDepthFormat(VkPhysicalDevice phys) {
  return findSupportedFormat(
      phys,
      {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

static glm::vec3 meshVertexColor() {
  return glm::vec3(0.5f);
}

// Vertex tags for warehouse props (fragment shader: distance match — avoids fragile inequalities).
static glm::vec3 shelfMetalVertexColor() {
  return glm::vec3(0.97f, 0.06f, 0.04f);
}
static glm::vec3 shelfWoodVertexColor() {
  return glm::vec3(0.03f, 0.98f, 0.04f);
}
static glm::vec3 shelfCrateVertexColor() {
  return glm::vec3(0.87f, 0.43f, 0.14f);
}
// shader.frag: isShelfPallet — deck pallet stacks (triplanar palletTex); tag range disjoint from crate/cardboard.
static glm::vec3 shelfPalletVertexColor() {
  return glm::vec3(0.66f, 0.50f, 0.28f);
}
// shader.frag: isShelfLadder — dark grey leaning prop (rarer spawn than crates).
static glm::vec3 shelfLadderVertexColor() {
  return glm::vec3(0.145f, 0.152f, 0.138f);
}
// shader.frag: isShelfBoxCutter — textured GLB on shelf decks (CMake: VULKAN_GAME_SHELF_DECK_PROPS).
#if defined(VULKAN_GAME_SHELF_BOX_CUTTER_GLB)
static glm::vec3 shelfBoxCutterVertexColor() {
  return glm::vec3(0.255f, 0.075f, 0.518f);
}
#endif
#if defined(VULKAN_GAME_SHELF_RUSTY_PIPE_GLB)
static glm::vec3 shelfRustyPipeVertexColor() {
  return glm::vec3(0.53f, 0.29f, 0.088f);
}
#endif
#if defined(VULKAN_GAME_PIZZA_SLICE_GLB)
static glm::vec3 deliPizzaVertexColor() {
  return glm::vec3(0.82f, 0.42f, 0.12f);
}
#endif
static glm::vec3 deliMeatballVertexColor() {
  // Keep this in a uniquely identifiable range for shader tag routing.
  return glm::vec3(0.372f, 0.772f, 0.612f);
}
static glm::vec3 fluorescentLightVertexColor() {
  // Avoid warm/orange world strips that can be mistaken for HUD bars at distance.
  return glm::vec3(0.72f, 0.78f, 0.30f);
}

static void meshAddBox(std::vector<Vertex>& out, const glm::vec3& mn, const glm::vec3& mx,
                       const glm::vec3& col) {
  const glm::vec4 rgba = vrgb(col);
  auto tri = [&](const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& n) {
    out.push_back({a, n, rgba, {}});
    out.push_back({c, n, rgba, {}});
    out.push_back({b, n, rgba, {}});
  };
  // +X
  tri({mx.x, mn.y, mn.z}, {mx.x, mx.y, mn.z}, {mx.x, mx.y, mx.z}, {1, 0, 0});
  tri({mx.x, mn.y, mn.z}, {mx.x, mx.y, mx.z}, {mx.x, mn.y, mx.z}, {1, 0, 0});
  // -X
  tri({mn.x, mn.y, mx.z}, {mn.x, mx.y, mx.z}, {mn.x, mx.y, mn.z}, {-1, 0, 0});
  tri({mn.x, mn.y, mx.z}, {mn.x, mx.y, mn.z}, {mn.x, mn.y, mn.z}, {-1, 0, 0});
  // +Y top
  tri({mn.x, mx.y, mn.z}, {mx.x, mx.y, mn.z}, {mx.x, mx.y, mx.z}, {0, 1, 0});
  tri({mn.x, mx.y, mn.z}, {mx.x, mx.y, mx.z}, {mn.x, mx.y, mx.z}, {0, 1, 0});
  // -Y bottom
  tri({mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z}, {mx.x, mn.y, mn.z}, {0, -1, 0});
  tri({mn.x, mn.y, mx.z}, {mx.x, mn.y, mn.z}, {mn.x, mn.y, mn.z}, {0, -1, 0});
  // +Z
  tri({mn.x, mn.y, mx.z}, {mn.x, mx.y, mx.z}, {mx.x, mx.y, mx.z}, {0, 0, 1});
  tri({mn.x, mn.y, mx.z}, {mx.x, mx.y, mx.z}, {mx.x, mn.y, mx.z}, {0, 0, 1});
  // -Z
  tri({mx.x, mn.y, mn.z}, {mx.x, mx.y, mn.z}, {mn.x, mx.y, mn.z}, {0, 0, -1});
  tri({mx.x, mn.y, mn.z}, {mn.x, mx.y, mn.z}, {mn.x, mn.y, mn.z}, {0, 0, -1});
}

static constexpr float kShelfMeshHalfW = 2.45f;
static constexpr float kShelfMeshHalfD = 1.1f;
// Staff vs shelf slot: skip post/deck/crate tests when centers are farther than this in XZ (cheap broadphase).
static constexpr float kShelfStaffXZReachSq =
    (std::sqrt(kShelfMeshHalfW * kShelfMeshHalfW + kShelfMeshHalfD * kShelfMeshHalfD) +
     std::sqrt(kStaffHitHalfW * kStaffHitHalfW + kStaffHitHalfD * kStaffHitHalfD) + 0.5f) *
    (std::sqrt(kShelfMeshHalfW * kShelfMeshHalfW + kShelfMeshHalfD * kShelfMeshHalfD) +
     std::sqrt(kStaffHitHalfW * kStaffHitHalfW + kStaffHitHalfD * kStaffHitHalfD) + 0.5f);
static constexpr float kShelfDeckThickness = 0.065f;
// GLB shelf props (box cutter, pipe): lift above deck top so they don’t z-fight the wood and read from the aisle.
static constexpr float kShelfDeckSmallPropLiftY = 0.025f;
// Vertical **clearance** between shelf boards (not overall building height).
static constexpr float kShelfGapBetweenLevels = 3.25f;
static constexpr int kShelfDeckCount = 4;
static constexpr float kShelfPostTopPad = 0.18f;
static constexpr float kShelfMeshHeight =
    0.12f + static_cast<float>(kShelfDeckCount - 1) * (kShelfGapBetweenLevels + kShelfDeckThickness) +
    kShelfDeckThickness + kShelfPostTopPad;
// Top walkable deck on a standard rack (world Y); used for respawn after lethal fall.
static constexpr float kShelfDeckYStepFall = kShelfGapBetweenLevels + kShelfDeckThickness;
static constexpr float kTopShelfDeckSurfaceY =
    kGroundY + 0.12f + static_cast<float>(kShelfDeckCount - 1) * kShelfDeckYStepFall + kShelfDeckThickness;
// Fall damage: Earth gravity (9.81 m/s²) for injury while motion uses arcade kGravity.
// Impact v = max(√(2·g_earth·h_excess), |vy|·√(g_earth/g_game)); h_excess = geometric drop − minDrop (roll/slop dead zone).
// Dmg ∝ (v−v_safe)² — Dying Light–style: generous safe band + ramp with height/speed, but scales and caps are
// tuned **lower** than DL so parkour mistakes are forgiving; top wood deck stays slightly harsher than aisles.
constexpr float kPlayerHealthMax = 100.f;
// Death: play front of knockdown/fall clip, hold pose, then respawn (see beginPlayerDeath).
constexpr float kPlayerDeathFallClipPortion = 0.68f;
constexpr float kPlayerDeathLandClipPortion = 0.40f;
constexpr float kPlayerDeathHoldBeforeRespawnSec = 1.45f;
constexpr float kPlayerDeathNoClipHoldSec = 1.85f;
constexpr float kPlayerHealthScreenEdgeCritical = 35.f;
constexpr float kPlayerHealthMercyCap = 35.f;
constexpr float kPlayerHealthMercyHealPerSec = 26.f;
constexpr float kPlayerHealthMercyHealDelaySec = 2.f;
constexpr float kPlayerHungerMax = 100.f;
constexpr float kPlayerHungerPizzaGain = 15.f;
constexpr float kPlayerHungerAutoHealPerSec = 5.5f;
constexpr float kPlayerHungerAutoHealThresholdFrac = 0.80f;
constexpr float kPlayerHungerDrainPerSec = 0.25f;
constexpr float kDeliPizzaReplenishSec = 300.f;
constexpr float kDeliFoodPickupRadius = 1.0f;
constexpr float kPlayerScreenDamagePulseRefDmg = 50.f;
constexpr float kPlayerScreenDamagePulseDecayPerSec = 2.05f;
constexpr float kFallDamageEarthG = 9.81f;
constexpr float kFallDamageVelToEarthScale =
    std::sqrt(kFallDamageEarthG / kGravity); // ~0.626: map in-game fall speed to earth-equivalent
// Fall injury model (earth-equivalent m/s throughout):
//   vImpact = max(measured downward speed, √(2·g·dropGeom))  — dropGeom = peak feet → landing support.
//   dv      = max(0, vImpact − vSafe); damage only if dv ≥ minExcess (filters integration noise).
// No separate “min drop” for energy: that used to fight vSafe and made short falls oddly binary.
// Landing tier (0..3): store floor + bottom shelf = low; 2nd / 3rd / 4th wood decks = mid-low / mid / high.
// Crate tops use nearest deck height for tier. See `playerFallDamageTierAtSupport`.
struct PlayerFallDamageTierParams {
  float safeImpactSpeed;
  float kineticScale;
  float singleHitCap;
  float minExcessImpact;
  float jumpArchMinDropM;
};
static constexpr int kPlayerFallDamageTierCount = 4;
static constexpr PlayerFallDamageTierParams kPlayerFallDamageTierParams[kPlayerFallDamageTierCount] = {
    // 1 tier to floor (~3.5m) ≈ negligible; 2 tiers (~6.8m) ≈ 3 HP;
    // top shelf to floor (~10m) ≈ 8 HP; ceiling falls cap at tier limit.
    {7.60f, 0.14f, 16.f, 0.45f, 7.50f},   // 0 low — ground + bottom shelf
    {7.20f, 0.17f, 20.f, 0.40f, 7.00f},   // 1 mid-low — second shelf
    {6.80f, 0.20f, 24.f, 0.35f, 6.50f},   // 2 mid — third shelf
    {6.40f, 0.24f, 30.f, 0.30f, 5.80f},   // 3 high — top shelf
};
// Doc alias: softest tier (floor / bottom deck); geometry selects stricter rows at landing.
constexpr float kPlayerFallDamageMinDvForHit = kPlayerFallDamageTierParams[0].minExcessImpact;
// Landing surface / rim: multipliers blend with excess impact speed (dv). Light landings barely over the
// safe threshold are similar on all surfaces; hard impacts spread concrete vs wood vs cardboard more.
constexpr float kPlayerFallLandHeightMatchEpsM = 0.09f;
constexpr float kPlayerFallLandFloorBandM = 0.055f;
constexpr float kPlayerFallLandEdgeRefM = 0.42f;
// (dv - minDvForSeverity) / this → severity 0..1 for surface/edge curves (earth-equiv m/s).
constexpr float kPlayerFallLandSeveritySpanDv = 12.5f;
constexpr float kPlayerFallLandDeckMulSoft = 0.95f;
constexpr float kPlayerFallLandDeckMulHard = 0.74f;
constexpr float kPlayerFallLandCrateMulSoft = 0.90f;
constexpr float kPlayerFallLandCrateMulHard = 0.58f;
constexpr float kPlayerFallLandEdgeRimSoft = 0.98f;
constexpr float kPlayerFallLandEdgeRimHard = 1.04f;
constexpr float kPlayerFallLandEdgeCenSoft = 0.92f;
constexpr float kPlayerFallLandEdgeCenHard = 0.76f;
// Jump landings: per-tier jumpArchMinDropM — higher shelves use a tighter arch so big drops still chip.
// Landing with tiny +velY from integration noise should not skip damage.
constexpr float kPlayerFallDamageLandVelYMaxEps = 0.07f;
// After HP loss from a fall, brief window where smaller drops (bunny hops / jump chains) deal no extra damage.
constexpr float kPlayerFallDamageChainImmuneSec = 5.2f;
constexpr float kPlayerFallDamageChainImmuneMaxDropM = 6.3f;
// Landings on shelf decks / crate tops with at most this fall distance: no damage (close ledge / low prop).
// Must cover one full rack tier (~kShelfDeckYStepFall); 2.85m alone made “walk down one deck” always injure.
constexpr float kPlayerFallNoDamageShelfOrCrateMaxDropM =
    std::max(3.15f, kShelfDeckYStepFall + 0.85f);
// Aisle runs along +Z: bay centers along the run. Must be **wider** than one rack footprint
// (2*kShelfMeshHalfW) or consecutive units sit edge-on-edge and overlap / z-fight in the screencast.
static constexpr float kShelfBayGapAlongRun = 1.05f;
static constexpr float kShelfAlongAislePitch = 2.0f * kShelfMeshHalfW + kShelfBayGapAlongRun;
// One aisle module = walk space + depth of both facing racks (centers spaced along X).
static constexpr float kShelfAisleModulePitch = kStoreAisleWidth + 2.0f * kShelfMeshHalfD;
// Absolute shelf grid indices overlapping a horizontal circle around (x,z). Avoids O(radius²)
// scans over the whole “infinite” store (was ~44k cells × many calls per frame → slideshow).
static void shelfGridWindowForRange(float x, float z, float rangeM, int& worldAisleMin,
                                    int& worldAisleMax, int& worldAlongMin, int& worldAlongMax) {
  const int aisle0 = static_cast<int>(std::floor(x / kShelfAisleModulePitch));
  const int along0 = static_cast<int>(std::floor(z / kShelfAlongAislePitch));
  const int dAisle = static_cast<int>(std::ceil(rangeM / kShelfAisleModulePitch)) + 1;
  const int dAlong = static_cast<int>(std::ceil(rangeM / kShelfAlongAislePitch)) + 1;
  worldAisleMin = aisle0 - dAisle;
  worldAisleMax = aisle0 + dAisle;
  worldAlongMin = along0 - dAlong;
  worldAlongMax = along0 + dAlong;
}

// Corner posts are 2*kShelfPostGauge thick; decks sit inside inner faces. Same inset on ±Z centers depth.
static constexpr float kShelfPostGauge = 0.055f;
static constexpr float kShelfDeckInset = 2.f * kShelfPostGauge + 0.018f;
static constexpr float kFluorescentGridCell = 17.5f;
static constexpr int kFluorescentGridRadius = 7;
// One vkCmdDraw with instanceCount = visible count (same mesh, per-instance translation).
constexpr uint32_t kMaxFluorescentInstances = 384u;
constexpr uint32_t kMaxPillarInstances = 128u;
constexpr uint32_t kMaxSignInstances = 384u;
// ~fixture bounds under ceiling (distance LOD uses center Y for 3D range to view eye).
static constexpr float kFluorescentCullCenterY = kCeilingY - 0.55f;

// Runtime perf (see loadGamePerfFromEnv): uncapped FPS by default; POTATO tightens draw distance / internal res.
struct GamePerfSettings {
  int fpsCap = 0; // 0 = no main-loop sleep (high refresh on strong GPUs)
  int sceneScalePct = 36;
  float shelfCullHardDist = kShelfCullHardDist;
  float shelfCullNominalDist = kShelfCullNominalDist;
  float shelfGridCpuScanRangeM = 148.f;
  float shelfDetailCullDist = kShelfCullHardDist;
  int fluorescentGridRadius = kFluorescentGridRadius;
  int pillarDrawGridRadius = kPillarDrawGridRadius;
  int signDrawGridRadius = kPillarDrawGridRadius;
  float staffCpuListDist = kStaffCpuListDist;
  /// 0 = use employee_mesh default (20k verts); POTATO sets a lower cap.
  size_t employeeMeshMaxVerts = 0;
  /// Cap for small instanced GLB props (box cutter, rusty pipe). Default 4k; POTATO lowers further.
  size_t propMeshMaxVerts = 4096;
};
static GamePerfSettings gGamePerf;

static void loadGamePerfFromEnv() {
  gGamePerf = GamePerfSettings{};
  if (const char* p = std::getenv("VULKAN_GAME_POTATO")) {
    if ((p[0] == '1' && p[1] == '\0') || std::strcmp(p, "yes") == 0 || std::strcmp(p, "true") == 0) {
      gGamePerf.sceneScalePct = 24;
      gGamePerf.fpsCap = 60;
      gGamePerf.shelfCullHardDist = 116.f;
      gGamePerf.shelfCullNominalDist = 98.f;
      gGamePerf.shelfGridCpuScanRangeM = 108.f;
      gGamePerf.shelfDetailCullDist = 96.f;
      gGamePerf.fluorescentGridRadius = 3;
      gGamePerf.pillarDrawGridRadius = 3;
      gGamePerf.signDrawGridRadius = 2;
      gGamePerf.staffCpuListDist = 100.f;
      gGamePerf.employeeMeshMaxVerts = 12'000;
      gGamePerf.propMeshMaxVerts = 4096;
      std::cerr << "[perf] VULKAN_GAME_POTATO: lower internal res, shorter shelf/staff/light draw distances.\n";
    }
  }
  if (const char* cap = std::getenv("VULKAN_GAME_FPS_CAP")) {
    int v = std::atoi(cap);
    if (v > 0 && v <= 480)
      gGamePerf.fpsCap = v;
  }
  if (const char* sp = std::getenv("VULKAN_GAME_SCENE_SCALE_PCT")) {
    int pct = std::atoi(sp);
    if (pct >= 20 && pct <= 95)
      gGamePerf.sceneScalePct = pct;
  }
  if (const char* sr = std::getenv("VULKAN_GAME_SIGN_GRID_RADIUS")) {
    int r = std::atoi(sr);
    if (r >= 1 && r <= 4)
      gGamePerf.signDrawGridRadius = r;
  }
  if (const char* sgs = std::getenv("VULKAN_GAME_SHELF_GRID_SCAN_M")) {
    float m = static_cast<float>(std::atof(sgs));
    if (m >= 40.f && m <= 180.f)
      gGamePerf.shelfGridCpuScanRangeM = m;
  }
  if (const char* emv = std::getenv("VULKAN_GAME_EMPLOYEE_MESH_MAX_VERTS")) {
    long v = std::strtol(emv, nullptr, 10);
    if (v >= 3000 && v <= 80'000)
      gGamePerf.employeeMeshMaxVerts = static_cast<size_t>(v);
  }
  if (const char* pmv = std::getenv("VULKAN_GAME_PROP_MESH_MAX_VERTS")) {
    long v = std::strtol(pmv, nullptr, 10);
    if (v >= 512 && v <= 65'000)
      gGamePerf.propMeshMaxVerts = static_cast<size_t>(v);
  }
  gGamePerf.propMeshMaxVerts = std::clamp(gGamePerf.propMeshMaxVerts, size_t{512}, size_t{65'000});
  gGamePerf.fluorescentGridRadius =
      std::clamp(gGamePerf.fluorescentGridRadius, 1, 8);
  gGamePerf.pillarDrawGridRadius = std::clamp(gGamePerf.pillarDrawGridRadius, 1, 4);
  gGamePerf.signDrawGridRadius = std::clamp(gGamePerf.signDrawGridRadius, 1, 4);
  gGamePerf.shelfGridCpuScanRangeM = std::clamp(gGamePerf.shelfGridCpuScanRangeM, 40.f, 180.f);
  gGamePerf.shelfDetailCullDist = std::clamp(gGamePerf.shelfDetailCullDist, 40.f, gGamePerf.shelfCullHardDist);
  if (gGamePerf.shelfCullNominalDist >= gGamePerf.shelfCullHardDist)
    gGamePerf.shelfCullNominalDist = gGamePerf.shelfCullHardDist * 0.88f;
}

struct FramePerfTrace {
  bool enabled = false;
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;
  TimePoint lastPrint{};
  double accumUpdate = 0.0, accumStaffCollision = 0.0;
  double accumStaffAIBlend = 0.0, accumStaffAIMain = 0.0;
  double accumRecordCmd = 0.0, accumShelfBuild = 0.0, accumSkinPalette = 0.0;
  double accumShelfOcc = 0.0;
  double accumSignDraws = 0.0;
  uint32_t frameCount = 0, totalVisibleStaff = 0, totalShelfInst = 0;
  uint32_t totalCrateInst = 0, totalNavProbes = 0, totalSupportQ = 0;
  uint32_t globalFrameCounter = 0;
  void init() {
    // VULKAN_GAME_PERF_TRACE=1 — stderr ms/frame (upd/rec/shelf/pal + staff AI splits + shelf occ).
    // Pair with perf(1), RenderDoc, or Tracy for deeper analysis; tune from measured lines, not guesses.
    const char* env = std::getenv("VULKAN_GAME_PERF_TRACE");
    enabled = env && env[0] == '1' && env[1] == '\0';
    lastPrint = Clock::now();
  }
  void endFrame() {
    ++globalFrameCounter;
    if (!enabled) return;
    ++frameCount;
    auto now = Clock::now();
    double elapsed = std::chrono::duration<double>(now - lastPrint).count();
    if (elapsed >= 1.0) {
      const double inv = 1000.0 / static_cast<double>(frameCount);
      std::fprintf(stderr,
          "[perf] FPS:%u  upd:%.2f  ai:%.2f  aibl:%.2f  aimn:%.2f  col:%.2f  rec:%.2f  shelf:%.2f  occ:%.2f  "
          "pal:%.2f  sign:%.2f ms\n"
          "       staff:%u  shelves:%u  crates:%u  navP:%u  supQ:%u\n",
          frameCount,
          accumUpdate * inv, (accumStaffAIBlend + accumStaffAIMain) * inv, accumStaffAIBlend * inv,
          accumStaffAIMain * inv,
          accumStaffCollision * inv, accumRecordCmd * inv, accumShelfBuild * inv, accumShelfOcc * inv,
          accumSkinPalette * inv, accumSignDraws * inv,
          totalVisibleStaff / std::max(1u, frameCount),
          totalShelfInst / std::max(1u, frameCount),
          totalCrateInst / std::max(1u, frameCount),
          totalNavProbes / std::max(1u, frameCount),
          totalSupportQ / std::max(1u, frameCount));
      accumUpdate = accumStaffCollision = 0.0;
      accumStaffAIBlend = accumStaffAIMain = 0.0;
      accumRecordCmd = accumShelfBuild = accumSkinPalette = accumShelfOcc = accumSignDraws = 0.0;
      frameCount = totalVisibleStaff = totalShelfInst = totalCrateInst = 0;
      totalNavProbes = totalSupportQ = 0;
      lastPrint = now;
    }
  }
};
static FramePerfTrace gPerfTrace;

struct PerfScopeTimer {
  double* accum;
  FramePerfTrace::TimePoint start;
  bool active;
  PerfScopeTimer(double* a, bool en)
      : accum(a),
        start(en ? FramePerfTrace::Clock::now() : FramePerfTrace::TimePoint{}),
        active(en) {}
  ~PerfScopeTimer() {
    if (active)
      *accum += std::chrono::duration<double>(FramePerfTrace::Clock::now() - start).count();
  }
};

// Deterministic scramble for infinite-store layout (no floating point).
static uint32_t scp3008ShelfHash(int a, int b, int salt) {
  uint32_t h = static_cast<uint32_t>(a + salt) * 374761393u ^ static_cast<uint32_t>(b) * 668265263u;
  h ^= h >> 13;
  h *= 1274126177u;
  return h ^ (h >> 16);
}

// Per-employee mesh + hitbox scale (stable from spawn key): short / tall / skinny / brute.
static glm::vec3 staffBodyScaleFromKey(uint64_t key) {
  const int ka = static_cast<int>(static_cast<uint32_t>(key >> 32));
  const int kb = static_cast<int>(static_cast<uint32_t>(key & 0xffffffffull));
  const uint32_t h0 = scp3008ShelfHash(ka, kb, 0x5CA1EB0D);
  const uint32_t h1 = scp3008ShelfHash(ka, kb, 0xB0D95501);
  const float u0 = static_cast<float>(h0 & 65535u) / 65535.f;
  const float u1 = static_cast<float>((h0 >> 16) & 65535u) / 65535.f;
  const float u2 = static_cast<float>(h1 & 65535u) / 65535.f;
  const uint32_t archetype = h0 % 4u;
  glm::vec3 s;
  switch (archetype) {
    case 0u: // short
      s = glm::vec3(0.92f + u0 * 0.12f, 0.76f + u1 * 0.14f, 0.90f + u2 * 0.12f);
      break;
    case 1u: // tall
      s = glm::vec3(0.90f + u0 * 0.12f, 1.06f + u1 * 0.16f, 0.88f + u2 * 0.12f);
      break;
    case 2u: // skinny
      s = glm::vec3(0.72f + u0 * 0.12f, 0.94f + u1 * 0.10f, 0.72f + u2 * 0.12f);
      break;
    default: // brute — wide X reads as long-arm / stocky silhouette on one skinned mesh
      s = glm::vec3(1.14f + u0 * 0.16f, 1.02f + u1 * 0.12f, 1.06f + u2 * 0.12f);
      break;
  }
  s.x = glm::clamp(s.x, 0.70f, 1.32f);
  s.y = glm::clamp(s.y, 0.72f, 1.28f);
  s.z = glm::clamp(s.z, 0.70f, 1.32f);
  return s;
}

// Deterministic staff class archetype from resident key:
// 0=short, 1=tall, 2=skinny, 3=brute.
static uint8_t staffClassArchetypeFromKey(uint64_t key) {
  const int ka = static_cast<int>(static_cast<uint32_t>(key >> 32));
  const int kb = static_cast<int>(static_cast<uint32_t>(key & 0xffffffffull));
  const uint32_t h0 = scp3008ShelfHash(ka, kb, 0x5CA1EB0D);
  return static_cast<uint8_t>(h0 % 4u);
}

static float staffClassMaxHp(uint8_t archetype) {
  switch (archetype & 3u) {
    case 0u:
      return 70.f;
    case 1u:
      return 105.f;
    case 2u:
      return 82.f;
    default:
      return 132.f;
  }
}

// Shelf staff: day wander unless shoved in daytime (staffPushAggro → day chase). Lose sight → chill
// immediately; calm timer still forgives while they can see you. Night shove uses staffNightShoveChase +
// staffNightShoveRevealRemain: for kStaffNightShoveRevealSec after a push they track your true XZ (radio /
// “they know”) then normal vision rules return (cleared when fluorescents on for day).
constexpr float kStaffDayPushAggroCalmSec = 24.f;
constexpr float kStaffDayPushAggroFarDistM = 22.f;
constexpr float kStaffDayPushAggroFarCalmMul = 2.1f;
// Skinned staff + local avatar share Meshy clips: walk, lean sprint (replaces run), optional crouch/slide.
constexpr float kShelfEmpWalkSpeed = 0.74f;
constexpr float kStaffGlobalSpeedMul = 1.45f;
constexpr float kStaffShortExtraSpeedMax = 0.22f;
// Steering: blend velocity toward desired direction (smoother turns, less wall sliding).
constexpr float kStaffSteerAccelWalk = 4.35f;
// Below this downward vertical speed (m/s), staff skips auto step-up onto shelf tops while falling — player-like
// ledge step-up rejects strong upward vel; here we avoid “catching” fast falls on deck overlaps.
constexpr float kStaffFallNoAutoStepVelY = -2.75f;
// Chase cap ramp — lower = staff take longer to hit max chase speed.
constexpr float kStaffSteerAccelChase = 21.f;
// When chasing: extra steering accel scales with velocity-vs-desired misalignment (sharp player turns).
constexpr float kStaffChaseMisalignAccelPerRad = 1.05f;
constexpr float kStaffChaseMisalignAccelMaxExtra = 2.05f;
// Yaw follows desired chase direction quickly when turning hard (still uses vel when nearly aligned).
constexpr float kStaffChaseYawFollowHz = 19.f;
// If XZ moves less than this for kShelfEmpStuckWindowS while “should be walking”, pick a new wander target.
constexpr float kShelfEmpStuckMoveEps = 0.26f;
constexpr float kShelfEmpStuckWindowS = 2.75f;
// Night chase: after stuck recovery, briefly steer toward new wander waypoint (ring around player), not through walls.
constexpr float kShelfEmpChaseUnstuckNavS = 2.5f;
constexpr float kShelfEmpChaseUnstuckNavSNight = 1.62f;
constexpr float kShelfEmpStuckWindowNightChaseS = 1.82f;
// Base chase; extra speed/accel when player is on shelves (see shelf chase boost below).
constexpr float kShelfEmpNightChaseSpeed = kSprintSpeed * 0.52f;
// Blackout pursuit only (not day shove chase): staff commit harder — closes distance faster, reads as a hunt.
constexpr float kStaffBlackoutPursuitSpeedMul = 1.12f;
constexpr float kStaffBlackoutPursuitAccelMul = 1.16f;
constexpr float kStaffChaseShelfSpeedBoost = 0.22f;
constexpr float kStaffChaseShelfAccelBoost = 0.34f;
constexpr float kStaffChaseShelfBoostRefHeightM = 5.2f;
constexpr float kStaffChaseClimbVertSpeed = 1.45f;
// Wider than pure melee range so staff still evaluate climb / pathing before they’re on top of you.
constexpr float kStaffChaseClimbMaxHorizToPlayer = 34.f;
// Longer nav probes when you’re on shelves — find gaps between racks toward the aisle under you.
constexpr float kStaffChaseNavLookaheadElevMul = 3.2f;
// Extra along-ray deck samples when vertical chase is urgent (pulling up / big height gap).
constexpr float kStaffChaseRelaxedGrabMultiSampleMul = 2.6f;
// Rises above a normal step use a short “ledge pull-up” (slow horizontal) instead of only vertical creep.
constexpr float kStaffChaseLedgeClimbMinRiseM = 0.085f;
// Fallback wall-clock if ladder clip missing; with clip, duration tracks full climb animation.
constexpr float kStaffChaseLedgeClimbDurationS = 0.98f;
constexpr float kStaffChaseLedgeDurationMul = 1.65f;
// After a mantel, brief cooldown + cap support snap so terrain probe cannot skip multiple tiers in one frame.
constexpr float kStaffChaseMantelCooldownSec = 0.04f;
constexpr float kStaffChasePostMantelSnapSlopM = 0.16f;
// If feet drop this far below the last mantel target deck, they left the ledge — stop post-mantel snap so
// gravity/fall continue and they can trigger a new pull-up (player-like: fall off = climb again).
constexpr float kStaffPostMantelFallOffDropM = 0.32f;
// Strong downward vel while off the mantel tier cancels snap even before drop threshold (first frames off lip).
constexpr float kStaffPostMantelCancelSnapFallVelY = -0.55f;
// Max vertical gain per mantel = one shelf tier (no multi-deck teleport in one clip).
constexpr float kStaffChaseLedgeClimbMaxSingleStepRiseM = kShelfDeckYStepFall + 0.24f;
constexpr float kStaffChaseLedgeClimbMoveMul = 0.56f;
constexpr float kStaffChaseLedgeClimbAccelMul = 0.56f;
// Mantel: jump clip for first slice of wall-clock, then ledge pull-up — u frac tuned for kLedgeGrabDuration
// (same wall-clock as player mantle) so most of the short pull is hands-on-ledge like the player.
constexpr float kStaffChaseMantelJumpAnimUFrac = 0.22f;
constexpr float kStaffChaseClimbPlayerFeetMinAbove = 0.18f;
// While climbing toward an elevated player, drift XZ toward them (m/s) so they mount toward you, not only up.
constexpr float kStaffChaseClimbDriftToPlayerMps = 2.6f;
// terrainSupportY probe: follow player across several shelf tiers.
constexpr float kStaffChaseClimbProbeMaxAboveFeetM = 15.5f;
// Active chasers pull unaware coworkers within this XZ radius into chase (shared “alert”).
constexpr float kStaffChaseNeighborAlertRadiusM = 20.0f;
// Night: after LMB shove, pursuer (and coworkers they alert) “know” the player’s position this long.
constexpr float kStaffNightShoveRevealSec = 10.f;
// Deck tops count as support this far outside the deck footprint (aisle = strict terrain misses shelves).
constexpr float kStaffChaseDeckSnapXZ = 2.65f;
// Ledge grab: only commit mantel when geometry at feet (or a short step toward the player) reaches targetY.
constexpr float kStaffChaseLedgeGrabDeckMatchEpsM = 0.17f;
constexpr float kStaffChaseLedgeGrabProbePadM = 0.28f;
// Forward sample along to-player so staff can start pull-up when sprinting onto a deck lip (not only when stopped).
constexpr float kStaffChaseLedgeGrabLeadMinM = 0.26f;
constexpr float kStaffChaseLedgeGrabLeadMaxM = 0.98f;
constexpr float kStaffChaseLedgeGrabLeadDistFrac = 0.34f;
// Facing / motion: avoid grabs while sideways or sliding away; close range bypasses strict facing.
constexpr float kStaffChaseLedgeGrabCloseBypassM = 1.32f;
constexpr float kStaffChaseLedgeGrabMinFaceCos = 0.08f;
constexpr float kStaffChaseLedgeGrabVelTowardPlayerMin = 0.14f;
// Looser than old tight cap so chasers can snap a mantle while dropping like the player (still not full -27).
constexpr float kStaffChaseLedgeGrabMaxStartVelY = -4.55f;
// After mantel, shorter cooldown if still far below player (chains multi-tier climbs faster).
constexpr float kStaffChaseMantelCooldownChainMul = 0.30f;
constexpr float kStaffChaseMantelChainPlayerFeetGapM = 0.42f;
// Abort pull-up early if deck support under feet disappears (player moved / bad approach).
constexpr float kStaffChaseLedgeGrabAbortEarlyU = 0.17f;
// Multi-sample lip search along to-player (aisle → shelf lip is often >1 m; old single-lead max was ~0.98 m).
constexpr float kStaffChaseLedgeGrabMultiSampleStepM = 0.30f;
constexpr float kStaffChaseLedgeGrabMultiSampleMaxM = 6.4f;
constexpr float kStaffChaseRunnerLedgeGrabMultiSampleMaxM = 7.2f;
// Chase “runner zombie” mantel (Dying Light–style): sprint in, run-jump into hands, faster pull, wider reach.
constexpr float kStaffChaseRunnerGrabMinHorizSpeed = 1.22f;
constexpr float kStaffChaseRunnerDriftMul = 1.10f;
constexpr float kStaffChaseRunnerLedgeMoveMul = 0.82f;
constexpr float kStaffChaseRunnerLedgeAccelMul = 0.82f;
constexpr float kStaffChaseRunnerLeadMaxM = 1.32f;
constexpr float kStaffChaseRunnerLeadMinM = 0.32f;
constexpr float kStaffChaseRunnerLeadDistFrac = 0.46f;
constexpr float kStaffChaseRunnerVelTowardPlayerMin = 0.095f;
constexpr float kStaffChaseRunnerGrabCloseBypassM = 1.62f;
constexpr float kStaffChaseRunnerMantelJumpUFrac = 0.18f;
constexpr float kStaffChaseRunnerMantelWallClockMul = 1.40f;
constexpr float kShelfEmpWanderReachEps = 0.38f;
// Simulation prune — large so chasing staff are not erased in an endless store (was 340 → visible depop).
constexpr float kShelfEmpPruneDist = 920.f;
// pairwise staff separation is O(n²); only NPCs within this radius of the player take part (sim is
// irrelevant far off-screen and the pruned map can hold hundreds of bay workers after long walks).
constexpr float kStaffSepMaxDistFromPlayer = 340.f;
constexpr float kStaffNightAlertMaxDistFromPlayer = 380.f;
// Night stealth: unaware until player is in forward vision cone (range + FOV) or very close behind.
// Then idle facing player; after kNightStaffChaseDelayS of continuous detection, chase.
constexpr float kNightStaffVisionRange = 17.4f;
constexpr float kNightStaffVisionCosHalfFov = 0.565f; // wider than ~50° — fewer “blind” sidesteps
// While chasing: wider / longer sight so staff do not drop the player at patrol cone limits.
constexpr float kNightStaffChaseVisionRange = 34.0f;
constexpr float kNightStaffChaseVisionCosHalfFov = 0.34f;
constexpr float kNightStaffChaseBehindSenseM = 2.5f;
// Eyes ~ on the floor plane; do not spot someone far overhead (high shelves / mantle).
constexpr float kNightStaffVisionEyeY = kGroundY + 1.52f;
constexpr float kNightStaffVisionMaxElevAboveDeg = 38.f;
// While chasing: look much further up so elevated shelf lanes do not break lock.
constexpr float kNightStaffChaseVisionMaxElevAboveDeg = 89.f;
// Feet this far above nominal floor: treat as shelf/ledge — within visRange, allow steeper look-up so
// short-range chase under the player does not fail atan2(dy, distXZ) vs a modest deg cap.
constexpr float kNightStaffVisionPlayerLedgedFeetAboveFloorM = 0.46f;
constexpr float kNightStaffVisionLedgedMaxElevAboveDeg = 89.f;
// Yaw follows velocity while steering; cone vs forward can miss the player while still moving toward them.
constexpr float kNightStaffChaseVisionVelConeMinSpeed = 0.034f;
constexpr float kNightStaffBehindSenseM = 1.24f;
// Night: continuous vision this long (phase 1 “watch”) before chase; lose sight → investigate last known → passive.
// Crouch stealth: scale vision range / behind-sense when player is crouched.
constexpr float kCrouchStealthVisionRangeMul = 0.42f;
constexpr float kCrouchStealthBehindSenseMul = 0.30f;
constexpr float kCrouchStealthChaseDelayMul = 1.8f;
constexpr float kNightStaffChaseDelayS = 2.2f;
// After losing sight (watching or chasing), walk toward last known player XZ for a short time.
constexpr float kNightStaffInvestigateReachM = 0.52f;
constexpr float kNightStaffInvestigateMaxS = 24.f;
constexpr float kNightStaffInvestigateSpeedMul = 1.14f;
// Lookahead obstacle avoidance: probe hitbox ahead; fan to side angles if blocked.
constexpr float kStaffNavLookahead = 0.68f;
constexpr int kStaffWanderClearMaxAttempts = 10;
// Night chase: stop and melee when this close; resume chase when farther.
constexpr float kStaffMeleePunchHoldRange = 1.12f;
constexpr float kStaffMeleePunchReleaseRange = 1.48f;
// Punch clip normalized phase [u0,u1] where contact counts (looping clip — i-frames prevent multi-tick spam).
constexpr double kStaffMeleeDamagePhaseU0 = 0.06;
constexpr double kStaffMeleeDamagePhaseU1 = 0.44;
constexpr float kStaffMeleePlayerDamage = 17.f;
// Body overlap while hostile (chase / day aggro): hurts if they’re moving or in punch stance — covers “bump” hits.
constexpr float kStaffContactPlayerDamage = 15.f;
constexpr float kStaffContactHitMinHorizSpeed = 0.28f;
constexpr float kStaffMeleePlayerInvulnSec = 0.52f;
// Horizontal bubble around the Shrek egg: staff melee off, proximity dance, full AI freeze + face egg.
constexpr float kStaffShrekProximityDanceRadiusM = 13.5f;
constexpr float kStaffMeleeSuppressedNearShrekRadiusM = kStaffShrekProximityDanceRadiusM;
constexpr float kStaffShoveMaxDist = 2.75f;
constexpr float kStaffShoveCosCone = 0.38f;
constexpr float kStaffShovePlayerRecoil = 1.15f;
// Horizontal knockback while hair / fall / stand clips play (decays each tick).
constexpr float kStaffShoveKnockbackSpeed = 3.15f;
constexpr float kStaffShoveKnockbackDecay = 5.0f;
// Landing on a staff hitbox from above: same knockdown/aggro as LMB shove (feet band around cap Y).
constexpr float kPlayerStaffBodySlamFeetBelowTopM = 0.44f;
constexpr float kPlayerStaffBodySlamFeetAboveTopM = 0.52f;
constexpr float kPlayerStaffBodySlamMaxVelY = 0.62f;
constexpr float kPlayerStaffBodySlamCooldownSec = 0.55f;
constexpr float kCrosshairShoveAnimDur = 0.13f;
constexpr float kCrosshairBaseScale = 0.56f;
constexpr float kMenuCursorScale = 0.50f;
// >1 runs step-push clip faster than wall-clock (playerPushAnimRemain countdown).
constexpr float kPushAnimPlaybackScale = 22.f;
constexpr float kKickAnimPlaybackScale = 2.5f;
constexpr float kDropKickMaxAirSec = 8.f;
constexpr float kDropKickGroundSlideSec = 0.8f;
constexpr float kDropKickGroundSlideDecay = 4.f;
constexpr float kDropKickLungeSpeed = 20.f;
constexpr float kDropKickMaxDist = 4.2f;
constexpr float kDropKickCosCone = 0.42f;
constexpr float kDropKickGravityScale = 0.70f;
constexpr float kDropKickMinRunSpeed = 2.5f;
constexpr float kDropKickKnockbackSpeed = 13.f;
constexpr float kDropKickNpcVerticalPop = 3.8f;
constexpr float kDropKickImpactPulse = 0.38f;
constexpr float kStaffHitDamageShove = 38.f;
constexpr float kStaffHitDamageKick = 12.f;
constexpr float kStaffHitDamageDropKick = 28.f;
constexpr float kStaffHitDamageBodySlam = 34.f;
// Crossfade when entering melee / changing knockdown clips (seconds).
constexpr float kStaffMeleeBlendSec = 0.22f;
// Fall / get-up clips often leave the torso/back well above the rig foot pivot when prone — sink draw Y.
// Scaled part tracks bodyScale.y; world bias keeps short archetypes from hovering (sy can be ~0.76).
constexpr float kStaffMeleeFallFeetSinkMax = 0.82f;
constexpr float kStaffMeleeFallFeetSinkEnd = 0.30f;
constexpr float kStaffMeleeFallFeetSinkWorldBias = 0.11f;
// Dead staff: tiny nudge vs z-fight only — large values float the corpse above support (see feet sink).
constexpr float kStaffRagdollBindVisualLiftY = 0.038f;
// Dead ragdoll uses bind pose + sim (no fall clip): light sink — mesh feet sit near bind origin.
constexpr float kStaffDeadRagdollBindFeetSink = 0.085f;
// Multi-point corpse vs terrain: lift feet so probed skin shell clears deck (meters, scales with bodyScale).
constexpr float kDeadCorpseGroundProbeSkinPad = 0.058f;
// Neck-only joint extras stay tight so PBD doesn’t corkscrew the head off the torso.
constexpr float kDeadRagdollNeckEulerClamp = 0.34f;
constexpr float kDeadRagdollHeadEulerClamp = 0.48f;
// Spine / pelvis: tiny euler band — large offsets read as impossible folds or “spikes” through the floor.
constexpr float kDeadRagdollCoreEulerClamp = 0.36f;
constexpr float kStaffMeleeHairFeetSink = 0.30f;
constexpr float kStaffMeleeHairFeetSinkWorldBias = 0.06f;
constexpr float kStaffMeleeStandFeetSinkStart = 0.30f;  // match kStaffMeleeFallFeetSinkEnd at fall→stand
// Third-person avatar: distance-synced loco; shorter crossfade = fluid parkour-style clip changes.
constexpr float kAvatarAnimPlaybackScale = 0.98f;
constexpr float kAvatarClipBlendSec = 0.22f;
// FP body: don’t rotate arms/torso 1:1 with view pitch — keeps lower-FOV hands calmer.
constexpr float kFpBodyPitchFollow = 0.65f;
constexpr float kFpBodyPitchMaxTilt = glm::radians(35.0f);
constexpr float kFpAvatarYawSmoothHz = 18.0f;
// Meters per foot event (must match footstep stride below): one full anim loop ≈ two steps.
constexpr float kAvatarStrideWalkM = 4.25f;
constexpr float kAvatarStrideRunM = 5.85f;
// Low-pass |v.xz| before mapping to stride phase (reduces walk/run cadence chatter from friction).
constexpr float kAvatarHorizSpeedSmoothHz = 12.f;
constexpr float kStaffNpcAnimPlaybackScale = 0.78f;
// Footsteps: stride matches staff walk/run speeds; vol falls off with horizontal distance to player.
constexpr float kStaffNpcFootstepStrideWalkM = 4.05f;
constexpr float kStaffNpcFootstepStrideRunM = 5.45f;
constexpr float kStaffNpcFootstepMinMoveSpeed = 0.17f;
constexpr int kStaffNpcFootstepsMaxPerFrame = 6;
constexpr float kStaffNpcFootstepHearRadiusM = 46.f;
constexpr float kStaffNpcFootstepBaseVolMul = 0.5f;
constexpr float kStaffNightPursuitFootstepVolMul = 1.45f;
constexpr float kStaffNightPursuitFootstepHearRadiusMul = 1.22f;

struct ShelfEmployeeNpc {
  glm::vec2 posXZ{0.f};
  float feetWorldY = kGroundY;
  float yaw = 0.f;
  glm::vec2 wanderTargetXZ{0.f};
  float aisleCenterX = 0.f;
  float aisleCenterZ = 0.f;
  float roamHalfX = 1.f;
  float roamHalfZ = 1.2f;
  bool inited = false;
  uint64_t residentKey = 0;
  uint32_t wanderSalt = 0u;
  float lastHorizSpeed = 0.f;
  glm::vec2 velXZ{0.f};
  // Night only: 0 = unaware, 1 = watching (idle, face player), 2 = chasing, 3 = investigate last known.
  uint8_t nightPhase = 0;
  float nightSpotTimer = 0.f;
  glm::vec2 nightLastKnownPlayerXZ{0.f};
  float nightInvestigateTimer = 0.f;
  glm::vec2 stuckRefXZ{0.f};
  float stuckTimer = 0.f;
  float chaseUnstuckTimer = 0.f;
  // Melee: 0 = normal, 1 = punching/kicking, 2 = knocked down, 3 = standing up,
  // 4 = hair/head shove wind-up (plays before fall when hair clip is loaded).
  uint8_t meleeState = 0;
  uint8_t meleeAttackPick = 0;
  double meleePhaseSec = 0.0;
  // 1 = no blend; <1 = lerp from snapshot clip/phase toward current draw clip.
  float meleeAnimBlend = 1.f;
  int meleeAnimFromClip = 0;
  double meleeAnimFromPhase = 0.0;
  uint8_t meleeAnimFromLoop = 1;
  glm::vec2 posXZPreResolve{0.f};
  glm::vec3 bodyScale{1.f, 1.f, 1.f};
  uint8_t staffClassArchetype = 0;
  float staffHp = 0.f;
  float staffHpMax = 0.f;
  bool staffDead = false;
  // Dead-only secondary body tilt (pseudo-ragdoll): damped pitch/roll from impacts while corpse slides.
  float deadRagdollPitch = 0.f;
  float deadRagdollRoll = 0.f;
  float deadRagdollPitchVel = 0.f;
  float deadRagdollRollVel = 0.f;
  float deadRagdollYaw = 0.f;
  float deadRagdollYawVel = 0.f;
  // Simple joint “physics”: euler offsets (rad) on a subset of bones, layered on fall-clip skin.
  bool deadRagdollJointSimInited = false;
  uint8_t deadRagdollJointSimCount = 0u;
  glm::vec3 deadRagdollJointEuler[kStaffRagdollSimMaxBones]{};
  glm::vec3 deadRagdollJointVel[kStaffRagdollSimMaxBones]{};
  // World-space PBD particles (corpse-only): distance constraints + ground, feedback into joint eulers.
  bool deadRagdollPbdInited = false;
  glm::vec3 deadRagdollPbdPosW[kStaffRagdollSimMaxBones]{};
  glm::vec3 deadRagdollPbdVelW[kStaffRagdollSimMaxBones]{};
  // Wall-clock target for shove hair = player step-push clip (sync knockdown to push).
  float shovePlayerPushDurSec = 0.f;
  glm::vec2 staffShoveKnockbackVelXZ{0.f};
  // Support probe height when knocked down — avoids upright chase probe snapping feet to a wrong shelf tier.
  float meleeKnockdownFeetAnchorY = kGroundY;
  // Shoved while store is lit: chase player during the day until calm timer expires (or pruned / far).
  bool staffPushAggro = false;
  float staffPushAggroCalmRemain = 0.f;
  // Shoved during blackout: chase at night without vision; cleared at dawn — no carry-over day chase.
  bool staffNightShoveChase = false;
  // Counts down only at night; while > 0, shove-pursuit ignores the vision cone (last known = true player XZ).
  float staffNightShoveRevealRemain = 0.f;
  // Night chase: mantel-style climb onto shelf tier (remaining seconds; <0 = idle).
  float chaseLedgeClimbRem = -1.f;
  float chaseLedgeClimbTotalDur = 0.f;
  float chaseLedgeClimbY0 = 0.f;
  float chaseLedgeClimbY1 = 0.f;
  float staffChaseMantelCooldownRem = 0.f;
  float staffLastMantelTargetY = kGroundY;
  // 0 = use full ladder clip length for palette phase during mantel; else cap (seconds) for short rises.
  float staffMantelAnimPhaseSpanSec = 0.f;
  // Chase: started mantel while sprinting toward player — run-jump segment, faster climb, stronger drift.
  uint8_t staffMantelRunnerChase = 0;
  glm::vec2 staffFootstepPrevXZ{0.f};
  bool staffFootstepHavePrev = false;
  float staffFootstepAccum = 0.f;
  float staffVelY = 0.f;
  // Air fall: jump clip timeline after leaving support (mirrors player jump remain / fall pose).
  float staffAirLocoRemain = 0.f;
  int staffAirFallClip = -1;
  // Grounded: hold end of jump clip after landing (walk-off or jump arc), like playerJumpPostLandRemain.
  float staffAirLandRemain = 0.f;
  int staffAirLandClip = -1;
  bool staffGroundedPrev = true;
  // Tall ledge fall: track max feet Y while airborne for landing ragdoll (melee fall clip).
  float staffFallPeakFeetY = kGroundY;
  uint8_t staffTallFallKnockdownPending = 0;
};

static void shelfEmpResetDeadJointSim(ShelfEmployeeNpc& e) {
  e.deadRagdollJointSimInited = false;
  e.deadRagdollJointSimCount = 0;
  e.deadRagdollPbdInited = false;
  for (int j = 0; j < kStaffRagdollSimMaxBones; ++j) {
    e.deadRagdollJointEuler[j] = glm::vec3(0.f);
    e.deadRagdollJointVel[j] = glm::vec3(0.f);
    e.deadRagdollPbdPosW[j] = glm::vec3(0.f);
    e.deadRagdollPbdVelW[j] = glm::vec3(0.f);
  }
}

// Patrol the open store: next waypoint is on a random ring around anchor (player), not the home bay.
static constexpr float kShelfEmpWanderRingMinM = 6.f;
static constexpr float kShelfEmpWanderRingMaxM = 920.f;

// Tight patrol in the home shelf bay (aisle slot). Night calm wander uses store-wide ring like day.
static void shelfEmpPickWanderLocalBay(ShelfEmployeeNpc& e, uint64_t key) {
  e.wanderSalt++;
  const int ka = static_cast<int>(static_cast<uint32_t>(key >> 32));
  const int kb = static_cast<int>(static_cast<uint32_t>(key & 0xffffffffull));
  uint32_t h = scp3008ShelfHash(ka, kb, static_cast<int>(e.wanderSalt ^ 0x51A11Bu));
  const float u1 = static_cast<float>(h & 65535u) / 65535.f;
  const float u2 = static_cast<float>((h >> 16) & 65535u) / 65535.f;
  // Bias along the aisle (+Z) so patrols read as walking the bay, not jittering in place.
  const float jx = (u1 * 2.f - 1.f) * e.roamHalfX * 0.98f;
  const float jz = (u2 * 2.f - 1.f) * e.roamHalfZ * 4.45f;
  e.wanderTargetXZ = glm::vec2(e.aisleCenterX + jx, e.aisleCenterZ + jz);
}

static void staffIntegrateSteering(ShelfEmployeeNpc& e, float dt, const glm::vec2& desiredDir2D, float maxSpeed,
                                   float accel, bool chaseFollowSharpTurns = false) {
  const float d2 = glm::dot(desiredDir2D, desiredDir2D);
  const glm::vec2 dir =
      d2 > 1e-8f ? desiredDir2D * (1.f / std::sqrt(d2)) : glm::vec2(std::sin(e.yaw), std::cos(e.yaw));
  float accelUse = accel;
  if (chaseFollowSharpTurns) {
    const float vlen0 = glm::length(e.velXZ);
    float misalignRad = 0.f;
    if (vlen0 > 0.045f) {
      const glm::vec2 vdn = e.velXZ * (1.f / vlen0);
      const float c = glm::clamp(glm::dot(vdn, dir), -1.f, 1.f);
      misalignRad = std::acos(c);
    } else
      misalignRad = glm::half_pi<float>() * 0.55f;
    accelUse *= 1.f + glm::min(misalignRad * kStaffChaseMisalignAccelPerRad, kStaffChaseMisalignAccelMaxExtra);
  }
  const glm::vec2 targetVel = dir * maxSpeed;
  glm::vec2 delta = targetVel - e.velXZ;
  const float step = accelUse * dt;
  const float dl = glm::length(delta);
  if (dl > step && dl > 1e-6f)
    delta *= step / dl;
  e.velXZ += delta;
  float vl = glm::length(e.velXZ);
  if (vl > maxSpeed)
    e.velXZ *= maxSpeed / std::max(vl, 1e-6f);
  e.posXZ += e.velXZ * dt;
  vl = glm::length(e.velXZ);
  if (chaseFollowSharpTurns) {
    const float yawGoal = std::atan2(dir.x, dir.y);
    float dy = yawGoal - e.yaw;
    while (dy > glm::pi<float>())
      dy -= glm::two_pi<float>();
    while (dy < -glm::pi<float>())
      dy += glm::two_pi<float>();
    const float ad = glm::abs(dy);
    const float urgency = glm::clamp(ad * (2.1f / glm::pi<float>()), 0.18f, 1.f);
    e.yaw += dy * glm::min(1.f, kStaffChaseYawFollowHz * dt * urgency);
  } else if (vl > 0.035f)
    e.yaw = std::atan2(e.velXZ.x, e.velXZ.y);
  e.lastHorizSpeed = vl;
}

static float staffSpeedScaleForBody(const ShelfEmployeeNpc& e) {
  const float shortT = glm::clamp((1.f - e.bodyScale.y) / 0.28f, 0.f, 1.f);
  return kStaffGlobalSpeedMul + kStaffShortExtraSpeedMax * shortT;
}

static void shelfEmpPickWanderStoreWide(ShelfEmployeeNpc& e, uint64_t key, const glm::vec2& anchorXZ) {
  e.wanderSalt++;
  const int ka = static_cast<int>(static_cast<uint32_t>(key >> 32));
  const int kb = static_cast<int>(static_cast<uint32_t>(key & 0xffffffffull));
  uint32_t h = scp3008ShelfHash(ka, kb, static_cast<int>(e.wanderSalt ^ 0xA11C0EEu));
  const float u1 = static_cast<float>(h & 65535u) / 65535.f;
  const float u2 = static_cast<float>((h >> 16) & 65535u) / 65535.f;
  const float theta = u1 * glm::two_pi<float>();
  const float rad = kShelfEmpWanderRingMinM + u2 * (kShelfEmpWanderRingMaxM - kShelfEmpWanderRingMinM);
  e.wanderTargetXZ = anchorXZ + glm::vec2(std::cos(theta), std::sin(theta)) * rad;
}

// patrolLocalBay: implemented as App::shelfEmpStepWanderTowardTarget (needs nav helpers on App).

static int shelfBiomeClusterCoord(int worldI, int span) {
  if (worldI >= 0)
    return worldI / span;
  return -((-worldI + span - 1) / span);
}

static bool deliClusterBase(int ca, int cl, int& baseA, int& baseL) {
  const uint32_t h = scp3008ShelfHash(ca, cl, 0xDE11B10E);
  if ((h % 8u) != 0u) return false;
  const uint32_t h2 = scp3008ShelfHash(ca, cl, 0xDE110FF5);
  constexpr int margin = 2;
  constexpr int range = kDeliBiomeClusterSpan - 2 * margin;
  static_assert(range >= 2, "cluster span too small for deli 2x2 + margin");
  baseA = ca * kDeliBiomeClusterSpan + margin + static_cast<int>(h2 % static_cast<uint32_t>(range));
  baseL = cl * kDeliBiomeClusterSpan + margin + static_cast<int>((h2 >> 8) % static_cast<uint32_t>(range));
  // Keep deli clusters farther from spawn/origin so edge-on deli food cannot leak into the spawn view.
  constexpr int kVoid = 10;
  for (int da = 0; da < 2; ++da)
    for (int dl = 0; dl < 2; ++dl)
      if (std::max(std::abs(baseA + da), std::abs(baseL + dl)) <= kVoid)
        return false;
  return true;
}

static bool cellInDeliClearZone(int worldAisleI, int worldAlongI) {
  const int ca = shelfBiomeClusterCoord(worldAisleI, kDeliBiomeClusterSpan);
  const int cl = shelfBiomeClusterCoord(worldAlongI, kDeliBiomeClusterSpan);
  int baseA, baseL;
  if (!deliClusterBase(ca, cl, baseA, baseL)) return false;
  const int da = worldAisleI - baseA;
  const int dl = worldAlongI - baseL;
  return da >= -kDeliClearPad && da <= 1 + kDeliClearPad &&
         dl >= -kDeliClearPad && dl <= 1 + kDeliClearPad;
}

// SCP-3008–style generation: no outer boundary — store “goes on forever” in fog.
// 75% / 25% is per cluster (kShelfBiomeClusterSpan² bays). Defined after pillarCollisionAABB
// so we can reject racks that intersect structural pillars.
static bool shelfSlotOccupied(int worldAisleI, int worldAlongI, int side);
static bool deliBarSlotOccupied(int worldAisleI, int worldAlongI);
// Rack-local: crate bottom on deck top yDeckTop; half-extents hx,hy,hz. ~1/17 occupied racks.
static bool shelfCrateLocalLayout(int worldAisleI, int worldAlongI, int side, float& lx, float& lz,
                                  float& yDeckTop, float& hx, float& hy, float& hz);
template <typename Fn>
static void forEachShelfPalletStackBoxLocal(int worldAisleI, int worldAlongI, int side, bool cratePresent,
                                            Fn&& fn);
// Floor ladder leaning into rack (~1/71 bays vs ~1/17 for crates).
static bool shelfLadderLocalLayout(int worldAisleI, int worldAlongI, int side, float& lx, float& lz,
                                   float& leanRad, glm::vec3& scaleWHD);
// Shelf deck box cutter (~1/263 of non-ladder bays vs ~1/71 for ladders).
static bool shelfBoxCutterLocalLayout(int worldAisleI, int worldAlongI, int side, float& lx, float& lz,
                                      float& yDeckTop, float& yawLocal, float& uniformScale);
// Shelf deck rusty pipe (~1/97 of non-ladder bays — more common than box cutter).
static bool shelfRustyPipeLocalLayout(int worldAisleI, int worldAlongI, int side, float& lx, float& lz,
                                      float& yDeckTop, float& yawLocal, float& uniformScale);

// [-1, 1] per bay — varies the distance cutoff so the rim isn’t a perfect circle.
static float shelfCullRimJitter(int worldAisleI, int worldAlongI) {
  uint32_t h = uint32_t(worldAisleI) * 73856093u;
  h ^= uint32_t(worldAlongI) * 19349663u;
  h ^= h >> 13;
  h *= 1274126177u;
  h ^= h >> 16;
  return (h & 0xFFFFu) * (2.0f / 65535.0f) - 1.0f;
}

// Conservative half-space cull (not full frustum): drops instances clearly *behind* the camera to cut
// CPU instance fills and GPU overdraw. XZ plane uses horizontal view dir so steep pitch doesn’t
// thrash culling (same idea as Shrek-egg facing).
constexpr float kLodBehindMarginShelf = 11.f;
constexpr float kLodBehindMarginPillar = 9.f;
constexpr float kLodBehindMarginFluor = 16.f;
constexpr float kLodBehindMarginSign = 10.f;
constexpr float kLodBehindMarginStaff = 7.5f;

static inline glm::vec2 lodHorizViewDir(const glm::vec3& viewFwd) {
  glm::vec2 h(viewFwd.x, viewFwd.z);
  const float ls = h.x * h.x + h.y * h.y;
  if (ls > 1e-12f) {
    const float inv = 1.f / std::sqrt(ls);
    return h * inv;
  }
  return glm::vec2(0.f, 1.f);
}

static inline bool lodBehindCameraXZ(float eyeX, float eyeZ, float wx, float wz, const glm::vec2& hfwd,
                                     float margin) {
  return (wx - eyeX) * hfwd.x + (wz - eyeZ) * hfwd.y < -margin;
}

static inline bool lodBehindCamera3D(const glm::vec3& eye, const glm::vec3& world, const glm::vec3& viewFwd,
                                       float margin) {
  return glm::dot(world - eye, viewFwd) < -margin;
}

static bool pointInShelfLocalXZ(const glm::vec3& shelfPos, float shelfYawRad, float wx, float wz,
                                float mnX, float mnZ, float mxX, float mxZ) {
  const glm::vec3 delta{wx - shelfPos.x, 0.f, wz - shelfPos.z};
  const glm::mat3 Rinv =
      glm::transpose(glm::mat3(glm::rotate(glm::mat4(1.f), shelfYawRad, glm::vec3(0.f, 1.f, 0.f))));
  const glm::vec3 L = Rinv * delta;
  return L.x >= mnX && L.x <= mxX && L.z >= mnZ && L.z <= mxZ;
}

// 2D distance in shelf-local XZ from world point to axis-aligned rectangle (0 if inside).
static float distWorldXZToShelfLocalRect(const glm::vec3& shelfPos, float shelfYawRad, float wx, float wz,
                                         float mnX, float mnZ, float mxX, float mxZ) {
  const glm::vec3 delta{wx - shelfPos.x, 0.f, wz - shelfPos.z};
  const glm::mat3 Rinv =
      glm::transpose(glm::mat3(glm::rotate(glm::mat4(1.f), shelfYawRad, glm::vec3(0.f, 1.f, 0.f))));
  const glm::vec3 L = Rinv * delta;
  float dx = 0.f, dz = 0.f;
  if (L.x < mnX)
    dx = mnX - L.x;
  else if (L.x > mxX)
    dx = L.x - mxX;
  if (L.z < mnZ)
    dz = mnZ - L.z;
  else if (L.z > mxZ)
    dz = L.z - mxZ;
  return std::sqrt(dx * dx + dz * dz);
}

// Shelf-local XZ for a world foot position (same basis as pointInShelfLocalXZ).
static glm::vec2 worldToShelfLocalXZ(const glm::vec3& shelfPos, float shelfYawRad, float wx, float wz) {
  const glm::vec3 delta{wx - shelfPos.x, 0.f, wz - shelfPos.z};
  const glm::mat3 Rinv =
      glm::transpose(glm::mat3(glm::rotate(glm::mat4(1.f), shelfYawRad, glm::vec3(0.f, 1.f, 0.f))));
  const glm::vec3 L = Rinv * delta;
  return {L.x, L.z};
}

// Distance from an interior point to the nearest axis-aligned edge in shelf-local XZ (0 if outside).
static float shelfLocalInteriorDistToRectEdge(float lx, float lz, float mnX, float mnZ, float mxX, float mxZ) {
  if (lx < mnX || lx > mxX || lz < mnZ || lz > mxZ)
    return 0.f;
  return std::min(std::min(lx - mnX, mxX - lx), std::min(lz - mnZ, mxZ - lz));
}

// Spatial cache for terrainSupportY (O(racks) per miss). Persists across frames: procedural layout is
// fixed per (x,z,feet band), and invalidating every frame made exploration/update hitch badly.
static constexpr int kTerrainYCacheSlots = 2048;
struct TerrainYCacheSlot {
  bool valid = false;
  int qx = 0;
  int qz = 0;
  int qf = 0;
  float y = 0.f;
};
static TerrainYCacheSlot gTerrainYCache[kTerrainYCacheSlots]{};

// Highest walkable support under (x,z) for grounding / snap (ground + shelf deck tops in range).
float terrainSupportY(float x, float z, float feetY) {
  const int qx = static_cast<int>(std::floor(x * 4.f));
  const int qz = static_cast<int>(std::floor(z * 4.f));
  // ~0.1 m buckets: view-bob was invalidating finer keys and forcing full shelf scans every tick.
  const int qf = static_cast<int>(std::floor(feetY * 10.f));
  {
    const size_t hi =
        (static_cast<size_t>(static_cast<uint32_t>(qx) * 1664525u ^
                              static_cast<uint32_t>(qz) * 1013904223u ^
                              static_cast<uint32_t>(qf) * 374761393u) &
        (static_cast<size_t>(kTerrainYCacheSlots) - 1u));
    const TerrainYCacheSlot& s = gTerrainYCache[hi];
    if (s.valid && s.qx == qx && s.qz == qz && s.qf == qf)
      return s.y;
  }
  float best = kGroundY;
  constexpr float kFeetBand = 0.12f;
  constexpr float kCullR2 = 15.5f * 15.5f;
  constexpr float kGridRangeM = 19.f;
  int waMin, waMax, wlMin, wlMax;
  shelfGridWindowForRange(x, z, kGridRangeM, waMin, waMax, wlMin, wlMax);
  for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
    const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
    const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
    const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
    for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
      const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
      for (int side = 0; side < 2; ++side) {
        if (!shelfSlotOccupied(worldAisle, worldAlong, side))
          continue;
        const float cx = side ? cxRight : cxLeft;
        const float yawDeg = side ? -90.0f : 90.0f;
        const float dx = x - cx;
        const float dz = z - cz;
        if (dx * dx + dz * dz > kCullR2)
          continue;
        const glm::vec3 shelfPos{cx, kGroundY, cz};
        const float shelfYawRad = glm::radians(yawDeg);
        const float hw = kShelfMeshHalfW;
        const float hd = kShelfMeshHalfD;
        const float shelfT = kShelfDeckThickness;
        const int numShelves = kShelfDeckCount;
        constexpr float yBase = 0.12f;
        const float yStep = kShelfGapBetweenLevels + shelfT;
        const float mnX = -hw + kShelfDeckInset;
        const float mnZ = -hd + kShelfDeckInset;
        const float mxX = hw - kShelfDeckInset;
        const float mxZ = hd - kShelfDeckInset;
        for (int si = 0; si < numShelves; ++si) {
          const float y0 = yBase + static_cast<float>(si) * yStep;
          const float y1 = y0 + shelfT;
          if (!pointInShelfLocalXZ(shelfPos, shelfYawRad, x, z, mnX, mnZ, mxX, mxZ))
            continue;
          const float top = kGroundY + y1;
          if (top <= feetY + kFeetBand)
            best = std::max(best, top);
        }
        float clx, clz, yDeck, chx, chy, chz;
        const bool hasCrate = shelfCrateLocalLayout(worldAisle, worldAlong, side, clx, clz, yDeck, chx, chy, chz);
        if (hasCrate) {
          if (pointInShelfLocalXZ(shelfPos, shelfYawRad, x, z, clx - chx, clz - chz, clx + chx,
                                  clz + chz)) {
            const float ctop = kGroundY + yDeck + 2.f * chy;
            if (ctop <= feetY + kFeetBand)
              best = std::max(best, ctop);
          }
        }
        forEachShelfPalletStackBoxLocal(worldAisle, worldAlong, side, hasCrate,
                                        [&](float mnx, float /*y0*/, float mnz, float mxx, float y1, float mxz) {
                                          if (!pointInShelfLocalXZ(shelfPos, shelfYawRad, x, z, mnx, mnz, mxx, mxz))
                                            return;
                                          const float ptop = kGroundY + y1;
                                          if (ptop <= feetY + kFeetBand)
                                            best = std::max(best, ptop);
                                        });
      }
    }
  }
  {
    const float deliCounterTop = kGroundY + kDeliCounterH + kDeliCounterTopH;
    const float deliRoofTop = kGroundY + kDeliCounterH + kDeliGlassHeight;
    const int daMin = static_cast<int>(std::floor((x - 3.f) / kShelfAisleModulePitch));
    const int daMax = static_cast<int>(std::ceil((x + 3.f) / kShelfAisleModulePitch));
    const int dlMin = static_cast<int>(std::floor((z - 3.f) / kShelfAlongAislePitch));
    const int dlMax = static_cast<int>(std::ceil((z + 3.f) / kShelfAlongAislePitch));
    const float hl = kDeliCounterHalfL + kDeliCounterTopOverhang;
    const float hd = kDeliCounterHalfD + kDeliCounterTopOverhang;
    const float ghl = kDeliCounterHalfL - 0.04f;
    const float innerZ = kDeliCounterHalfD - 0.01f - 0.38f;
    for (int da = daMin; da <= daMax; ++da)
      for (int dl = dlMin; dl <= dlMax; ++dl) {
        if (!deliBarSlotOccupied(da, dl)) continue;
        const float cx = (static_cast<float>(da) + 0.5f) * kShelfAisleModulePitch;
        const float cz = (static_cast<float>(dl) + 0.5f) * kShelfAlongAislePitch;
        if (deliCounterTop <= feetY + kFeetBand &&
            x >= cx - hl && x <= cx + hl && z >= cz - hd && z <= cz + hd)
          best = std::max(best, deliCounterTop);
        if (deliRoofTop <= feetY + kFeetBand &&
            x >= cx - ghl && x <= cx + ghl && z >= cz - innerZ && z <= cz + innerZ)
          best = std::max(best, deliRoofTop);
      }
  }
  {
    const size_t hi =
        (static_cast<size_t>(static_cast<uint32_t>(qx) * 1664525u ^
                              static_cast<uint32_t>(qz) * 1013904223u ^
                              static_cast<uint32_t>(qf) * 374761393u) &
        (static_cast<size_t>(kTerrainYCacheSlots) - 1u));
    gTerrainYCache[hi] = {true, qx, qz, qf, best};
  }
  return best;
}

// First- and third-person: keep eye above terrain support and below ceiling.
static void resolveCameraEyeSupportAndCeiling(glm::vec3& eye) {
  const float sup = terrainSupportY(eye.x, eye.z, eye.y);
  eye.y = std::max(eye.y, sup + kThirdPersonCamMinAboveSupportM);
  constexpr float kCameraEyeBelowCeilingM = 0.4f;
  eye.y = std::min(eye.y, kCeilingY - kCameraEyeBelowCeilingM);
}

static void resolveThirdPersonEyeAboveFloor(glm::vec3& eye) {
  resolveCameraEyeSupportAndCeiling(eye);
}

// Multi-sample max support height along a unit direction in XZ (vec2.x = world X, .y = world Z).
// Single-point probes over the void between shelf units read the floor — falsely huge drop.
// Samples must extend past the next deck (~kShelfAlongAislePitch along bays, ~kShelfAisleModulePitch
// across aisles); otherwise every sample between decks hits void and max() becomes floor → "infinite" drop
// and walk-off fall animations fire while stepping across shelf rows.
//
// terrainSupportY() only accepts a deck/crate top T if T <= feetY + 0.12f. playerTerrainSupportY() passes
// feetY = actual_feet + kStaffTerrainStepProbe so nearby tops qualify. Using lastSupportY - 0.17 here made
// same-height deck tops ahead *invisible* (T <= lastSupportY - 0.05 never held), so probes only saw floor.
//
// Classify “small shelf hop” vs “real fall”:
// - Same-tier hits at the *current* feet position and nearby planks zero the drop if we max() blindly.
// - Tier path: only same-tier deck/crate tops at kWalkOffGapSameTierMinForwardM+ count as “next ledge ahead”
//   (small gap → walk/run in air, no fall clips).
// - Else: max support whose height differs from lastSupportY by more than kWalkOffGapExcludeSameSurfaceEpsM
//   (void/floor/lower tier) → large drop → walk-off pre-fall / fall mid-pose / etc.
static float playerWalkOffProbeDropBelowDir(float lastSupportY, float wx, float wz,
                                            const glm::vec2& ndUnit) {
  const float feetYForTerrain = lastSupportY + kStaffTerrainStepProbe;
  const glm::vec2 nd = glm::length(ndUnit) > 1e-4f ? glm::normalize(ndUnit) : glm::vec2(0.f, 1.f);
  float bestAhead = -1e30f;
  float bestSameTierFarAhead = -1e30f;
  float bestNonSameSurface = -1e30f;
  constexpr float kTierBelow = 0.62f;
  constexpr float kTierAbove = 0.48f;
  const float tierLo = lastSupportY - kTierBelow;
  const float tierHi = lastSupportY + kTierAbove;
  // Fewer samples: each calls terrainSupportY() (shelf grid on miss). Walk-off runs ~5×/frame when coyote fires.
  constexpr float kAlongM[] = {
      0.44f, 0.58f, 0.74f, 0.95f, 1.2f, 1.55f, 1.95f, 2.45f, 3.1f, 4.0f, 5.2f,
      kShelfAlongAislePitch * 0.5f, kShelfAlongAislePitch * 0.95f, kShelfAisleModulePitch * 0.68f,
      kShelfAisleModulePitch * 1.02f, 8.5f};
  for (float al : kAlongM) {
    if (al < kWalkOffGapProbeForwardMinM)
      continue;
    const glm::vec2 p(wx + nd.x * al, wz + nd.y * al);
    const float y = terrainSupportY(p.x, p.y, feetYForTerrain);
    bestAhead = std::max(bestAhead, y);
    if (al >= kWalkOffGapSameTierMinForwardM && y >= tierLo && y <= tierHi)
      bestSameTierFarAhead = std::max(bestSameTierFarAhead, y);
    // Do not treat “void sample = floor” a few decimeters ahead as the landing surface — that made
    // bestNonSameSurface = kGroundY and returned a huge drop for every narrow plank gap.
    if (al >= kWalkOffGapNonSameMinForwardM &&
        std::abs(y - lastSupportY) > kWalkOffGapExcludeSameSurfaceEpsM)
      bestNonSameSurface = std::max(bestNonSameSurface, y);
  }
  if (bestSameTierFarAhead > -1e29f)
    return std::max(0.f, lastSupportY - bestSameTierFarAhead);
  if (bestNonSameSurface > -1e29f)
    return std::max(0.f, lastSupportY - bestNonSameSurface);
  if (bestAhead > -1e29f)
    return std::max(0.f, lastSupportY - bestAhead);
  return 1e30f;
}

// Find which standard rack wood deck (world grid) the feet are on, matching lastSupportY to a deck tier.
static bool shelfFindWoodDeckSlotAtSupport(float wx, float wz, float lastSupportY, int& outWa, int& outWl,
                                           int& outSide, int& outSi) {
  constexpr float kMatchEps = 0.16f;
  constexpr float kCullR2 = 15.5f * 15.5f;
  constexpr float kGridRangeM = 19.f;
  int waMin, waMax, wlMin, wlMax;
  shelfGridWindowForRange(wx, wz, kGridRangeM, waMin, waMax, wlMin, wlMax);
  for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
    const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
    const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
    const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
    for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
      const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
      for (int side = 0; side < 2; ++side) {
        if (!shelfSlotOccupied(worldAisle, worldAlong, side))
          continue;
        const float cx = side ? cxRight : cxLeft;
        const float yawDeg = side ? -90.0f : 90.0f;
        const float dx = wx - cx;
        const float dz = wz - cz;
        if (dx * dx + dz * dz > kCullR2)
          continue;
        const glm::vec3 shelfPos{cx, kGroundY, cz};
        const float shelfYawRad = glm::radians(yawDeg);
        const float hw = kShelfMeshHalfW;
        const float hd = kShelfMeshHalfD;
        const float shelfT = kShelfDeckThickness;
        const int numShelves = kShelfDeckCount;
        constexpr float yBase = 0.12f;
        const float yStep = kShelfGapBetweenLevels + shelfT;
        const float mnX = -hw + kShelfDeckInset;
        const float mnZ = -hd + kShelfDeckInset;
        const float mxX = hw - kShelfDeckInset;
        const float mxZ = hd - kShelfDeckInset;
        for (int si = 0; si < numShelves; ++si) {
          const float y0 = yBase + static_cast<float>(si) * yStep;
          const float y1 = y0 + shelfT;
          if (!pointInShelfLocalXZ(shelfPos, shelfYawRad, wx, wz, mnX, mnZ, mxX, mxZ))
            continue;
          const float top = kGroundY + y1;
          if (std::abs(top - lastSupportY) > kMatchEps)
            continue;
          outWa = worldAisle;
          outWl = worldAlong;
          outSide = side;
          outSi = si;
          return true;
        }
      }
    }
  }
  return false;
}

// Primary path for “small shelf hop vs big fall”: use infinite-store grid neighbors (same tier = small gap).
// Ray probes alone are unreliable on repeating decks. Returns true if feet are on a matched wood deck.
static bool playerWalkOffShelfGridTryForwardDrop(float lastSupportY, float wx, float wz,
                                                 const glm::vec2& velXZ, const glm::vec2& forwardXZ,
                                                 float& outDrop) {
  int wa, wl, side, si;
  if (!shelfFindWoodDeckSlotAtSupport(wx, wz, lastSupportY, wa, wl, side, si))
    return false;
  const glm::vec2 vd =
      glm::length(velXZ) > 0.048f
          ? glm::normalize(velXZ)
          : (glm::length(forwardXZ) > 1e-4f ? glm::normalize(forwardXZ) : glm::vec2(0.f, 1.f));
  // Cone toward neighbor *centers* — near deck edges, toN is skewed; 0.08 was too strict and never
  // classified small bay hops. Stay above ~0 so perpendicular “side” racks (dot≈0) never win over void.
  constexpr float kMinForwardDot = 0.045f;
  float bestMinDrop = 1e30f;
  bool anyForward = false;

  auto neighborTierTop = [&](int nwa, int nwl, int nside, float& outTop) -> bool {
    if (!shelfSlotOccupied(nwa, nwl, nside))
      return false;
    constexpr float yBase = 0.12f;
    const float yStep = kShelfGapBetweenLevels + kShelfDeckThickness;
    const float y0 = yBase + static_cast<float>(si) * yStep;
    const float y1 = y0 + kShelfDeckThickness;
    outTop = kGroundY + y1;
    return true;
  };

  auto considerNeighbor = [&](int nwa, int nwl, int nside) {
    float nTop = 0.f;
    if (!neighborTierTop(nwa, nwl, nside, nTop))
      return;
    const float aisleCX = (static_cast<float>(nwa) + 0.5f) * kShelfAisleModulePitch;
    const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
    const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
    const float cz = (static_cast<float>(nwl) + 0.5f) * kShelfAlongAislePitch;
    const float cx = nside ? cxRight : cxLeft;
    glm::vec2 toN(cx - wx, cz - wz);
    const float len2 = glm::dot(toN, toN);
    if (len2 < 0.0016f)
      return;
    toN *= 1.f / std::sqrt(len2);
    if (glm::dot(toN, vd) < kMinForwardDot)
      return;
    anyForward = true;
    bestMinDrop = std::min(bestMinDrop, std::max(0.f, lastSupportY - nTop));
  };

  considerNeighbor(wa, wl + 1, side);
  considerNeighbor(wa, wl - 1, side);
  considerNeighbor(wa + 1, wl, side);
  considerNeighbor(wa - 1, wl, side);
  considerNeighbor(wa, wl, 1 - side);

  // Single cardinal step in the dominant move direction (no min over all neighbors): fixes edge cases
  // where dot-to-center misses but the next bay / aisle cell along motion is still the right probe.
  // Matches playerWalkOffProbeDropBelowShelfBayGap vs cross-aisle: Z stride when not strongly X-dominant.
  if (!anyForward) {
    const float vx = velXZ.x, vz = velXZ.y;
    const float avx = std::abs(vx), avz = std::abs(vz);
    constexpr float kStrideDom = 0.88f;
    auto considerOneCell = [&](int nwa, int nwl, int nside) {
      float nTop = 0.f;
      if (!neighborTierTop(nwa, nwl, nside, nTop))
        return;
      anyForward = true;
      bestMinDrop = std::min(bestMinDrop, std::max(0.f, lastSupportY - nTop));
    };
    if (avx <= avz * kStrideDom + 1e-5f) {
      int dz = 0;
      if (avz > 0.052f)
        dz = vz > 0.f ? 1 : -1;
      else if (std::abs(forwardXZ.y) > 0.11f)
        dz = forwardXZ.y > 0.f ? 1 : -1;
      if (dz != 0)
        considerOneCell(wa, wl + dz, side);
    } else if (avz <= avx * kStrideDom + 1e-5f) {
      int dwa = 0;
      if (avx > 0.052f)
        dwa = vx > 0.f ? 1 : -1;
      else if (std::abs(forwardXZ.x) > 0.11f)
        dwa = forwardXZ.x > 0.f ? 1 : -1;
      if (dwa != 0)
        considerOneCell(wa + dwa, wl, side);
    }
  }

  outDrop = anyForward ? bestMinDrop : 1e30f;
  return true;
}

// Small ledge-gap helpers (walk in air, skip coyote jump, mantle skip, pre-fall delay): aisles run along +Z;
// narrow bay-to-bay breaks are along ±Z. Probe along ±Z using motion when not cross-aisle dominant; else
// use view forward so we don’t return ∞ and lose the bay stride ray on diagonal approaches.
static float playerWalkOffProbeDropBelowShelfBayGap(float lastSupportY, float wx, float wz,
                                                    const glm::vec2& velXZ, const glm::vec2& forwardXZ) {
  constexpr float kCrossAisleDom = 0.92f;
  const float avx = std::abs(velXZ.x), avz = std::abs(velXZ.y);
  glm::vec2 nd(0.f, 1.f);
  if (avz > 0.06f)
    nd.y = glm::sign(velXZ.y);
  else if (std::abs(forwardXZ.y) > 0.15f)
    nd.y = glm::sign(forwardXZ.y);
  if (avx > avz * kCrossAisleDom + 1e-5f) {
    if (std::abs(forwardXZ.y) > 0.12f)
      nd.y = glm::sign(forwardXZ.y);
    else
      nd.y = nd.y > 0.f ? 1.f : -1.f;
  }
  return playerWalkOffProbeDropBelowDir(lastSupportY, wx, wz, nd);
}

// Smallest drop to next support: prefer shelf grid (reliable on racks); else ray probes (floor / crates / odd).
static float playerWalkOffEffectiveWalkableGapDrop(float lastSupportY, float wx, float wz,
                                                 const glm::vec2& velXZ, const glm::vec2& forwardXZ) {
  constexpr float kGridUnclassifiedDrop = 1e29f;
  float gridDrop = 1e30f;
  // Only trust grid when it matched a forward neighbor; otherwise outDrop stays huge — fall through to rays
  // so big gaps don’t get mis-tagged as “small stride” from an unclassified grid result.
  if (playerWalkOffShelfGridTryForwardDrop(lastSupportY, wx, wz, velXZ, forwardXZ, gridDrop) &&
      gridDrop < kGridUnclassifiedDrop)
    return gridDrop;
  // First air frame often leaves the deck AABB before shelfFind matches — step back along motion / view.
  const glm::vec2 vm =
      glm::length(velXZ) > 0.055f
          ? glm::normalize(velXZ)
          : (glm::length(forwardXZ) > 1e-4f ? glm::normalize(forwardXZ) : glm::vec2(0.f, 1.f));
  constexpr float kShelfGapBacktrackM = 0.74f;
  for (int i = 1; i <= 3; ++i) {
    const float s = kShelfGapBacktrackM * static_cast<float>(i);
    gridDrop = 1e30f;
    if (playerWalkOffShelfGridTryForwardDrop(lastSupportY, wx - vm.x * s, wz - vm.y * s, velXZ, forwardXZ,
                                             gridDrop) &&
        gridDrop < kGridUnclassifiedDrop)
      return gridDrop;
  }
  const float bay = playerWalkOffProbeDropBelowShelfBayGap(lastSupportY, wx, wz, velXZ, forwardXZ);
  const float vlen = glm::length(velXZ);
  const glm::vec2 alongDir =
      vlen > 0.048f ? velXZ * (1.f / vlen)
                    : (glm::length(forwardXZ) > 1e-4f ? glm::normalize(forwardXZ) : glm::vec2(0.f, 1.f));
  const float along = playerWalkOffProbeDropBelowDir(lastSupportY, wx, wz, alongDir);
  // Extra pass along camera / intent forward — catches ledges where velocity is skewed but next surface
  // lies under view direction (common on crates and non-grid props).
  const glm::vec2 fwdN =
      glm::length(forwardXZ) > 1e-4f ? glm::normalize(forwardXZ) : glm::vec2(0.f, 1.f);
  const float alongView = playerWalkOffProbeDropBelowDir(lastSupportY, wx, wz, fwdN);
  float rayMin = std::min(bay, std::min(along, alongView));
  // Orthogonal samples: many ledges are short in view-forward but clear along ±90° (crates, corners).
  const glm::vec2 ortho(-fwdN.y, fwdN.x);
  rayMin = std::min(rayMin, playerWalkOffProbeDropBelowDir(lastSupportY, wx, wz, ortho));
  rayMin = std::min(rayMin, playerWalkOffProbeDropBelowDir(lastSupportY, wx, wz, -ortho));
  return rayMin;
}

// Feet on any shelf deck level or crate top (not aisle floor); used to forgive short falls onto ledges/props.
static bool playerFallSupportIsShelfDeckOrCrateTop(float wx, float wz, float landedSupportY) {
  if (landedSupportY <= kGroundY + kPlayerFallLandFloorBandM)
    return false;
  constexpr float kCullR2 = 15.5f * 15.5f;
  constexpr float kGridRangeM = 19.f;
  int waMin, waMax, wlMin, wlMax;
  shelfGridWindowForRange(wx, wz, kGridRangeM, waMin, waMax, wlMin, wlMax);
  for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
    const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
    const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
    const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
    for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
      const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
      for (int side = 0; side < 2; ++side) {
        if (!shelfSlotOccupied(worldAisle, worldAlong, side))
          continue;
        const float cx = side ? cxRight : cxLeft;
        const float yawDeg = side ? -90.0f : 90.0f;
        const float dx = wx - cx;
        const float dz = wz - cz;
        if (dx * dx + dz * dz > kCullR2)
          continue;
        const glm::vec3 shelfPos{cx, kGroundY, cz};
        const float shelfYawRad = glm::radians(yawDeg);
        const float hw = kShelfMeshHalfW;
        const float hd = kShelfMeshHalfD;
        const float shelfT = kShelfDeckThickness;
        const int numShelves = kShelfDeckCount;
        constexpr float yBase = 0.12f;
        const float yStep = kShelfGapBetweenLevels + shelfT;
        const float mnX = -hw + kShelfDeckInset;
        const float mnZ = -hd + kShelfDeckInset;
        const float mxX = hw - kShelfDeckInset;
        const float mxZ = hd - kShelfDeckInset;
        for (int si = 0; si < numShelves; ++si) {
          const float y0 = yBase + static_cast<float>(si) * yStep;
          const float y1 = y0 + shelfT;
          if (!pointInShelfLocalXZ(shelfPos, shelfYawRad, wx, wz, mnX, mnZ, mxX, mxZ))
            continue;
          const float top = kGroundY + y1;
          if (std::abs(top - landedSupportY) > kPlayerFallLandHeightMatchEpsM)
            continue;
          return true;
        }
        float clx, clz, yDeck, chx, chy, chz;
        if (shelfCrateLocalLayout(worldAisle, worldAlong, side, clx, clz, yDeck, chx, chy, chz)) {
          const float cmnX = clx - chx;
          const float cmnZ = clz - chz;
          const float cmxX = clx + chx;
          const float cmxZ = clz + chz;
          if (!pointInShelfLocalXZ(shelfPos, shelfYawRad, wx, wz, cmnX, cmnZ, cmxX, cmxZ))
            continue;
          const float ctop = kGroundY + yDeck + 2.f * chy;
          if (std::abs(ctop - landedSupportY) > kPlayerFallLandHeightMatchEpsM)
            continue;
          return true;
        }
      }
    }
  }
  return false;
}

// Random respawn: aisle floor only (not decks/crates), clear of pillar posts in XZ.
static bool playerRespawnFloorXZClear(float x, float z) {
  constexpr float kProbeFeetY = kGroundY + 0.62f;
  const float fy = terrainSupportY(x, z, kProbeFeetY + kStaffTerrainStepProbe);
  if (!std::isfinite(fy) || fy > kGroundY + 0.11f || fy < kGroundY - 0.04f)
    return false;
  if (playerFallSupportIsShelfDeckOrCrateTop(x, z, fy))
    return false;
  const int gcx = static_cast<int>(std::floor(x / kPillarSpacing));
  const int gcz = static_cast<int>(std::floor(z / kPillarSpacing));
  constexpr float kPad = 0.11f;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dz = -1; dz <= 1; ++dz) {
      const float px = static_cast<float>(gcx + dx) * kPillarSpacing;
      const float pz = static_cast<float>(gcz + dz) * kPillarSpacing;
      if (std::abs(x - px) < kPillarHalfW + kPlayerHalfXZ + kPad &&
          std::abs(z - pz) < kPillarHalfD + kPlayerHalfXZ + kPad)
        return false;
    }
  }
  return true;
}

// 0 = low (floor + bottom shelf), 1..3 = second through top wood deck; crate tops → nearest deck index.
static int playerFallDamageTierAtSupport(float wx, float wz, float landedSupportY) {
  if (landedSupportY <= kGroundY + kPlayerFallLandFloorBandM)
    return 0;
  constexpr float kCullR2 = 15.5f * 15.5f;
  constexpr float kGridRangeM = 19.f;
  int waMin, waMax, wlMin, wlMax;
  shelfGridWindowForRange(wx, wz, kGridRangeM, waMin, waMax, wlMin, wlMax);
  for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
    const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
    const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
    const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
    for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
      const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
      for (int side = 0; side < 2; ++side) {
        if (!shelfSlotOccupied(worldAisle, worldAlong, side))
          continue;
        const float cx = side ? cxRight : cxLeft;
        const float yawDeg = side ? -90.0f : 90.0f;
        const float dx = wx - cx;
        const float dz = wz - cz;
        if (dx * dx + dz * dz > kCullR2)
          continue;
        const glm::vec3 shelfPos{cx, kGroundY, cz};
        const float shelfYawRad = glm::radians(yawDeg);
        const float hw = kShelfMeshHalfW;
        const float hd = kShelfMeshHalfD;
        const float shelfT = kShelfDeckThickness;
        const int numShelves = kShelfDeckCount;
        constexpr float yBase = 0.12f;
        const float yStep = kShelfGapBetweenLevels + shelfT;
        const float mnX = -hw + kShelfDeckInset;
        const float mnZ = -hd + kShelfDeckInset;
        const float mxX = hw - kShelfDeckInset;
        const float mxZ = hd - kShelfDeckInset;
        for (int si = 0; si < numShelves; ++si) {
          const float y0 = yBase + static_cast<float>(si) * yStep;
          const float y1 = y0 + shelfT;
          if (!pointInShelfLocalXZ(shelfPos, shelfYawRad, wx, wz, mnX, mnZ, mxX, mxZ))
            continue;
          const float top = kGroundY + y1;
          if (std::abs(top - landedSupportY) > kPlayerFallLandHeightMatchEpsM)
            continue;
          return si;
        }
        float clx, clz, yDeck, chx, chy, chz;
        if (shelfCrateLocalLayout(worldAisle, worldAlong, side, clx, clz, yDeck, chx, chy, chz)) {
          const float cmnX = clx - chx;
          const float cmnZ = clz - chz;
          const float cmxX = clx + chx;
          const float cmxZ = clz + chz;
          if (!pointInShelfLocalXZ(shelfPos, shelfYawRad, wx, wz, cmnX, cmnZ, cmxX, cmxZ))
            continue;
          const float ctop = kGroundY + yDeck + 2.f * chy;
          if (std::abs(ctop - landedSupportY) > kPlayerFallLandHeightMatchEpsM)
            continue;
          int bestSi = 0;
          float bestErr = 1e9f;
          for (int si = 0; si < numShelves; ++si) {
            const float y0c = yBase + static_cast<float>(si) * yStep;
            const float y1c = y0c + shelfT;
            const float deckTop = kGroundY + y1c;
            const float err = std::abs(ctop - deckTop);
            if (err < bestErr) {
              bestErr = err;
              bestSi = si;
            }
          }
          return glm::clamp(bestSi, 0, kShelfDeckCount - 1);
        }
      }
    }
  }
  return 0;
}

// Nearest shelf deck index from Y alone (no XZ footprint). Fixes takeoff tier when the first air frame is
// already past the deck inset; also used with max feet height across a multi-bounce drop to the floor.
static int playerFallDamageTierFromSupportWorldY(float supportOrFeetY) {
  if (!std::isfinite(supportOrFeetY) || supportOrFeetY <= kGroundY + kPlayerFallLandFloorBandM)
    return 0;
  constexpr float yBase = 0.12f;
  const float shelfT = kShelfDeckThickness;
  const float yStep = kShelfGapBetweenLevels + shelfT;
  int bestSi = 0;
  float bestErr = 1e9f;
  for (int si = 0; si < kShelfDeckCount; ++si) {
    const float y0 = yBase + static_cast<float>(si) * yStep;
    const float top = kGroundY + y0 + shelfT;
    const float err = std::abs(top - supportOrFeetY);
    if (err < bestErr) {
      bestErr = err;
      bestSi = si;
    }
  }
  constexpr float kMaxDeckTierErrM = 0.26f;
  if (bestErr > kMaxDeckTierErrM)
    return 0;
  return bestSi;
}

// One-shot query when applying fall damage: surface + rim factors scaled by impact severity (earth-equiv dv).
static void playerFallLandingDamageMultipliers(float wx, float wz, float landedSupportY,
                                               float excessImpactDvEarth, float minDvForSeverity,
                                               float& surfaceMul, float& edgeMul) {
  surfaceMul = 1.f;
  edgeMul = 1.f;
  const float sev =
      glm::clamp((excessImpactDvEarth - minDvForSeverity) / std::max(1e-4f, kPlayerFallLandSeveritySpanDv),
                 0.f, 1.f);
  if (landedSupportY <= kGroundY + kPlayerFallLandFloorBandM)
    return;

  constexpr float kCullR2 = 15.5f * 15.5f;
  constexpr float kGridRangeM = 19.f;
  int waMin, waMax, wlMin, wlMax;
  shelfGridWindowForRange(wx, wz, kGridRangeM, waMin, waMax, wlMin, wlMax);
  bool deckHit = false;
  float deckEdgeMin = 1e9f;
  bool crateHit = false;
  float crateEdgeMin = 1e9f;
  for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
    const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
    const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
    const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
    for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
      const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
      for (int side = 0; side < 2; ++side) {
        if (!shelfSlotOccupied(worldAisle, worldAlong, side))
          continue;
        const float cx = side ? cxRight : cxLeft;
        const float yawDeg = side ? -90.0f : 90.0f;
        const float dx = wx - cx;
        const float dz = wz - cz;
        if (dx * dx + dz * dz > kCullR2)
          continue;
        const glm::vec3 shelfPos{cx, kGroundY, cz};
        const float shelfYawRad = glm::radians(yawDeg);
        const float hw = kShelfMeshHalfW;
        const float hd = kShelfMeshHalfD;
        const float shelfT = kShelfDeckThickness;
        const int numShelves = kShelfDeckCount;
        constexpr float yBase = 0.12f;
        const float yStep = kShelfGapBetweenLevels + shelfT;
        const float mnX = -hw + kShelfDeckInset;
        const float mnZ = -hd + kShelfDeckInset;
        const float mxX = hw - kShelfDeckInset;
        const float mxZ = hd - kShelfDeckInset;
        for (int si = 0; si < numShelves; ++si) {
          const float y0 = yBase + static_cast<float>(si) * yStep;
          const float y1 = y0 + shelfT;
          if (!pointInShelfLocalXZ(shelfPos, shelfYawRad, wx, wz, mnX, mnZ, mxX, mxZ))
            continue;
          const float top = kGroundY + y1;
          if (std::abs(top - landedSupportY) > kPlayerFallLandHeightMatchEpsM)
            continue;
          const glm::vec2 L = worldToShelfLocalXZ(shelfPos, shelfYawRad, wx, wz);
          const float dEdge = shelfLocalInteriorDistToRectEdge(L.x, L.y, mnX, mnZ, mxX, mxZ);
          deckHit = true;
          deckEdgeMin = std::min(deckEdgeMin, dEdge);
        }
        float clx, clz, yDeck, chx, chy, chz;
        if (shelfCrateLocalLayout(worldAisle, worldAlong, side, clx, clz, yDeck, chx, chy, chz)) {
          const float cmnX = clx - chx;
          const float cmnZ = clz - chz;
          const float cmxX = clx + chx;
          const float cmxZ = clz + chz;
          if (!pointInShelfLocalXZ(shelfPos, shelfYawRad, wx, wz, cmnX, cmnZ, cmxX, cmxZ))
            continue;
          const float ctop = kGroundY + yDeck + 2.f * chy;
          if (std::abs(ctop - landedSupportY) > kPlayerFallLandHeightMatchEpsM)
            continue;
          const glm::vec2 Lc = worldToShelfLocalXZ(shelfPos, shelfYawRad, wx, wz);
          const float dEdgeC = shelfLocalInteriorDistToRectEdge(Lc.x, Lc.y, cmnX, cmnZ, cmxX, cmxZ);
          crateHit = true;
          crateEdgeMin = std::min(crateEdgeMin, dEdgeC);
        }
      }
    }
  }
  const float deckMul = glm::mix(kPlayerFallLandDeckMulSoft, kPlayerFallLandDeckMulHard, sev);
  const float crateMul = glm::mix(kPlayerFallLandCrateMulSoft, kPlayerFallLandCrateMulHard, sev);
  const float rimMul = glm::mix(kPlayerFallLandEdgeRimSoft, kPlayerFallLandEdgeRimHard, sev);
  const float cenMul = glm::mix(kPlayerFallLandEdgeCenSoft, kPlayerFallLandEdgeCenHard, sev);
  if (crateHit) {
    surfaceMul = crateMul;
    const float u = glm::clamp(crateEdgeMin / kPlayerFallLandEdgeRefM, 0.f, 1.f);
    edgeMul = glm::mix(rimMul, cenMul, u);
  } else if (deckHit) {
    surfaceMul = deckMul;
    const float u = glm::clamp(deckEdgeMin / kPlayerFallLandEdgeRefM, 0.f, 1.f);
    edgeMul = glm::mix(rimMul, cenMul, u);
  }
}

// Same normalized height blend as player advanceLedgeClimb (ledge grab S-curve).
static float playerLedgeGrabHeightS(float t) {
  t = glm::clamp(t, 0.f, 1.f);
  if (t <= kLedgeGrabAnimLiftPhaseEnd) {
    const float u = t / kLedgeGrabAnimLiftPhaseEnd;
    return (1.f - std::pow(1.f - u, kLedgeGrabEaseOutPow)) * kLedgeGrabAnimLiftHeightFrac;
  }
  const float u = (t - kLedgeGrabAnimLiftPhaseEnd) / (1.f - kLedgeGrabAnimLiftPhaseEnd);
  const float settle = u * u * (3.f - 2.f * u);
  return kLedgeGrabAnimLiftHeightFrac + (1.f - kLedgeGrabAnimLiftHeightFrac) * settle;
}

// Horizontal point to steer toward when the player is elevated: blend player XZ with the aisle centerline
// under them (+Z matches bay) so chasers path through walkable floor toward a mantle lip.
static glm::vec2 staffChaseElevatedApproachPointXZ(const glm::vec2& playerXZ, float playerFeetY,
                                                   float staffFeetY) {
  if (playerFeetY <= staffFeetY + 0.46f)
    return playerXZ;
  const float pitch = kShelfAisleModulePitch;
  const int ai = static_cast<int>(std::floor(playerXZ.x / pitch));
  const float aisleCX = (static_cast<float>(ai) + 0.5f) * pitch;
  const glm::vec2 onAisle(aisleCX, playerXZ.y);
  const float dy = glm::clamp(playerFeetY - staffFeetY, 0.f, 12.f);
  const float w = glm::clamp(0.25f + dy * 0.055f, 0.25f, 0.82f);
  return glm::mix(playerXZ, onAisle, w);
}

// Mantel feet: y0→y1 follows player ledge grab height curve (no extra hop arc).
static float staffChaseMantelFeetWorldY(float y0, float y1, float uLinear) {
  uLinear = glm::clamp(uLinear, 0.f, 1.f);
  return glm::mix(y0, y1, playerLedgeGrabHeightS(uLinear));
}

static constexpr int kStaffSupportCacheSlots = 512;
struct StaffSupportCacheSlot {
  bool valid = false;
  int qx = 0, qz = 0, qf = 0;
  float y = 0.f;
};
static StaffSupportCacheSlot gStaffSupportCache[kStaffSupportCacheSlots]{};

static float staffChaseClimbSupportY(float x, float z, float feetY) {
  const int qx = static_cast<int>(std::floor(x * 2.f));
  const int qz = static_cast<int>(std::floor(z * 2.f));
  const int qf = static_cast<int>(std::floor(feetY * 5.f));
  {
    const size_t hi =
        (static_cast<size_t>(static_cast<uint32_t>(qx) * 2654435761u ^
                              static_cast<uint32_t>(qz) * 2246822519u ^
                              static_cast<uint32_t>(qf) * 374761393u) &
        (static_cast<size_t>(kStaffSupportCacheSlots) - 1u));
    const StaffSupportCacheSlot& s = gStaffSupportCache[hi];
    if (s.valid && s.qx == qx && s.qz == qz && s.qf == qf)
      return s.y;
  }
  float best = kGroundY;
  constexpr float kFeetBand = 0.12f;
  constexpr float kCullR2 = 15.5f * 15.5f;
  constexpr float kGridRangeM = 19.f;
  int waMin, waMax, wlMin, wlMax;
  shelfGridWindowForRange(x, z, kGridRangeM, waMin, waMax, wlMin, wlMax);
  for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
    const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
    const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
    const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
    for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
      const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
      for (int side = 0; side < 2; ++side) {
        if (!shelfSlotOccupied(worldAisle, worldAlong, side))
          continue;
        const float cx = side ? cxRight : cxLeft;
        const float yawDeg = side ? -90.0f : 90.0f;
        const float dx = x - cx;
        const float dz = z - cz;
        if (dx * dx + dz * dz > kCullR2)
          continue;
        const glm::vec3 shelfPos{cx, kGroundY, cz};
        const float shelfYawRad = glm::radians(yawDeg);
        const float hw = kShelfMeshHalfW;
        const float hd = kShelfMeshHalfD;
        const float shelfT = kShelfDeckThickness;
        const int numShelves = kShelfDeckCount;
        constexpr float yBase = 0.12f;
        const float yStep = kShelfGapBetweenLevels + shelfT;
        const float mnX = -hw + kShelfDeckInset;
        const float mnZ = -hd + kShelfDeckInset;
        const float mxX = hw - kShelfDeckInset;
        const float mxZ = hd - kShelfDeckInset;
        for (int si = 0; si < numShelves; ++si) {
          const float y0 = yBase + static_cast<float>(si) * yStep;
          const float y1 = y0 + shelfT;
          if (distWorldXZToShelfLocalRect(shelfPos, shelfYawRad, x, z, mnX, mnZ, mxX, mxZ) > kStaffChaseDeckSnapXZ)
            continue;
          const float top = kGroundY + y1;
          if (top <= feetY + kFeetBand)
            best = std::max(best, top);
        }
        float clx, clz, yDeck, chx, chy, chz;
        if (shelfCrateLocalLayout(worldAisle, worldAlong, side, clx, clz, yDeck, chx, chy, chz)) {
          if (distWorldXZToShelfLocalRect(shelfPos, shelfYawRad, x, z, clx - chx, clz - chz, clx + chx,
                                          clz + chz) > kStaffChaseDeckSnapXZ)
            continue;
          const float ctop = kGroundY + yDeck + 2.f * chy;
          if (ctop <= feetY + kFeetBand)
            best = std::max(best, ctop);
        }
      }
    }
  }
  {
    const size_t hi =
        (static_cast<size_t>(static_cast<uint32_t>(qx) * 2654435761u ^
                              static_cast<uint32_t>(qz) * 2246822519u ^
                              static_cast<uint32_t>(qf) * 374761393u) &
        (static_cast<size_t>(kStaffSupportCacheSlots) - 1u));
    gStaffSupportCache[hi] = {true, qx, qz, qf, best};
  }
  return best;
}

// Shelf/crate top at (x,z) reaches roughly wantFeetY (same horizontal snap as chase climb).
static bool staffChaseHasSupportNearHeight(float x, float z, float wantFeetY) {
  const float got = staffChaseClimbSupportY(x, z, wantFeetY + kStaffChaseLedgeGrabProbePadM);
  return got >= wantFeetY - kStaffChaseLedgeGrabDeckMatchEpsM;
}

// Walk the approach toward the player: lip is often offset in XZ from aisle center (DL-style runner pathing).
static bool staffChaseLedgeGrabMultiSampleSupport(const ShelfEmployeeNpc& e, float targetY,
                                                  const glm::vec2& toP, float distPXZ, float maxAlong) {
  if (staffChaseHasSupportNearHeight(e.posXZ.x, e.posXZ.y, targetY))
    return true;
  if (distPXZ < 0.14f)
    return false;
  const glm::vec2 n = toP * (1.f / distPXZ);
  const float cap = glm::min(distPXZ * 0.94f, maxAlong);
  for (float s = kStaffChaseLedgeGrabMultiSampleStepM; s <= cap + 1e-4f;
       s += kStaffChaseLedgeGrabMultiSampleStepM) {
    if (staffChaseHasSupportNearHeight(e.posXZ.x + n.x * s, e.posXZ.y + n.y * s, targetY))
      return true;
  }
  return false;
}

// Feet under target deck now, or along the run toward the player (short legacy lead + dense samples).
static bool staffChaseLedgeGrabAligned(const ShelfEmployeeNpc& e, float targetY, const glm::vec2& toP,
                                       float distPXZ) {
  if (staffChaseLedgeGrabMultiSampleSupport(e, targetY, toP, distPXZ, kStaffChaseLedgeGrabMultiSampleMaxM))
    return true;
  if (distPXZ < 0.2f)
    return false;
  const glm::vec2 n = toP * (1.f / distPXZ);
  const float lead =
      glm::clamp(kStaffChaseLedgeGrabLeadDistFrac * distPXZ, kStaffChaseLedgeGrabLeadMinM,
                 kStaffChaseLedgeGrabLeadMaxM);
  return staffChaseHasSupportNearHeight(e.posXZ.x + n.x * lead, e.posXZ.y + n.y * lead, targetY);
}

// Sprinting: longer along-ray search so grabs fire on the run, not after stopping at the lip.
static bool staffChaseLedgeGrabAlignedRunner(const ShelfEmployeeNpc& e, float targetY, const glm::vec2& toP,
                                             float distPXZ) {
  if (staffChaseLedgeGrabMultiSampleSupport(e, targetY, toP, distPXZ,
                                            kStaffChaseRunnerLedgeGrabMultiSampleMaxM))
    return true;
  if (distPXZ < 0.2f)
    return false;
  const glm::vec2 n = toP * (1.f / distPXZ);
  const float lead = glm::clamp(kStaffChaseRunnerLeadDistFrac * distPXZ, kStaffChaseRunnerLeadMinM,
                                kStaffChaseRunnerLeadMaxM);
  return staffChaseHasSupportNearHeight(e.posXZ.x + n.x * lead, e.posXZ.y + n.y * lead, targetY);
}

// Facing player, moving toward them, or already very close — avoids “side grab” starts.
static bool staffChaseLedgeGrabOrientOk(const ShelfEmployeeNpc& e, const glm::vec2& toP, float distPXZ) {
  if (distPXZ < kStaffChaseLedgeGrabCloseBypassM)
    return true;
  const glm::vec2 toN = toP * (1.f / std::max(distPXZ, 1e-4f));
  const glm::vec2 fwd(std::sin(e.yaw), std::cos(e.yaw));
  if (glm::dot(fwd, toN) >= kStaffChaseLedgeGrabMinFaceCos)
    return true;
  const float vlen = glm::length(e.velXZ);
  if (vlen > 0.12f && glm::dot(e.velXZ * (1.f / vlen), toN) >= kStaffChaseLedgeGrabVelTowardPlayerMin)
    return true;
  return false;
}

// Sprinting chase: trust velocity-toward-player more than facing (DL runners often grab at an angle).
static bool staffChaseLedgeGrabOrientRunnerOk(const ShelfEmployeeNpc& e, const glm::vec2& toP, float distPXZ,
                                              float vToward) {
  if (distPXZ < kStaffChaseRunnerGrabCloseBypassM)
    return true;
  if (vToward >= kStaffChaseRunnerVelTowardPlayerMin)
    return true;
  return staffChaseLedgeGrabOrientOk(e, toP, distPXZ);
}

// Foot support for gravity / grounding / post-mantel snap. Must NOT use the player's feet height as a
// vertical ceiling: staffChaseClimbSupportY picks the highest deck with top <= probe+band within XZ snap;
// inflating probe toward the player made chasers on the floor snap onto the player's shelf when backing
// away. Climb targeting still uses probeFeet in the chase-ledges block (deckAtPlayer / targetY).
static float staffNpcFootSupportY(const ShelfEmployeeNpc& e,
                                  [[maybe_unused]] float playerFeetHint =
                                      std::numeric_limits<float>::quiet_NaN()) {
  if (e.meleeState >= 2) {
    const float anchor = std::max(kGroundY + 0.02f, e.meleeKnockdownFeetAnchorY);
    // Knockdowns slide in XZ; terrain only returns surfaces with top <= feetY + band. A low probe
    // (anchor-only) misses shelf decks when the corpse moves onto wood or anchor lags feet height.
    const float baseY = std::max(anchor, e.feetWorldY);
    const float probeTop =
        baseY + kEmployeeVisualHeight * e.bodyScale.y + kStaffTerrainStepProbe;
    return terrainSupportY(e.posXZ.x, e.posXZ.y, probeTop);
  }
  const float probe = e.feetWorldY + kStaffTerrainStepProbe;
  if (e.nightPhase == 2 && e.meleeState == 0)
    return staffChaseClimbSupportY(e.posXZ.x, e.posXZ.y, probe);
  return terrainSupportY(e.posXZ.x, e.posXZ.y, probe);
}

// Tall ledge: geometric drop from peak feet and/or hard impact before melee fall clip (skip low hops).
static constexpr float kStaffTallFallRagdollMinDropM = 3.08f;
static constexpr float kStaffTallFallRagdollMinVelY = -6.85f;
// Treat drops ≤ this like a step-down: snap feet, no gravity arc, no jump-in-air / land clips.
static constexpr float kStaffLowDropSnapM = kMaxStepHeight + 0.44f;
// Landings shallower than this (from airborne peak) still cancel in-air loco if impact was soft.
static constexpr float kStaffShortFallSilenceAirM = 0.78f;
static constexpr float kStaffShortFallSilenceVelY = -3.95f;
// Jump/fall-in-air clip: need real separation from support or meaningful downward speed (tiny step-offs stay walk-like).
static constexpr float kStaffAirFallMinClearanceAboveSupportM = 0.98f;
static constexpr float kStaffAirFallMinDownVelForClip = -1.02f;

static void staffNpcNotePossibleTallFallLanding(ShelfEmployeeNpc& e, float landSy, float vyBeforeZero) {
  if (e.meleeState >= 2 || e.chaseLedgeClimbRem > 0.f) {
    e.staffFallPeakFeetY = landSy;
    return;
  }
  const float drop = std::max(0.f, e.staffFallPeakFeetY - landSy);
  const bool hardVy = vyBeforeZero <= kStaffTallFallRagdollMinVelY;
  const bool highDrop = drop >= kStaffTallFallRagdollMinDropM;
  if (hardVy || highDrop)
    e.staffTallFallKnockdownPending = 1u;
  e.staffFallPeakFeetY = landSy;
}

static void staffNpcAfterLandSnap(ShelfEmployeeNpc& e, float landSy, float vyLand) {
  if (e.meleeState >= 2)
    return;
  const float dropLand = std::max(0.f, e.staffFallPeakFeetY - landSy);
  if (dropLand < kStaffShortFallSilenceAirM && vyLand > kStaffShortFallSilenceVelY) {
    e.staffAirLocoRemain = 0.f;
    e.staffAirFallClip = -1;
    e.staffAirLandRemain = 0.f;
    e.staffAirLandClip = -1;
  }
  staffNpcNotePossibleTallFallLanding(e, landSy, vyLand);
}

// Gravity + landing: player-parity — kFeetSnapDownSlop pull-down, glue band, vel thresholds like isGroundedUsingSupport.
static void staffNpcIntegrateVerticalPhysics(ShelfEmployeeNpc& e, float dt, float playerFeetHint) {
  if (e.chaseLedgeClimbRem > 0.f)
    return;
  const float sy = staffNpcFootSupportY(e, playerFeetHint);
  if (e.meleeState < 2) {
    const float above = e.feetWorldY - sy;
    if (above > kGroundedFeetAboveSupport + 0.02f && above <= kStaffLowDropSnapM &&
        e.staffVelY <= 0.11f) {
      e.feetWorldY = sy;
      e.staffVelY = 0.f;
      e.staffFallPeakFeetY = sy;
      e.staffAirLocoRemain = 0.f;
      e.staffAirFallClip = -1;
      e.staffAirLandRemain = 0.f;
      e.staffAirLandClip = -1;
      return;
    }
  }
  const float groundedBand = kGroundedFeetAboveSupport + 0.045f;
  if (e.feetWorldY > sy && e.feetWorldY <= sy + groundedBand && e.staffVelY <= 0.12f) {
    e.feetWorldY = sy;
    e.staffVelY = 0.f;
    if (e.meleeState < 2)
      e.staffFallPeakFeetY = e.feetWorldY;
    return;
  }
  if (e.meleeState < 2) {
    const bool groundedLike =
        e.feetWorldY <= sy + kGroundedFeetAboveSupport && e.staffVelY <= 0.01f;
    if (!groundedLike)
      e.staffFallPeakFeetY = std::max(e.staffFallPeakFeetY, e.feetWorldY);
  }
  e.staffVelY -= kGravity * dt;
  e.feetWorldY += e.staffVelY * dt;
  if (e.feetWorldY < sy) {
    const float vyLand = e.staffVelY;
    e.feetWorldY = sy;
    e.staffVelY = 0.f;
    staffNpcAfterLandSnap(e, sy, vyLand);
  } else if (e.feetWorldY <= sy + kFeetSnapDownSlop && e.staffVelY <= 0.04f) {
    const float vyLand = e.staffVelY;
    e.feetWorldY = sy;
    if (e.staffVelY < 0.f)
      e.staffVelY = 0.f;
    staffNpcAfterLandSnap(e, sy, vyLand);
  } else if (e.feetWorldY > sy && e.feetWorldY <= sy + groundedBand && e.staffVelY <= 0.12f) {
    const float vyLand = e.staffVelY;
    e.feetWorldY = sy;
    e.staffVelY = 0.f;
    staffNpcAfterLandSnap(e, sy, vyLand);
  }
  const float headTop = e.feetWorldY + kEmployeeVisualHeight * e.bodyScale.y;
  if (headTop > kCeilingY - 0.18f) {
    e.feetWorldY = kCeilingY - 0.18f - kEmployeeVisualHeight * e.bodyScale.y;
    e.staffVelY = std::min(0.f, e.staffVelY);
  }
}

// Matches player isGroundedUsingSupport primary rule (feet band + small downward vel).
static bool staffNpcIsGroundedLikePlayer(const ShelfEmployeeNpc& e, float sy) {
  if (e.chaseLedgeClimbRem > 0.f || e.meleeState >= 2)
    return true;
  return e.feetWorldY <= sy + kGroundedFeetAboveSupport && e.staffVelY <= 0.01f;
}

// Include deck/crate tops up to step height above feet (same idea as staff NPCs). Raw feet alone can miss
// surfaces slightly above the last query band, which flickers grounded state and breaks walk/sprint on rises.
static float playerTerrainSupportY(float x, float z, float feetY) {
  return terrainSupportY(x, z, feetY + kStaffTerrainStepProbe);
}

static std::vector<Vertex> makeWarehouseShelfMesh() {
  const glm::vec3 metal = shelfMetalVertexColor();
  const glm::vec3 wood = shelfWoodVertexColor();
  std::vector<Vertex> v;
  const float hw = kShelfMeshHalfW;
  const float hd = kShelfMeshHalfD;
  const float ph = kShelfPostGauge;
  const float H = kShelfMeshHeight;
  const float shelfT = kShelfDeckThickness;
  const int numShelves = kShelfDeckCount;
  const float yBase = 0.12f;
  const float yStep = kShelfGapBetweenLevels + shelfT;
  // Dark steel posts
  meshAddBox(v, {-hw, 0.f, -hd}, {-hw + 2.f * ph, H, -hd + 2.f * ph}, metal);
  meshAddBox(v, {hw - 2.f * ph, 0.f, -hd}, {hw, H, -hd + 2.f * ph}, metal);
  meshAddBox(v, {-hw, 0.f, hd - 2.f * ph}, {-hw + 2.f * ph, H, hd}, metal);
  meshAddBox(v, {hw - 2.f * ph, 0.f, hd - 2.f * ph}, {hw, H, hd}, metal);
  // OSB / particle-board decks — symmetric Z inset (matches inner post faces, not skewed toward the back).
  for (int i = 0; i < numShelves; ++i) {
    const float y0 = yBase + static_cast<float>(i) * yStep;
    const float y1 = y0 + shelfT;
    meshAddBox(v, {-hw + kShelfDeckInset, y0, -hd + kShelfDeckInset},
               {hw - kShelfDeckInset, y1, hd - kShelfDeckInset}, wood);
  }
  return v;
}

static std::vector<Vertex> makeShelfCrateUnitMesh() {
  std::vector<Vertex> v;
  meshAddBox(v, {-0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}, shelfCrateVertexColor());
  return v;
}

static std::vector<Vertex> makeShelfPalletUnitMesh() {
  std::vector<Vertex> v;
  meshAddBox(v, {-0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, 0.5f}, shelfPalletVertexColor());
  return v;
}

// Unit: x in [-0.5,0.5], y in [0,1], z in [-0.5,0.5] — bottom on ground, scaled per instance.
static std::vector<Vertex> makeShelfLadderUnitMesh() {
  std::vector<Vertex> v;
  const glm::vec3 col = shelfLadderVertexColor();
  constexpr float z0 = -0.06f;
  constexpr float z1 = 0.06f;
  meshAddBox(v, {-0.5f, 0.f, z0}, {-0.38f, 1.f, z1}, col);
  meshAddBox(v, {0.38f, 0.f, z0}, {0.5f, 1.f, z1}, col);
  for (int i = 0; i < 8; ++i) {
    const float y0 = 0.07f + static_cast<float>(i) * 0.118f;
    const float y1 = y0 + 0.038f;
    meshAddBox(v, {-0.38f, y0, z0}, {0.38f, y1, z1}, col);
  }
  return v;
}

static std::vector<Vertex> makeDeliCounterMesh() {
  std::vector<Vertex> v;
  const glm::vec3 cabinet{0.14f, 0.92f, 0.14f};
  const glm::vec3 top{0.15f, 0.08f, 0.92f};
  const float hl = kDeliCounterHalfL;
  const float hd = kDeliCounterHalfD;
  const float ch = kDeliCounterH;
  const float oh = kDeliCounterTopOverhang;
  const float th = kDeliCounterTopH;
  meshAddBox(v, {-hl, 0.f, -hd}, {hl, ch, hd}, cabinet);
  meshAddBox(v, {-hl - oh, ch, -hd - oh}, {hl + oh, ch + th, hd + oh}, top);
  const float glassBase = ch + th;
  const float gh = kDeliGlassHeight;
  const glm::vec3 glass{0.72f, 0.78f, 0.82f};
  const glm::vec3 frame{0.15f, 0.08f, 0.92f};
  const glm::vec4 gRgba{glass.x, glass.y, glass.z, 0.38f};
  const float gl = hl - 0.04f;
  const float roofDepth = 0.38f;
  const float peakY = glassBase + gh;
  auto quad = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, glm::vec3 n, glm::vec4 col) {
    v.push_back({a, n, col, {}});
    v.push_back({b, n, col, {}});
    v.push_back({c, n, col, {}});
    v.push_back({a, n, col, {}});
    v.push_back({c, n, col, {}});
    v.push_back({d, n, col, {}});
  };
  const float standH = 0.32f;
  const float standW = 0.03f;
  const float glassBottomY = glassBase + standH;
  for (float zSign : {-1.f, 1.f}) {
    const float edgeZ = zSign * (hd - 0.01f);
    const float tipZ  = zSign * (hd - 0.01f - roofDepth);
    const glm::vec3 nFace = glm::normalize(glm::vec3(0.f, roofDepth, gh * zSign));
    quad({-gl, glassBottomY, edgeZ}, {gl, glassBottomY, edgeZ},
         {gl, peakY, tipZ},          {-gl, peakY, tipZ},
         nFace, gRgba);
    quad({-gl, peakY, tipZ},         {gl, peakY, tipZ},
         {gl, glassBottomY, edgeZ},  {-gl, glassBottomY, edgeZ},
         -nFace, gRgba);
    for (float sx : {-1.f, 1.f}) {
      const float ex = sx * gl;
      meshAddBox(v, {ex - standW, glassBase, edgeZ - standW},
                    {ex + standW, glassBottomY + 0.01f, edgeZ + standW}, frame);
    }
  }
  const float roofInner = hd - 0.01f - roofDepth;
  meshAddBox(v, {-gl, peakY - 0.04f, -roofInner}, {gl, peakY, roofInner}, top);
  const glm::vec3 deliFloor{0.92f, 0.14f, 0.14f};
  const float fhx = kShelfAisleModulePitch * 0.5f;
  const float fhz = kShelfAlongAislePitch * 0.5f;
  meshAddBox(v, {-fhx, -0.01f, -fhz}, {fhx, 0.005f, fhz}, deliFloor);
  return v;
}

#if defined(VULKAN_GAME_PIZZA_SLICE_GLB)
// Simple procedural wedge so deli pizza shape is deterministic (no odd GLB topology).
static std::vector<Vertex> makeDeliPizzaSliceMesh() {
  std::vector<Vertex> v;
  const glm::vec4 rgba = vrgb(deliPizzaVertexColor());
  auto tri = [&](const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
    glm::vec3 n = glm::cross(b - a, c - a);
    const float nl = glm::length(n);
    n = (nl > 1e-6f) ? (n / nl) : glm::vec3(0.f, 1.f, 0.f);
    v.push_back({a, n, rgba, {}});
    v.push_back({c, n, rgba, {}});
    v.push_back({b, n, rgba, {}});
  };

  // 60-degree wedge centered at local origin (for 6-slice circle with no radial offset).
  const float tipX = 0.0f;
  const float crustX = 0.32f;
  const float halfW = 0.185f;
  const float h = 0.028f;
  const glm::vec3 tip0{tipX, 0.f, 0.f};
  const glm::vec3 l0{crustX, 0.f, -halfW};
  const glm::vec3 r0{crustX, 0.f, halfW};
  const glm::vec3 tip1{tipX, h, 0.f};
  const glm::vec3 l1{crustX, h, -halfW};
  const glm::vec3 r1{crustX, h, halfW};

  tri(tip1, l1, r1); // top
  tri(tip0, r0, l0); // bottom
  tri(tip0, tip1, l1);
  tri(tip0, l1, l0); // left side
  tri(tip0, r1, tip1);
  tri(tip0, r0, r1); // right side
  tri(l0, l1, r1);
  tri(l0, r1, r0);   // crust edge
  return v;
}
#endif

// Procedural low-poly sphere for deli meatballs (avoids problematic GLB topology/import artifacts).
static std::vector<Vertex> makeDeliMeatballMesh() {
  std::vector<Vertex> v;
  const glm::vec4 rgba = vrgb(deliMeatballVertexColor());
  constexpr int kLat = 8;
  constexpr int kLon = 12;
  constexpr float rx = 0.048f;
  constexpr float ry = 0.042f;
  constexpr float rz = 0.048f;
  v.reserve(static_cast<size_t>(kLat * kLon * 6));
  auto p = [](float u, float vv) {
    const float th = u * glm::two_pi<float>();
    const float ph = (vv - 0.5f) * glm::pi<float>();
    return glm::vec3(std::cos(ph) * std::cos(th), std::sin(ph), std::cos(ph) * std::sin(th));
  };
  auto emit = [&](const glm::vec3& an, const glm::vec3& bn, const glm::vec3& cn) {
    const glm::vec3 a{an.x * rx, an.y * ry, an.z * rz};
    const glm::vec3 b{bn.x * rx, bn.y * ry, bn.z * rz};
    const glm::vec3 c{cn.x * rx, cn.y * ry, cn.z * rz};
    // Match UI/world winding used elsewhere in this file.
    v.push_back({a, glm::normalize(glm::vec3(an.x / rx, an.y / ry, an.z / rz)), rgba, {}});
    v.push_back({c, glm::normalize(glm::vec3(cn.x / rx, cn.y / ry, cn.z / rz)), rgba, {}});
    v.push_back({b, glm::normalize(glm::vec3(bn.x / rx, bn.y / ry, bn.z / rz)), rgba, {}});
  };
  for (int iy = 0; iy < kLat; ++iy) {
    const float v0 = static_cast<float>(iy) / static_cast<float>(kLat);
    const float v1 = static_cast<float>(iy + 1) / static_cast<float>(kLat);
    for (int ix = 0; ix < kLon; ++ix) {
      const float u0 = static_cast<float>(ix) / static_cast<float>(kLon);
      const float u1 = static_cast<float>(ix + 1) / static_cast<float>(kLon);
      const glm::vec3 a = p(u0, v0);
      const glm::vec3 b = p(u1, v0);
      const glm::vec3 c = p(u1, v1);
      const glm::vec3 d = p(u0, v1);
      emit(a, b, c);
      emit(a, c, d);
    }
  }
  return v;
}

// Hanging fluorescent: thin wire + housing (local +Y up; placed under ceiling in world space).
static std::vector<Vertex> makeFluorescentFixtureMesh() {
  const glm::vec3 col = fluorescentLightVertexColor();
  std::vector<Vertex> v;
  const float hw = 0.95f;
  const float hd = 0.14f;
  meshAddBox(v, {-0.03f, -0.55f, -0.03f}, {0.03f, 0.0f, 0.03f}, shelfMetalVertexColor());
  meshAddBox(v, {-hw, -0.62f, -hd}, {hw, -0.5f, hd}, col);
  return v;
}


struct AABB {
  glm::vec3 min, max;
};

struct ParkourWallHit {
  bool hit = false;
  bool pillar = false;
  bool shelf = false;
  float t = 1e30f;
  int side = 0;
  AABB box{};
  glm::vec3 normal{0.f};
  glm::vec3 point{0.f};
};

struct SpeedVaultTarget {
  bool hit = false;
  float t = 1e30f;
  AABB box{};
  glm::vec2 dir{0.f};
  float topY = 0.f;
};

// One axis-aligned box in rack-local space → world AABB after Y rotation (matches visible mesh).
static AABB shelfLocalBoxWorldAABB(const glm::vec3& shelfPos, float shelfYawRad, const glm::vec3& mn,
                                   const glm::vec3& mx) {
  const glm::vec3 L[8] = {
      {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z}, {mx.x, mn.y, mx.z}, {mn.x, mn.y, mx.z},
      {mn.x, mx.y, mn.z}, {mx.x, mx.y, mn.z}, {mx.x, mx.y, mx.z}, {mn.x, mx.y, mx.z},
  };
  const glm::mat3 Rs = glm::mat3(glm::rotate(glm::mat4(1.f), shelfYawRad, glm::vec3(0.f, 1.f, 0.f)));
  glm::vec3 wmn(1e30f);
  glm::vec3 wmx(-1e30f);
  for (int i = 0; i < 8; ++i) {
    const glm::vec3 w = Rs * L[i] + shelfPos;
    wmn = glm::min(wmn, w);
    wmx = glm::max(wmx, w);
  }
  return AABB{wmn, wmx};
}

// Ladder prop: unit mesh x∈[-0.5,0.5], y∈[0,1], z thin; slightly inflated for collision.
static constexpr glm::vec3 kShelfLadderMeshColMin{-0.51f, -0.02f, -0.07f};
static constexpr glm::vec3 kShelfLadderMeshColMax{0.51f, 1.02f, 0.07f};

static AABB obbModelMatrixToWorldAABB(const glm::mat4& M, const glm::vec3& localMin,
                                      const glm::vec3& localMax) {
  const glm::vec3 L[8] = {
      {localMin.x, localMin.y, localMin.z}, {localMax.x, localMin.y, localMin.z},
      {localMax.x, localMin.y, localMax.z}, {localMin.x, localMin.y, localMax.z},
      {localMin.x, localMax.y, localMin.z}, {localMax.x, localMax.y, localMin.z},
      {localMax.x, localMax.y, localMax.z}, {localMin.x, localMax.y, localMax.z},
  };
  glm::vec3 wmn(1e30f);
  glm::vec3 wmx(-1e30f);
  for (int i = 0; i < 8; ++i) {
    const glm::vec4 wp = M * glm::vec4(L[i], 1.f);
    const glm::vec3 p(wp);
    wmn = glm::min(wmn, p);
    wmx = glm::max(wmx, p);
  }
  return AABB{wmn, wmx};
}

static glm::mat4 shelfLadderWorldModel(float cx, float cz, float yawDeg, float llx, float llz, float leanR,
                                     const glm::vec3& ladderScl) {
  const float shelfYawRad = glm::radians(yawDeg);
  return glm::translate(glm::mat4(1.f), glm::vec3(cx, kGroundY, cz)) *
         glm::rotate(glm::mat4(1.f), shelfYawRad, glm::vec3(0.f, 1.f, 0.f)) *
         glm::translate(glm::mat4(1.f), glm::vec3(llx, 0.f, llz)) *
         glm::rotate(glm::mat4(1.f), leanR, glm::vec3(1.f, 0.f, 0.f)) *
         glm::scale(glm::mat4(1.f), ladderScl);
}

static AABB shelfLadderCollisionAABB(const glm::mat4& ladderModel) {
  return obbModelMatrixToWorldAABB(ladderModel, kShelfLadderMeshColMin, kShelfLadderMeshColMax);
}

static AABB staffNpcWorldHitbox(float wx, float wz, float yawRad, float feetWorldY,
                                const glm::vec3& bodyScale = glm::vec3(1.f)) {
  const glm::vec3 feet(wx, feetWorldY, wz);
  const float sx = bodyScale.x;
  const float sy = bodyScale.y;
  const float sz = bodyScale.z;
  return shelfLocalBoxWorldAABB(feet, yawRad, {-kStaffHitHalfW * sx, 0.f, -kStaffHitHalfD * sz},
                                {kStaffHitHalfW * sx, kEmployeeVisualHeight * sy, kStaffHitHalfD * sz});
}

// Inflated vs staffNpcWorldHitbox: nudgeShelfEmployeesFromPlayer runs before melee damage, so physics AABB
// overlap is usually false even when a punch should land; this box approximates reach + post-nudge slack.
static AABB staffNpcMeleeDamageHitbox(const ShelfEmployeeNpc& e) {
  AABB b = staffNpcWorldHitbox(e.posXZ.x, e.posXZ.y, e.yaw, e.feetWorldY, e.bodyScale);
  constexpr float kMeleeHitPadXZ = 0.48f;
  constexpr float kMeleeHitPadYTop = 0.4f;
  b.min.x -= kMeleeHitPadXZ;
  b.max.x += kMeleeHitPadXZ;
  b.min.z -= kMeleeHitPadXZ;
  b.max.z += kMeleeHitPadXZ;
  b.max.y += kMeleeHitPadYTop;
  return b;
}

// Max XZ distance from feet pivot to local footprint corner (yaw rotation keeps this bound valid).
static float staffNpcFootprintRadiusXZ(const glm::vec3& bodyScale) {
  const float hx = kStaffHitHalfW * bodyScale.x;
  const float hz = kStaffHitHalfD * bodyScale.z;
  return std::sqrt(hx * hx + hz * hz);
}

bool aabbOverlap(const AABB& a, const AABB& b) {
  return a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y && a.max.y >= b.min.y &&
         a.min.z <= b.max.z && a.max.z >= b.min.z;
}

static float distSqPointXZToRect(float px, float pz, float minX, float maxX, float minZ, float maxZ) {
  const float qx = std::clamp(px, minX, maxX);
  const float qz = std::clamp(pz, minZ, maxZ);
  const float dx = px - qx;
  const float dz = pz - qz;
  return dx * dx + dz * dz;
}

// Shortest distance in XZ from a point to the perimeter of an axis-aligned rectangle.
static float distPointXZToRectPerimeter(float px, float pz, float minX, float maxX, float minZ, float maxZ) {
  const float dx0 = px - minX;
  const float dx1 = maxX - px;
  const float dz0 = pz - minZ;
  const float dz1 = maxZ - pz;
  if (dx0 >= -1e-4f && dx1 >= -1e-4f && dz0 >= -1e-4f && dz1 >= -1e-4f)
    return std::min(std::min(dx0, dx1), std::min(dz0, dz1));
  const float qx = std::clamp(px, minX, maxX);
  const float qz = std::clamp(pz, minZ, maxZ);
  const float dx = px - qx;
  const float dz = pz - qz;
  return std::sqrt(dx * dx + dz * dz);
}

static bool rayAABBFirstHit(const glm::vec3& ro, const glm::vec3& rd, const AABB& box, float& tOut) {
  float tmin = -1e30f;
  float tmax = 1e30f;
  for (int a = 0; a < 3; ++a) {
    const float o = ro[a];
    const float d = rd[a];
    if (std::abs(d) < 1e-7f) {
      if (o < box.min[a] - 1e-5f || o > box.max[a] + 1e-5f)
        return false;
      continue;
    }
    const float inv = 1.f / d;
    float t1 = (box.min[a] - o) * inv;
    float t2 = (box.max[a] - o) * inv;
    if (t1 > t2)
      std::swap(t1, t2);
    tmin = std::max(tmin, t1);
    tmax = std::min(tmax, t2);
    if (tmin > tmax)
      return false;
  }
  if (tmax < 0.f)
    return false;
  tOut = tmin >= 0.f ? tmin : tmax;
  return tOut >= 0.f;
}

static glm::vec3 outwardNormalFromAabbHitPointXZ(const glm::vec3& hitPoint, const AABB& box) {
  const float dxMin = std::abs(hitPoint.x - box.min.x);
  const float dxMax = std::abs(box.max.x - hitPoint.x);
  const float dzMin = std::abs(hitPoint.z - box.min.z);
  const float dzMax = std::abs(box.max.z - hitPoint.z);
  const float best = std::min(std::min(dxMin, dxMax), std::min(dzMin, dzMax));
  if (best == dxMin)
    return {-1.f, 0.f, 0.f};
  if (best == dxMax)
    return {1.f, 0.f, 0.f};
  if (best == dzMin)
    return {0.f, 0.f, -1.f};
  return {0.f, 0.f, 1.f};
}

// When lip-aim rays all miss, still allow grab if the screen-center ray hits the deck top somewhere
// nearby (no visible crosshair / hands).
static bool mantleAssistDeckTopHit(const glm::vec3& ro, const glm::vec3& rd, const AABB& deck,
                                   float topSlop, float& outT, float minRdY) {
  if (rd.y < minRdY)
    return false;
  const float tTop = (deck.max.y - ro.y) / rd.y;
  const float tMax = mantleLedgeTopPlaneTMax(ro, deck.max.y, rd);
  if (tTop <= 0.f || tTop > tMax)
    return false;
  const glm::vec3 ph = ro + rd * tTop;
  const float px = kLedgeGrabAssistXZPad + topSlop;
  if (ph.x < deck.min.x - px || ph.x > deck.max.x + px || ph.z < deck.min.z - px ||
      ph.z > deck.max.z + px)
    return false;
  outT = tTop;
  return true;
}

// Last-resort: hugging shelf, huge XZ pad — only needs a slightly upward view ray to hit deck top.
static bool mantleMegaAssistDeckTop(const glm::vec3& ro, const glm::vec3& rd, const AABB& deck,
                                    float topSlop, float& outT) {
  if (rd.y < 0.0006f)
    return false;
  const float tTop = (deck.max.y - ro.y) / rd.y;
  if (tTop <= 0.f)
    return false;
  const float tMax = mantleLedgeTopPlaneTMax(ro, deck.max.y, rd);
  if (tTop > tMax)
    return false;
  const glm::vec3 ph = ro + rd * tTop;
  const float px = kLedgeGrabMegaAssistXZPad + topSlop;
  if (ph.x < deck.min.x - px || ph.x > deck.max.x + px || ph.z < deck.min.z - px ||
      ph.z > deck.max.z + px)
    return false;
  outT = tTop;
  return true;
}

// When close to shelf, aim a ray from the eye toward deck-top center (magnetic grab).
static bool mantleSnapRayToDeckTop(const glm::vec3& ro, const AABB& deck, float topSlop, float& outT,
                                   glm::vec3& outRd) {
  const glm::vec3 target((deck.min.x + deck.max.x) * 0.5f, deck.max.y + std::max(0.07f, topSlop * 0.4f),
                         (deck.min.z + deck.max.z) * 0.5f);
  glm::vec3 rd = target - ro;
  const float len = glm::length(rd);
  if (len < 0.012f)
    return false;
  rd *= 1.f / len;
  if (rd.y < 0.0055f)
    return false;
  const float tTop = (deck.max.y - ro.y) / rd.y;
  if (tTop <= 0.f)
    return false;
  const float tMax = mantleLedgeTopPlaneTMax(ro, deck.max.y, rd);
  if (tTop > tMax)
    return false;
  const glm::vec3 ph = ro + rd * tTop;
  const float px = topSlop + 0.62f;
  if (ph.x < deck.min.x - px || ph.x > deck.max.x + px || ph.z < deck.min.z - px || ph.z > deck.max.z + px)
    return false;
  outT = tTop;
  outRd = rd;
  return true;
}

static constexpr int kMantleCrosshairRayCountMax = 30;

// Center ray plus three rings in the view plane — dense cone so aim doesn’t need to be precise.
static int buildMantleCrosshairRayFan(const glm::vec3& rdCenter, const glm::vec3& camRight,
                                      const glm::vec3& camUp, float halfAngleRad,
                                      std::array<glm::vec3, kMantleCrosshairRayCountMax>& out) {
  if (halfAngleRad < 1e-5f) {
    out[0] = rdCenter;
    return 1;
  }
  out[0] = rdCenter;
  int n = 1;
  const float tau = 6.283185307179586f;
  const float tanOuter = std::tan(halfAngleRad);
  constexpr int kOuterRing = 12;
  for (int i = 0; i < kOuterRing; ++i) {
    const float ang = (tau * static_cast<float>(i)) / static_cast<float>(kOuterRing);
    const glm::vec3 off = camRight * (std::cos(ang) * tanOuter) + camUp * (std::sin(ang) * tanOuter);
    out[n++] = glm::normalize(rdCenter + off);
  }
  const float tanMid = std::tan(halfAngleRad * 0.7f);
  constexpr int kMidRing = 8;
  for (int i = 0; i < kMidRing; ++i) {
    const float ang = (tau * (static_cast<float>(i) + 0.5f)) / static_cast<float>(kMidRing);
    const glm::vec3 off = camRight * (std::cos(ang) * tanMid) + camUp * (std::sin(ang) * tanMid);
    out[n++] = glm::normalize(rdCenter + off);
  }
  const float tanInner = std::tan(halfAngleRad * 0.42f);
  constexpr int kInnerRing = 6;
  for (int i = 0; i < kInnerRing; ++i) {
    const float ang = (tau * static_cast<float>(i)) / static_cast<float>(kInnerRing);
    const glm::vec3 off = camRight * (std::cos(ang) * tanInner) + camUp * (std::sin(ang) * tanInner);
    out[n++] = glm::normalize(rdCenter + off);
  }
  return n;
}

// Shared shelf-deck / crate-top mantle test (horizontal slab; max.y = stand surface).
struct MantleProbeParams {
  float feet;
  float velY;
  float maxVelUp;
  float minFallGrab;
  float reach2;
  glm::vec3 camPos;
  glm::vec3 ro;
  glm::vec3 rdCenter;
  glm::vec3 camRight;
  glm::vec3 camUp;
  glm::vec2 horizVel;
  float runSp;
  float runTEff;
  float eyeHeight;
};

static void mantleConsiderHorizontalLedge(const MantleProbeParams& mp, const AABB& ledgeTop,
                                          float aisleCX, float rackCenterX, float& bestTHit,
                                          glm::vec3& bestEnd, bool& have, AABB& bestDeckOut,
                                          bool crateMantle = false,
                                          float nextShelfDeckBottomYWorld = 1e30f) {
  const float rise = ledgeTop.max.y - mp.feet;
  if (rise < kLedgeGrabMinRise || rise > kLedgeGrabMaxRise)
    return;
  if (mp.velY > mp.maxVelUp || mp.velY < mp.minFallGrab)
    return;
  const float dsq = distSqPointXZToRect(mp.camPos.x, mp.camPos.z, ledgeTop.min.x, ledgeTop.max.x,
                                        ledgeTop.min.z, ledgeTop.max.z);
  const float reachCap2 = crateMantle ? mp.reach2 * 1.1f * 1.1f : mp.reach2;
  if (dsq < 1e-5f || dsq > reachCap2)
    return;
  const float dcx = (ledgeTop.min.x + ledgeTop.max.x) * 0.5f;
  const float dcz = (ledgeTop.min.z + ledgeTop.max.z) * 0.5f;
  // Standing room on crate top (world AABB); both axes must fit player footprint + pull margin.
  if (crateMantle) {
    constexpr float kStandPad = 0.11f;
    const float needXZ = 2.f * kPlayerHalfXZ + kStandPad;
    const float wx = ledgeTop.max.x - ledgeTop.min.x;
    const float wz = ledgeTop.max.z - ledgeTop.min.z;
    if (wx < needXZ || wz < needXZ)
      return;
    // Head must clear the next shelf deck above (when there is one).
    if (nextShelfDeckBottomYWorld < 1e29f &&
        ledgeTop.max.y + mp.eyeHeight + 0.15f >= nextShelfDeckBottomYWorld - 0.07f)
      return;
  }
  if (!crateMantle) {
    const glm::vec2 towardAisle(aisleCX - rackCenterX, 0.f);
    const float taLen = glm::length(towardAisle);
    if (taLen > 1e-4f) {
      const glm::vec2 fn = towardAisle * (1.f / taLen);
      const float towardAisleDot = glm::dot(glm::vec2(mp.camPos.x - dcx, mp.camPos.z - dcz), fn);
      if (towardAisleDot < kLedgeGrabMinTowardAisleDotXZ)
        return;
    }
  }
  {
    const float minLookToward =
        crateMantle ? 0.035f : kLedgeGrabMinLookTowardDeckDotXZ;
    glm::vec2 fwdH(mp.rdCenter.x, mp.rdCenter.z);
    const float fhl = glm::length(fwdH);
    if (fhl > 1e-4f) {
      fwdH *= 1.f / fhl;
      glm::vec2 toDeck(dcx - mp.ro.x, dcz - mp.ro.z);
      const float tdl = glm::length(toDeck);
      if (tdl > 0.035f) {
        toDeck *= 1.f / tdl;
        if (glm::dot(fwdH, toDeck) < minLookToward)
          return;
      }
    }
  }
  const float footLipDist =
      distPointXZToRectPerimeter(mp.camPos.x, mp.camPos.z, ledgeTop.min.x, ledgeTop.max.x,
                                 ledgeTop.min.z, ledgeTop.max.z);
  const bool lipRelaxed =
      footLipDist < (crateMantle ? kLedgeGrabRelaxLipFootM * 1.35f : kLedgeGrabRelaxLipFootM);

  const float riseSpan = kLedgeGrabMaxRise - kLedgeGrabMinRise;
  const float riseNorm =
      glm::clamp((rise - kLedgeGrabMinRise) / glm::max(riseSpan, 0.04f), 0.f, 1.f);
  const float aimTilt = glm::mix(-kLedgeGrabAimTiltLowDown, kLedgeGrabAimTiltHighUp, riseNorm);
  glm::vec3 rdAim = mp.rdCenter + mp.camUp * aimTilt;
  const float rdAimLen = glm::length(rdAim);
  if (rdAimLen > 1e-5f)
    rdAim *= 1.f / rdAimLen;
  else
    rdAim = mp.rdCenter;
  const float coneHalfDeg =
      glm::mix(kLedgeGrabConeHalfLowRiseDeg, kLedgeGrabConeHalfHighRiseDeg, riseNorm);
  std::array<glm::vec3, kMantleCrosshairRayCountMax> rayDirs{};
  const int nMantleRays =
      buildMantleCrosshairRayFan(rdAim, mp.camRight, mp.camUp, glm::radians(coneHalfDeg), rayDirs);
  const float assistMinRdY =
      glm::mix(kLedgeGrabAssistMinLookYLowRise, kLedgeGrabAssistMinLookYHighRise, riseNorm);

  float bestTHitDeck = 1e30f;
  glm::vec3 bestRd = rdAim;
  const float sl = kLedgeCrosshairTopSlop;
  const float edgeBand = crateMantle ? kLedgeCrosshairEdgeBand * 1.35f : kLedgeCrosshairEdgeBand;
  for (int ri = 0; ri < nMantleRays; ++ri) {
    const glm::vec3& rd = rayDirs[static_cast<size_t>(ri)];
    float tHit = 0.f;
    bool aimOk = false;
    if (std::abs(rd.y) > kLedgeGrabRayMinYForTopPlane) {
      const float tTop = (ledgeTop.max.y - mp.ro.y) / rd.y;
      const float tTopMax = mantleLedgeTopPlaneTMax(mp.ro, ledgeTop.max.y, rd);
      if (tTop > 0.f && tTop <= tTopMax) {
        const glm::vec3 ph = mp.ro + rd * tTop;
        if (ph.x >= ledgeTop.min.x - sl && ph.x <= ledgeTop.max.x + sl && ph.z >= ledgeTop.min.z - sl &&
            ph.z <= ledgeTop.max.z + sl) {
          const float ed = distPointXZToRectPerimeter(ph.x, ph.z, ledgeTop.min.x, ledgeTop.max.x,
                                                      ledgeTop.min.z, ledgeTop.max.z);
          if (lipRelaxed || ed <= edgeBand) {
            tHit = tTop;
            aimOk = true;
          }
        }
      }
    } else if (rayAABBFirstHit(mp.ro, rd, ledgeTop, tHit) && tHit > 0.f &&
               tHit <= mantleLedgeTopPlaneTMax(mp.ro, ledgeTop.max.y, rd)) {
      const glm::vec3 ph = mp.ro + rd * tHit;
      if (ph.y >= ledgeTop.min.y - sl && ph.y <= ledgeTop.max.y + sl) {
        const float ed = distPointXZToRectPerimeter(ph.x, ph.z, ledgeTop.min.x, ledgeTop.max.x,
                                                    ledgeTop.min.z, ledgeTop.max.z);
        if (lipRelaxed || ed <= edgeBand)
          aimOk = true;
      }
    }
    if (aimOk && tHit < bestTHitDeck) {
      bestTHitDeck = tHit;
      bestRd = rd;
    }
  }
  if (bestTHitDeck >= 1e29f) {
    float tAssist = 0.f;
    glm::vec3 snapRd = rdAim;
    if (mantleAssistDeckTopHit(mp.ro, rdAim, ledgeTop, sl, tAssist, assistMinRdY)) {
      bestTHitDeck = tAssist;
      bestRd = rdAim;
    } else if (footLipDist < kLedgeGrabNearLipForAssistM &&
               mantleAssistDeckTopHit(mp.ro, rdAim, ledgeTop, sl, tAssist,
                                      kLedgeGrabAssistMinLookYNearLip)) {
      bestTHitDeck = tAssist;
      bestRd = rdAim;
    } else if (footLipDist < kLedgeGrabMegaAssistFootLipM &&
               mantleMegaAssistDeckTop(mp.ro, rdAim, ledgeTop, sl, tAssist)) {
      bestTHitDeck = tAssist;
      bestRd = rdAim;
    } else if (footLipDist < kLedgeGrabSnapRayFootLipM &&
               mantleSnapRayToDeckTop(mp.ro, ledgeTop, sl, tAssist, snapRd)) {
      bestTHitDeck = tAssist;
      bestRd = snapRd;
    } else if (footLipDist < kLedgeGrabHugAutoFootLipM) {
      bestTHitDeck = 8.f;
      bestRd = rdAim;
    }
  }
  if (bestTHitDeck >= 1e29f)
    return;

  {
    glm::vec2 fwdH(mp.rdCenter.x, mp.rdCenter.z);
    const float fhl = glm::length(fwdH);
    if (fhl > 1e-4f) {
      fwdH *= 1.f / fhl;
      glm::vec2 rdH(bestRd.x, bestRd.z);
      const float rhl = glm::length(rdH);
      if (rhl > 0.11f) {
        rdH *= 1.f / rhl;
        if (glm::dot(fwdH, rdH) <
            (crateMantle ? 0.02f : kLedgeGrabMinMantleRayForwardDotXZ))
          return;
      }
    }
  }

  if (ledgeTop.max.y + mp.eyeHeight + 0.12f >= kCeilingY - 0.05f)
    return;
  const float mx = kPlayerHalfXZ + (crateMantle ? 0.08f : 0.1f);
  if (!crateMantle) {
    if (ledgeTop.max.x - ledgeTop.min.x < mx * 2.f + 0.05f ||
        ledgeTop.max.z - ledgeTop.min.z < mx * 2.f + 0.05f)
      return;
  }

  glm::vec2 pullH(bestRd.x, bestRd.z);
  float plen = glm::length(pullH);
  if (plen < 0.08f) {
    const glm::vec2 ctr((ledgeTop.min.x + ledgeTop.max.x) * 0.5f, (ledgeTop.min.z + ledgeTop.max.z) * 0.5f);
    const glm::vec2 p(mp.camPos.x, mp.camPos.z);
    glm::vec2 v = ctr - p;
    if (glm::length(v) < 1e-4f)
      return;
    pullH = glm::normalize(v);
  } else
    pullH *= 1.f / plen;
  if (mp.runSp > 0.28f) {
    const float momT =
        glm::clamp(std::max(mp.runTEff, mp.runSp / std::max(kSprintSpeed, 0.01f) * 0.78f), 0.f, 1.f);
    const float hvl = glm::length(mp.horizVel);
    const glm::vec2 hv = hvl > 1e-4f ? mp.horizVel * (1.f / hvl) : pullH;
    pullH = glm::normalize(pullH + hv * (0.64f * momT));
  }

  float endX = mp.camPos.x + pullH.x * kLedgeGrabFwdPull;
  float endZ = mp.camPos.z + pullH.y * kLedgeGrabFwdPull;
  endX = std::clamp(endX, ledgeTop.min.x + mx, ledgeTop.max.x - mx);
  endZ = std::clamp(endZ, ledgeTop.min.z + mx, ledgeTop.max.z - mx);
  const float endFeetY = ledgeTop.max.y + 0.04f;
  const glm::vec3 cand(endX, endFeetY + mp.eyeHeight, endZ);
  if (bestTHitDeck < bestTHit) {
    bestTHit = bestTHitDeck;
    bestEnd = cand;
    bestDeckOut = ledgeTop;
    have = true;
  }
}

void resolveAABBMinPenetration(AABB& player, const AABB& box) {
  if (!aabbOverlap(player, box))
    return;
  const float ox = std::min(player.max.x, box.max.x) - std::max(player.min.x, box.min.x);
  const float oy = std::min(player.max.y, box.max.y) - std::max(player.min.y, box.min.y);
  const float oz = std::min(player.max.z, box.max.z) - std::max(player.min.z, box.min.z);
  if (ox < oy && ox < oz) {
    const float pc = (player.min.x + player.max.x) * 0.5f;
    const float bc = (box.min.x + box.max.x) * 0.5f;
    const float push = (pc < bc) ? -ox : ox;
    player.min.x += push;
    player.max.x += push;
  } else if (oy < oz) {
    const float pc = (player.min.y + player.max.y) * 0.5f;
    const float bc = (box.min.y + box.max.y) * 0.5f;
    const float push = (pc < bc) ? -oy : oy;
    player.min.y += push;
    player.max.y += push;
  } else {
    const float pc = (player.min.z + player.max.z) * 0.5f;
    const float bc = (box.min.z + box.max.z) * 0.5f;
    const float push = (pc < bc) ? -oz : oz;
    player.min.z += push;
    player.max.z += push;
  }
}

// Staff stay on kGroundY — push only on X or Z so decks don’t lift NPCs vertically.
static void resolveAABBMinPenetrationXZ(AABB& mover, const AABB& box) {
  if (!aabbOverlap(mover, box))
    return;
  const float oy = std::min(mover.max.y, box.max.y) - std::max(mover.min.y, box.min.y);
  if (oy <= 0.f)
    return;
  const float ox = std::min(mover.max.x, box.max.x) - std::max(mover.min.x, box.min.x);
  const float oz = std::min(mover.max.z, box.max.z) - std::max(mover.min.z, box.min.z);
  if (ox <= 0.f || oz <= 0.f)
    return;
  if (ox <= oz) {
    const float pc = (mover.min.x + mover.max.x) * 0.5f;
    const float bc = (box.min.x + box.max.x) * 0.5f;
    const float push = (pc < bc) ? -ox : ox;
    mover.min.x += push;
    mover.max.x += push;
  } else {
    const float pc = (mover.min.z + mover.max.z) * 0.5f;
    const float bc = (box.min.z + box.max.z) * 0.5f;
    const float push = (pc < bc) ? -oz : oz;
    mover.min.z += push;
    mover.max.z += push;
  }
}

// Deck / crate volumes: stand on top or auto step-up (player kMaxStepHeight); do not XZ-shove off thin ledges.
// customMaxRise > 0: staff in night chase may step a full shelf tier when overlapping a deck/crate (resolveStaffNpc).
static void resolveStaffAgainstShelfWalkableSurface(AABB& s, ShelfEmployeeNpc& e, const AABB& box,
                                                    float customMaxRise = -1.f) {
  if (!aabbOverlap(s, box))
    return;
  constexpr float kOnTopSlop = 0.14f;
  const float feet = s.min.y;
  const float top = box.max.y;
  if (feet >= top - kOnTopSlop)
    return;
  const float rise = top - feet;
  const float maxRise =
      (customMaxRise > 0.f) ? customMaxRise : (kMaxStepHeight + 0.1f);
  if (rise > 0.f && rise <= maxRise) {
    if (e.staffVelY < kStaffFallNoAutoStepVelY) {
      resolveAABBMinPenetrationXZ(s, box);
      return;
    }
    s.min.y = top;
    s.max.y += rise;
    e.feetWorldY = top;
    e.staffVelY = 0.f;
    return;
  }
  resolveAABBMinPenetrationXZ(s, box);
}

void syncCamFromPlayerAABB(const AABB& p, glm::vec3& camPos, float eyeHeight) {
  camPos.x = (p.min.x + p.max.x) * 0.5f;
  camPos.z = (p.min.z + p.max.z) * 0.5f;
  camPos.y = p.min.y + eyeHeight;
}

// Thin horizontal ledges (shelf decks): step up within kMaxStepHeight; skip when already on top.
static void resolveShortLedgeStepUp(AABB& player, float velY, const AABB& ledge) {
  constexpr float kOnTopEps = 0.03f;
  constexpr float kVelRejectStep = 0.2f;
  if (!aabbOverlap(player, ledge))
    return;
  const float feet = player.min.y;
  const float top = ledge.max.y;
  if (feet >= top - kOnTopEps)
    return;
  const float rise = top - feet;
  if (rise > kMaxStepHeight + 0.02f) {
    resolveAABBMinPenetration(player, ledge);
    return;
  }
  if (velY > kVelRejectStep) {
    resolveAABBMinPenetration(player, ledge);
    return;
  }
  if (player.max.y + rise >= kCeilingY - 0.08f) {
    resolveAABBMinPenetration(player, ledge);
    return;
  }
  player.min.y = top;
  player.max.y += rise;
}

static AABB pillarCollisionAABB(float px, float pz) {
  return AABB{{px - kPillarHalfW, kGroundY, pz - kPillarHalfD},
              {px + kPillarHalfW, kGroundY + kPillarHeight, pz + kPillarHalfD}};
}

static bool shelfRackIntersectsAnyPillar(float cx, float cz, float yawDeg) {
  const glm::vec3 shelfPos{cx, kGroundY, cz};
  const float yawRad = glm::radians(yawDeg);
  const float hw = kShelfMeshHalfW;
  const float hd = kShelfMeshHalfD;
  const float H = kShelfMeshHeight;
  const AABB shelfBox =
      shelfLocalBoxWorldAABB(shelfPos, yawRad, {-hw, 0.f, -hd}, {hw, H, hd});
  const int ix = static_cast<int>(std::floor(cx / kPillarSpacing));
  const int iz = static_cast<int>(std::floor(cz / kPillarSpacing));
  for (int dz = -2; dz <= 2; ++dz) {
    for (int dx = -2; dx <= 2; ++dx) {
      const float px = static_cast<float>(ix + dx) * kPillarSpacing;
      const float pz = static_cast<float>(iz + dz) * kPillarSpacing;
      if (aabbOverlap(shelfBox, pillarCollisionAABB(px, pz)))
        return true;
    }
  }
  return false;
}

static bool deliBarSlotOccupied(int worldAisleI, int worldAlongI) {
  const int ca = shelfBiomeClusterCoord(worldAisleI, kDeliBiomeClusterSpan);
  const int cl = shelfBiomeClusterCoord(worldAlongI, kDeliBiomeClusterSpan);
  int baseA, baseL;
  if (!deliClusterBase(ca, cl, baseA, baseL)) return false;
  const int da = worldAisleI - baseA;
  const int dl = worldAlongI - baseL;
  if (da < 0 || da > 1 || dl < 0 || dl > 1) return false;
  const float cx = (static_cast<float>(worldAisleI) + 0.5f) * kShelfAisleModulePitch;
  const float cz = (static_cast<float>(worldAlongI) + 0.5f) * kShelfAlongAislePitch;
  const float hl = kDeliCounterHalfL + kDeliCounterTopOverhang;
  const float hd = kDeliCounterHalfD + kDeliCounterTopOverhang;
  const AABB deliBox = {{cx - hl, kGroundY, cz - hd},
                        {cx + hl, kGroundY + 1.f, cz + hd}};
  const int ix = static_cast<int>(std::floor(cx / kPillarSpacing));
  const int iz = static_cast<int>(std::floor(cz / kPillarSpacing));
  for (int dz = -1; dz <= 1; ++dz)
    for (int dx = -1; dx <= 1; ++dx) {
      const float px = static_cast<float>(ix + dx) * kPillarSpacing;
      const float pz = static_cast<float>(iz + dz) * kPillarSpacing;
      if (aabbOverlap(deliBox, pillarCollisionAABB(px, pz)))
        return false;
    }
  return true;
}

struct ShelfOccSlot {
  uint64_t gen = 0;
  uint64_t mix = 0;
  int32_t wa = 0;
  int32_t wl = 0;
  int8_t side = 0;
  bool occ = false;
};
// Open addressing (linear probe), fixed allocation; generation bump instead of unordered_map::clear spikes.
static std::vector<ShelfOccSlot> gShelfOccTable;
static uint64_t gShelfOccGen = 1;
static size_t gShelfOccCount = 0;
static constexpr size_t kShelfSlotOccTableSize = size_t{1} << 17;
static constexpr size_t kShelfSlotOccTableMask = kShelfSlotOccTableSize - 1;
static constexpr size_t kShelfSlotOccCacheMaxEntries = 98304u;

static uint64_t shelfSlotOccMixKey(int wa, int wl, int side) {
  uint64_t x = static_cast<uint64_t>(static_cast<uint32_t>(wa));
  x ^= static_cast<uint64_t>(static_cast<uint32_t>(wl)) + 0x9e3779b97f4a7c15ull + (x << 6) + (x >> 2);
  x ^= static_cast<uint64_t>(static_cast<uint8_t>(side)) + 0x9e3779b97f4a7c15ull + (x << 6) + (x >> 2);
  return x;
}

static void shelfOccEnsureTable() {
  if (gShelfOccTable.empty())
    gShelfOccTable.resize(kShelfSlotOccTableSize);
}

static void shelfOccBumpGeneration() {
  ++gShelfOccGen;
  if (gShelfOccGen == 0)
    gShelfOccGen = 1;
  gShelfOccCount = 0;
}

static bool shelfSlotOccupied(int worldAisleI, int worldAlongI, int side) {
  PerfScopeTimer _occScope(&gPerfTrace.accumShelfOcc, gPerfTrace.enabled);
  constexpr int kEntranceVoidHalf = 3;
  if (std::max(std::abs(worldAisleI), std::abs(worldAlongI)) <= kEntranceVoidHalf)
    return false;
  shelfOccEnsureTable();
  const uint64_t mix = shelfSlotOccMixKey(worldAisleI, worldAlongI, side);
  const auto computeOcc = [&]() -> bool {
    if (cellInDeliClearZone(worldAisleI, worldAlongI))
      return false;
    const int ca = shelfBiomeClusterCoord(worldAisleI, kShelfBiomeClusterSpan);
    const int cAlong = shelfBiomeClusterCoord(worldAlongI, kShelfBiomeClusterSpan);
    const uint32_t cell = scp3008ShelfHash(ca, cAlong, 0);
    if ((cell & 3u) != 0u)
      return false;
    const float aisleCX = (static_cast<float>(worldAisleI) + 0.5f) * kShelfAisleModulePitch;
    const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
    const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
    const float rackZ = (static_cast<float>(worldAlongI) + 0.5f) * kShelfAlongAislePitch;
    const float cx = side ? cxRight : cxLeft;
    const float yawDeg = side ? -90.0f : 90.0f;
    if (shelfRackIntersectsAnyPillar(cx, rackZ, yawDeg))
      return false;
    return true;
  };
  for (int rep = 0; rep < 2; ++rep) {
    if (gShelfOccCount >= kShelfSlotOccCacheMaxEntries)
      shelfOccBumpGeneration();
    size_t idx = static_cast<size_t>(mix) & kShelfSlotOccTableMask;
    for (size_t probe = 0; probe < kShelfSlotOccTableSize; ++probe) {
      ShelfOccSlot& s = gShelfOccTable[idx];
      if (s.gen != gShelfOccGen) {
        const bool occ = computeOcc();
        s.gen = gShelfOccGen;
        s.mix = mix;
        s.wa = worldAisleI;
        s.wl = worldAlongI;
        s.side = static_cast<int8_t>(side);
        s.occ = occ;
        ++gShelfOccCount;
        return occ;
      }
      if (s.mix == mix && s.wa == worldAisleI && s.wl == worldAlongI && s.side == side)
        return s.occ;
      idx = (idx + 1) & kShelfSlotOccTableMask;
    }
    shelfOccBumpGeneration();
  }
  return computeOcc();
}

static bool shelfCrateLocalLayout(int worldAisleI, int worldAlongI, int side, float& lx, float& lz,
                                  float& yDeckTop, float& hx, float& hy, float& hz) {
  if (!shelfSlotOccupied(worldAisleI, worldAlongI, side))
    return false;
  if (scp3008ShelfHash(worldAisleI, worldAlongI, 0xCAAE1234) % 17 != 0)
    return false;
  const uint32_t H0 = scp3008ShelfHash(worldAisleI, worldAlongI, 0x71FEu ^ static_cast<uint32_t>(side * 19));
  const int deckIdx = static_cast<int>(H0 % static_cast<uint32_t>(kShelfDeckCount));
  constexpr float yBase = 0.12f;
  const float yStep = kShelfGapBetweenLevels + kShelfDeckThickness;
  const float y0 = yBase + static_cast<float>(deckIdx) * yStep;
  yDeckTop = y0 + kShelfDeckThickness;
  const float hw = kShelfMeshHalfW;
  const float hd = kShelfMeshHalfD;
  const float mnX = -hw + kShelfDeckInset;
  const float mnZ = -hd + kShelfDeckInset;
  const float mxX = hw - kShelfDeckInset;
  const float mxZ = hd - kShelfDeckInset;
  const uint32_t H = scp3008ShelfHash(worldAisleI, worldAlongI, 0xDEDA0001u ^ static_cast<uint32_t>(side + 1u));
  auto u01 = [&](unsigned sh) {
    return static_cast<float>((H >> (sh % 22u)) & 0x3FFu) / 1023.f;
  };
  hx = glm::mix(0.94f, 1.12f, u01(0));
  hz = glm::mix(0.60f, 0.82f, u01(3));
  hy = glm::mix(1.12f, 1.48f, u01(6));
  const float maxHalfW = (mxX - mnX) * 0.5f - 0.07f;
  const float maxHalfD = (mxZ - mnZ) * 0.5f - 0.06f;
  hx = std::min(hx, maxHalfW);
  hz = std::min(hz, maxHalfD);
  hy = std::min(hy, kShelfGapBetweenLevels * 0.46f);
  lx = glm::mix(mnX + hx + 0.09f, mxX - hx - 0.09f, u01(9));
  lz = glm::mix(mnZ + hz + 0.07f, mxZ - hz - 0.07f, u01(12));
  return true;
}

// Pallets: higher spawn than crates; never placed on crate-occupied racks.
static bool shelfPalletLocalLayout(int worldAisleI, int worldAlongI, int side, bool cratePresent, float& lx,
                                   float& lz, float& yDeckTop, float& hx, float& hy, float& hz,
                                   int& stackCount, float& yaw0, float& yaw1, float& stackOffX,
                                   float& stackOffZ) {
  if (cratePresent || !shelfSlotOccupied(worldAisleI, worldAlongI, side))
    return false;
  const uint32_t Hs =
      scp3008ShelfHash(worldAisleI, worldAlongI, 0xA11E7701u ^ static_cast<uint32_t>(side * 37));
  if ((Hs % 100u) >= 78u)
    return false;

  const uint32_t H0 = scp3008ShelfHash(worldAisleI, worldAlongI, 0x511F0A9Du ^ static_cast<uint32_t>(side * 13));
  const int deckIdx = static_cast<int>(H0 % static_cast<uint32_t>(kShelfDeckCount));
  constexpr float yBase = 0.12f;
  const float yStep = kShelfGapBetweenLevels + kShelfDeckThickness;
  const float y0 = yBase + static_cast<float>(deckIdx) * yStep;
  yDeckTop = y0 + kShelfDeckThickness;

  const float hw = kShelfMeshHalfW;
  const float hd = kShelfMeshHalfD;
  const float mnX = -hw + kShelfDeckInset;
  const float mnZ = -hd + kShelfDeckInset;
  const float mxX = hw - kShelfDeckInset;
  const float mxZ = hd - kShelfDeckInset;
  const uint32_t H = scp3008ShelfHash(worldAisleI, worldAlongI, 0x61E7703Du ^ static_cast<uint32_t>(side + 3u));
  auto u01 = [&](unsigned sh) {
    return static_cast<float>((H >> (sh % 22u)) & 0x3FFu) / 1023.f;
  };

  hx = glm::mix(0.82f, 1.04f, u01(0));
  hz = glm::mix(0.46f, 0.68f, u01(3));
  hy = glm::mix(0.08f, 0.13f, u01(6));
  const float maxHalfW = (mxX - mnX) * 0.5f - 0.05f;
  const float maxHalfD = (mxZ - mnZ) * 0.5f - 0.05f;
  hx = std::min(hx, maxHalfW);
  hz = std::min(hz, maxHalfD);
  hy = std::min(hy, 0.15f);
  lx = glm::mix(mnX + hx + 0.05f, mxX - hx - 0.05f, u01(9));
  lz = glm::mix(mnZ + hz + 0.05f, mxZ - hz - 0.05f, u01(12));

  stackCount = ((Hs >> 7u) & 1u) ? 2 : 1;
  yaw0 = (u01(14) - 0.5f) * glm::radians(8.0f);
  // Keep the second pallet more stable: excessive twist/offset made 2-stack collision feel buggy.
  yaw1 = (u01(16) - 0.5f) * glm::radians(10.0f);
  stackOffX = (u01(18) - 0.5f) * (2.f * hx * 0.14f);
  stackOffZ = (u01(20) - 0.5f) * (2.f * hz * 0.14f);
  return true;
}

static bool shelfLadderLocalLayout(int worldAisleI, int worldAlongI, int side, float& lx, float& lz,
                                   float& leanRad, glm::vec3& scaleWHD) {
  if (!shelfSlotOccupied(worldAisleI, worldAlongI, side))
    return false;
  const uint32_t Hr =
      scp3008ShelfHash(worldAisleI, worldAlongI, 0xBADD10CEu ^ static_cast<uint32_t>(side * 101));
  // Crates use ~1/17 (~5.9%); ladders much rarer.
  if ((Hr % 71u) != 0u)
    return false;
  const float hw = kShelfMeshHalfW;
  const float hd = kShelfMeshHalfD;
  const float ph = kShelfPostGauge;
  // Uniform ladder profile so every spawned ladder is the same.
  leanRad = glm::radians(18.0f);
  const float cosL = std::cos(leanRad);
  // Tall enough that the top meets the second shelf deck (mantle tier si=1), along the leaned rail.
  constexpr float yBaseDeck = 0.12f;
  const float yStepL = kShelfGapBetweenLevels + kShelfDeckThickness;
  const float ySecondShelfTop = yBaseDeck + 1.f * yStepL + kShelfDeckThickness;
  const float h = (ySecondShelfTop + 0.12f) / std::max(0.50f, cosL);
  const float w = 0.42f;
  const float d = 0.076f;
  scaleWHD = glm::vec3(w, h, d);
  const float reachZ = h * std::sin(leanRad);
  // Keep a fixed corner per rack side (no per-bay random variation).
  const uint32_t corner = side ? 1u : 0u;
  float px = 0.f;
  float pz = 0.f;
  switch (corner) {
    case 0u:
      px = -hw + ph;
      pz = -hd + ph;
      break;
    case 1u:
      px = hw - ph;
      pz = -hd + ph;
      break;
    case 2u:
      px = -hw + ph;
      pz = hd - ph;
      break;
    default:
      px = hw - ph;
      pz = hd - ph;
      break;
  }
  lx = px;
  // Base sits “behind” the post along local −Z from the top contact (Rx lean tips +Z), top ~ at post (px,pz).
  lz = pz - reachZ - 0.055f;
  return true;
}

static bool shelfBoxCutterLocalLayout(int worldAisleI, int worldAlongI, int side, float& lx, float& lz,
                                      float& yDeckTop, float& yawLocal, float& uniformScale) {
#if !defined(VULKAN_GAME_SHELF_BOX_CUTTER_GLB)
  (void)worldAisleI;
  (void)worldAlongI;
  (void)side;
  (void)lx;
  (void)lz;
  (void)yDeckTop;
  (void)yawLocal;
  (void)uniformScale;
  return false;
#else
  if (!gShelfBoxCutterMeshLoaded)
    return false;
  if (!shelfSlotOccupied(worldAisleI, worldAlongI, side))
    return false;
  const uint32_t Hr =
      scp3008ShelfHash(worldAisleI, worldAlongI, 0xBADD10CEu ^ static_cast<uint32_t>(side * 101));
  if ((Hr % 71u) == 0u)
    return false;
  if ((Hr % 17u) != 0u)
    return false;
  const uint32_t Hd =
      scp3008ShelfHash(worldAisleI, worldAlongI, 0xC471F00Du ^ static_cast<uint32_t>(side + 41u));
  const int deckIdx = static_cast<int>(Hd % static_cast<uint32_t>(kShelfDeckCount));
  constexpr float yBase = 0.12f;
  const float yStep = kShelfGapBetweenLevels + kShelfDeckThickness;
  const float y0 = yBase + static_cast<float>(deckIdx) * yStep;
  yDeckTop = y0 + kShelfDeckThickness;
  const uint32_t H = Hd;
  auto u01 = [&](unsigned sh) {
    return static_cast<float>((H >> (sh % 22u)) & 0x3FFu) / 1023.f;
  };
  const float hw = kShelfMeshHalfW;
  const float hd = kShelfMeshHalfD;
  const float mnX = -hw + kShelfDeckInset + 0.14f;
  const float mnZ = -hd + kShelfDeckInset + 0.14f;
  const float mxX = hw - kShelfDeckInset - 0.14f;
  const float mxZ = hd - kShelfDeckInset - 0.14f;
  constexpr float kLedgeInset = 0.055f;
  const bool ledgeSidePos = u01(2) > 0.5f;
  lx = glm::mix(mnX + 0.16f, mxX - 0.16f, u01(6));
  lz = ledgeSidePos ? (mxZ - kLedgeInset) : (mnZ + kLedgeInset);
  yawLocal = (u01(14) - 0.5f) * glm::radians(28.f);
  uniformScale = glm::mix(0.55f, 0.75f, u01(18));
  return true;
#endif
}

static bool shelfRustyPipeLocalLayout(int worldAisleI, int worldAlongI, int side, float& lx, float& lz,
                                      float& yDeckTop, float& yawLocal, float& uniformScale) {
#if !defined(VULKAN_GAME_SHELF_RUSTY_PIPE_GLB)
  (void)worldAisleI;
  (void)worldAlongI;
  (void)side;
  (void)lx;
  (void)lz;
  (void)yDeckTop;
  (void)yawLocal;
  (void)uniformScale;
  return false;
#else
  if (!gShelfRustyPipeMeshLoaded)
    return false;
  if (!shelfSlotOccupied(worldAisleI, worldAlongI, side))
    return false;
  const uint32_t Hr =
      scp3008ShelfHash(worldAisleI, worldAlongI, 0xBADD10CEu ^ static_cast<uint32_t>(side * 101));
  if ((Hr % 71u) == 0u)
    return false;
  const uint32_t Hp =
      scp3008ShelfHash(worldAisleI, worldAlongI, 0x517ED1FEu ^ static_cast<uint32_t>(side * 127));
  if ((Hp % 7u) != 0u)
    return false;
  const uint32_t Hd =
      scp3008ShelfHash(worldAisleI, worldAlongI, 0x2A1E70E1u ^ static_cast<uint32_t>(side + 67u));
  const int deckIdx = static_cast<int>(Hd % static_cast<uint32_t>(kShelfDeckCount));
  constexpr float yBase = 0.12f;
  const float yStep = kShelfGapBetweenLevels + kShelfDeckThickness;
  const float y0 = yBase + static_cast<float>(deckIdx) * yStep;
  yDeckTop = y0 + kShelfDeckThickness;
  const uint32_t H = Hd;
  auto u01 = [&](unsigned sh) {
    return static_cast<float>((H >> (sh % 22u)) & 0x3FFu) / 1023.f;
  };
  const float hw = kShelfMeshHalfW;
  const float hd = kShelfMeshHalfD;
  const float mnX = -hw + kShelfDeckInset + 0.14f;
  const float mnZ = -hd + kShelfDeckInset + 0.14f;
  const float mxX = hw - kShelfDeckInset - 0.14f;
  const float mxZ = hd - kShelfDeckInset - 0.14f;
  constexpr float kLedgeInset = 0.07f;
  const bool ledgeSidePos = u01(3) > 0.5f;
  lx = glm::mix(mnX + 0.18f, mxX - 0.18f, u01(5));
  lz = ledgeSidePos ? (mxZ - kLedgeInset) : (mnZ + kLedgeInset);
  yawLocal = (u01(15) - 0.5f) * glm::radians(24.f);
  uniformScale = glm::mix(0.85f, 1.15f, u01(19));
  return true;
#endif
}

template <typename Fn>
static void forEachShelfPalletStackBoxLocal(int worldAisleI, int worldAlongI, int side, bool cratePresent,
                                            Fn&& fn) {
  float lx, lz, yDeckTop, hx, hy, hz, yaw0, yaw1, offX, offZ;
  int stackCount = 0;
  if (!shelfPalletLocalLayout(worldAisleI, worldAlongI, side, cratePresent, lx, lz, yDeckTop, hx, hy, hz,
                              stackCount, yaw0, yaw1, offX, offZ))
    return;
  for (int si = 0; si < stackCount; ++si) {
    const float ox = (si == 0) ? 0.f : offX;
    const float oz = (si == 0) ? 0.f : offZ;
    const float y0 = yDeckTop + static_cast<float>(si) * (2.f * hy + 0.015f);
    const float y1 = y0 + 2.f * hy;
    fn(lx + ox - hx, y0, lz + oz - hz, lx + ox + hx, y1, lz + oz + hz);
  }
}

// Ray o + t*d (XZ as vec2.x=world X, vec2.y=world Z), t >= 0, vs axis-aligned XZ rectangle.
static bool rayXZHitAabbPositiveT(glm::vec2 o, glm::vec2 d, float xmin, float xmax, float zmin, float zmax,
                                  float& outTEnter, float& outTExit) {
  constexpr float kEps = 1e-7f;
  float t0 = -1e30f;
  float t1 = 1e30f;
  if (std::abs(d.x) < kEps) {
    if (o.x < xmin || o.x > xmax)
      return false;
  } else {
    const float inv = 1.f / d.x;
    float ta = (xmin - o.x) * inv;
    float tb = (xmax - o.x) * inv;
    if (ta > tb)
      std::swap(ta, tb);
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
  }
  if (std::abs(d.y) < kEps) {
    if (o.y < zmin || o.y > zmax)
      return false;
  } else {
    const float inv = 1.f / d.y;
    float ta = (zmin - o.y) * inv;
    float tb = (zmax - o.y) * inv;
    if (ta > tb)
      std::swap(ta, tb);
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
  }
  if (t1 < t0 || t1 < 0.f)
    return false;
  outTEnter = std::max(0.f, t0);
  outTExit = t1;
  return outTExit > outTEnter + 1e-5f;
}

bool segmentIntersectsRectXZ(glm::vec2 a, glm::vec2 b, glm::vec2 rmin, glm::vec2 rmax) {
  const glm::vec2 d = b - a;
  float tmin = 0.0f;
  float tmax = 1.0f;
  for (int axis = 0; axis < 2; ++axis) {
    const float p = (axis == 0) ? a.x : a.y;
    const float q = (axis == 0) ? d.x : d.y;
    const float mn = (axis == 0) ? rmin.x : rmin.y;
    const float mx = (axis == 0) ? rmax.x : rmax.y;
    if (std::abs(q) < 1e-6f) {
      if (p < mn || p > mx)
        return false;
      continue;
    }
    const float invQ = 1.0f / q;
    float t1 = (mn - p) * invQ;
    float t2 = (mx - p) * invQ;
    if (t1 > t2)
      std::swap(t1, t2);
    tmin = std::max(tmin, t1);
    tmax = std::min(tmax, t2);
    if (tmin > tmax)
      return false;
  }
  return true;
}

bool cameraOffsetHitsPillar(const glm::vec3& baseEye, const glm::vec3& offsetEye) {
  const int gcx = static_cast<int>(std::floor(baseEye.x / kPillarSpacing));
  const int gcz = static_cast<int>(std::floor(baseEye.z / kPillarSpacing));
  const float eyeY = offsetEye.y;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dz = -1; dz <= 1; ++dz) {
      const float px = static_cast<float>(gcx + dx) * kPillarSpacing;
      const float pz = static_cast<float>(gcz + dz) * kPillarSpacing;
      const AABB pillar = pillarCollisionAABB(px, pz);
      if (eyeY < pillar.min.y || eyeY > pillar.max.y)
        continue;
      const glm::vec2 rmin{pillar.min.x + 0.02f, pillar.min.z + 0.02f};
      const glm::vec2 rmax{pillar.max.x - 0.02f, pillar.max.z - 0.02f};
      const bool eyeInside = offsetEye.x >= rmin.x && offsetEye.x <= rmax.x && offsetEye.z >= rmin.y &&
                             offsetEye.z <= rmax.y;
      if (eyeInside)
        return true;
      if (segmentIntersectsRectXZ(glm::vec2(baseEye.x, baseEye.z),
                                  glm::vec2(offsetEye.x, offsetEye.z), rmin, rmax)) {
        return true;
      }
    }
  }
  return false;
}

void resolveEyeAgainstPillars(glm::vec3& eye) {
  const int gcx = static_cast<int>(std::floor(eye.x / kPillarSpacing));
  const int gcz = static_cast<int>(std::floor(eye.z / kPillarSpacing));
  for (int pass = 0; pass < 4; ++pass) {
    bool moved = false;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dz = -1; dz <= 1; ++dz) {
        const float px = static_cast<float>(gcx + dx) * kPillarSpacing;
        const float pz = static_cast<float>(gcz + dz) * kPillarSpacing;
        const AABB pillar = pillarCollisionAABB(px, pz);
        if (eye.y < pillar.min.y || eye.y > pillar.max.y)
          continue;
        const float minX = pillar.min.x + kCameraClipRadius;
        const float maxX = pillar.max.x - kCameraClipRadius;
        const float minZ = pillar.min.z + kCameraClipRadius;
        const float maxZ = pillar.max.z - kCameraClipRadius;
        if (eye.x < minX || eye.x > maxX || eye.z < minZ || eye.z > maxZ)
          continue;
        const float pushNegX = eye.x - minX;
        const float pushPosX = maxX - eye.x;
        const float pushNegZ = eye.z - minZ;
        const float pushPosZ = maxZ - eye.z;
        const float minPush = std::min(std::min(pushNegX, pushPosX), std::min(pushNegZ, pushPosZ));
        if (minPush == pushNegX)
          eye.x = minX;
        else if (minPush == pushPosX)
          eye.x = maxX;
        else if (minPush == pushNegZ)
          eye.z = minZ;
        else
          eye.z = maxZ;
        moved = true;
      }
    }
    if (!moved)
      break;
  }
}

std::vector<Vertex> makePillarMesh() {
  const float hw = kPillarHalfW;
  const float hd = kPillarHalfD;
  const float h = kPillarHeight;
  const glm::vec3 col = meshVertexColor();

  // Triangles must be clockwise in screen space (matches VK_FRONT_FACE_CLOCKWISE + glm Y flip).
  auto quad = [&](glm::vec3 n, glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {
    const glm::vec4 rgba = vrgb(col);
    return std::array<Vertex, 6>{
        Vertex{a, n, rgba, {}}, Vertex{c, n, rgba, {}}, Vertex{b, n, rgba, {}},
        Vertex{a, n, rgba, {}}, Vertex{d, n, rgba, {}}, Vertex{c, n, rgba, {}},
    };
  };

  std::vector<Vertex> v;
  // +X
  for (auto x : quad(glm::vec3(1, 0, 0), glm::vec3(hw, 0, -hd), glm::vec3(hw, h, -hd),
                     glm::vec3(hw, h, hd), glm::vec3(hw, 0, hd)))
    v.push_back(x);
  // -X
  for (auto x : quad(glm::vec3(-1, 0, 0), glm::vec3(-hw, 0, hd), glm::vec3(-hw, h, hd),
                     glm::vec3(-hw, h, -hd), glm::vec3(-hw, 0, -hd)))
    v.push_back(x);
  // +Y top
  for (auto x : quad(glm::vec3(0, 1, 0), glm::vec3(-hw, h, -hd), glm::vec3(hw, h, -hd),
                     glm::vec3(hw, h, hd), glm::vec3(-hw, h, hd)))
    v.push_back(x);
  // -Y bottom
  for (auto x : quad(glm::vec3(0, -1, 0), glm::vec3(-hw, 0, hd), glm::vec3(hw, 0, hd),
                     glm::vec3(hw, 0, -hd), glm::vec3(-hw, 0, -hd)))
    v.push_back(x);
  // +Z
  for (auto x : quad(glm::vec3(0, 0, 1), glm::vec3(-hw, 0, hd), glm::vec3(-hw, h, hd),
                     glm::vec3(hw, h, hd), glm::vec3(hw, 0, hd)))
    v.push_back(x);
  // -Z
  for (auto x : quad(glm::vec3(0, 0, -1), glm::vec3(hw, 0, -hd), glm::vec3(hw, h, -hd),
                     glm::vec3(-hw, h, -hd), glm::vec3(-hw, 0, -hd)))
    v.push_back(x);
  return v;
}

std::vector<Vertex> makeHandQuadMesh(const glm::vec3& markerColor) {
  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  const glm::vec3 a{-1.0f, -1.0f, 0.0f};
  const glm::vec3 b{1.0f, -1.0f, 0.0f};
  const glm::vec3 c{1.0f, 1.0f, 0.0f};
  const glm::vec3 d{-1.0f, 1.0f, 0.0f};
  return {
      Vertex{a, n, vrgb(markerColor), {}}, Vertex{c, n, vrgb(markerColor), {}},
      Vertex{b, n, vrgb(markerColor), {}}, Vertex{a, n, vrgb(markerColor), {}},
      Vertex{d, n, vrgb(markerColor), {}}, Vertex{c, n, vrgb(markerColor), {}},
  };
}

// NDC quad [-1,1]²; shader.vert tags fragColor — see crosshair branch in shader.frag.
static std::vector<Vertex> makeCrosshairQuadMesh() {
  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  const glm::vec4 col = glm::vec4(0.98f, 0.02f, 0.98f, 1.f);
  const glm::vec3 a{-1.0f, -1.0f, 0.0f};
  const glm::vec3 b{1.0f, -1.0f, 0.0f};
  const glm::vec3 c{1.0f, 1.0f, 0.0f};
  const glm::vec3 d{-1.0f, 1.0f, 0.0f};
  return {
      Vertex{a, n, col, {}}, Vertex{c, n, col, {}}, Vertex{b, n, col, {}},
      Vertex{a, n, col, {}}, Vertex{d, n, col, {}}, Vertex{c, n, col, {}},
  };
}

static const glm::vec4 kUiBackdropTag{0.02f, 0.997f, 0.997f, 1.f};
static const glm::vec4 kUiTextTag{0.997f, 0.5f, 0.045f, 1.f};
// Must match shader.vert / shader.frag UI NDC branch + health bar colors.
static const glm::vec4 kUiHealthTrackTag{0.02f, 0.35f, 0.996f, 1.f};
static const glm::vec4 kUiHealthFillTag{0.03f, 0.91f, 0.07f, 1.f};
static const glm::vec4 kUiHealthFillCritTag{0.058f, 0.24f, 0.088f, 1.f};
static const glm::vec4 kUiHealthFrameTag{0.07f, 0.118f, 0.996f, 1.f};
// Hunger piggybacks the stable HP HUD tag path for guaranteed visibility; fill uses crit/orange palette.
static const glm::vec4 kUiHungerTrackTag{0.02f, 0.35f, 0.996f, 1.f};
static const glm::vec4 kUiHungerFillTag{0.058f, 0.24f, 0.088f, 1.f};
static const glm::vec4 kUiHungerFrameTag{0.07f, 0.118f, 0.996f, 1.f};
// Pip-boy style top HUD (must match shader.vert / shader.frag isUiPipHud + pip* branches).
static const glm::vec4 kUiPipHudBgTag{0.068f, 0.996f, 0.020f, 1.f};
static const glm::vec4 kUiPipHudLineBrightTag{0.068f, 0.996f, 0.043f, 1.f};
static const glm::vec4 kUiPipHudLineDimTag{0.068f, 0.996f, 0.063f, 1.f};
static const glm::vec4 kUiPipHudFillCritTag{0.068f, 0.996f, 0.083f, 1.f};
static const glm::vec4 kUiPipHudTextTag{0.068f, 0.996f, 0.103f, 1.f};
// TrueType HUD (shader.vert / shader.frag: isUiHudFont + hudFontTex). .g selects primary vs accent tint.
static const glm::vec4 kUiHudFontPri{0.0788f, 0.901f, 0.071f, 1.f};
static const glm::vec4 kUiHudFontAcc{0.0788f, 0.9035f, 0.071f, 1.f};
// Fullscreen death dim (alpha in .a). .g kept < 0.96 so we do not match kUiBackdropTag in shader.frag.
static const glm::vec4 kUiHudVignetteTag{0.0188f, 0.88f, 0.9925f, 0.86f};
// Start / pause “sign” panels (shader: uiIkeaPanel / uiIkeaFont).
static const glm::vec4 kUiIkeaPanelTag{0.011f, 0.318f, 0.717f, 1.f};
static const glm::vec4 kUiIkeaFontPri{0.0845f, 0.9040f, 0.109f, 1.f};
static const glm::vec4 kUiIkeaFontAcc{0.0845f, 0.9056f, 0.109f, 1.f};
// Menu option lines (shader: uiIkeaFont + tag.r > 0.0865 → neutral gray).
static const glm::vec4 kUiIkeaFontOpt{0.0872f, 0.9048f, 0.109f, 1.f};
// Death screen title “YOU DIED” — bright red (shader: uiDeathTitleFont).
static const glm::vec4 kUiIkeaFontDeathTitle{0.0905f, 0.9042f, 0.109f, 1.f};
// Rounded button background behind each menu option (shader: uiOptionBtn).
static const glm::vec4 kUiOptionBtnTag{0.0125f, 0.318f, 0.635f, 1.f};
// Rounded stroke on menu panel perimeter (shader: uiMenuFrame; drawn after panel quad).
static const glm::vec4 kUiMenuFrameTag{0.0115f, 0.318f, 0.505f, 1.f};
// Title-screen IKEA wordmark: texture + retro anim in shader (uiIkeaLogo); time = ubo.employeeFadeH.z.
static const glm::vec4 kUiIkeaLogoTag{0.012f, 0.319f, 0.516f, 1.f};
// Death-only radial vignette (shader: uiDeathVignette). Bands avoid uiHudVignette / health HUD tags.
static const glm::vec4 kUiDeathVignetteTag{0.0145f, 0.798f, 0.991f, 0.90f};

static constexpr int kHudFontFirstChar = 32;
static constexpr int kHudFontCharCount = 95; // ' ' .. '~'
// Extra advance (font pixels) after each glyph on IKEA-style sign menus — improves ALL CAPS readability.
static constexpr float kIkeaMenuFontTrackPx = 1.45f;
static constexpr float kIkeaMenuOptionLineSkipMul = 2.0f;
// Title main menu only: wider gaps between CONTINUE / NEW GAME / EXIT than other menus.
static constexpr float kTitleMenuOptionLineSkipMul = 2.65f;
// Death menu: extra vertical gap between RETRY / QUIT (on top of kIkeaMenuOptionLineSkipMul).
static constexpr float kDeathMenuOptionLineExtraMul = 1.45f;
// Pause menu: extra vertical gap between RESUME / TITLE MENU.
static constexpr float kPauseMenuOptionLineExtraMul = 1.45f;
static stbtt_packedchar gHudUiFontPacked[kHudFontCharCount];
static int gHudUiFontAtlasW = 1024;
static int gHudUiFontAtlasH = 1024;
static float gHudUiFontSizePx = 36.f;
static float gHudUiFontLineSkipPx = 42.f;
static bool gHudUiFontReady = false;

static void appendHudFontRun(std::vector<Vertex>& mesh, const glm::vec3& n, const glm::vec4& fontTag,
                             const char* text, size_t textLen, float originXNdc, float originYNdc,
                             float pixelsToNdc, float extraAdvancePx = 0.f) {
  if (!gHudUiFontReady || !text || textLen == 0)
    return;
  float xpos = 0.f;
  float ypos = 0.f;
  for (size_t i = 0; i < textLen; ++i) {
    unsigned char uc = static_cast<unsigned char>(text[i]);
    if (uc < static_cast<unsigned char>(kHudFontFirstChar) ||
        uc >= static_cast<unsigned char>(kHudFontFirstChar + kHudFontCharCount))
      uc = static_cast<unsigned char>('?');
    const int idx = static_cast<int>(uc) - kHudFontFirstChar;
    stbtt_aligned_quad q{};
    stbtt_GetPackedQuad(gHudUiFontPacked, gHudUiFontAtlasW, gHudUiFontAtlasH, idx, &xpos, &ypos, &q, 0);
    xpos += extraAdvancePx;
    const float x0n = originXNdc + q.x0 * pixelsToNdc;
    const float x1n = originXNdc + q.x1 * pixelsToNdc;
    const float yTop = originYNdc - q.y0 * pixelsToNdc;
    const float yBot = originYNdc - q.y1 * pixelsToNdc;
    const glm::vec2 u00(q.s0, q.t0), u10(q.s1, q.t0), u11(q.s1, q.t1), u01(q.s0, q.t1);
    mesh.push_back({glm::vec3(x0n, yTop, 0.f), n, fontTag, u00});
    mesh.push_back({glm::vec3(x1n, yBot, 0.f), n, fontTag, u11});
    mesh.push_back({glm::vec3(x1n, yTop, 0.f), n, fontTag, u10});
    mesh.push_back({glm::vec3(x0n, yTop, 0.f), n, fontTag, u00});
    mesh.push_back({glm::vec3(x0n, yBot, 0.f), n, fontTag, u01});
    mesh.push_back({glm::vec3(x1n, yBot, 0.f), n, fontTag, u11});
  }
}

static void appendHudFontMultiline(std::vector<Vertex>& mesh, const glm::vec3& n, const glm::vec4& fontTag,
                                   const char* text, float originXNdc, float originYNdc, float pixelsToNdc,
                                   float extraAdvancePx = 0.f, float lineSpacingMul = 1.f) {
  if (!gHudUiFontReady || !text)
    return;
  const char* line = text;
  float lineY = originYNdc;
  const float step = gHudUiFontLineSkipPx * pixelsToNdc * lineSpacingMul;
  for (const char* p = text;; ++p) {
    if (*p == '\n' || *p == '\0') {
      appendHudFontRun(mesh, n, fontTag, line, static_cast<size_t>(p - line), originXNdc, lineY, pixelsToNdc,
                       extraAdvancePx);
      if (*p == '\0')
        break;
      line = p + 1;
      lineY -= step;
    }
  }
}

static float measureHudFontRunPx(const char* text, size_t textLen, float extraAdvancePx);

static void appendHudFontMultilineCentered(std::vector<Vertex>& mesh, const glm::vec3& n,
                                           const glm::vec4& fontTag, const char* text,
                                           float centerXNdc, float originYNdc, float pixelsToNdc,
                                           float extraAdvancePx = 0.f, float lineSpacingMul = 1.f) {
  if (!gHudUiFontReady || !text)
    return;
  const char* line = text;
  float lineY = originYNdc;
  const float step = gHudUiFontLineSkipPx * pixelsToNdc * lineSpacingMul;
  for (const char* p = text;; ++p) {
    if (*p == '\n' || *p == '\0') {
      const size_t len = static_cast<size_t>(p - line);
      const float lineW = measureHudFontRunPx(line, len, extraAdvancePx) * pixelsToNdc;
      appendHudFontRun(mesh, n, fontTag, line, len, centerXNdc - 0.5f * lineW, lineY, pixelsToNdc,
                       extraAdvancePx);
      if (*p == '\0')
        break;
      line = p + 1;
      lineY -= step;
    }
  }
}

static float measureHudFontRunPx(const char* text, size_t textLen, float extraAdvancePx = 0.f) {
  if (!gHudUiFontReady || !text || textLen == 0)
    return 0.f;
  float xpos = 0.f;
  float ypos = 0.f;
  for (size_t i = 0; i < textLen; ++i) {
    unsigned char uc = static_cast<unsigned char>(text[i]);
    if (uc < static_cast<unsigned char>(kHudFontFirstChar) ||
        uc >= static_cast<unsigned char>(kHudFontFirstChar + kHudFontCharCount))
      uc = static_cast<unsigned char>('?');
    const int idx = static_cast<int>(uc) - kHudFontFirstChar;
    stbtt_aligned_quad q{};
    stbtt_GetPackedQuad(gHudUiFontPacked, gHudUiFontAtlasW, gHudUiFontAtlasH, idx, &xpos, &ypos, &q, 0);
    xpos += extraAdvancePx;
  }
  return xpos;
}

static void appendStbEasyQuads(std::vector<Vertex>& mesh, const glm::vec3& n, const glm::vec4& colorTag,
                               const char* str, float ox, float oy, float scale) {
  stb_easy_font_spacing(-0.5f);
  std::vector<char> buf(65536);
  const int nq = static_cast<int>(
      stb_easy_font_print(0.f, 0.f, const_cast<char*>(str), nullptr, buf.data(), static_cast<int>(buf.size())));
  for (int q = 0; q < nq; ++q) {
    const char* p = buf.data() + q * 64;
    const auto corner = [&](int vi) {
      const float* f = reinterpret_cast<const float*>(p + vi * 16);
      return glm::vec2(ox + f[0] * scale, oy - f[1] * scale);
    };
    const glm::vec2 p0 = corner(0), p1 = corner(1), p2 = corner(2), p3 = corner(3);
    const auto tri = [&](glm::vec2 x, glm::vec2 y, glm::vec2 z) {
      mesh.push_back({glm::vec3(x.x, x.y, 0.f), n, colorTag, {}});
      mesh.push_back({glm::vec3(y.x, y.y, 0.f), n, colorTag, {}});
      mesh.push_back({glm::vec3(z.x, z.y, 0.f), n, colorTag, {}});
    };
    tri(p0, p1, p2);
    tri(p0, p2, p3);
  }
}

static void appendPanelQuad(std::vector<Vertex>& mesh, const glm::vec3& n, const glm::vec4& tag,
                            const glm::vec3& pa, const glm::vec3& pb,
                            const glm::vec3& pc, const glm::vec3& pd);
static void appendMenuFrameQuad(std::vector<Vertex>& mesh, const glm::vec3& n, const glm::vec3& pa,
                                const glm::vec3& pb, const glm::vec3& pc, const glm::vec3& pd);
static void appendOptionBtnQuads(std::vector<Vertex>& mesh, const glm::vec3& n, const char* text,
                                 float originYNdc, float pixelsToNdc, float lineSpacingMul, float btnHalfW);

static std::vector<Vertex> buildControlsHelpOverlayVertices() {
  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  std::vector<Vertex> mesh;

  static char kTitle[] = "CONTROLS";
  static char kBody[] = "WASD / ARROWS      MOVE\n"
                        "MOUSE              LOOK\n"
                        "SPACE              JUMP\n"
                        "SHIFT              SPRINT\n"
                        "C                  CROUCH\n"
                        "C + SHIFT          SLIDE\n"
                        "\n"
                        "SPACE AT LEDGE     GRAB / CLIMB\n"
                        "SPACE AT WALL      WALL JUMP / CLIMB\n"
                        "SHIFT AT WALL      WALL RUN\n"
                        "X                  CANCEL CLIMB\n"
                        "\n"
                        "LMB                SHOVE STAFF\n"
                        "ESC                PAUSE";
  static char kDismiss[] = "PRESS ANY KEY TO CONTINUE";
  constexpr float kHelpPxToNdc = 0.00052f;

  {
    const glm::vec4 dimTag = kUiHudVignetteTag;
    const glm::vec3 da{-1.f, -1.f, 0.f};
    const glm::vec3 db{ 1.f, -1.f, 0.f};
    const glm::vec3 dc{ 1.f,  1.f, 0.f};
    const glm::vec3 dd{-1.f,  1.f, 0.f};
    mesh.push_back({da, n, dimTag, {}});
    mesh.push_back({dc, n, dimTag, {}});
    mesh.push_back({db, n, dimTag, {}});
    mesh.push_back({da, n, dimTag, {}});
    mesh.push_back({dd, n, dimTag, {}});
    mesh.push_back({dc, n, dimTag, {}});
  }

  float bodyW = 0.f, bodyH = 0.f;
  float titleW = 0.f, titleH = 0.f;
  float dismissW = 0.f, dismissH = 0.f;
  constexpr float kBodyLineSpacing = 1.45f;
  if (gHudUiFontReady) {
    int bodyLines = 1;
    for (const char* q = kBody; *q; ++q)
      if (*q == '\n') ++bodyLines;
    float maxLineW = 0.f;
    const char* line = kBody;
    for (const char* p = kBody;; ++p) {
      if (*p == '\n' || *p == '\0') {
        maxLineW = std::max(maxLineW, measureHudFontRunPx(line, static_cast<size_t>(p - line)));
        if (*p == '\0') break;
        line = p + 1;
      }
    }
    bodyW = maxLineW * kHelpPxToNdc;
    bodyH = static_cast<float>(bodyLines) * gHudUiFontLineSkipPx * kHelpPxToNdc * kBodyLineSpacing;
    titleW = measureHudFontRunPx(kTitle, std::strlen(kTitle)) * kHelpPxToNdc;
    titleH = gHudUiFontLineSkipPx * kHelpPxToNdc;
    dismissW = measureHudFontRunPx(kDismiss, std::strlen(kDismiss)) * kHelpPxToNdc;
    dismissH = gHudUiFontLineSkipPx * kHelpPxToNdc;
  } else {
    stb_easy_font_spacing(-0.5f);
    constexpr float scale = 0.0031f;
    bodyW = static_cast<float>(stb_easy_font_width(kBody)) * scale;
    bodyH = static_cast<float>(stb_easy_font_height(kBody)) * scale * kBodyLineSpacing;
    titleW = static_cast<float>(stb_easy_font_width(kTitle)) * scale;
    titleH = static_cast<float>(stb_easy_font_height(kTitle)) * scale;
    dismissW = static_cast<float>(stb_easy_font_width(kDismiss)) * scale;
    dismissH = static_cast<float>(stb_easy_font_height(kDismiss)) * scale;
  }

  constexpr float padX = 0.14f;
  constexpr float padY = 0.08f;
  constexpr float titleGap = 0.06f;
  constexpr float dismissGap = 0.07f;
  const float contentW = std::max({bodyW, titleW, dismissW});
  const float totalH = titleH + titleGap + bodyH + dismissGap + dismissH;
  const float panelHalfW = std::max(0.5f * contentW + padX, 0.48f);
  const float panelHalfH = 0.5f * totalH + padY;

  {
    const glm::vec3 pa{-panelHalfW, -panelHalfH, 0.f};
    const glm::vec3 pb{ panelHalfW, -panelHalfH, 0.f};
    const glm::vec3 pc{ panelHalfW,  panelHalfH, 0.f};
    const glm::vec3 pd{-panelHalfW,  panelHalfH, 0.f};
    appendPanelQuad(mesh, n, kUiIkeaPanelTag, pa, pb, pc, pd);
    appendMenuFrameQuad(mesh, n, pa, pb, pc, pd);
  }

  const float topY = panelHalfH - padY;

  if (gHudUiFontReady) {
    appendHudFontMultiline(mesh, n, kUiIkeaFontAcc, kTitle, -0.5f * titleW, topY, kHelpPxToNdc);
    const float bodyTopY = topY - titleH - titleGap;
    appendHudFontMultiline(mesh, n, kUiHudFontPri, kBody, -0.5f * bodyW, bodyTopY, kHelpPxToNdc, 0.f, kBodyLineSpacing);
    const float dismissY = bodyTopY - bodyH - dismissGap;
    appendOptionBtnQuads(mesh, n, kDismiss, dismissY, kHelpPxToNdc, 1.f, panelHalfW - 0.03f);
    appendHudFontMultiline(mesh, n, kUiIkeaFontAcc, kDismiss, -0.5f * dismissW, dismissY, kHelpPxToNdc);
  } else {
    constexpr float scale = 0.0031f;
    appendStbEasyQuads(mesh, n, kUiTextTag, kTitle, -0.5f * titleW, topY, scale);
    const float bodyTopY = topY - titleH - titleGap;
    appendStbEasyQuads(mesh, n, kUiTextTag, kBody, -0.5f * bodyW, bodyTopY, scale);
    const float dismissY = bodyTopY - bodyH - dismissGap;
    appendStbEasyQuads(mesh, n, kUiTextTag, kDismiss, -0.5f * dismissW, dismissY, scale);
  }

  return mesh;
}

static void appendPanelQuad(std::vector<Vertex>& mesh, const glm::vec3& n, const glm::vec4& tag,
                            const glm::vec3& pa, const glm::vec3& pb,
                            const glm::vec3& pc, const glm::vec3& pd) {
  const glm::vec2 uv00{0.f, 0.f}, uv10{1.f, 0.f}, uv11{1.f, 1.f}, uv01{0.f, 1.f};
  mesh.push_back({pa, n, tag, uv00});
  mesh.push_back({pc, n, tag, uv11});
  mesh.push_back({pb, n, tag, uv10});
  mesh.push_back({pa, n, tag, uv00});
  mesh.push_back({pd, n, tag, uv01});
  mesh.push_back({pc, n, tag, uv11});
}

static void appendMenuFrameQuad(std::vector<Vertex>& mesh, const glm::vec3& n, const glm::vec3& pa,
                                const glm::vec3& pb, const glm::vec3& pc, const glm::vec3& pd) {
  appendPanelQuad(mesh, n, kUiMenuFrameTag, pa, pb, pc, pd);
}

static void appendOptionBtnQuads(std::vector<Vertex>& mesh, const glm::vec3& n,
                                  const char* text, float originYNdc, float pixelsToNdc,
                                  float lineSpacingMul, float btnHalfW) {
  if (!gHudUiFontReady || !text)
    return;
  const float lineH = gHudUiFontLineSkipPx * pixelsToNdc;
  const float step = lineH * lineSpacingMul;
  // Pill height ~ one menu line; HUD text uses stbtt baseline at originYNdc — caps sit mostly above baseline,
  // so center the chip slightly above baseline (older cy = lineY - 0.40*lineH sat too low vs the glyphs).
  const float btnHalfH = lineH * 1.02f;
  float lineY = originYNdc;
  for (const char* p = text;; ++p) {
    if (*p == '\n' || *p == '\0') {
      constexpr float kBtnCenterAboveBaselineMul = 0.17f;
      const float cy = lineY + lineH * kBtnCenterAboveBaselineMul;
      glm::vec3 ba{-btnHalfW, cy - btnHalfH, 0.f};
      glm::vec3 bb{ btnHalfW, cy - btnHalfH, 0.f};
      glm::vec3 bc{ btnHalfW, cy + btnHalfH, 0.f};
      glm::vec3 bd{-btnHalfW, cy + btnHalfH, 0.f};
      appendPanelQuad(mesh, n, kUiOptionBtnTag, ba, bb, bc, bd);
      if (*p == '\0')
        break;
      lineY -= step;
    }
  }
}

static std::vector<Vertex> buildDeathMenuOverlayVertices() {
  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  std::vector<Vertex> mesh;
  mesh.reserve(800);
  static const char kTitle[] = "YOU DIED";
  static const char kSub[] = "RETRY\nEXIT";
  constexpr float kDeathTitlePx = 0.00095f;
  constexpr float kDeathSubPx = 0.00056f;
  const float deathLineMul = kIkeaMenuOptionLineSkipMul * kDeathMenuOptionLineExtraMul;

  {
    const glm::vec3 v0{-1.f, -1.f, 0.f};
    const glm::vec3 v1{1.f, -1.f, 0.f};
    const glm::vec3 v2{1.f, 1.f, 0.f};
    const glm::vec3 v3{-1.f, 1.f, 0.f};
    mesh.push_back({v0, n, kUiDeathVignetteTag, {}});
    mesh.push_back({v2, n, kUiDeathVignetteTag, {}});
    mesh.push_back({v1, n, kUiDeathVignetteTag, {}});
    mesh.push_back({v0, n, kUiDeathVignetteTag, {}});
    mesh.push_back({v3, n, kUiDeathVignetteTag, {}});
    mesh.push_back({v2, n, kUiDeathVignetteTag, {}});
  }

  float titleW = 0.f;
  float subW = 0.f;
  if (gHudUiFontReady) {
    titleW = measureHudFontRunPx(kTitle, std::strlen(kTitle), kIkeaMenuFontTrackPx) * kDeathTitlePx;
    const char* sn = kSub;
    for (const char* p = kSub;; ++p) {
      if (*p == '\n' || *p == '\0') {
        subW = std::max(
            subW, measureHudFontRunPx(sn, static_cast<size_t>(p - sn), kIkeaMenuFontTrackPx) * kDeathSubPx);
        if (*p == '\0')
          break;
        sn = p + 1;
      }
    }
  } else {
    static char kTxtFb[] = "YOU DIED\nRETRY\nEXIT";
    stb_easy_font_spacing(-0.5f);
    constexpr float scale = 0.0046f;
    const int fontW = stb_easy_font_width(kTxtFb);
    const float textW = static_cast<float>(fontW) * scale;
    titleW = textW;
    subW = textW;
  }

  constexpr float panelPadX = 0.22f;
  constexpr float panelPadY = 0.12f;
  const float panelHalfW = std::max(0.5f * std::max(titleW, subW) + panelPadX, 0.58f);
  constexpr float panelTopY = 0.12f;
  const float titleBlockH = gHudUiFontReady ? gHudUiFontLineSkipPx * kDeathTitlePx : 0.09f;
  const float subBlockH =
      gHudUiFontReady ? (2.f * gHudUiFontLineSkipPx * kDeathSubPx * deathLineMul + 0.06f) : 0.24f;
  const float textStackH = titleBlockH + subBlockH + (gHudUiFontReady ? 0.14f : 0.05f);
  const float panelBotY = panelTopY - textStackH - 2.f * panelPadY;

  const glm::vec3 pa{-panelHalfW, panelBotY, 0.f};
  const glm::vec3 pb{panelHalfW, panelBotY, 0.f};
  const glm::vec3 pc{panelHalfW, panelTopY, 0.f};
  const glm::vec3 pd{-panelHalfW, panelTopY, 0.f};
  appendPanelQuad(mesh, n, kUiIkeaPanelTag, pa, pb, pc, pd);
  appendMenuFrameQuad(mesh, n, pa, pb, pc, pd);

  const float line1Y = panelTopY - panelPadY - titleBlockH * 0.2f;
  const float line2Y = line1Y - titleBlockH - (gHudUiFontReady ? 0.08f : 0.04f);
  if (gHudUiFontReady) {
    const float tw = measureHudFontRunPx(kTitle, std::strlen(kTitle), kIkeaMenuFontTrackPx) * kDeathTitlePx;
    appendHudFontRun(mesh, n, kUiIkeaFontDeathTitle, kTitle, std::strlen(kTitle), -0.5f * tw, line1Y, kDeathTitlePx,
                     kIkeaMenuFontTrackPx);
    appendOptionBtnQuads(mesh, n, kSub, line2Y, kDeathSubPx, deathLineMul, panelHalfW - 0.02f);
    appendHudFontMultilineCentered(mesh, n, kUiIkeaFontOpt, kSub, 0.f, line2Y, kDeathSubPx,
                                   kIkeaMenuFontTrackPx, deathLineMul);
  } else {
    static char kTxt[] = "YOU DIED\nRETRY\nEXIT";
    constexpr float scale = 0.0046f;
    stb_easy_font_spacing(-0.5f);
    const int fontW = stb_easy_font_width(kTxt);
    const int fontH = stb_easy_font_height(kTxt);
    const float textW = static_cast<float>(fontW) * scale;
    const float textH = static_cast<float>(fontH) * scale;
    const float ox = -0.5f * textW;
    const float oy = panelTopY - panelPadY;
    (void)textH;
    appendStbEasyQuads(mesh, n, kUiTextTag, kTxt, ox, oy, scale);
  }
  return mesh;
}

static std::vector<Vertex> buildPauseMenuOverlayVertices() {
  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  std::vector<Vertex> mesh;
  mesh.reserve(800);
  static const char kTitle[] = "PAUSED";
  static const char kTagline[] = "THE STORE CAN WAIT";
  static const char kSub[] = "RESUME\nEXIT";
  constexpr float kPauseTitlePx = 0.00095f;
  constexpr float kPauseTaglinePx = 0.00052f;
  constexpr float kPauseSubPx = 0.00056f;
  const float pauseLineMul = kIkeaMenuOptionLineSkipMul * kPauseMenuOptionLineExtraMul;

  {
    const glm::vec3 v0{-1.f, -1.f, 0.f};
    const glm::vec3 v1{1.f, -1.f, 0.f};
    const glm::vec3 v2{1.f, 1.f, 0.f};
    const glm::vec3 v3{-1.f, 1.f, 0.f};
    mesh.push_back({v0, n, kUiHudVignetteTag, {}});
    mesh.push_back({v2, n, kUiHudVignetteTag, {}});
    mesh.push_back({v1, n, kUiHudVignetteTag, {}});
    mesh.push_back({v0, n, kUiHudVignetteTag, {}});
    mesh.push_back({v3, n, kUiHudVignetteTag, {}});
    mesh.push_back({v2, n, kUiHudVignetteTag, {}});
  }

  float titleW = 0.f;
  float subW = 0.f;
  if (gHudUiFontReady) {
    titleW = measureHudFontRunPx(kTitle, std::strlen(kTitle), kIkeaMenuFontTrackPx) * kPauseTitlePx;
    subW = std::max(subW,
                    measureHudFontRunPx(kTagline, std::strlen(kTagline), kIkeaMenuFontTrackPx) * kPauseTaglinePx);
    const char* sn = kSub;
    for (const char* p = kSub;; ++p) {
      if (*p == '\n' || *p == '\0') {
        subW = std::max(
            subW, measureHudFontRunPx(sn, static_cast<size_t>(p - sn), kIkeaMenuFontTrackPx) * kPauseSubPx);
        if (*p == '\0')
          break;
        sn = p + 1;
      }
    }
  } else {
    static char kTxtFb[] = "PAUSED\nRESUME\nEXIT";
    stb_easy_font_spacing(-0.5f);
    constexpr float scale = 0.0046f;
    const int fontW = stb_easy_font_width(kTxtFb);
    const float textW = static_cast<float>(fontW) * scale;
    titleW = textW;
    subW = textW;
  }

  constexpr float panelPadX = 0.22f;
  constexpr float panelPadY = 0.12f;
  const float panelHalfW = std::max(0.5f * std::max(titleW, subW) + panelPadX, 0.58f);
  constexpr float panelTopY = 0.12f;
  const float titleBlockH = gHudUiFontReady ? gHudUiFontLineSkipPx * kPauseTitlePx : 0.09f;
  const float taglineBlockH = gHudUiFontReady ? gHudUiFontLineSkipPx * kPauseTaglinePx : 0.f;
  const float subBlockH = gHudUiFontReady
                              ? (2.f * gHudUiFontLineSkipPx * kPauseSubPx * pauseLineMul + 0.02f)
                              : 0.19f;
  const float textStackH = titleBlockH + taglineBlockH + subBlockH + (gHudUiFontReady ? 0.13f : 0.04f);
  const float panelBotY = panelTopY - textStackH - 2.f * panelPadY;

  const glm::vec3 pa{-panelHalfW, panelBotY, 0.f};
  const glm::vec3 pb{panelHalfW, panelBotY, 0.f};
  const glm::vec3 pc{panelHalfW, panelTopY, 0.f};
  const glm::vec3 pd{-panelHalfW, panelTopY, 0.f};
  appendPanelQuad(mesh, n, kUiIkeaPanelTag, pa, pb, pc, pd);
  appendMenuFrameQuad(mesh, n, pa, pb, pc, pd);

  const float line1Y = panelTopY - panelPadY - titleBlockH * 0.2f;
  const float lineTagY = line1Y - titleBlockH - 0.025f;
  const float line2Y = lineTagY - taglineBlockH - (gHudUiFontReady ? 0.045f : 0.03f);
  if (gHudUiFontReady) {
    const float tw = measureHudFontRunPx(kTitle, std::strlen(kTitle), kIkeaMenuFontTrackPx) * kPauseTitlePx;
    appendHudFontRun(mesh, n, kUiIkeaFontAcc, kTitle, std::strlen(kTitle), -0.5f * tw, line1Y, kPauseTitlePx,
                     kIkeaMenuFontTrackPx);
    const float tgw =
        measureHudFontRunPx(kTagline, std::strlen(kTagline), kIkeaMenuFontTrackPx) * kPauseTaglinePx;
    appendHudFontRun(mesh, n, kUiIkeaFontPri, kTagline, std::strlen(kTagline), -0.5f * tgw, lineTagY,
                     kPauseTaglinePx, kIkeaMenuFontTrackPx);
    appendOptionBtnQuads(mesh, n, kSub, line2Y, kPauseSubPx, pauseLineMul, panelHalfW - 0.02f);
    appendHudFontMultilineCentered(mesh, n, kUiIkeaFontOpt, kSub, 0.f, line2Y, kPauseSubPx,
                                   kIkeaMenuFontTrackPx, pauseLineMul);
  } else {
    static char kTxt[] = "PAUSED\nRESUME\nEXIT";
    constexpr float scale = 0.0046f;
    stb_easy_font_spacing(-0.5f);
    const int fontW = stb_easy_font_width(kTxt);
    const int fontH = stb_easy_font_height(kTxt);
    const float textW = static_cast<float>(fontW) * scale;
    const float textH = static_cast<float>(fontH) * scale;
    const float ox = -0.5f * textW;
    const float oy = panelTopY - panelPadY;
    (void)textH;
    appendStbEasyQuads(mesh, n, kUiTextTag, kTxt, ox, oy, scale);
  }
  return mesh;
}

static std::vector<std::pair<std::string, int>> buildInventoryStacks(
    const std::vector<std::string>& items) {
  std::vector<std::pair<std::string, int>> stacks;
  stacks.reserve(items.size());
  std::unordered_map<std::string, size_t> idxByName;
  idxByName.reserve(items.size());
  for (const std::string& it : items) {
    auto found = idxByName.find(it);
    if (found == idxByName.end()) {
      idxByName.emplace(it, stacks.size());
      stacks.emplace_back(it, 1);
    } else {
      ++stacks[found->second].second;
    }
  }
  return stacks;
}

static std::vector<Vertex> buildInventoryOverlayVertices(const std::vector<std::string>& items,
                                                         int scrollRow) {
  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  std::vector<Vertex> mesh;
  mesh.reserve(2200);

  {
    const glm::vec3 v0{-1.f, -1.f, 0.f};
    const glm::vec3 v1{1.f, -1.f, 0.f};
    const glm::vec3 v2{1.f, 1.f, 0.f};
    const glm::vec3 v3{-1.f, 1.f, 0.f};
    mesh.push_back({v0, n, kUiHudVignetteTag, {}});
    mesh.push_back({v2, n, kUiHudVignetteTag, {}});
    mesh.push_back({v1, n, kUiHudVignetteTag, {}});
    mesh.push_back({v0, n, kUiHudVignetteTag, {}});
    mesh.push_back({v3, n, kUiHudVignetteTag, {}});
    mesh.push_back({v2, n, kUiHudVignetteTag, {}});
  }

  constexpr float panelHalfW = 0.76f;
  constexpr float panelTopY = 0.72f;
  constexpr float panelBotY = -0.72f;
  const glm::vec3 pa{-panelHalfW, panelBotY, 0.f};
  const glm::vec3 pb{panelHalfW, panelBotY, 0.f};
  const glm::vec3 pc{panelHalfW, panelTopY, 0.f};
  const glm::vec3 pd{-panelHalfW, panelTopY, 0.f};
  appendPanelQuad(mesh, n, kUiIkeaPanelTag, pa, pb, pc, pd);
  appendMenuFrameQuad(mesh, n, pa, pb, pc, pd);

  if (gHudUiFontReady) {
    constexpr float kTitlePx = 0.00088f;
    constexpr float kSubPx = 0.00050f;
    constexpr float kItemPx = 0.00052f;
    constexpr float kTrack = kIkeaMenuFontTrackPx;
    static const char kTitle[] = "INVENTORY";
    static const char kSub[] = "TAB SCROLLABLE LOOT LIST";
    const float tw = measureHudFontRunPx(kTitle, std::strlen(kTitle), kTrack) * kTitlePx;
    const float sw = measureHudFontRunPx(kSub, std::strlen(kSub), kTrack) * kSubPx;
    appendHudFontRun(mesh, n, kUiIkeaFontAcc, kTitle, std::strlen(kTitle), -0.5f * tw, 0.60f, kTitlePx,
                     kTrack);
    appendHudFontRun(mesh, n, kUiIkeaFontPri, kSub, std::strlen(kSub), -0.5f * sw, 0.52f, kSubPx, kTrack);

    // Scrollable list window (Dying Light-inspired stash list).
    constexpr float listHalfW = 0.64f;
    constexpr float listTopY = 0.42f;
    constexpr float listBotY = -0.50f;
    const glm::vec3 la{-listHalfW, listBotY, 0.f};
    const glm::vec3 lb{listHalfW, listBotY, 0.f};
    const glm::vec3 lc{listHalfW, listTopY, 0.f};
    const glm::vec3 ld{-listHalfW, listTopY, 0.f};
    appendPanelQuad(mesh, n, kUiOptionBtnTag, la, lb, lc, ld);
    appendMenuFrameQuad(mesh, n, la, lb, lc, ld);

    constexpr int kVisibleRows = 8;
    const auto stacks = buildInventoryStacks(items);
    const int totalRows = static_cast<int>(stacks.size());
    const int maxScroll = std::max(0, totalRows - kVisibleRows);
    const int start = std::clamp(scrollRow, 0, maxScroll);
    const float rowStep = (listTopY - listBotY - 0.12f) / static_cast<float>(kVisibleRows);
    float rowY = listTopY - 0.09f;
    if (items.empty()) {
      static const char kEmpty[] = "EMPTY";
      const float ew = measureHudFontRunPx(kEmpty, std::strlen(kEmpty), kTrack) * kItemPx;
      appendHudFontRun(mesh, n, kUiIkeaFontPri, kEmpty, std::strlen(kEmpty), -0.5f * ew, -0.02f, kItemPx,
                       kTrack);
    } else {
      for (int r = 0; r < kVisibleRows; ++r) {
        const int idx = start + r;
        if (idx >= totalRows)
          break;
        const std::string row = std::to_string(idx + 1) + ". " +
                                stacks[static_cast<size_t>(idx)].first + " x" +
                                std::to_string(stacks[static_cast<size_t>(idx)].second);
        const float rw = measureHudFontRunPx(row.c_str(), row.size(), kTrack) * kItemPx;
        appendHudFontRun(mesh, n, kUiIkeaFontOpt, row.c_str(), row.size(),
                         -std::min(0.58f, 0.5f * rw), rowY, kItemPx, kTrack);
        rowY -= rowStep;
      }
    }

    const std::string hint =
        "MOUSE WHEEL / UP-DOWN SCROLL  |  STACKS " + std::to_string(stacks.size()) +
        "  ITEMS " + std::to_string(items.size());
    const float hw = measureHudFontRunPx(hint.c_str(), hint.size(), kTrack) * kSubPx;
    appendHudFontRun(mesh, n, kUiIkeaFontPri, hint.c_str(), hint.size(), -0.5f * hw, -0.58f, kSubPx,
                     kTrack);
  }

  return mesh;
}

static constexpr int kGameSaveSlotCount = 4;

static inline void windowPixelsToUiNdc(int mx, int my, int winW, int winH, float* outX, float* outY) {
  const float fw = static_cast<float>(std::max(winW, 1));
  const float fh = static_cast<float>(std::max(winH, 1));
  *outX = (static_cast<float>(mx) + 0.5f) / fw * 2.f - 1.f;
  *outY = 1.f - (static_cast<float>(my) + 0.5f) / fh * 2.f;
}

/// Map SDL window-client mouse to UI NDC. Menus draw in the present pass with viewport = swapchainExtent
/// (not necessarily equal to SDL_Vulkan_GetDrawableSize). Use the same dimensions for hit-testing.
static inline void sdlWindowMouseToUiNdc(SDL_Window* window, int mx, int my, uint32_t viewportW,
                                         uint32_t viewportH, float* outX, float* outY) {
  int logicalW = 0, logicalH = 0;
  SDL_GetWindowSize(window, &logicalW, &logicalH);
  const int vw = static_cast<int>(std::max(viewportW, 1u));
  const int vh = static_cast<int>(std::max(viewportH, 1u));
  if (logicalW <= 0 || logicalH <= 0) {
    windowPixelsToUiNdc(mx, my, vw, vh, outX, outY);
    return;
  }
  const float px =
      (static_cast<float>(mx) + 0.5f) / static_cast<float>(logicalW) * static_cast<float>(vw);
  const float py =
      (static_cast<float>(my) + 0.5f) / static_cast<float>(logicalH) * static_cast<float>(vh);
  const float fw = static_cast<float>(vw);
  const float fh = static_cast<float>(vh);
  *outX = px / fw * 2.f - 1.f;
  *outY = 1.f - py / fh * 2.f;
}

static inline bool uiPointInRect(float x, float y, float xmin, float xmax, float ymin, float ymax) {
  return x >= xmin && x <= xmax && y >= ymin && y <= ymax;
}

struct UiMenuClickLayout {
  float panelHalfW = 0.f;
  float line2Y = 0.f;
  float lineSkipNdc = 0.052f;
  int optionLines = 0;
};

static UiMenuClickLayout computeTitleMenuMainClickLayout(bool showContinue) {
  UiMenuClickLayout L{};
  constexpr float kSubPx = 0.00062f;
  float subW = 0.f;
  int subLines = 1;
  if (gHudUiFontReady) {
    const char* kSub =
        showContinue ? "CONTINUE\nNEW GAME\nEXIT" : "NEW GAME\nEXIT";
    for (const char* q = kSub; *q; ++q)
      if (*q == '\n')
        ++subLines;
    const char* sn = kSub;
    for (const char* p = kSub;; ++p) {
      if (*p == '\n' || *p == '\0') {
        subW = std::max(subW, measureHudFontRunPx(sn, static_cast<size_t>(p - sn), kIkeaMenuFontTrackPx) * kSubPx);
        if (*p == '\0')
          break;
        sn = p + 1;
      }
    }
    L.optionLines = showContinue ? 3 : 2;
  } else {
    subW = 0.5f;
    subLines = showContinue ? 3 : 2;
    L.optionLines = subLines;
  }
  constexpr float panelPadX = 0.22f;
  constexpr float panelPadY = 0.14f;
  L.panelHalfW = std::max(0.5f * subW + panelPadX, 0.62f);
  constexpr float panelTopY = 0.14f;
  const float subBlockH =
      gHudUiFontReady ? (static_cast<float>(subLines) * gHudUiFontLineSkipPx * kSubPx * kTitleMenuOptionLineSkipMul +
                         0.02f)
                      : 0.26f;
  const float textStackH = subBlockH + (gHudUiFontReady ? 0.06f : 0.04f);
  const float panelBotY = panelTopY - textStackH - 2.f * panelPadY;
  (void)panelBotY;
  L.line2Y = panelTopY - panelPadY - (gHudUiFontReady ? 0.02f : 0.03f);
  L.lineSkipNdc = (gHudUiFontReady ? gHudUiFontLineSkipPx * kSubPx : std::max(gHudUiFontLineSkipPx * kSubPx, 0.048f)) *
                  kTitleMenuOptionLineSkipMul;
  return L;
}

static UiMenuClickLayout computeTitleMenuSlotPickerClickLayout(const std::array<bool, 4>& slotUsed) {
  UiMenuClickLayout L{};
  char block[384];
  int pos = 0;
  pos += std::snprintf(block + pos, sizeof(block) - pos, "CHOOSE SAVE SLOT\n");
  for (int i = 0; i < kGameSaveSlotCount; ++i) {
    pos += std::snprintf(block + pos, sizeof(block) - pos, "%d  SLOT %d  %s\n", i + 1, i + 1,
                         slotUsed[static_cast<size_t>(i)] ? "(SAVED)" : "(EMPTY)");
  }
  std::snprintf(block + pos, sizeof(block) - pos, "\nBACK (ESC)\nLMB: LOAD   RMB: DELETE");
  static const char kTitle[] = "START";
  constexpr float kTitlePx = 0.00112f;
  constexpr float kSubPx = 0.00058f;
  float titleW = 0.f;
  float subW = 0.f;
  int subLines = 1;
  if (gHudUiFontReady) {
    titleW = measureHudFontRunPx(kTitle, std::strlen(kTitle), kIkeaMenuFontTrackPx) * kTitlePx;
    for (const char* q = block; *q; ++q)
      if (*q == '\n')
        ++subLines;
    const char* sn = block;
    for (const char* p = block;; ++p) {
      if (*p == '\n' || *p == '\0') {
        subW = std::max(
            subW, measureHudFontRunPx(sn, static_cast<size_t>(p - sn), kIkeaMenuFontTrackPx) * kSubPx);
        if (*p == '\0')
          break;
        sn = p + 1;
      }
    }
  } else {
    titleW = 0.4f;
    subW = 0.55f;
    subLines = 7;
  }
  constexpr float panelPadX = 0.22f;
  constexpr float panelPadY = 0.12f;
  L.panelHalfW = std::max(0.5f * std::max(titleW, subW) + panelPadX, 0.64f);
  constexpr float panelTopY = 0.16f;
  const float titleBlockH = gHudUiFontReady ? gHudUiFontLineSkipPx * kTitlePx : 0.09f;
  const float subBlockH =
      gHudUiFontReady ? (static_cast<float>(subLines) * gHudUiFontLineSkipPx * kSubPx * kIkeaMenuOptionLineSkipMul +
                         0.02f)
                      : 0.38f;
  const float textStackH = titleBlockH + subBlockH + 0.04f;
  const float panelBotY = panelTopY - textStackH - 2.f * panelPadY;
  (void)panelBotY;
  const float line1Y = panelTopY - panelPadY - titleBlockH * 0.2f;
  const float line2Y = line1Y - titleBlockH - 0.03f;
  L.line2Y = line2Y;
  L.lineSkipNdc = (gHudUiFontReady ? gHudUiFontLineSkipPx * kSubPx : std::max(gHudUiFontLineSkipPx * kSubPx, 0.048f)) *
                  kIkeaMenuOptionLineSkipMul;
  L.optionLines = subLines;
  return L;
}

static UiMenuClickLayout computeDeathMenuClickLayout() {
  UiMenuClickLayout L{};
  static const char kTitle[] = "YOU DIED";
  constexpr float kDeathTitlePx = 0.00095f;
  constexpr float kDeathSubPx = 0.00056f;
  const float deathLineMul = kIkeaMenuOptionLineSkipMul * kDeathMenuOptionLineExtraMul;
  float titleW = 0.f;
  float subW = 0.f;
  constexpr int subLines = 2;
  if (gHudUiFontReady) {
    titleW = measureHudFontRunPx(kTitle, std::strlen(kTitle), kIkeaMenuFontTrackPx) * kDeathTitlePx;
  static const char kSub[] = "RETRY\nEXIT";
    const char* sn = kSub;
    for (const char* p = kSub;; ++p) {
      if (*p == '\n' || *p == '\0') {
        subW = std::max(
            subW, measureHudFontRunPx(sn, static_cast<size_t>(p - sn), kIkeaMenuFontTrackPx) * kDeathSubPx);
        if (*p == '\0')
          break;
        sn = p + 1;
      }
    }
  } else {
    titleW = 0.5f;
    subW = 0.5f;
  }
  L.optionLines = 2;
  constexpr float panelPadX = 0.22f;
  constexpr float panelPadY = 0.12f;
  L.panelHalfW = std::max(0.5f * std::max(titleW, subW) + panelPadX, 0.58f);
  constexpr float panelTopY = 0.12f;
  const float titleBlockH = gHudUiFontReady ? gHudUiFontLineSkipPx * kDeathTitlePx : 0.09f;
  const float subBlockH =
      gHudUiFontReady ? (static_cast<float>(subLines) * gHudUiFontLineSkipPx * kDeathSubPx * deathLineMul + 0.06f)
                      : 0.24f;
  const float textStackH = titleBlockH + subBlockH + (gHudUiFontReady ? 0.1f : 0.05f);
  const float panelBotY = panelTopY - textStackH - 2.f * panelPadY;
  (void)panelBotY;
  const float line1Y = panelTopY - panelPadY - titleBlockH * 0.2f;
  L.line2Y = line1Y - titleBlockH - (gHudUiFontReady ? 0.055f : 0.03f);
  L.lineSkipNdc = (gHudUiFontReady ? gHudUiFontLineSkipPx * kDeathSubPx
                                   : std::max(gHudUiFontLineSkipPx * kDeathSubPx, 0.048f)) *
                  deathLineMul;
  return L;
}

static UiMenuClickLayout computePauseMenuClickLayout() {
  UiMenuClickLayout L{};
  static const char kTitle[] = "PAUSED";
  static const char kTagline[] = "THE STORE CAN WAIT";
  constexpr float kPauseTitlePx = 0.00095f;
  constexpr float kPauseTaglinePx = 0.00052f;
  constexpr float kPauseSubPx = 0.00056f;
  const float pauseLineMul = kIkeaMenuOptionLineSkipMul * kPauseMenuOptionLineExtraMul;
  float titleW = 0.f;
  float subW = 0.f;
  int subLines = 2;
  if (gHudUiFontReady) {
    titleW = measureHudFontRunPx(kTitle, std::strlen(kTitle), kIkeaMenuFontTrackPx) * kPauseTitlePx;
    subW = std::max(subW,
                    measureHudFontRunPx(kTagline, std::strlen(kTagline), kIkeaMenuFontTrackPx) * kPauseTaglinePx);
    static const char kSub[] = "RESUME\nEXIT";
    const char* sn = kSub;
    for (const char* p = kSub;; ++p) {
      if (*p == '\n' || *p == '\0') {
        subW = std::max(
            subW, measureHudFontRunPx(sn, static_cast<size_t>(p - sn), kIkeaMenuFontTrackPx) * kPauseSubPx);
        if (*p == '\0')
          break;
        sn = p + 1;
      }
    }
  } else {
    titleW = 0.5f;
    subW = 0.5f;
  }
  L.optionLines = 2;
  constexpr float panelPadX = 0.22f;
  constexpr float panelPadY = 0.12f;
  L.panelHalfW = std::max(0.5f * std::max(titleW, subW) + panelPadX, 0.58f);
  constexpr float panelTopY = 0.12f;
  const float titleBlockH = gHudUiFontReady ? gHudUiFontLineSkipPx * kPauseTitlePx : 0.09f;
  const float taglineBlockH = gHudUiFontReady ? gHudUiFontLineSkipPx * kPauseTaglinePx : 0.f;
  const float subBlockH =
      gHudUiFontReady ? (static_cast<float>(subLines) * gHudUiFontLineSkipPx * kPauseSubPx * pauseLineMul +
                         0.02f)
                      : 0.19f;
  const float textStackH = titleBlockH + taglineBlockH + subBlockH + (gHudUiFontReady ? 0.09f : 0.04f);
  const float panelBotY = panelTopY - textStackH - 2.f * panelPadY;
  (void)panelBotY;
  const float line1Y = panelTopY - panelPadY - titleBlockH * 0.2f;
  const float lineTagY = line1Y - titleBlockH - 0.014f;
  L.line2Y = lineTagY - taglineBlockH - (gHudUiFontReady ? 0.028f : 0.03f);
  L.lineSkipNdc = (gHudUiFontReady ? gHudUiFontLineSkipPx * kPauseSubPx
                                   : std::max(gHudUiFontLineSkipPx * kPauseSubPx, 0.048f)) *
                  pauseLineMul;
  return L;
}

static std::vector<Vertex> buildIntroSplashVertices() {
  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  std::vector<Vertex> mesh;
  mesh.reserve(256);

  {
    const glm::vec3 v0{-1.f, -1.f, 0.f}, v1{1.f, -1.f, 0.f};
    const glm::vec3 v2{1.f, 1.f, 0.f}, v3{-1.f, 1.f, 0.f};
    mesh.push_back({v0, n, kUiBackdropTag, {}});
    mesh.push_back({v2, n, kUiBackdropTag, {}});
    mesh.push_back({v1, n, kUiBackdropTag, {}});
    mesh.push_back({v0, n, kUiBackdropTag, {}});
    mesh.push_back({v3, n, kUiBackdropTag, {}});
    mesh.push_back({v2, n, kUiBackdropTag, {}});
  }

  static const char kLine1[] = "Made by";
  static const char kLine2[] = "memesdudeguy";
  constexpr float kPx = 0.0014f;

  if (gHudUiFontReady) {
    const float w1 = measureHudFontRunPx(kLine1, std::strlen(kLine1), kIkeaMenuFontTrackPx) * kPx;
    const float w2 = measureHudFontRunPx(kLine2, std::strlen(kLine2), kIkeaMenuFontTrackPx) * kPx;
    const float lineSkip = gHudUiFontLineSkipPx * kPx * 1.3f;
    appendHudFontRun(mesh, n, kUiHudFontPri, kLine1, std::strlen(kLine1),
                     -0.5f * w1, 0.05f + lineSkip * 0.5f, kPx, kIkeaMenuFontTrackPx);
    appendHudFontRun(mesh, n, kUiHudFontPri, kLine2, std::strlen(kLine2),
                     -0.5f * w2, 0.05f - lineSkip * 0.5f, kPx, kIkeaMenuFontTrackPx);
  } else {
    appendStbEasyQuads(mesh, n, kUiTextTag, "Made by\nmemesdudeguy", -0.35f, 0.06f, 0.0008f);
  }
  return mesh;
}

static std::vector<Vertex> buildLoadingScreenVertices() {
  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  std::vector<Vertex> mesh;
  mesh.reserve(200);

  {
    const glm::vec3 v0{-1.f, -1.f, 0.f}, v1{1.f, -1.f, 0.f};
    const glm::vec3 v2{1.f, 1.f, 0.f}, v3{-1.f, 1.f, 0.f};
    const glm::vec4 greyBg = kUiDeathVignetteTag;
    mesh.push_back({v0, n, greyBg, {}});
    mesh.push_back({v2, n, greyBg, {}});
    mesh.push_back({v1, n, greyBg, {}});
    mesh.push_back({v0, n, greyBg, {}});
    mesh.push_back({v3, n, greyBg, {}});
    mesh.push_back({v2, n, greyBg, {}});
  }

  static const char kText[] = "LOADING...";
  constexpr float kPx = 0.0011f;
  if (gHudUiFontReady) {
    const float w = measureHudFontRunPx(kText, std::strlen(kText), kIkeaMenuFontTrackPx) * kPx;
    appendHudFontRun(mesh, n, kUiHudFontPri, kText, std::strlen(kText),
                     -0.5f * w, -0.02f, kPx, kIkeaMenuFontTrackPx);
  } else {
    appendStbEasyQuads(mesh, n, kUiTextTag, kText, -0.25f, 0.0f, 0.0006f);
  }
  return mesh;
}

static std::vector<Vertex> buildTitleMenuMainOverlayVertices(bool showContinue) {
  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  std::vector<Vertex> mesh;
  mesh.reserve(1200);
  const char* kSub = showContinue ? "CONTINUE\nNEW GAME\nEXIT" : "NEW GAME\nEXIT";
  constexpr float kSubPx = 0.00062f;

  {
    const glm::vec3 v0{-1.f, -1.f, 0.f};
    const glm::vec3 v1{1.f, -1.f, 0.f};
    const glm::vec3 v2{1.f, 1.f, 0.f};
    const glm::vec3 v3{-1.f, 1.f, 0.f};
    mesh.push_back({v0, n, kUiHudVignetteTag, {}});
    mesh.push_back({v2, n, kUiHudVignetteTag, {}});
    mesh.push_back({v1, n, kUiHudVignetteTag, {}});
    mesh.push_back({v0, n, kUiHudVignetteTag, {}});
    mesh.push_back({v3, n, kUiHudVignetteTag, {}});
    mesh.push_back({v2, n, kUiHudVignetteTag, {}});
  }

  {
    constexpr float loX = -0.5f, hiX = 0.5f, loY = 0.56f, hiY = 0.94f;
    const glm::vec3 la{loX, loY, 0.f}, lb{hiX, loY, 0.f}, lc{hiX, hiY, 0.f}, ld{loX, hiY, 0.f};
    const glm::vec2 uvBl(0.f, 1.f), uvBr(1.f, 1.f), uvTr(1.f, 0.f), uvTl(0.f, 0.f);
    mesh.push_back({la, n, kUiIkeaLogoTag, uvBl});
    mesh.push_back({lc, n, kUiIkeaLogoTag, uvTr});
    mesh.push_back({lb, n, kUiIkeaLogoTag, uvBr});
    mesh.push_back({la, n, kUiIkeaLogoTag, uvBl});
    mesh.push_back({ld, n, kUiIkeaLogoTag, uvTl});
    mesh.push_back({lc, n, kUiIkeaLogoTag, uvTr});
  }

  float subW = 0.f;
  int subLines = 1;
  if (gHudUiFontReady) {
    for (const char* q = kSub; *q; ++q)
      if (*q == '\n')
        ++subLines;
    const char* sn = kSub;
    for (const char* p = kSub;; ++p) {
      if (*p == '\n' || *p == '\0') {
        subW = std::max(subW, measureHudFontRunPx(sn, static_cast<size_t>(p - sn), kIkeaMenuFontTrackPx) * kSubPx);
        if (*p == '\0')
          break;
        sn = p + 1;
      }
    }
  } else {
    subW = 0.5f;
    subLines = showContinue ? 3 : 2;
  }

  constexpr float panelPadX = 0.22f;
  constexpr float panelPadY = 0.14f;
  const float panelHalfW = std::max(0.5f * subW + panelPadX, 0.62f);
  constexpr float panelTopY = 0.14f;
  const float subBlockH =
      gHudUiFontReady ? (static_cast<float>(subLines) * gHudUiFontLineSkipPx * kSubPx * kTitleMenuOptionLineSkipMul +
                         0.02f)
                      : 0.26f;
  const float textStackH = subBlockH + (gHudUiFontReady ? 0.06f : 0.04f);
  const float panelBotY = panelTopY - textStackH - 2.f * panelPadY;

  const glm::vec3 pa{-panelHalfW, panelBotY, 0.f};
  const glm::vec3 pb{panelHalfW, panelBotY, 0.f};
  const glm::vec3 pc{panelHalfW, panelTopY, 0.f};
  const glm::vec3 pd{-panelHalfW, panelTopY, 0.f};
  appendPanelQuad(mesh, n, kUiIkeaPanelTag, pa, pb, pc, pd);
  appendMenuFrameQuad(mesh, n, pa, pb, pc, pd);

  const float line2Y = panelTopY - panelPadY - (gHudUiFontReady ? 0.02f : 0.03f);
  if (gHudUiFontReady) {
    appendOptionBtnQuads(mesh, n, kSub, line2Y, kSubPx, kTitleMenuOptionLineSkipMul, panelHalfW - 0.02f);
    appendHudFontMultilineCentered(mesh, n, kUiIkeaFontOpt, kSub, 0.f, line2Y, kSubPx,
                                   kIkeaMenuFontTrackPx, kTitleMenuOptionLineSkipMul);
  } else {
    const char* fb = showContinue ? "CONTINUE\nNEW GAME\nEXIT" : "NEW GAME\nEXIT";
    constexpr float scale = 0.0048f;
    stb_easy_font_spacing(-0.5f);
    const int fontW = stb_easy_font_width(const_cast<char*>(fb));
    const float textW = static_cast<float>(fontW) * scale;
    appendStbEasyQuads(mesh, n, kUiTextTag, fb, -0.5f * textW, panelTopY - panelPadY, scale);
  }
  return mesh;
}

static std::vector<Vertex> buildTitleMenuSlotPickerVertices(const std::array<bool, 4>& slotUsed) {
  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  std::vector<Vertex> mesh;
  mesh.reserve(1400);
  char block[384];
  int pos = 0;
  pos += std::snprintf(block + pos, sizeof(block) - pos, "CHOOSE SAVE SLOT\n");
  for (int i = 0; i < kGameSaveSlotCount; ++i) {
    pos += std::snprintf(block + pos, sizeof(block) - pos, "%d  SLOT %d  %s\n", i + 1, i + 1,
                         slotUsed[static_cast<size_t>(i)] ? "(SAVED)" : "(EMPTY)");
  }
  std::snprintf(block + pos, sizeof(block) - pos, "\nBACK (ESC)\nLMB: LOAD   RMB: DELETE");
  static const char kTitle[] = "START";
  constexpr float kTitlePx = 0.00112f;
  constexpr float kSubPx = 0.00058f;

  {
    const glm::vec3 v0{-1.f, -1.f, 0.f};
    const glm::vec3 v1{1.f, -1.f, 0.f};
    const glm::vec3 v2{1.f, 1.f, 0.f};
    const glm::vec3 v3{-1.f, 1.f, 0.f};
    mesh.push_back({v0, n, kUiHudVignetteTag, {}});
    mesh.push_back({v2, n, kUiHudVignetteTag, {}});
    mesh.push_back({v1, n, kUiHudVignetteTag, {}});
    mesh.push_back({v0, n, kUiHudVignetteTag, {}});
    mesh.push_back({v3, n, kUiHudVignetteTag, {}});
    mesh.push_back({v2, n, kUiHudVignetteTag, {}});
  }

  float titleW = 0.f;
  float subW = 0.f;
  int subLines = 1;
  if (gHudUiFontReady) {
    titleW = measureHudFontRunPx(kTitle, std::strlen(kTitle), kIkeaMenuFontTrackPx) * kTitlePx;
    for (const char* q = block; *q; ++q)
      if (*q == '\n')
        ++subLines;
    const char* sn = block;
    for (const char* p = block;; ++p) {
      if (*p == '\n' || *p == '\0') {
        subW = std::max(
            subW, measureHudFontRunPx(sn, static_cast<size_t>(p - sn), kIkeaMenuFontTrackPx) * kSubPx);
        if (*p == '\0')
          break;
        sn = p + 1;
      }
    }
  } else {
    titleW = 0.4f;
    subW = 0.55f;
    subLines = 7;
  }

  constexpr float panelPadX = 0.22f;
  constexpr float panelPadY = 0.12f;
  const float panelHalfW = std::max(0.5f * std::max(titleW, subW) + panelPadX, 0.64f);
  constexpr float panelTopY = 0.16f;
  const float titleBlockH = gHudUiFontReady ? gHudUiFontLineSkipPx * kTitlePx : 0.09f;
  const float subBlockH =
      gHudUiFontReady ? (static_cast<float>(subLines) * gHudUiFontLineSkipPx * kSubPx * kIkeaMenuOptionLineSkipMul +
                         0.02f)
                      : 0.38f;
  const float textStackH = titleBlockH + subBlockH + 0.04f;
  const float panelBotY = panelTopY - textStackH - 2.f * panelPadY;

  const glm::vec3 pa{-panelHalfW, panelBotY, 0.f};
  const glm::vec3 pb{panelHalfW, panelBotY, 0.f};
  const glm::vec3 pc{panelHalfW, panelTopY, 0.f};
  const glm::vec3 pd{-panelHalfW, panelTopY, 0.f};
  appendPanelQuad(mesh, n, kUiIkeaPanelTag, pa, pb, pc, pd);
  appendMenuFrameQuad(mesh, n, pa, pb, pc, pd);

  const float line1Y = panelTopY - panelPadY - titleBlockH * 0.2f;
  const float line2Y = line1Y - titleBlockH - 0.03f;
  if (gHudUiFontReady) {
    const float tw = measureHudFontRunPx(kTitle, std::strlen(kTitle), kIkeaMenuFontTrackPx) * kTitlePx;
    appendHudFontRun(mesh, n, kUiIkeaFontAcc, kTitle, std::strlen(kTitle), -0.5f * tw, line1Y, kTitlePx,
                     kIkeaMenuFontTrackPx);
    appendOptionBtnQuads(mesh, n, block, line2Y, kSubPx, kIkeaMenuOptionLineSkipMul, panelHalfW - 0.02f);
    appendHudFontMultilineCentered(mesh, n, kUiIkeaFontOpt, block, 0.f, line2Y, kSubPx,
                                   kIkeaMenuFontTrackPx, kIkeaMenuOptionLineSkipMul);
  } else {
    constexpr float scale = 0.0042f;
    stb_easy_font_spacing(-0.5f);
    appendStbEasyQuads(mesh, n, kUiTextTag, block, -0.5f * subW, line2Y, scale);
  }
  return mesh;
}

#pragma pack(push, 1)
struct GameSaveFileV1 {
  uint32_t magic;
  uint32_t version;
  float camX, camY, camZ;
  float yaw, pitch;
  float playerHealth;
  float eyeHeight;
};
// v2 on disk: audio blob without storeDayMusicTrackIdx (60 bytes).
struct AudioStoreCycleSaveStateV1Disk {
  uint32_t version;
  uint32_t storePhase;
  uint32_t flags;
  uint64_t storeCursorFrames;
  uint64_t horrorCursorFrames;
  uint64_t chaseCursorFrames;
  uint64_t shrekCursorFrames;
  uint64_t blackoutRemainingMs;
  uint64_t dayRestoreRemainingMs;
};
struct GameSaveFileV2Read {
  uint32_t magic;
  uint32_t version;
  float camX, camY, camZ;
  float yaw, pitch;
  float playerHealth;
  float eyeHeight;
  AudioStoreCycleSaveStateV1Disk audio;
};
struct GameSaveFileV3 {
  uint32_t magic;
  uint32_t version;
  float camX, camY, camZ;
  float yaw, pitch;
  float playerHealth;
  float eyeHeight;
  AudioStoreCycleSaveState audioState;
};
#pragma pack(pop)
static_assert(sizeof(AudioStoreCycleSaveStateV1Disk) == 60);
static_assert(sizeof(GameSaveFileV2Read) == 96);
static_assert(sizeof(GameSaveFileV3) == 100);
static constexpr uint32_t kGameSaveMagic = 0x31474B56u;  // 'VKG1'
static constexpr uint32_t kGameSaveVersion = 3u;
static constexpr uint32_t kInventoryStateSaveMagic = 0x31564E49u;  // 'INV1'
static constexpr uint32_t kInventoryStateSaveVersion = 2u;

static float wrapAnglePi(float a) {
  const float pi = glm::pi<float>();
  const float tpi = glm::two_pi<float>();
  while (a > pi)
    a -= tpi;
  while (a < -pi)
    a += tpi;
  return a;
}

static void buildHealthHudOverlayVertices(float hp, float hpMax, float hunger, float hungerMax, float yawRad,
                                          int dayCount, bool showInteractHint, std::vector<Vertex>& mesh) {
  const glm::vec3 n{0.0f, 0.0f, 1.0f};
  mesh.clear();
  mesh.reserve(2200);
  const float bearingRad = std::atan2(std::cos(yawRad), std::sin(yawRad));

  // Top-right HUD: high mesh Y → top of screen after shader y flip.
  constexpr float hudR = 0.97f;
  constexpr float barW = 0.42f;
  const float hudL = hudR - barW;
  const float hudCx = 0.5f * (hudL + hudR);
  constexpr float barH = 0.012f;
  constexpr float barTop = 0.756f;
  const float barBot = barTop - barH;
  constexpr float tickW = 0.0018f;
  const float fillT = glm::clamp(hp / std::max(hpMax, 1e-4f), 0.f, 1.f);
  const bool crit =
      hpMax > 1e-4f && hp / hpMax < (kPlayerHealthScreenEdgeCritical / std::max(kPlayerHealthMax, 1e-4f));
  const glm::vec4 healthFillTag = crit ? kUiHealthFillCritTag : kUiHealthFillTag;
  const float hungerFillT = glm::clamp(hunger / std::max(hungerMax, 1e-4f), 0.f, 1.f);

  const auto quad = [&](const glm::vec3& bl, const glm::vec3& br, const glm::vec3& tr, const glm::vec3& tl,
                        const glm::vec4& col) {
    const glm::vec2 u00(0.f, 0.f), u10(1.f, 0.f), u11(1.f, 1.f), u01(0.f, 1.f);
    mesh.push_back({bl, n, col, u00});
    mesh.push_back({tr, n, col, u11});
    mesh.push_back({br, n, col, u10});
    mesh.push_back({bl, n, col, u00});
    mesh.push_back({tl, n, col, u01});
    mesh.push_back({tr, n, col, u11});
  };

  constexpr float panelTop = 0.865f;
  constexpr float padY = 0.018f;
  quad(glm::vec3(hudL - 0.01f, barBot - padY - 0.048f, 0.f), glm::vec3(hudR + 0.012f, barBot - padY - 0.048f, 0.f),
       glm::vec3(hudR + 0.012f, panelTop + padY * 0.38f, 0.f), glm::vec3(hudL - 0.01f, panelTop + padY * 0.38f, 0.f),
       kUiPipHudBgTag);

  quad(glm::vec3(hudL - 0.0015f, barBot - 0.0012f, 0.f), glm::vec3(hudR + 0.0015f, barBot - 0.0012f, 0.f),
       glm::vec3(hudR + 0.0015f, barTop + 0.0012f, 0.f), glm::vec3(hudL - 0.0015f, barTop + 0.0012f, 0.f),
       kUiHealthFrameTag);
  quad(glm::vec3(hudL, barBot, 0.f), glm::vec3(hudR, barBot, 0.f), glm::vec3(hudR, barTop, 0.f),
       glm::vec3(hudL, barTop, 0.f), kUiHealthTrackTag);

  const float fillR = hudL + barW * fillT;
  if (fillT > 1e-4f) {
    quad(glm::vec3(hudL, barBot + 0.0018f, 0.f), glm::vec3(fillR, barBot + 0.0018f, 0.f),
         glm::vec3(fillR, barTop - 0.0018f, 0.f), glm::vec3(hudL, barTop - 0.0018f, 0.f), healthFillTag);
  }

  constexpr float hungerH = 0.016f;
  const float hungerTop = barBot - 0.034f;
  const float hungerBot = hungerTop - hungerH;
  quad(glm::vec3(hudL - 0.0015f, hungerBot - 0.0012f, 0.f),
       glm::vec3(hudR + 0.0015f, hungerBot - 0.0012f, 0.f),
       glm::vec3(hudR + 0.0015f, hungerTop + 0.0012f, 0.f),
       glm::vec3(hudL - 0.0015f, hungerTop + 0.0012f, 0.f), kUiHungerFrameTag);
  quad(glm::vec3(hudL, hungerBot, 0.f), glm::vec3(hudR, hungerBot, 0.f),
       glm::vec3(hudR, hungerTop, 0.f), glm::vec3(hudL, hungerTop, 0.f), kUiHungerTrackTag);
  const float hungerR = hudL + barW * hungerFillT;
  if (hungerFillT > 1e-4f) {
    quad(glm::vec3(hudL, hungerBot + 0.0018f, 0.f), glm::vec3(hungerR, hungerBot + 0.0018f, 0.f),
         glm::vec3(hungerR, hungerTop - 0.0018f, 0.f), glm::vec3(hudL, hungerTop - 0.0018f, 0.f),
         kUiHungerFillTag);
  }

  constexpr float kNdcPerRad = 0.42f / glm::radians(52.f);
  // Compass ticks / letters centered on the health bar panel (not screen center).
  const float compassHalfW = barW * 0.44f;
  const float compMidY = 0.818f;
  quad(glm::vec3(hudL, compMidY - 0.0014f, 0.f), glm::vec3(hudR, compMidY - 0.0014f, 0.f),
       glm::vec3(hudR, compMidY + 0.0014f, 0.f), glm::vec3(hudL, compMidY + 0.0014f, 0.f), kUiPipHudLineDimTag);

  // Fixed caret at panel center = facing; scrolling ticks are world headings.
  {
    const glm::vec2 u00(0.f, 0.f), u10(1.f, 0.f), u11(1.f, 1.f);
    const float yTip = compMidY + 0.019f;
    const float yBase = compMidY + 0.002f;
    const float hw = 0.007f;
    const glm::vec3 tip{hudCx, yTip, 0.f};
    const glm::vec3 br{hudCx + hw, yBase, 0.f};
    const glm::vec3 bl{hudCx - hw, yBase, 0.f};
    mesh.push_back({tip, n, kUiPipHudLineBrightTag, u00});
    mesh.push_back({br, n, kUiPipHudLineBrightTag, u11});
    mesh.push_back({bl, n, kUiPipHudLineBrightTag, u10});
    mesh.push_back({tip, n, kUiPipHudLineBrightTag, u00});
    mesh.push_back({bl, n, kUiPipHudLineBrightTag, u10});
    mesh.push_back({br, n, kUiPipHudLineBrightTag, u11});
  }

  for (int deg = 0; deg < 360; deg += 10) {
    const float theta = glm::radians(static_cast<float>(deg));
    const float delta = wrapAnglePi(bearingRad - theta);
    const float off = delta * kNdcPerRad;
    if (std::fabs(off) > compassHalfW)
      continue;
    const bool major = (deg % 30) == 0;
    const float tickH = major ? 0.014f : 0.007f;
    const float tx = hudCx + off;
    quad(glm::vec3(tx - tickW * 0.5f, compMidY, 0.f), glm::vec3(tx + tickW * 0.5f, compMidY, 0.f),
         glm::vec3(tx + tickW * 0.5f, compMidY + tickH, 0.f), glm::vec3(tx - tickW * 0.5f, compMidY + tickH, 0.f),
         major ? kUiPipHudLineBrightTag : kUiPipHudLineDimTag);
  }

  static const char* kCardLabels[4] = {"N", "E", "S", "W"};
  static const float kCardAngles[4] = {0.f, glm::pi<float>() * 0.5f, glm::pi<float>(), -glm::pi<float>() * 0.5f};

  constexpr float kHudLabelPx = 0.00038f;
  constexpr float kHudHpPx = 0.00062f;
  const float hpTextY = barTop + 0.004f;
  const float compLabelY = compMidY + 0.012f;
  const float hungerTextY = hungerBot - 0.012f;
  if (gHudUiFontReady) {
    static const char kHealthLbl[] = "HP";
    static const char kHungerLbl[] = "HUNGER";
    appendHudFontRun(mesh, n, kUiHudFontAcc, kHealthLbl, std::strlen(kHealthLbl), hudL, hpTextY, kHudLabelPx);
    char hpBuf[24];
    std::snprintf(hpBuf, sizeof(hpBuf), "%.0f", hp);
    const float hw = measureHudFontRunPx(hpBuf, std::strlen(hpBuf)) * kHudHpPx;
    appendHudFontRun(mesh, n, kUiHudFontPri, hpBuf, std::strlen(hpBuf), hudR - hw, hpTextY, kHudHpPx);
    constexpr float kHungerLabelPx = 0.00034f;
    constexpr float kHungerValPx = 0.00054f;
    appendHudFontRun(mesh, n, kUiHudFontAcc, kHungerLbl, std::strlen(kHungerLbl), hudL, hungerTextY,
                     kHungerLabelPx);
    char hungerBuf[24];
    std::snprintf(hungerBuf, sizeof(hungerBuf), "%.0f", hunger);
    const float huw = measureHudFontRunPx(hungerBuf, std::strlen(hungerBuf)) * kHungerValPx;
    appendHudFontRun(mesh, n, kUiHudFontPri, hungerBuf, std::strlen(hungerBuf), hudR - huw, hungerTextY,
                     kHungerValPx);
    constexpr float kCardPx = 0.00034f;
    for (int i = 0; i < 4; ++i) {
      const float delta = wrapAnglePi(bearingRad - kCardAngles[i]);
      const float off = delta * kNdcPerRad;
      if (std::fabs(off) > compassHalfW * 0.92f)
        continue;
      const char* cl = kCardLabels[i];
      const float tw = measureHudFontRunPx(cl, std::strlen(cl)) * kCardPx;
      appendHudFontRun(mesh, n, kUiHudFontPri, cl, std::strlen(cl), hudCx + off - 0.5f * tw, compLabelY, kCardPx);
    }
  } else {
    char line[64];
    std::snprintf(line, sizeof(line), "HP: %.0f", hp);
    appendStbEasyQuads(mesh, n, kUiPipHudTextTag, line, hudL, hpTextY, 0.00115f);
    constexpr float scaleCard = 0.00105f;
    stb_easy_font_spacing(-0.5f);
    for (int i = 0; i < 4; ++i) {
      const float delta = wrapAnglePi(bearingRad - kCardAngles[i]);
      const float off = delta * kNdcPerRad;
      if (std::fabs(off) > compassHalfW * 0.92f)
        continue;
      const int w = stb_easy_font_width(const_cast<char*>(kCardLabels[i]));
      const float tw = static_cast<float>(w) * scaleCard;
      appendStbEasyQuads(mesh, n, kUiPipHudTextTag, kCardLabels[i], hudCx + off - 0.5f * tw, compLabelY, scaleCard);
    }
  }

  {
    char dayBuf[32];
    std::snprintf(dayBuf, sizeof(dayBuf), "DAY %d", dayCount);
    const float dayY = hungerTextY - 0.022f;
    if (gHudUiFontReady) {
      const float dayW = measureHudFontRunPx(dayBuf, std::strlen(dayBuf)) * kHudLabelPx;
      appendHudFontRun(mesh, n, kUiHudFontPri, dayBuf, std::strlen(dayBuf), hudR - dayW, dayY, kHudLabelPx);
    } else {
      appendStbEasyQuads(mesh, n, kUiPipHudTextTag, dayBuf, hudL, dayY, 0.00115f);
    }
  }

  if (showInteractHint) {
    static const char kHint[] = "PRESS E TO PICK UP";
    constexpr float kHintPx = 0.00056f;
    constexpr float hintY = -0.15f;
    if (gHudUiFontReady) {
      const float hintW = measureHudFontRunPx(kHint, std::strlen(kHint)) * kHintPx;
      appendHudFontRun(mesh, n, kUiHudFontAcc, kHint, std::strlen(kHint), -0.5f * hintW, hintY, kHintPx);
    } else {
      const int w = stb_easy_font_width(const_cast<char*>(kHint));
      const float tw = static_cast<float>(w) * 0.0011f;
      appendStbEasyQuads(mesh, n, kUiPipHudTextTag, kHint, -0.5f * tw, hintY, 0.0011f);
    }
  }
}

void buildCeilingMesh(int centerChunkX, int centerChunkZ, std::vector<Vertex>& out) {
  out.clear();
  const float chunkWorld = static_cast<float>(kChunkCellCount) * kCellSize;
  const int side = kChunkRadius * 2 + 1;
  out.reserve(static_cast<size_t>(side * side * 6));

  for (int cz = -kChunkRadius; cz <= kChunkRadius; ++cz) {
    for (int cx = -kChunkRadius; cx <= kChunkRadius; ++cx) {
      const int gcx = centerChunkX + cx;
      const int gcz = centerChunkZ + cz;
      const float ox = static_cast<float>(gcx) * chunkWorld;
      const float oz = static_cast<float>(gcz) * chunkWorld;
      const float x1 = ox + chunkWorld;
      const float z1 = oz + chunkWorld;

      const glm::vec3 n{0, -1, 0};
      glm::vec3 p00{ox, kCeilingY, oz};
      glm::vec3 p10{x1, kCeilingY, oz};
      glm::vec3 p11{x1, kCeilingY, z1};
      glm::vec3 p01{ox, kCeilingY, z1};

      const glm::vec4 vc = vrgb(meshVertexColor());

      // Winding for visible underside (CW with proj Y flip) — opposite of floor.
      out.push_back({p00, n, vc, {}});
      out.push_back({p01, n, vc, {}});
      out.push_back({p11, n, vc, {}});
      out.push_back({p00, n, vc, {}});
      out.push_back({p11, n, vc, {}});
      out.push_back({p10, n, vc, {}});
    }
  }
}

void buildTerrainMesh(int centerChunkX, int centerChunkZ, std::vector<Vertex>& out) {
  out.clear();
  const float chunkWorld = static_cast<float>(kChunkCellCount) * kCellSize;
  const int side = kChunkRadius * 2 + 1;
  out.reserve(static_cast<size_t>(side * side * 6));

  for (int cz = -kChunkRadius; cz <= kChunkRadius; ++cz) {
    for (int cx = -kChunkRadius; cx <= kChunkRadius; ++cx) {
      const int gcx = centerChunkX + cx;
      const int gcz = centerChunkZ + cz;
      const float ox = static_cast<float>(gcx) * chunkWorld;
      const float oz = static_cast<float>(gcz) * chunkWorld;
      const float x1 = ox + chunkWorld;
      const float z1 = oz + chunkWorld;

      const glm::vec3 n{0, 1, 0};
      glm::vec3 p00{ox, kGroundY, oz};
      glm::vec3 p10{x1, kGroundY, oz};
      glm::vec3 p11{x1, kGroundY, z1};
      glm::vec3 p01{ox, kGroundY, z1};

      const glm::vec4 vc = vrgb(meshVertexColor());

      out.push_back({p00, n, vc, {}});
      out.push_back({p10, n, vc, {}});
      out.push_back({p11, n, vc, {}});
      out.push_back({p00, n, vc, {}});
      out.push_back({p11, n, vc, {}});
      out.push_back({p01, n, vc, {}});
    }
  }
}

static void writeTerrainChunkVerts(int gcx, int gcz, Vertex* dst) {
  const float chunkWorld = static_cast<float>(kChunkCellCount) * kCellSize;
  const float ox = static_cast<float>(gcx) * chunkWorld;
  const float oz = static_cast<float>(gcz) * chunkWorld;
  const float x1 = ox + chunkWorld;
  const float z1 = oz + chunkWorld;
  const glm::vec3 n{0, 1, 0};
  glm::vec3 p00{ox, kGroundY, oz};
  glm::vec3 p10{x1, kGroundY, oz};
  glm::vec3 p11{x1, kGroundY, z1};
  glm::vec3 p01{ox, kGroundY, z1};
  const glm::vec4 vc = vrgb(meshVertexColor());
  dst[0] = {p00, n, vc, {}};
  dst[1] = {p10, n, vc, {}};
  dst[2] = {p11, n, vc, {}};
  dst[3] = {p00, n, vc, {}};
  dst[4] = {p11, n, vc, {}};
  dst[5] = {p01, n, vc, {}};
}

static void writeCeilingChunkVerts(int gcx, int gcz, Vertex* dst) {
  const float chunkWorld = static_cast<float>(kChunkCellCount) * kCellSize;
  const float ox = static_cast<float>(gcx) * chunkWorld;
  const float oz = static_cast<float>(gcz) * chunkWorld;
  const float x1 = ox + chunkWorld;
  const float z1 = oz + chunkWorld;
  const glm::vec3 n{0, -1, 0};
  glm::vec3 p00{ox, kCeilingY, oz};
  glm::vec3 p10{x1, kCeilingY, oz};
  glm::vec3 p11{x1, kCeilingY, z1};
  glm::vec3 p01{ox, kCeilingY, z1};
  const glm::vec4 vc = vrgb(meshVertexColor());
  dst[0] = {p00, n, vc, {}};
  dst[1] = {p01, n, vc, {}};
  dst[2] = {p11, n, vc, {}};
  dst[3] = {p00, n, vc, {}};
  dst[4] = {p11, n, vc, {}};
  dst[5] = {p10, n, vc, {}};
}

struct App {
  SDL_Window* window = nullptr;
  int winW = kWidth;
  int winH = kHeight;

  VkInstance instance = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue graphicsQueue = VK_NULL_HANDLE;
  VkQueue presentQueue = VK_NULL_HANDLE;

  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  std::vector<VkImage> swapchainImages;
  VkFormat swapchainImageFormat{};
  VkExtent2D swapchainExtent{};
  std::vector<VkImageView> swapchainImageViews;

  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkRenderPass presentRenderPass = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline graphicsPipeline = VK_NULL_HANDLE;
  VkPipeline uiPipeline = VK_NULL_HANDLE;
  VkPipeline uiPresentPipeline = VK_NULL_HANDLE;
  VkPipelineLayout postPipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout postDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipeline postProcessPipeline = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> postDescriptorSets;
  VkPipelineCache pipelineCache = VK_NULL_HANDLE;
  std::string pipelineCachePath;

  VkExtent2D sceneExtent{};
  std::vector<VkImage> depthImages;
  std::vector<VkDeviceMemory> depthMemories;
  std::vector<VkImageView> depthViews;
  std::vector<VkImage> sceneColorImages;
  std::vector<VkDeviceMemory> sceneColorMemories;
  std::vector<VkImageView> sceneColorViews;
  std::vector<VkFramebuffer> sceneFramebuffers;
  std::vector<bool> sceneColorWasSampled;
  std::vector<bool> depthGpuReady;
  VkSampler sceneRenderSampler = VK_NULL_HANDLE;
  float horrorPresentTime = 0.f;
  float postNightHorrorWeight = 0.f;
  float postNightPursuitMix = 0.f;
  VkImage sceneTextureImage = VK_NULL_HANDLE;
  VkDeviceMemory sceneTextureMemory = VK_NULL_HANDLE;
  VkImageView sceneTextureView = VK_NULL_HANDLE;
  VkSampler sceneTextureSampler = VK_NULL_HANDLE;
  VkImage signTextureImage = VK_NULL_HANDLE;
  VkDeviceMemory signTextureMemory = VK_NULL_HANDLE;
  VkImageView signTextureView = VK_NULL_HANDLE;
  VkSampler signTextureSampler = VK_NULL_HANDLE;
  VkImage shelfRackTextureImage = VK_NULL_HANDLE;
  VkDeviceMemory shelfRackTextureMemory = VK_NULL_HANDLE;
  VkImageView shelfRackTextureView = VK_NULL_HANDLE;
  VkSampler shelfRackTextureSampler = VK_NULL_HANDLE;
  VkImage crateTextureImage = VK_NULL_HANDLE;
  VkDeviceMemory crateTextureMemory = VK_NULL_HANDLE;
  VkImageView crateTextureView = VK_NULL_HANDLE;
  VkSampler crateTextureSampler = VK_NULL_HANDLE;
  VkImage palletTextureImage = VK_NULL_HANDLE;
  VkDeviceMemory palletTextureMemory = VK_NULL_HANDLE;
  VkImageView palletTextureView = VK_NULL_HANDLE;
  VkSampler palletTextureSampler = VK_NULL_HANDLE;
  VkImage deliMetalTextureImage = VK_NULL_HANDLE;
  VkDeviceMemory deliMetalTextureMemory = VK_NULL_HANDLE;
  VkImageView deliMetalTextureView = VK_NULL_HANDLE;
  VkSampler deliMetalTextureSampler = VK_NULL_HANDLE;
  VkImage deliBaseTextureImage = VK_NULL_HANDLE;
  VkDeviceMemory deliBaseTextureMemory = VK_NULL_HANDLE;
  VkImageView deliBaseTextureView = VK_NULL_HANDLE;
  VkSampler deliBaseTextureSampler = VK_NULL_HANDLE;

  struct ExtraTexSlot {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
  };
  std::array<ExtraTexSlot, kMaxExtraTextures> extraTexSlots{};
  uint32_t extraTexturesLoadedCount = 0;

  std::vector<VkFramebuffer> framebuffers;

  VkCommandPool commandPool = VK_NULL_HANDLE;
  std::vector<VkCommandBuffer> commandBuffers;

  std::vector<VkSemaphore> imageAvailableSemaphores;
  std::vector<VkSemaphore> renderFinishedSemaphores;
  std::vector<VkFence> inFlightFences;
  uint32_t currentFrame = 0;

  VkBuffer groundVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory groundVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t groundVertexCount = 0;

  VkBuffer ceilingVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory ceilingVertexBufferMemory = VK_NULL_HANDLE;
  void* ceilingMapped = nullptr;
  uint32_t ceilingVertexCount = 0;

  VkBuffer pillarVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory pillarVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t pillarVertexCount = 0;
  VkBuffer pillarInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory pillarInstanceBufferMemory = VK_NULL_HANDLE;
  void* pillarInstanceMapped = nullptr;
  std::vector<glm::mat4> pillarInstanceScratch;
  VkBuffer signInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory signInstanceBufferMemory = VK_NULL_HANDLE;
  void* signInstanceMapped = nullptr;
  std::vector<glm::mat4> signBoardInstanceScratch;
  std::vector<glm::mat4> signStringInstanceScratch;
  VkBuffer crosshairVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory crosshairVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t crosshairVertexCount = 0;
  VkBuffer controlsHelpVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory controlsHelpVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t controlsHelpVertexCount = 0;
  VkBuffer deathMenuVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory deathMenuVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t deathMenuVertexCount = 0;
  VkBuffer pauseMenuVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory pauseMenuVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t pauseMenuVertexCount = 0;
  VkBuffer inventoryMenuVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory inventoryMenuVertexBufferMemory = VK_NULL_HANDLE;
  void* inventoryMenuVertexMapped = nullptr;
  VkDeviceSize inventoryMenuVertexBufferBytes = 0;
  std::vector<Vertex> inventoryMenuVertexCache;
  uint32_t inventoryMenuVertexCount = 0;
  int inventoryMenuCacheScroll = -1;
  uint32_t inventoryMenuCacheRevision = 0;
  VkBuffer titleMenuMainVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory titleMenuMainVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t titleMenuMainVertexCount = 0;
  VkBuffer titleMenuSlotVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory titleMenuSlotVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t titleMenuSlotVertexCount = 0;
  bool inIntroSplash = true;
  float introSplashTime = 0.f;
  VkBuffer introSplashVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory introSplashVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t introSplashVertexCount = 0;

  bool inLoadingScreen = false;
  float loadingScreenTime = 0.f;
  int loadingScreenSlot = -1;
  bool loadingScreenIsNewSave = false;
  VkBuffer loadingScreenVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory loadingScreenVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t loadingScreenVertexCount = 0;

  bool inTitleMenu = false;
  bool titleMenuPickSlot = false;
  bool titleMenuHasContinue = false;
  int titleMenuLastSlot = 0;
  float titleMenuSceneTime = 0.f;
  float titleMenuSlideTime = 0.f;
  bool titleMenuSlideWasSlot = false;
  glm::vec3 titleMenuSceneAnchor{0.f, kTopShelfDeckSurfaceY + kEyeHeight, 12.f};
  int activeSaveSlot = 0;
  bool pendingLoadedAudioStateValid = false;
  AudioStoreCycleSaveState pendingLoadedAudioState{};
  bool pendingLoadedSaveWasDead = false;
  SDL_Cursor* yellowMenuCursor = nullptr;
  VkBuffer healthHudVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory healthHudVertexBufferMemory = VK_NULL_HANDLE;
  void* healthHudVertexMapped = nullptr;
  VkDeviceSize healthHudVertexBufferBytes = 0;
  std::vector<Vertex> healthHudVertexCache;
  uint32_t healthHudCachedVertexCount = 0;
  float healthHudCacheHp = -1e25f;
  float healthHudCacheHpMax = -1.f;
  float healthHudCacheHunger = -1e25f;
  float healthHudCacheHungerMax = -1.f;
  float healthHudCacheYaw = 1e10f;
  int healthHudCacheDayCount = -1;
  bool healthHudCacheInteractHint = false;
  int uboCachedExtraBlend = 0;
  int uboCachedStaffTexBlend = 255;
  VkBuffer signVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory signVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t signVertexCount = 0;
  VkBuffer signStringVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory signStringVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t signStringVertexCount = 0;
  VkBuffer shelfVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shelfVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t shelfVertexCount = 0;
  VkBuffer identityInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory identityInstanceBufferMemory = VK_NULL_HANDLE;
  VkBuffer shelfInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shelfInstanceBufferMemory = VK_NULL_HANDLE;
  void* shelfInstanceMapped = nullptr;
  std::vector<glm::mat4> shelfInstanceScratch;
  VkImage staffGlbDiffuseImage = VK_NULL_HANDLE;
  VkDeviceMemory staffGlbDiffuseMemory = VK_NULL_HANDLE;
  VkImageView staffGlbDiffuseView = VK_NULL_HANDLE;
  VkSampler staffGlbDiffuseSampler = VK_NULL_HANDLE;
  int staffGlbDiffuseActive = 0;
  VkImage hudFontTextureImage = VK_NULL_HANDLE;
  VkDeviceMemory hudFontTextureMemory = VK_NULL_HANDLE;
  VkImageView hudFontTextureView = VK_NULL_HANDLE;
  VkSampler hudFontTextureSampler = VK_NULL_HANDLE;
  VkImage titleIkeaLogoImage = VK_NULL_HANDLE;
  VkDeviceMemory titleIkeaLogoMemory = VK_NULL_HANDLE;
  VkImageView titleIkeaLogoView = VK_NULL_HANDLE;
  VkSampler titleIkeaLogoSampler = VK_NULL_HANDLE;
  VkBuffer shelfCrateVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shelfCrateVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t shelfCrateVertexCount = 0;
  VkBuffer shelfCrateInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shelfCrateInstanceBufferMemory = VK_NULL_HANDLE;
  void* shelfCrateInstanceMapped = nullptr;
  std::vector<glm::mat4> shelfCrateInstanceScratch;
  VkBuffer shelfPalletVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shelfPalletVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t shelfPalletVertexCount = 0;
  VkBuffer shelfPalletInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shelfPalletInstanceBufferMemory = VK_NULL_HANDLE;
  void* shelfPalletInstanceMapped = nullptr;
  std::vector<glm::mat4> shelfPalletInstanceScratch;
  VkBuffer shelfLadderVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shelfLadderVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t shelfLadderVertexCount = 0;
  VkBuffer shelfLadderInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shelfLadderInstanceBufferMemory = VK_NULL_HANDLE;
  void* shelfLadderInstanceMapped = nullptr;
  std::vector<glm::mat4> shelfLadderInstanceScratch;
  VkBuffer shelfBoxCutterVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shelfBoxCutterVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t shelfBoxCutterVertexCount = 0;
  VkBuffer shelfBoxCutterInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shelfBoxCutterInstanceBufferMemory = VK_NULL_HANDLE;
  void* shelfBoxCutterInstanceMapped = nullptr;
  std::vector<glm::mat4> shelfBoxCutterInstanceScratch;
  VkImage shelfBoxCutterTextureImage = VK_NULL_HANDLE;
  VkDeviceMemory shelfBoxCutterTextureMemory = VK_NULL_HANDLE;
  VkImageView shelfBoxCutterTextureView = VK_NULL_HANDLE;
  VkSampler shelfBoxCutterTextureSampler = VK_NULL_HANDLE;
  VkBuffer shelfRustyPipeVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shelfRustyPipeVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t shelfRustyPipeVertexCount = 0;
  VkBuffer shelfRustyPipeInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shelfRustyPipeInstanceBufferMemory = VK_NULL_HANDLE;
  void* shelfRustyPipeInstanceMapped = nullptr;
  std::vector<glm::mat4> shelfRustyPipeInstanceScratch;
  VkImage shelfRustyPipeTextureImage = VK_NULL_HANDLE;
  VkDeviceMemory shelfRustyPipeTextureMemory = VK_NULL_HANDLE;
  VkImageView shelfRustyPipeTextureView = VK_NULL_HANDLE;
  VkSampler shelfRustyPipeTextureSampler = VK_NULL_HANDLE;
  VkBuffer marketVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory marketVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t marketVertexCount = 0;
  VkBuffer marketInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory marketInstanceBufferMemory = VK_NULL_HANDLE;
  void* marketInstanceMapped = nullptr;
  std::vector<glm::mat4> marketInstanceScratch;
  VkBuffer deliPizzaVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory deliPizzaVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t deliPizzaVertexCount = 0;
  VkBuffer deliPizzaInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory deliPizzaInstanceBufferMemory = VK_NULL_HANDLE;
  void* deliPizzaInstanceMapped = nullptr;
  std::vector<glm::mat4> deliPizzaInstanceScratch;
  VkBuffer deliMeatballVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory deliMeatballVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t deliMeatballVertexCount = 0;
  VkBuffer deliMeatballInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory deliMeatballInstanceBufferMemory = VK_NULL_HANDLE;
  void* deliMeatballInstanceMapped = nullptr;
  std::vector<glm::mat4> deliMeatballInstanceScratch;
  VkBuffer fluorescentVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory fluorescentVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t fluorescentVertexCount = 0;
  VkBuffer fluorescentInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory fluorescentInstanceBufferMemory = VK_NULL_HANDLE;
  void* fluorescentInstanceMapped = nullptr;
  std::vector<glm::mat4> fluorescentInstanceScratch;
  VkBuffer employeeVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory employeeVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t employeeVertexCount = 0;
  VkBuffer employeeInstanceBuffer = VK_NULL_HANDLE;
  VkDeviceMemory employeeInstanceBufferMemory = VK_NULL_HANDLE;
  void* employeeInstanceMapped = nullptr;
  std::vector<StaffNpcDrawSlot> staffNpcDrawBuild;
  VkPipeline graphicsPipelineStaffSkinned = VK_NULL_HANDLE;
  VkBuffer staffBoneSsbBuffer = VK_NULL_HANDLE;
  VkDeviceMemory staffBoneSsbMemory = VK_NULL_HANDLE;
  void* staffBoneSsbMapped = nullptr;
  bool staffSkinnedActive = false;
  staff_skin::Rig staffRig{};
  int staffRigBoneCount = 0;
  int staffClipMeleePunch = -1;
  int staffClipMeleeKick = -1;
  int staffClipMeleeFall = -1;
  int staffClipMeleeStand = -1;
  int staffClipShoveHair = -1;
  // Retargeted from VULKAN_GAME_STAFF_SHREK_PROXIMITY_DANCE_GLB or Shrek egg GLB; phase syncs shrekEggAnimPhase.
  int staffClipShrekProximityDance = -1;
  bool pendingStaffShoveLmb = false;
  bool pendingPlayerKick = false;
  uint32_t debugRagdollSpawnSeq = 1u;
  float playerKickAnimRemain = 0.f;
  bool dropKickActive = false;
  float dropKickTimer = 0.f;
  glm::vec2 dropKickDir{0.f};
  bool dropKickHitApplied = false;
  float crosshairShoveAnimRemain = 0.f;
  // True when any staff is in day push-aggro chase (nightPhase 2 while store lit) — matches run gait in shader.
  bool shelfEmpAnyDayPushChase = false;
  bool shelfEmpNightPursuitActive = false;
  glm::vec4 employeeBounds{0.88f, 1.39f, 0.90f, 0.24f};
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
  staff_skin::Rig shrekEggRig{};
  VkBuffer shrekEggVertexBuffer = VK_NULL_HANDLE;
  VkDeviceMemory shrekEggVertexBufferMemory = VK_NULL_HANDLE;
  uint32_t shrekEggVertexCount = 0;
  bool shrekEggAssetLoaded = false;
  bool shrekEggActive = false;
  glm::vec3 shrekEggPos{0.f};
  float shrekEggYaw = 0.f;
  float shrekEggAnimPhase = 0.f;
  float shrekEggLookAwayAccum = 0.f;
  bool shrekEggLookAwayPrimed = false;
  int shrekEggLookAwayStrikes = 0;
  int shrekEggRegionX = INT_MAX;
  int shrekEggRegionZ = INT_MAX;
  bool shrekEggDidAutoSpawnOnce = false;
  VkImage shrekEggDiffuseImage = VK_NULL_HANDLE;
  VkDeviceMemory shrekEggDiffuseMemory = VK_NULL_HANDLE;
  VkImageView shrekEggDiffuseView = VK_NULL_HANDLE;
  VkSampler shrekEggDiffuseSampler = VK_NULL_HANDLE;
  bool shrekEggDiffuseLoaded = false;
  int shrekEggAnimClipIndex = 0;
#endif

  std::vector<VkBuffer> uniformBuffers;
  std::vector<VkDeviceMemory> uniformBuffersMemory;
  std::vector<void*> uniformBuffersMapped;

  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> descriptorSets;

  bool framebufferResized = false;

  // Camera / input / physics
  glm::vec3 camPos{0, kGroundY + kEyeHeight, 6};
  float yaw = -glm::pi<float>() * 0.5f;
  float pitch = 0;
  bool showControlsOverlay = false;
  float newGameControlsPopupDelay = -1.f;
  bool mouseGrab = true;
  // Orbit camera for inspecting player animations; most gameplay rays stay first-person (mantle uses render
  // view in 3rd person so grabs match the on-screen camera).
  bool thirdPersonTestMode = false;
  float thirdPersonCamDist = 3.35f;
  float lastDrawFrameDt = 1.f / 60.f;
  float fpAvatarYawSmooth = 0.f;
  float avatarHorizSpeedSmoothed = 0.f;
  // Distance-driven phase (seconds of clip timeline) for idle/walk/sprint/crouch — synced to horizDist.
  double avatarLocoPhaseSec = 0.0;
  // Smooth grounded for locomotion clip selection (reduces sprint/walk/idle flicker on support edges).
  float avatarLocoGroundedSmoothed = 1.f;
  float velY = 0;
  glm::vec2 horizVel{0.0f};
  float bobPhase = 0;
  float bobOffsetY = 0;
  float bobSideOffset = 0;
  float walkPitchOsc = 0;
  float swayPitch = 0;
  float swayRoll = 0;
  float idlePhase = 0;
  float idlePitch = 0;
  float idleRoll = 0;
  float idleBobY = 0;
  float idleSide = 0;
  float randomSwayPhase = 0;
  float randomSwayPitch = 0;
  float randomSwayRoll = 0;
  float randomSwayBobY = 0;
  float randomSwaySide = 0;
  float runSideSway = 0;
  float runAnimBlend = 0;
  float parkourPs1PresentMix = kPs1ParkourBaselineMix;
  bool playerAirWalkSmallGap = false;
  // Per-frame: min(bay-stride, move-dir) drop; used to gate walk-off fall anims (incl. cross-aisle steps).
  float playerWalkOffWalkableGapDropCached = 1e30f;
  float idleAnimBlend = 1;
  float eyeHeight = kEyeHeight;
  int fpHeadBoneIdx = -1;
  glm::vec3 fpHeadBindPos{0.f};
  glm::vec3 fpHeadAnimOffset{0.f};
  float viewStepSmoothY = 0.f;
  bool viewStepSmoothYInited = false;
  int fpNeckBoneIdx = -1;
  glm::vec3 fpNeckBindPos{0.f};
  std::vector<int> fpHeadTiltBoneIndices;
  bool slideActive = false;
  float slideTimer = 0;
  float slideCooldownTimer = 0;
  glm::vec2 slideDir{1.0f, 0.0f};
  // When Meshy slide clips load: one slide = one full clip; elapsed drives anim phase and end time.
  int slideAnimClip = -1;
  float slideAnimDurSec = 0.f;
  float slideAnimElapsed = 0.f;
  float slideStartSpeed = 0.f;
  bool slideClearClipNextFrame = false;
  float landingPitchOfs = 0;
  float groundEase = 1.f;
  float coyoteTime = 0;
  float jumpBuffer = 0;
  float playerJumpSquatCharge = 0.f;
  bool playerJumpSquatCharging = false;
  float playerDepthJumpWindowRemain = 0.f;
  bool spaceWasDown = false;
  float playerJumpMinIntervalRem = 0.f;
  float playerJumpRepeatWindowRem = 0.f;
  // One slide trigger per physical C keydown (avoids double-edge from merged keyboard state).
  bool pendingSlideCrouchEdge = false;
  bool wasGrounded = true;
  bool playerLastGroundedOnShelfDeck = false;
  float playerHealth = kPlayerHealthMax;
  float playerHunger = kPlayerHungerMax;
  bool playerDeathActive = false;
  bool playerDeathPlayingFallClip = false;
  double playerDeathAnimTime = 0.0;
  float playerDeathHoldRemain = 0.f;
  int playerDeathClipIndex = -1;
  float playerDeathClipFracEnd = kPlayerDeathFallClipPortion;
  bool playerDeathShowMenu = false;
  bool showPauseMenu = false;
  bool showInventoryMenu = false;
  int inventoryScrollRow = 0;
  uint32_t inventoryRevision = 1;
  std::vector<std::string> inventoryItems;
  std::unordered_map<uint64_t, uint8_t> deliPizzaSlicesBySlot;
  std::unordered_map<uint64_t, float> deliPizzaReplenishTimerBySlot;
  std::unordered_map<uint64_t, uint8_t> deliMeatballsBySlot;
  std::unordered_map<uint64_t, float> deliMeatballReplenishTimerBySlot;
  float autoSaveAccumSec = 0.f;
  bool playerDanceEmoteActive = false;
  float playerDanceEmoteStopGraceRemain = 0.f;
  float playerAirFeetPeakY = kGroundY;
  float playerFallLastGroundedSupportY = kGroundY;
  // Set on first airborne frame; 1 = second shelf — debug / backup for skip below.
  int playerFallTakeoffDamageTier = -1;
  // Max feet Y this fall chain (aisle → shelf skim → floor); cleared on store-floor landing / mantle / respawn.
  float playerFallChainMaxFeetY = std::numeric_limits<float>::quiet_NaN();
  bool playerTrackingAirFall = false;
  float playerFallDamageChainImmuneRemain = 0.f;
  float playerStaffBodySlamCooldownRem = 0.f;
  float playerScreenDamagePulse = 0.f;
  float playerStaffMeleeInvulnRem = 0.f;
  float playerMercyHealDelayRemain = 0.f;
  bool playerInMercyHealthZone = false;
  // Written in update(); drawFrame reuses for shadow UBO (same tick as final support solve).
  float lastShadowGroundUnderY = kGroundY;
  float ledgeClimbT = -1.f;
  float ledgeClimbVisPhase = 0.f;
  glm::vec3 ledgeClimbStartCam{0.f};
  glm::vec3 ledgeClimbEndCam{0.f};
  glm::vec2 ledgeClimbExitHoriz{0.f};
  glm::vec2 ledgeClimbApproachVel{0.f};
  bool ladderClimbActive = false;
  float ladderClimbT = 0.f;
  glm::vec3 ladderClimbStartCam{0.f};
  glm::vec3 ladderClimbEndCam{0.f};
  glm::vec2 ladderClimbExitHoriz{0.f};
  glm::vec2 ladderClimbApproachVel{0.f};
  bool ledgeHangActive = false;
  float ledgeHangGrabTimer = 0.f;
  glm::vec3 ledgeHangStartCam{0.f};
  glm::vec3 ledgeHangCamPos{0.f};
  glm::vec3 ledgeHangEndCam{0.f};
  glm::vec2 ledgeHangExitHoriz{0.f};
  glm::vec2 ledgeHangApproachVel{0.f};
  float ledgeHangShimmyPhase = 0.f;
  int ledgeHangShimmyDir = 0;
  float ledgeHangSettledFeetY = 0.f;
  bool ledgeHangFeetYCaptured = false;
  AABB ledgeHangDeckAABB{};
  float ledgeHangTargetYaw = 0.f;
  float ledgeHangStartYaw = 0.f;
  int lastTerrainChunkX = INT_MAX;
  int lastTerrainChunkZ = INT_MAX;
  void* groundMapped = nullptr;
  float footstepDistAccum = 0.f;
  float staffSimTime = 0.f;
  // Local third-person avatar clip indices (same rig as staff; set when GLBs load).
  int avClipIdle = 0;
  int avClipWalk = 1;
  int avClipSprint = 2;
  int avClipSlideRight = -1;
  int avClipCrouchLeft = -1;
  int avClipCrouchFwd = -1;
  int avClipCrouchBack = -1;
  int avClipCrouchRight = -1;
  int avClipSlideLight = -1;
  int avClipStepPush = -1;
  int avClipCrouchIdleBow = -1;
  int avClipLedgeClimb = -1;
  int avClipLedgeGrab = -1;
  int avClipShimmyLeft = -1;
  int avClipShimmyRight = -1;
  int avClipJump = -1;
  int avClipJumpRun = -1;
  int avClipLand = -1;
  float playerPushAnimRemain = 0.f;
  float playerJumpAnimRemain = 0.f;
  // True from space-jump until landing footstep (even if jump clip finished early in air).
  bool playerJumpArchActive = false;
  // 2·vy/g style hang for current arch; drives jump clip playback rate (charged jump / depth jump).
  float playerJumpAirTimeTargetSec = 0.f;
  float playerJumpPostLandRemain = 0.f;
  // When >0, post-land scrubs over this wall duration; 0 = hold last jump frame (legacy).
  float playerJumpPostLandDurationInit = 0.f;
  // If true, post-land scrubs the jump clip from kJumpClipLedgeFirstHalfFrac → end (touchdown outro).
  bool playerJumpPostLandSecondHalfScrub = false;
  // Sprint jump: regular jump through apex, then Jump_Run clip until landing.
  bool playerJumpRunTailActive = false;
  int playerJumpPostLandClipIndex = -1;
  // Non-jump ledge step-off: countdown + feet lock Y; then jump remain starts at (1-frac)*clip length.
  float playerPreFallAnimRemain = 0.f;
  float playerPreFallFeetLockY = 0.f;
  bool playerPreFallUseRunClip = false;
  float playerFallAnimTime = 0.f;
  // After walk-off pre-fall (first half of jump clip), wait until near landing to play second half.
  bool playerJumpAwaitPreLandSecondHalf = false;
  // Second half is playing (walk-off); use lead-window playback rate instead of full-jump air-time warp.
  bool playerJumpLedgeSecondHalfAir = false;
  // Ground jump over a crate/box ahead: jump clip scrubs a bit faster for the short arc; cleared on land.
  bool playerVaultCrateJumpActive = false;
  bool wallRunActive = false;
  float wallRunTimer = 0.f;
  float wallRunCooldownTimer = 0.f;
  glm::vec3 wallRunNormal{0.f};
  glm::vec3 wallRunDir{0.f};
  float wallRunCameraTilt = 0.f;
  int wallRunSide = 0; // -1 left, +1 right
  bool wallClimbActive = false;
  float wallClimbTimer = 0.f;
  float wallClimbCooldownTimer = 0.f;
  glm::vec3 wallClimbNormal{0.f};
  float wallClimbStartY = 0.f;
  float wallJumpCooldownTimer = 0.f;
  bool speedVaultActive = false;
  float speedVaultTimer = 0.f;
  glm::vec2 speedVaultDir{0.f};
  float speedVaultStartY = 0.f;
  float speedVaultTargetY = 0.f;
  int playerAvatarClip = 0;
  bool playerWalkAnimReverse = false;
  bool playerWalkReverseHold = false;
  int playerAvatarBlendFromClip = 0;
  float playerAvatarClipBlend = 1.f;
  // Wayland / fullscreen often miss keys in SDL_GetKeyboardState alone; merge with KEYDOWN/KEYUP.
  std::array<bool, SDL_NUM_SCANCODES> scancodeDown{};
  std::vector<ShelfEmployeeNpc> shelfEmpPool;
  std::vector<uint32_t> shelfEmpActiveSlots;
  std::vector<uint32_t> shelfEmpFreeSlots;
  std::unordered_map<uint64_t, uint32_t> shelfEmpKeyToSlot;
  std::vector<ShelfEmployeeNpc*> shelfSepEmpScratch;
  std::vector<float> shelfSepFootRScratch;

  bool staffNpcAABBTouchesWorld(const AABB& s) const {
    const float ecx = 0.5f * (s.min.x + s.max.x);
    const float ecz = 0.5f * (s.min.z + s.max.z);
    const float scaleFromAabb = std::max(
        {(s.max.x - s.min.x) / (2.f * kStaffHitHalfW + 1e-6f),
         (s.max.y - s.min.y) / (kEmployeeVisualHeight + 1e-6f),
         (s.max.z - s.min.z) / (2.f * kStaffHitHalfD + 1e-6f)});
    const float shelfReachSq = kShelfStaffXZReachSq * scaleFromAabb * scaleFromAabb;
    const int gcx = static_cast<int>(std::floor(ecx / kPillarSpacing));
    const int gcz = static_cast<int>(std::floor(ecz / kPillarSpacing));
    for (int dx = -kPillarGridRadius; dx <= kPillarGridRadius; ++dx) {
      for (int dz = -kPillarGridRadius; dz <= kPillarGridRadius; ++dz) {
        const float px = static_cast<float>(gcx + dx) * kPillarSpacing;
        const float pz = static_cast<float>(gcz + dz) * kPillarSpacing;
        if (aabbOverlap(s, pillarCollisionAABB(px, pz)))
          return true;
      }
    }
    constexpr float kStaffShelfGridRangeM = 22.f;
    int waMin, waMax, wlMin, wlMax;
    shelfGridWindowForRange(ecx, ecz, kStaffShelfGridRangeM, waMin, waMax, wlMin, wlMax);
    for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
      const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
      const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
      const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
      for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
        const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
        for (int side = 0; side < 2; ++side) {
          if (!shelfSlotOccupied(worldAisle, worldAlong, side))
            continue;
          const float cx = side ? cxRight : cxLeft;
          const float yawDeg = side ? -90.0f : 90.0f;
          const float dxw = ecx - cx;
          const float dzw = ecz - cz;
          if (dxw * dxw + dzw * dzw > shelfReachSq)
            continue;
          const glm::vec3 shelfPos{cx, kGroundY, cz};
          const float shelfYawRad = glm::radians(yawDeg);
          const float hw = kShelfMeshHalfW;
          const float hd = kShelfMeshHalfD;
          const float ph = kShelfPostGauge;
          const float H = kShelfMeshHeight;
          const float shelfT = kShelfDeckThickness;
          const int numShelves = kShelfDeckCount;
          constexpr float yBase = 0.12f;
          const float yStep = kShelfGapBetweenLevels + shelfT;
          auto overlapPost = [&](const glm::vec3& mn, const glm::vec3& mx) {
            const AABB box = shelfLocalBoxWorldAABB(shelfPos, shelfYawRad, mn, mx);
            return aabbOverlap(s, box);
          };
          if (overlapPost({-hw, 0.f, -hd}, {-hw + 2.f * ph, H, -hd + 2.f * ph}))
            return true;
          if (overlapPost({hw - 2.f * ph, 0.f, -hd}, {hw, H, -hd + 2.f * ph}))
            return true;
          if (overlapPost({-hw, 0.f, hd - 2.f * ph}, {-hw + 2.f * ph, H, hd}))
            return true;
          if (overlapPost({hw - 2.f * ph, 0.f, hd - 2.f * ph}, {hw, H, hd}))
            return true;
          for (int si = 0; si < numShelves; ++si) {
            const float y0 = yBase + static_cast<float>(si) * yStep;
            const float y1 = y0 + shelfT;
            const AABB deck =
                shelfLocalBoxWorldAABB(shelfPos, shelfYawRad,
                                       {-hw + kShelfDeckInset, y0, -hd + kShelfDeckInset},
                                       {hw - kShelfDeckInset, y1, hd - kShelfDeckInset});
            if (aabbOverlap(s, deck)) {
              constexpr float kStandSlop = 0.14f;
              if (s.min.y < deck.max.y - kStandSlop)
                return true;
            }
          }
          float lx, lz, yDeck, chx, chy, chz;
          const bool hasCrate =
              shelfCrateLocalLayout(worldAisle, worldAlong, side, lx, lz, yDeck, chx, chy, chz);
          if (hasCrate) {
            const AABB crate = shelfLocalBoxWorldAABB(
                shelfPos, shelfYawRad, {lx - chx, yDeck, lz - chz},
                {lx + chx, yDeck + 2.f * chy, lz + chz});
            if (aabbOverlap(s, crate)) {
              constexpr float kStandSlop = 0.14f;
              if (s.min.y < crate.max.y - kStandSlop)
                return true;
            }
            const float topY = yDeck + 2.f * chy;
            const AABB crateTop = shelfLocalBoxWorldAABB(
                shelfPos, shelfYawRad,
                {lx - chx + 0.04f, topY - 0.06f, lz - chz + 0.04f},
                {lx + chx - 0.04f, topY, lz + chz - 0.04f});
            if (aabbOverlap(s, crateTop)) {
              constexpr float kStandSlop = 0.14f;
              if (s.min.y < crateTop.max.y - kStandSlop)
                return true;
            }
          }
          bool palletOverlap = false;
          forEachShelfPalletStackBoxLocal(worldAisle, worldAlong, side, hasCrate,
                                          [&](float mnx, float y0, float mnz, float mxx, float y1, float mxz) {
                                            if (palletOverlap)
                                              return;
                                            const AABB pallet = shelfLocalBoxWorldAABB(
                                                shelfPos, shelfYawRad, {mnx, y0, mnz}, {mxx, y1, mxz});
                                            if (aabbOverlap(s, pallet)) {
                                              constexpr float kStandSlop = 0.14f;
                                              if (s.min.y < pallet.max.y - kStandSlop) {
                                                palletOverlap = true;
                                                return;
                                              }
                                            }
                                            const AABB palletTop = shelfLocalBoxWorldAABB(
                                                shelfPos, shelfYawRad,
                                                {mnx + 0.035f, y1 - 0.05f, mnz + 0.035f},
                                                {mxx - 0.035f, y1, mxz - 0.035f});
                                            if (aabbOverlap(s, palletTop)) {
                                              constexpr float kStandSlop = 0.14f;
                                              if (s.min.y < palletTop.max.y - kStandSlop)
                                                palletOverlap = true;
                                            }
                                          });
          if (palletOverlap)
            return true;
          float llx, llz, leanR;
          glm::vec3 ladderScl;
          if (shelfLadderLocalLayout(worldAisle, worldAlong, side, llx, llz, leanR, ladderScl)) {
            const glm::mat4 ladderM = shelfLadderWorldModel(cx, cz, yawDeg, llx, llz, leanR, ladderScl);
            const AABB ladderBox = shelfLadderCollisionAABB(ladderM);
            if (aabbOverlap(s, ladderBox)) {
              constexpr float kStandSlop = 0.14f;
              if (s.min.y < ladderBox.max.y - kStandSlop)
                return true;
            }
          }
        }
      }
    }
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
    if (shrekEggAssetLoaded && shrekEggActive && shrekEggVertexCount > 0u && staffSkinnedActive) {
      constexpr float kShrekEggTargetHeightM = 1.78f;
      const float shrekScaleY = kShrekEggTargetHeightM / std::max(1e-4f, kEmployeeVisualHeight);
      const glm::vec3 shrekBodyScale(1.f, shrekScaleY, 1.f);
      const AABB shrekBox = staffNpcWorldHitbox(shrekEggPos.x, shrekEggPos.z, shrekEggYaw, shrekEggPos.y,
                                                shrekBodyScale);
      if (aabbOverlap(s, shrekBox))
        return true;
    }
#endif
    return false;
  }

  bool staffNpcNavClearAtXZYaw(float wx, float wz, float yawRad, float feetWorldY,
                               const glm::vec3& bodyScale = glm::vec3(1.f)) const {
    return !staffNpcAABBTouchesWorld(staffNpcWorldHitbox(wx, wz, yawRad, feetWorldY, bodyScale));
  }

  glm::vec2 staffNavAdjustedDesiredXZ(const ShelfEmployeeNpc& e, const glm::vec2& desiredW,
                                      float distToPlayer, float lookaheadMul = 1.f,
                                      bool wideObstacleFan = false) const {
    const float ld = glm::length(desiredW);
    if (ld < 1e-6f)
      return desiredW;
    if (distToPlayer > kStaffNavLookaheadSkipDist)
      return desiredW;
    const glm::vec2 g = desiredW / ld;
    const float la = kStaffNavLookahead * lookaheadMul;
    const auto probeClear = [&](const glm::vec2& dirU) -> bool {
      const float lu = glm::length(dirU);
      if (lu < 1e-6f)
        return false;
      const glm::vec2 du = dirU / lu;
      const float yh = std::atan2(du.x, du.y);
      const glm::vec2 p = e.posXZ + du * la;
      const float probeFeet = e.feetWorldY + kStaffTerrainStepProbe;
      return staffNpcNavClearAtXZYaw(p.x, p.y, yh, probeFeet, e.bodyScale);
    };
    if (probeClear(g))
      return desiredW;
    float bestDot = -2.f;
    glm::vec2 bestG = g;
    constexpr float kNavProbeLodFarDist = 42.f;
    const bool farLod = distToPlayer > kNavProbeLodFarDist;
    const std::initializer_list<float> degList =
        wideObstacleFan ? (farLod ? std::initializer_list<float>{50.f, -50.f, 100.f, -100.f, 180.f}
                                  : std::initializer_list<float>{35.f,  -35.f,  50.f,  -50.f,  72.f,  -72.f,
                                                                 100.f, -100.f, 125.f, -125.f, 180.f})
                        : (farLod ? std::initializer_list<float>{55.f, -55.f, 180.f}
                                  : std::initializer_list<float>{35.f, -35.f, 72.f, -72.f, 180.f});
    for (float dg : degList) {
      const float rad = glm::radians(dg);
      const float c = std::cos(rad), sn = std::sin(rad);
      const glm::vec2 d(g.x * c - g.y * sn, g.x * sn + g.y * c);
      if (!probeClear(d))
        continue;
      const float dotg = glm::dot(d, g);
      if (dotg > bestDot) {
        bestDot = dotg;
        bestG = d;
      }
    }
    if (bestDot < -1.5f)
      return desiredW;
    return bestG * ld;
  }

  void shelfEmpEnsureWanderTargetClear(ShelfEmployeeNpc& e, uint64_t key, bool localBay,
                                       const glm::vec2& anchorXZ) {
    for (int attempt = 0; attempt < kStaffWanderClearMaxAttempts; ++attempt) {
      const float tgtFeet = terrainSupportY(e.wanderTargetXZ.x, e.wanderTargetXZ.y,
                                            kGroundY + kStaffTerrainStepProbe);
      if (staffNpcNavClearAtXZYaw(e.wanderTargetXZ.x, e.wanderTargetXZ.y, e.yaw, tgtFeet, e.bodyScale))
        return;
      if (localBay)
        shelfEmpPickWanderLocalBay(e, key);
      else
        shelfEmpPickWanderStoreWide(e, key, anchorXZ);
    }
  }

  void shelfEmpStepWanderTowardTarget(ShelfEmployeeNpc& e, float dt, uint64_t key, const glm::vec2& anchorXZ,
                                      bool patrolLocalBay,
                                      float playerFeetHint = std::numeric_limits<float>::quiet_NaN()) {
    const float distPlayer = glm::length(anchorXZ - e.posXZ);
    glm::vec2 toT = e.wanderTargetXZ - e.posXZ;
    float dT = glm::length(toT);
    if (dT < kShelfEmpWanderReachEps) {
      e.velXZ *= std::exp(-5.f * dt);
      if (patrolLocalBay)
        shelfEmpPickWanderLocalBay(e, key);
      else
        shelfEmpPickWanderStoreWide(e, key, anchorXZ);
      shelfEmpEnsureWanderTargetClear(e, key, patrolLocalBay, anchorXZ);
      toT = e.wanderTargetXZ - e.posXZ;
      dT = glm::length(toT);
    }
    if (dT > 1e-5f) {
      glm::vec2 steer = toT;
      if (glm::dot(steer, steer) > 1e-8f)
        steer = staffNavAdjustedDesiredXZ(e, steer, distPlayer);
      const float sy = staffNpcFootSupportY(e, playerFeetHint);
      const bool airLike = !staffNpcIsGroundedLikePlayer(e, sy);
      float acW = kStaffSteerAccelWalk;
      if (airLike)
        acW *= kAirAccel / kWalkAccel;
      staffIntegrateSteering(e, dt, steer, kShelfEmpWalkSpeed * staffSpeedScaleForBody(e), acW);
      if (airLike)
        e.velXZ *= std::exp(-kAirDrag * dt);
    } else {
      e.velXZ *= std::exp(-6.f * dt);
      e.lastHorizSpeed = glm::length(e.velXZ);
    }
  }

  // Prefer idle jump for mantel; run-jump only if idle clip missing.
  int staffMantelJumpClipIndex() const {
    if (avClipJump >= 0 && static_cast<size_t>(avClipJump) < staffRig.clips.size())
      return avClipJump;
    if (avClipJumpRun >= 0 && static_cast<size_t>(avClipJumpRun) < staffRig.clips.size())
      return avClipJumpRun;
    return -1;
  }

  // Mantel: optional jump clip into ledge pull-up; phases share the same wall-clock as feet Y arc.
  void staffNpcMantelAnimPhase(const ShelfEmployeeNpc& npc, float uLinear, int& clipIdx, double& ph,
                               bool& loopClip) const {
    loopClip = false;
    const bool runnerMantel = npc.staffMantelRunnerChase != 0 && npc.nightPhase == 2;
    const float uJ =
        runnerMantel ? kStaffChaseRunnerMantelJumpUFrac : kStaffChaseMantelJumpAnimUFrac;
    int jClip = staffMantelJumpClipIndex();
    if (runnerMantel && avClipJumpRun >= 0 && static_cast<size_t>(avClipJumpRun) < staffRig.clips.size())
      jClip = avClipJumpRun;
    const bool haveJump = jClip >= 0;
    const bool haveLedge = avClipLedgeClimb >= 0 &&
                          static_cast<size_t>(avClipLedgeClimb) < staffRig.clips.size();

    if (!haveLedge && haveJump) {
      clipIdx = jClip;
      const double durJ = staff_skin::clipDuration(staffRig, jClip);
      ph = std::clamp(static_cast<double>(uLinear) * durJ, 0.0, std::max(1e-6, durJ - 1e-6));
      return;
    }
    if (haveLedge && haveJump && uLinear < uJ - 1e-5f) {
      clipIdx = jClip;
      const double durJ = staff_skin::clipDuration(staffRig, jClip);
      const float t = glm::clamp(uLinear / std::max(1e-4f, uJ), 0.f, 1.f);
      ph = std::clamp(static_cast<double>(t) * durJ, 0.0, std::max(1e-6, durJ - 1e-6));
      return;
    }
    if (haveLedge) {
      clipIdx = avClipLedgeClimb;
      const double durLc = staff_skin::clipDuration(staffRig, avClipLedgeClimb);
      const double halfDur = durLc * static_cast<double>(kLedgeClimbAnimClipFrac);
      const double span =
          npc.staffMantelAnimPhaseSpanSec > 1e-4f
              ? std::min(halfDur, static_cast<double>(npc.staffMantelAnimPhaseSpanSec))
              : halfDur;
      const float uPull = (haveJump && uJ < 0.999f)
          ? glm::clamp((uLinear - uJ) / std::max(1e-4f, 1.f - uJ), 0.f, 1.f)
          : uLinear;
      // Match player resolvePlayerAvatarPhase: ladder phase advances linearly over first half of clip.
      ph = std::clamp(static_cast<double>(uPull) * span, 0.0, std::max(1e-6, span - 1e-6));
      return;
    }
    clipIdx = 0;
    ph = 0.0;
    loopClip = true;
  }

  void staffNpcLocomotionClip(const ShelfEmployeeNpc& npc, uint64_t key, int& outClip, double& outPhase,
                              bool& outLoop) const {
    outLoop = true;
    outClip = 0;
    outPhase = 0.0;
    if (npc.chaseLedgeClimbRem > 0.f) {
      const float td = npc.chaseLedgeClimbTotalDur > 1e-4f ? npc.chaseLedgeClimbTotalDur
                                                           : kStaffChaseLedgeClimbDurationS;
      const float u = glm::clamp(1.f - npc.chaseLedgeClimbRem / td, 0.f, 1.f);
      staffNpcMantelAnimPhase(npc, u, outClip, outPhase, outLoop);
      return;
    }
    if (npc.staffAirLandClip >= 0 && npc.staffAirLandRemain > 1e-4f && npc.meleeState < 2) {
      outClip = npc.staffAirLandClip;
      const double durJ = staff_skin::clipDuration(staffRig, npc.staffAirLandClip);
      outPhase = std::clamp(durJ - 1e-4, 0.0, std::max(1e-6, durJ - 1e-6));
      outLoop = false;
      return;
    }
    if (npc.staffAirFallClip >= 0 && npc.staffAirLocoRemain > 1e-4f && npc.meleeState < 2) {
      outClip = npc.staffAirFallClip;
      const double durJ = staff_skin::clipDuration(staffRig, npc.staffAirFallClip);
      outPhase = std::clamp(static_cast<double>(durJ) - static_cast<double>(npc.staffAirLocoRemain), 0.0,
                            std::max(1e-6, durJ - 1e-6));
      outLoop = false;
      return;
    }
    const int wa = static_cast<int>(static_cast<uint32_t>(key >> 32));
    const int wl = static_cast<int>(static_cast<uint32_t>(key & 0xffffffffull));
    const uint32_t ehAnim = scp3008ShelfHash(wa, wl, 0xE3910EE5u);
    const bool storeLit = audioAreStoreFluorescentsOn();
    // Lit store: walk wander uses clip 1; day push-aggro chase still uses nightPhase 2 → run like blackout chase.
    if (storeLit && npc.nightPhase != 2) {
      if (npc.lastHorizSpeed > 0.04f && staffRig.clips.size() > 1)
        outClip = 1;
    } else if (npc.nightPhase == 2) {
      if (npc.lastHorizSpeed > 0.04f && staffRig.clips.size() > 2)
        outClip = 2;
      else if (npc.lastHorizSpeed > 0.04f && staffRig.clips.size() > 1)
        outClip = 1;
    } else if (npc.nightPhase == 0 || npc.nightPhase == 3) {
      // Phase 3: walk toward last known spot after losing sight — same forward walk as patrol.
      if (npc.lastHorizSpeed > 0.04f && staffRig.clips.size() > 1)
        outClip = 1;
    }
    const double dur = staff_skin::clipDuration(staffRig, outClip);
    outPhase = std::fmod(static_cast<double>(staffSimTime) *
                             (outClip >= 2 ? 1.02 : outClip == 1 ? 0.92 : 0.78) *
                             static_cast<double>(kStaffNpcAnimPlaybackScale) +
                         static_cast<double>(ehAnim) * 0.000173,
                         dur);
    if (outPhase < 0)
      outPhase += dur;
  }

  void staffNpcResolveDrawAnim(const ShelfEmployeeNpc& npc, uint64_t key, bool storeLitForStaffAnim,
                               int& clipIdx, double& ph, bool& loopClip) const {
    clipIdx = 0;
    ph = 0.0;
    loopClip = true;
    if (npc.meleeState == 4 && staffClipShoveHair >= 0) {
      clipIdx = staffClipShoveHair;
      ph = npc.meleePhaseSec;
      loopClip = false;
    } else if (npc.meleeState == 2 && staffClipMeleeFall >= 0) {
      clipIdx = staffClipMeleeFall;
      ph = npc.meleePhaseSec;
      loopClip = false;
    } else if (npc.meleeState == 3 && staffClipMeleeStand >= 0) {
      clipIdx = staffClipMeleeStand;
      ph = npc.meleePhaseSec;
      loopClip = false;
    } else if (npc.meleeState == 1 && (staffClipMeleePunch >= 0 || staffClipMeleeKick >= 0)) {
      clipIdx = (npc.meleeAttackPick == 1 && staffClipMeleeKick >= 0) ? staffClipMeleeKick
                                                                       : staffClipMeleePunch;
      ph = npc.meleePhaseSec;
      loopClip = true;
    } else if (npc.chaseLedgeClimbRem > 0.f) {
      const float td = npc.chaseLedgeClimbTotalDur > 1e-4f ? npc.chaseLedgeClimbTotalDur
                                                           : kStaffChaseLedgeClimbDurationS;
      const float u = glm::clamp(1.f - npc.chaseLedgeClimbRem / td, 0.f, 1.f);
      staffNpcMantelAnimPhase(npc, u, clipIdx, ph, loopClip);
    } else {
      if (npc.staffAirLandClip >= 0 && npc.staffAirLandRemain > 1e-4f && npc.meleeState < 2) {
        clipIdx = npc.staffAirLandClip;
        const double durJ = staff_skin::clipDuration(staffRig, npc.staffAirLandClip);
        ph = std::clamp(durJ - 1e-4, 0.0, std::max(1e-6, durJ - 1e-6));
        loopClip = false;
      } else if (npc.staffAirFallClip >= 0 && npc.staffAirLocoRemain > 1e-4f && npc.meleeState < 2) {
        clipIdx = npc.staffAirFallClip;
        const double durJ = staff_skin::clipDuration(staffRig, npc.staffAirFallClip);
        ph = std::clamp(static_cast<double>(durJ) - static_cast<double>(npc.staffAirLocoRemain), 0.0,
                        std::max(1e-6, durJ - 1e-6));
        loopClip = false;
      } else {
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
        const bool shrekNearDance = staffNpcShouldHoldShrekDancePose(npc, camPos.y - eyeHeight) &&
                                  npc.chaseLedgeClimbRem <= 0.f;
        if (shrekNearDance) {
          clipIdx = staffClipShrekProximityDance;
          const double dur = staff_skin::clipDuration(staffRig, clipIdx);
          ph = std::fmod(static_cast<double>(shrekEggAnimPhase), std::max(dur, 1e-6));
          if (ph < 0)
            ph += dur;
          loopClip = true;
        }
        if (!shrekNearDance) {
#else
        {
#endif
          if (storeLitForStaffAnim && npc.nightPhase != 2) {
            if (npc.lastHorizSpeed > 0.04f && staffRig.clips.size() > 1)
              clipIdx = 1;
          } else if (npc.nightPhase == 2) {
            if (npc.lastHorizSpeed > 0.04f && staffRig.clips.size() > 2)
              clipIdx = 2;
            else if (npc.lastHorizSpeed > 0.04f && staffRig.clips.size() > 1)
              clipIdx = 1;
          } else if (npc.nightPhase == 0 || npc.nightPhase == 3) {
            if (npc.lastHorizSpeed > 0.04f && staffRig.clips.size() > 1)
              clipIdx = 1;
          }
          const int wa = static_cast<int>(static_cast<uint32_t>(key >> 32));
          const int wl = static_cast<int>(static_cast<uint32_t>(key & 0xffffffffull));
          const uint32_t ehAnim = scp3008ShelfHash(wa, wl, 0xE3910EE5u);
          const double dur = staff_skin::clipDuration(staffRig, clipIdx);
          ph = std::fmod(static_cast<double>(staffSimTime) *
                             (clipIdx >= 2 ? 1.02 : clipIdx == 1 ? 0.92 : 0.78) *
                             static_cast<double>(kStaffNpcAnimPlaybackScale) +
                         static_cast<double>(ehAnim) * 0.000173,
                         dur);
          if (ph < 0)
            ph += dur;
        }
      }
    }
  }

  float staffMeleeDrawFeetSinkY(const ShelfEmployeeNpc& npc) const {
    const float sy = npc.bodyScale.y;
    // Match end-of-fall sole height; dead NPCs no longer advance meleePhaseSec with the clip.
    if (npc.staffDead)
      return kStaffDeadRagdollBindFeetSink * sy + kStaffMeleeFallFeetSinkWorldBias * 0.62f;
    if (npc.meleeState == 4 && staffClipShoveHair >= 0)
      return kStaffMeleeHairFeetSink * sy + kStaffMeleeHairFeetSinkWorldBias;
    if (npc.meleeState == 2 && staffClipMeleeFall >= 0) {
      const double dF = staff_skin::clipDuration(staffRig, staffClipMeleeFall);
      const float u =
          dF > 1e-6 ? glm::clamp(static_cast<float>(npc.meleePhaseSec / dF), 0.f, 1.f) : 0.f;
      return glm::mix(kStaffMeleeFallFeetSinkMax, kStaffMeleeFallFeetSinkEnd, u) * sy +
             kStaffMeleeFallFeetSinkWorldBias;
    }
    if (npc.meleeState == 3 && staffClipMeleeStand >= 0) {
      const double dU = staff_skin::clipDuration(staffRig, staffClipMeleeStand);
      const float u =
          dU > 1e-6 ? glm::clamp(static_cast<float>(npc.meleePhaseSec / dU), 0.f, 1.f) : 0.f;
      return glm::mix(kStaffMeleeStandFeetSinkStart, 0.f, u) * sy +
             glm::mix(kStaffMeleeFallFeetSinkWorldBias, 0.f, u);
    }
    if (npc.staffAirLandRemain > 1e-4f && npc.staffAirLandClip >= 0 && npc.meleeState == 0)
      return kAvatarJumpLandFeetVisualDown;
    if (npc.staffAirLocoRemain > 1e-4f && npc.staffVelY < -0.16f && npc.meleeState == 0 &&
        npc.chaseLedgeClimbRem <= 0.f && npc.staffAirFallClip >= 0)
      return kAvatarJumpFallFeetVisualDown;
    return 0.f;
  }

  // Player visible to staff for pursuit (same geometry as night chase: wide cone, behind sense, shelf elev).
  bool staffPlayerVisibleInChaseCone(const ShelfEmployeeNpc& e, float distP, const glm::vec2& toPlayer,
                                      float playerFeetYVis) const {
    const glm::vec2 fwd(std::sin(e.yaw), std::cos(e.yaw));
    glm::vec2 toPn(0.f);
    if (distP > 1e-4f)
      toPn = toPlayer * (1.f / distP);
    const float dotF = glm::dot(toPn, fwd);
    float dotCone = dotF;
    const float vCh = glm::length(e.velXZ);
    if (vCh > kNightStaffChaseVisionVelConeMinSpeed)
      dotCone = std::max(dotF, glm::dot(toPn, e.velXZ * (1.f / vCh)));
    const float staffEyeY =
        e.feetWorldY + (kNightStaffVisionEyeY - kGroundY) * e.bodyScale.y;
    const float dyPlayerEyes = camPos.y - staffEyeY;
    const float elevAboveHoriz = std::atan2(dyPlayerEyes, std::max(distP, 1e-3f));
    const float crouchT = glm::clamp((kEyeHeight - eyeHeight) / (kEyeHeight - kCrouchEyeHeight), 0.f, 1.f);
    const float visRange = kNightStaffChaseVisionRange * glm::mix(1.f, kCrouchStealthVisionRangeMul, crouchT);
    const float visCosHalf = kNightStaffChaseVisionCosHalfFov;
    const float behindSenseM = kNightStaffChaseBehindSenseM * glm::mix(1.f, kCrouchStealthBehindSenseMul, crouchT);
    const float maxElevAboveDeg = kNightStaffChaseVisionMaxElevAboveDeg;
    const bool ledgedUpForVision =
        playerFeetYVis >= kGroundY + kNightStaffVisionPlayerLedgedFeetAboveFloorM;
    const float elevCapRad =
        (ledgedUpForVision && distP <= visRange)
            ? glm::radians(kNightStaffVisionLedgedMaxElevAboveDeg)
            : glm::radians(maxElevAboveDeg);
    const bool withinElevSight = elevAboveHoriz <= elevCapRad;
    const bool inFrontCone =
        withinElevSight && distP <= visRange && dotCone >= visCosHalf;
    const bool closeBehind =
        withinElevSight && distP <= behindSenseM && dotF < 0.12f;
    return inFrontCone || closeBehind;
  }

  // Wall-clock for one staff mantel: same player curve, but slower for readability/gameplay balance.
  // Feet still use playerLedgeGrabHeightS over normalized u.
  float staffChaseLedgePullUpDurationSec() const { return kLedgeGrabDuration * kStaffChaseLedgeDurationMul; }

  void staffNpcUpdateAirFallLoco(ShelfEmployeeNpc& e, float dt, float playerFeetHint) {
    if (e.meleeState >= 2 || e.chaseLedgeClimbRem > 0.f) {
      e.staffAirLocoRemain = 0.f;
      e.staffAirFallClip = -1;
      e.staffAirLandRemain = 0.f;
      e.staffAirLandClip = -1;
      e.staffGroundedPrev = true;
      return;
    }
    const float sy = staffNpcFootSupportY(e, playerFeetHint);
    const bool grounded =
        e.feetWorldY <= sy + kGroundedFeetAboveSupport && e.staffVelY <= 0.01f;

    if (grounded) {
      if (e.staffAirLandRemain > 1e-4f) {
        e.staffAirLandRemain = std::max(0.f, e.staffAirLandRemain - dt);
        if (e.staffAirLandRemain <= 1e-4f) {
          e.staffAirLandRemain = 0.f;
          e.staffAirLandClip = -1;
        }
        e.staffAirLocoRemain = 0.f;
        e.staffAirFallClip = -1;
        e.staffGroundedPrev = true;
        return;
      }
      if (!e.staffGroundedPrev) {
        const bool hadAirJumpAnim = e.staffAirFallClip >= 0 || e.staffAirLocoRemain > 1e-3f;
        if (hadAirJumpAnim && staffSkinnedActive && (avClipJump >= 0 || avClipJumpRun >= 0)) {
          int lc = e.staffAirFallClip;
          if (lc < 0 || static_cast<size_t>(lc) >= staffRig.clips.size())
            lc = (avClipJump >= 0) ? avClipJump : avClipJumpRun;
          e.staffAirLandClip = lc;
          const bool runLand = (avClipJumpRun >= 0 && lc == avClipJumpRun);
          e.staffAirLandRemain = runLand ? kJumpLandPoseHoldSecRunJump : kJumpLandPoseHoldSec;
        }
      }
      e.staffAirLocoRemain = 0.f;
      e.staffAirFallClip = -1;
      e.staffGroundedPrev = true;
      return;
    }

    e.staffAirLandRemain = 0.f;
    e.staffAirLandClip = -1;

    const auto staffAirFallSignificant = [&]() {
      const float clearAboveSupport = e.feetWorldY - sy;
      return clearAboveSupport > kStaffAirFallMinClearanceAboveSupportM ||
             e.staffVelY <= kStaffAirFallMinDownVelForClip;
    };
    const auto staffAirFallAssignClip = [&]() {
      const bool wantRun = e.nightPhase == 2 && e.lastHorizSpeed > 0.14f && avClipJumpRun >= 0 &&
                           static_cast<size_t>(avClipJumpRun) < staffRig.clips.size();
      if (wantRun)
        e.staffAirFallClip = avClipJumpRun;
      else if (avClipJump >= 0 && static_cast<size_t>(avClipJump) < staffRig.clips.size())
        e.staffAirFallClip = avClipJump;
      else
        e.staffAirFallClip = -1;
      if (e.staffAirFallClip >= 0)
        e.staffAirLocoRemain =
            static_cast<float>(staff_skin::clipDuration(staffRig, e.staffAirFallClip));
    };

    if (e.staffGroundedPrev) {
      if (staffAirFallSignificant())
        staffAirFallAssignClip();
      else {
        e.staffAirFallClip = -1;
        e.staffAirLocoRemain = 0.f;
      }
    } else if (!grounded && e.staffAirFallClip < 0 && e.staffAirLocoRemain <= 1e-4f &&
               staffAirFallSignificant()) {
      // First airborne frame can still see the old support height under the feet; pick up fall pose once gap opens.
      staffAirFallAssignClip();
    }
    e.staffGroundedPrev = false;
    if (e.staffAirFallClip >= 0 && e.staffAirLocoRemain > 0.f) {
      const double durJ = staff_skin::clipDuration(staffRig, e.staffAirFallClip);
      const float tPred = std::max(kJumpPredictedAirTimeSec, 0.12f);
      const float rate =
          glm::clamp(static_cast<float>(durJ) / tPred, 0.52f, 2.05f) * kJumpAnimPlaybackScale;
      e.staffAirLocoRemain = std::max(0.f, e.staffAirLocoRemain - dt * rate);
    }
  }

  // Shared chase + melee when nightPhase == 2 (night chase or day push-aggro).
  // blackoutPursuit: fluorescents off — extra speed/accel so the run reads as an active hunt (day shove uses false).
  void shelfEmployeeRunChasePhase2(ShelfEmployeeNpc& e, float dt, const glm::vec2& pXZ, float distP,
                                   const glm::vec2& toPlayer, float playerFeetY, bool blackoutPursuit,
                                   bool suppressMeleeNearShrek) {
    if (suppressMeleeNearShrek) {
      if (e.meleeState == 1) {
        e.meleeState = 0;
        e.meleeAnimBlend = 1.f;
        e.shovePlayerPushDurSec = 0.f;
        e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
      }
    } else if ((staffClipMeleePunch >= 0 || staffClipMeleeKick >= 0) && distP <= kStaffMeleePunchHoldRange) {
      if (e.meleeState != 1) {
        e.meleeAttackPick = static_cast<uint8_t>((e.meleeAttackPick + 1) % 2);
        if (e.meleeAttackPick == 1 && staffClipMeleeKick < 0)
          e.meleeAttackPick = 0;
        if (e.meleeAttackPick == 0 && staffClipMeleePunch < 0)
          e.meleeAttackPick = 1;
      }
      e.meleeState = 1;
    } else if (distP > kStaffMeleePunchReleaseRange) {
      e.meleeState = 0;
      e.meleeAnimBlend = 1.f;
      e.shovePlayerPushDurSec = 0.f;
      e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
    }
    if (e.meleeState == 1) {
      e.chaseLedgeClimbRem = -1.f;
      e.chaseLedgeClimbTotalDur = 0.f;
      e.staffMantelAnimPhaseSpanSec = 0.f;
      e.staffMantelRunnerChase = 0;
      e.velXZ *= std::exp(-14.f * dt);
      e.lastHorizSpeed = glm::length(e.velXZ);
      e.yaw = std::atan2(toPlayer.x, toPlayer.y);
    } else if (distP > 0.12f) {
      glm::vec2 aim = toPlayer;
      const float leadMinD = blackoutPursuit ? 1.08f : 1.35f;
      const float leadMaxD = blackoutPursuit ? 30.f : 24.f;
      if (distP > leadMinD && distP < leadMaxD) {
        const float leadT =
            blackoutPursuit ? glm::clamp(0.26f + distP * 0.041f, 0.19f, 1.18f)
                            : glm::clamp(0.2f + distP * 0.03f, 0.16f, 0.95f);
        const glm::vec2 lead = pXZ + horizVel * leadT - e.posXZ;
        const float dL = glm::length(lead);
        if (dL > (blackoutPursuit ? 0.16f : 0.22f))
          aim = lead;
      }
      if (playerFeetY > e.feetWorldY + 0.38f) {
        const glm::vec2 ap = staffChaseElevatedApproachPointXZ(pXZ, playerFeetY, e.feetWorldY);
        const glm::vec2 aimElev = ap - e.posXZ;
        const float ael = glm::length(aimElev);
        if (ael > 0.055f) {
          const float dy = glm::clamp(playerFeetY - e.feetWorldY, 0.f, 7.f);
          const float blendW = glm::clamp(0.3f + dy * 0.048f, 0.3f, 0.74f);
          aim = glm::mix(aim, aimElev, blendW);
        }
      }
      glm::vec2 dir;
      if (e.chaseUnstuckTimer > 0.f) {
        e.chaseUnstuckTimer = std::max(0.f, e.chaseUnstuckTimer - dt);
        glm::vec2 toW = e.wanderTargetXZ - e.posXZ;
        const float dW = glm::length(toW);
        if (dW > 0.08f)
          dir = toW / dW;
        else
          dir = glm::length(aim) > 1e-4f ? aim * (1.f / glm::length(aim)) : glm::vec2(0.f, 1.f);
      } else {
        dir = glm::length(aim) > 1e-4f ? aim * (1.f / glm::length(aim)) : glm::vec2(0.f, 1.f);
      }
      float navLookMul = blackoutPursuit ? 1.36f : 1.f;
      if (playerFeetY > e.feetWorldY + 0.62f)
        navLookMul = std::max(navLookMul, kStaffChaseNavLookaheadElevMul);
      if (glm::dot(dir, dir) > 1e-8f)
        dir = staffNavAdjustedDesiredXZ(e, dir, distP, navLookMul, playerFeetY > e.feetWorldY + 0.72f);
      const bool ledgePull = e.chaseLedgeClimbRem > 0.f;
      const bool runnerMantelPull = ledgePull && e.staffMantelRunnerChase != 0;
      float spMul = ledgePull ? (kStaffChaseLedgeClimbMoveMul * (runnerMantelPull ? kStaffChaseRunnerLedgeMoveMul
                                                                                  : 1.f))
                              : 1.f;
      float acMul = ledgePull ? (kStaffChaseLedgeClimbAccelMul * (runnerMantelPull ? kStaffChaseRunnerLedgeAccelMul
                                                                                   : 1.f))
                              : 1.f;
      const float syCh = staffNpcFootSupportY(e, playerFeetY);
      const bool airLike = !ledgePull && !staffNpcIsGroundedLikePlayer(e, syCh);
      if (airLike)
        acMul *= kAirAccel / kWalkAccel;
      const float shelfBoost = glm::clamp(
          (playerFeetY - kGroundY) / std::max(kStaffChaseShelfBoostRefHeightM, 0.01f), 0.f, 1.f);
      const float pursuitMulSp = blackoutPursuit ? kStaffBlackoutPursuitSpeedMul : 1.f;
      const float pursuitMulAc = blackoutPursuit ? kStaffBlackoutPursuitAccelMul : 1.f;
      const float chaseSp = kShelfEmpNightChaseSpeed * staffSpeedScaleForBody(e) * pursuitMulSp * spMul *
                            (1.f + kStaffChaseShelfSpeedBoost * shelfBoost);
      const float chaseAc = kStaffSteerAccelChase * pursuitMulAc * acMul *
                            (1.f + kStaffChaseShelfAccelBoost * shelfBoost);
      staffIntegrateSteering(e, dt, dir, chaseSp, chaseAc, true);
      if (airLike)
        e.velXZ *= std::exp(-kAirDrag * dt);
    } else {
      e.velXZ *= std::exp(-8.f * dt);
      e.lastHorizSpeed = glm::length(e.velXZ);
    }
  }

#if defined(VULKAN_GAME_SHREK_EGG_GLB)
  bool staffNpcShouldHoldShrekDancePose(const ShelfEmployeeNpc& e, float playerFeetHint) const {
    if (!staffSkinnedActive || staffClipShrekProximityDance < 0 || !shrekEggAssetLoaded || !shrekEggActive ||
        e.meleeState >= 2)
      return false;
    const glm::vec2 d(e.posXZ.x - shrekEggPos.x, e.posXZ.y - shrekEggPos.z);
    const float r2 = kStaffShrekProximityDanceRadiusM * kStaffShrekProximityDanceRadiusM;
    if (glm::dot(d, d) > r2)
      return false;
    const float sy = staffNpcFootSupportY(e, playerFeetHint);
    if (!staffNpcIsGroundedLikePlayer(e, sy))
      return false;
    if ((e.staffAirLandClip >= 0 && e.staffAirLandRemain > 1e-4f) ||
        (e.staffAirFallClip >= 0 && e.staffAirLocoRemain > 1e-4f))
      return false;
    return true;
  }

  void updateShrekEggEaster(float dt) {
    if (!shrekEggAssetLoaded || !staffSkinnedActive || dt <= 0.f)
      return;
    constexpr float kEggRegionM = 132.f;
    constexpr float kEggMinDistFromOrigin = 92.f;
    if (shrekEggActive) {
      shrekEggAnimPhase += dt;
      glm::vec3 ro, fwd, right, up;
      getFirstPersonViewBasis(ro, fwd, right, up);
      const glm::vec2 toE(shrekEggPos.x - ro.x, shrekEggPos.z - ro.z);
      const float lenE = glm::length(toE);
      const glm::vec2 fXZ(fwd.x, fwd.z);
      const float fl = glm::length(fXZ);
      // Looking up/down makes fwd.xz ~ 0; old code left dotLook = -1 and despawned in 0.62s while he was in front.
      const glm::vec2 fn =
          fl > 1e-3f ? fXZ * (1.f / fl)
                     : glm::vec2(std::cos(yaw), std::sin(yaw)); // horizontal body forward (matches move basis)
      float dotLook = 1.f;
      if (lenE > 0.05f)
        dotLook = glm::dot(fn, toE * (1.f / lenE));
      constexpr float kLookAtMin = 0.38f;
      constexpr float kLookAwayMax = 0.22f;
      constexpr float kAwayHoldS = 0.42f;
      if (dotLook > kLookAtMin) {
        if (shrekEggLookAwayPrimed) {
          ++shrekEggLookAwayStrikes;
          shrekEggLookAwayPrimed = false;
          shrekEggLookAwayAccum = 0.f;
          if (shrekEggLookAwayStrikes >= 3) {
            shrekEggActive = false;
            audioSetShrekEggDanceActive(false);
            return;
          }
        } else
          shrekEggLookAwayAccum = 0.f;
      } else if (dotLook < kLookAwayMax) {
        shrekEggLookAwayAccum += dt;
        if (shrekEggLookAwayAccum >= kAwayHoldS)
          shrekEggLookAwayPrimed = true;
      }
      audioSetShrekEggDanceActive(true);
      const float distSong = glm::length(ro - shrekEggPos);
      audioUpdateShrekEggVolumeByDistance(distSong);
      return;
    }
    audioSetShrekEggDanceActive(false);
    const int rx = static_cast<int>(std::floor(camPos.x / kEggRegionM));
    const int rz = static_cast<int>(std::floor(camPos.z / kEggRegionM));
    if (rx == shrekEggRegionX && rz == shrekEggRegionZ)
      return;
    shrekEggRegionX = rx;
    shrekEggRegionZ = rz;
    if (glm::length(glm::vec2(camPos.x, camPos.z)) < kEggMinDistFromOrigin)
      return;
    const uint32_t h = scp3008ShelfHash(rx + 919, rz + 503, 0x1337C0DE);
    if ((h % 30u) != 0u)
      return;
    glm::vec3 ro, fwd, right, up;
    getFirstPersonViewBasis(ro, fwd, right, up);
    glm::vec2 back(-fwd.x, -fwd.z);
    const float bl = glm::length(back);
    if (bl < 1e-4f)
      return;
    back /= bl;
    const glm::vec2 side(-back.y, back.x);
    const float jig = static_cast<float>(static_cast<int32_t>(h >> 3) % 2000) / 2000.f;
    const float backM = 34.f + jig * 24.f;
    const float sideM = jig * 14.f - 7.f;
    const glm::vec2 pXZ = glm::vec2(camPos.x, camPos.z) + back * backM + side * sideM;
    const float feetY = playerTerrainSupportY(pXZ.x, pXZ.y, kGroundY + 2.8f);
    if (!std::isfinite(feetY) || feetY < kGroundY - 0.05f || feetY > kCeilingY - 2.5f)
      return;
    shrekEggPos = glm::vec3(pXZ.x, feetY, pXZ.y);
    shrekEggYaw = std::atan2(camPos.x - shrekEggPos.x, camPos.z - shrekEggPos.z);
    shrekEggAnimPhase = 0.f;
    shrekEggLookAwayAccum = 0.f;
    shrekEggLookAwayPrimed = false;
    shrekEggLookAwayStrikes = 0;
    shrekEggActive = true;
    audioSetShrekEggDanceActive(true);
  }

  // ] key: spawn ahead of view in XZ (no map hash); syncs region coords so walking doesn’t instantly re-roll rare egg.
  void spawnShrekEggNearPlayer() {
    if (!shrekEggAssetLoaded) {
      std::fprintf(stderr,
                   "[easter] Shrek GLB did not load — rebuild with the Shrek .glb path in CMake, or check console "
                   "for [easter] Shrek GLB: ... errors.\n");
      return;
    }
    if (!staffSkinnedActive) {
      std::fprintf(stderr, "[easter] Shrek: skinned staff pipeline is off (staff mesh not active).\n");
      return;
    }
    glm::vec3 ro, fwd, right, up;
    getFirstPersonViewBasis(ro, fwd, right, up);
    glm::vec2 f2(fwd.x, fwd.z);
    float fl = glm::length(f2);
    if (fl < 1e-4f)
      f2 = glm::vec2(std::cos(yaw), std::sin(yaw));
    else
      f2 *= 1.f / fl;
    // Right in front of the player (body XZ = cam XZ); minimal gap past capsule so feet don’t overlap.
    constexpr float kAheadM = kPlayerHalfXZ + 0.06f;
    const glm::vec2 pXZ = glm::vec2(camPos.x, camPos.z) + f2 * kAheadM;
    const float playerFeetProbe = camPos.y - eyeHeight;
    float feetY = playerTerrainSupportY(pXZ.x, pXZ.y, kGroundY + 2.8f);
    if (!std::isfinite(feetY) || feetY < kGroundY - 0.05f || feetY > kCeilingY - 2.5f)
      feetY = playerTerrainSupportY(pXZ.x, pXZ.y, playerFeetProbe);
    if (!std::isfinite(feetY) || feetY < kGroundY - 0.05f || feetY > kCeilingY - 2.5f)
      feetY = std::clamp(playerFeetProbe, kGroundY, kCeilingY - 2.5f);
    shrekEggPos = glm::vec3(pXZ.x, feetY, pXZ.y);
    shrekEggYaw = std::atan2(camPos.x - shrekEggPos.x, camPos.z - shrekEggPos.z);
    shrekEggAnimPhase = 0.f;
    shrekEggLookAwayAccum = 0.f;
    shrekEggLookAwayPrimed = false;
    shrekEggLookAwayStrikes = 0;
    shrekEggActive = true;
    audioSetShrekEggDanceActive(true);
    constexpr float kEggRegionM = 132.f;
    shrekEggRegionX = static_cast<int>(std::floor(camPos.x / kEggRegionM));
    shrekEggRegionZ = static_cast<int>(std::floor(camPos.z / kEggRegionM));
    std::fprintf(stderr,
                 "[easter] Shrek spawned ~%.2f m in front (%s). Look away 3 times (away ~0.4s, then look back) "
                 "to despawn. All Star plays louder near him if MP3 found.\n",
                 static_cast<double>(kAheadM),
                 shrekEggDiffuseLoaded ? "GLB mesh + diffuse + anim" : "mesh/anim (no diffuse in GLB)");
  }
#endif

  void spawnDeadRagdollStaffNearPlayer() {
    if (!staffSkinnedActive || employeeVertexCount == 0u) {
      std::fprintf(stderr, "[debug] staff ragdoll spawn: staff mesh/rig not active.\n");
      return;
    }
    glm::vec3 ro, fwd, right, up;
    getFirstPersonViewBasis(ro, fwd, right, up);
    glm::vec2 f2(fwd.x, fwd.z);
    float fl = glm::length(f2);
    if (fl < 1e-4f)
      f2 = glm::vec2(std::cos(yaw), std::sin(yaw));
    else
      f2 *= 1.f / fl;
    const glm::vec2 side(-f2.y, f2.x);
    const float aheadM = kPlayerHalfXZ + 0.82f;
    const float sideJitter = (((debugRagdollSpawnSeq & 1u) == 0u) ? -1.f : 1.f) * 0.22f;
    const glm::vec2 pXZ = glm::vec2(camPos.x, camPos.z) + f2 * aheadM + side * sideJitter;
    const float playerFeetProbe = camPos.y - eyeHeight;
    float feetY = playerTerrainSupportY(pXZ.x, pXZ.y, playerFeetProbe);
    if (!std::isfinite(feetY))
      feetY = std::clamp(playerFeetProbe, kGroundY, kCeilingY - 2.5f);

    const uint32_t seq = debugRagdollSpawnSeq++;
    const uint32_t hi = 0x7F000000u | ((seq >> 8) & 0x00FFFFFFu);
    const uint32_t lo = seq ^ 0x5A11C0DEu;
    const uint64_t key = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    ShelfEmployeeNpc& npc = shelfEmpAcquire(key);
    npc.inited = true;
    npc.posXZ = pXZ;
    npc.feetWorldY = feetY;
    npc.yaw = std::atan2(-f2.x, -f2.y);
    npc.aisleCenterX = pXZ.x;
    npc.aisleCenterZ = pXZ.y;
    npc.roamHalfX = 0.8f;
    npc.roamHalfZ = 0.8f;
    npc.wanderTargetXZ = pXZ;
    npc.velXZ = glm::vec2(0.f);
    npc.staffVelY = 0.f;
    npc.bodyScale = staffBodyScaleFromKey(key);
    npc.staffClassArchetype = staffClassArchetypeFromKey(key);
    npc.staffHpMax = staffClassMaxHp(npc.staffClassArchetype);
    npc.staffHp = 0.f;
    npc.staffDead = true;
    npc.meleeState = 2;
    const double dFall = staffClipMeleeFall >= 0 ? staff_skin::clipDuration(staffRig, staffClipMeleeFall) : 0.0;
    npc.meleePhaseSec = dFall > 1e-6 ? dFall * 0.62 : 0.0;
    npc.deadRagdollPitch = 0.16f;
    npc.deadRagdollRoll = (((seq >> 1) & 1u) != 0u) ? -0.42f : 0.42f;
    npc.deadRagdollPitchVel = 1.35f;
    npc.deadRagdollRollVel = (((seq >> 1) & 1u) != 0u) ? -1.2f : 1.2f;
    npc.deadRagdollYaw = npc.yaw;
    npc.deadRagdollYawVel = (((seq >> 1) & 1u) != 0u) ? -2.4f : 2.4f;
    npc.staffShoveKnockbackVelXZ = f2 * 0.75f;
    npc.staffPushAggro = false;
    npc.staffPushAggroCalmRemain = 0.f;
    npc.staffNightShoveChase = false;
    npc.staffNightShoveRevealRemain = 0.f;
    npc.nightPhase = 0;
    npc.nightSpotTimer = 0.f;
    npc.nightInvestigateTimer = 0.f;
    npc.chaseLedgeClimbRem = -1.f;
    npc.chaseLedgeClimbTotalDur = 0.f;
    npc.staffMantelAnimPhaseSpanSec = 0.f;
    npc.staffMantelRunnerChase = 0;
    npc.staffAirLocoRemain = 0.f;
    npc.staffAirFallClip = -1;
    npc.staffAirLandRemain = 0.f;
    npc.staffAirLandClip = -1;
    npc.staffFallPeakFeetY = feetY;
    npc.staffTallFallKnockdownPending = 0u;
    std::fprintf(stderr, "[debug] spawned dead ragdoll staff (\\).\n");
  }

  ShelfEmployeeNpc* shelfEmpFind(uint64_t key) {
    const auto it = shelfEmpKeyToSlot.find(key);
    if (it == shelfEmpKeyToSlot.end())
      return nullptr;
    return &shelfEmpPool[it->second];
  }
  ShelfEmployeeNpc& shelfEmpAcquire(uint64_t key) {
    if (const auto it = shelfEmpKeyToSlot.find(key); it != shelfEmpKeyToSlot.end())
      return shelfEmpPool[it->second];
    uint32_t slot = 0;
    if (!shelfEmpFreeSlots.empty()) {
      slot = shelfEmpFreeSlots.back();
      shelfEmpFreeSlots.pop_back();
    } else {
      slot = static_cast<uint32_t>(shelfEmpPool.size());
      shelfEmpPool.emplace_back();
    }
    shelfEmpKeyToSlot[key] = slot;
    ShelfEmployeeNpc& e = shelfEmpPool[slot];
    e = ShelfEmployeeNpc{};
    e.residentKey = key;
    shelfEmpActiveSlots.push_back(slot);
    return e;
  }

  void updateShelfEmployees(float dt) {
    const glm::vec2 pXZ(camPos.x, camPos.z);
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
    const glm::vec2 dEgg = pXZ - glm::vec2(shrekEggPos.x, shrekEggPos.z);
    const float shrekProxR2 = kStaffShrekProximityDanceRadiusM * kStaffShrekProximityDanceRadiusM;
    const bool staffSuppressMeleeNearShrekThisFrame =
        shrekEggAssetLoaded && shrekEggActive && glm::dot(dEgg, dEgg) <= shrekProxR2;
#else
    const bool staffSuppressMeleeNearShrekThisFrame = false;
#endif
    const float playerFeetForStaffSupport = camPos.y - eyeHeight;
    const bool day = audioAreStoreFluorescentsOn();
    if (playerStaffMeleeInvulnRem > 0.f)
      playerStaffMeleeInvulnRem = std::max(0.f, playerStaffMeleeInvulnRem - dt);
    const float pruneSq = kShelfEmpPruneDist * kShelfEmpPruneDist;
    // Distance-based AI stagger: far NPCs in wander mode tick every Nth frame.
    // Compensate dt so walk speed stays correct over the stagger window.
    static uint32_t aiFrameCounter = 0;
    ++aiFrameCounter;
    constexpr float kStaffAiMidDistSq = 40.f * 40.f;
    constexpr float kStaffAiFarDistSq = 70.f * 70.f;
    {
      PerfScopeTimer _aiBlend(&gPerfTrace.accumStaffAIBlend, gPerfTrace.enabled);
      for (uint32_t si : shelfEmpActiveSlots) {
        ShelfEmployeeNpc& e = shelfEmpPool[si];
        if (!e.inited)
          continue;
        if (e.staffHpMax <= 1e-4f) {
          e.staffClassArchetype = staffClassArchetypeFromKey(e.residentKey);
          e.staffHpMax = staffClassMaxHp(e.staffClassArchetype);
          e.staffHp = e.staffHpMax;
          e.staffDead = false;
          e.deadRagdollPitch = 0.f;
          e.deadRagdollRoll = 0.f;
          e.deadRagdollPitchVel = 0.f;
          e.deadRagdollRollVel = 0.f;
          e.deadRagdollYaw = e.yaw;
          e.deadRagdollYawVel = 0.f;
          shelfEmpResetDeadJointSim(e);
        } else {
          if (e.staffDead)
            e.staffHp = 0.f;
          else
            e.staffHp = glm::clamp(e.staffHp, 0.f, e.staffHpMax);
        }
        if (e.staffDead && e.meleeState < 2) {
          e.meleeState = 2;
          e.meleeKnockdownFeetAnchorY = e.feetWorldY;
        }
        if (e.meleeAnimBlend < 1.f)
          e.meleeAnimBlend = std::min(1.f, e.meleeAnimBlend + dt * (1.f / kStaffMeleeBlendSec));
        e.staffChaseMantelCooldownRem = std::max(0.f, e.staffChaseMantelCooldownRem - dt);
        if (!day && e.staffNightShoveRevealRemain > 0.f)
          e.staffNightShoveRevealRemain = std::max(0.f, e.staffNightShoveRevealRemain - dt);
      }
    }
    {
      PerfScopeTimer _aiMain(&gPerfTrace.accumStaffAIMain, gPerfTrace.enabled);
      for (uint32_t si : shelfEmpActiveSlots) {
        ShelfEmployeeNpc& e = shelfEmpPool[si];
      if (!e.inited)
        continue;
      e.lastHorizSpeed = 0.f;
      const glm::vec2 toPlayer = pXZ - e.posXZ;
      const float distP = glm::length(toPlayer);
      const float distPSq = distP * distP;
      // Stagger AI for distant wander-mode NPCs. Aggressive/melee NPCs always tick.
      const bool npcIsAggressive = (e.meleeState != 0) || e.staffPushAggro || e.staffNightShoveChase ||
                                   (e.nightPhase >= 2);
      if (!npcIsAggressive && distPSq > kStaffAiMidDistSq) {
        const uint32_t slotHash = static_cast<uint32_t>(e.residentKey ^ (e.residentKey >> 32));
        if (distPSq > kStaffAiFarDistSq) {
          if (((aiFrameCounter + slotHash) & 3u) != 0u)
            continue;
        } else {
          if (((aiFrameCounter + slotHash) & 1u) != 0u)
            continue;
        }
      }
      if (day && e.staffPushAggro) {
        const float calmRate =
            distP > kStaffDayPushAggroFarDistM ? kStaffDayPushAggroFarCalmMul : 1.f;
        e.staffPushAggroCalmRemain -= dt * calmRate;
        if (e.staffPushAggroCalmRemain <= 0.f) {
          e.staffPushAggro = false;
          e.staffPushAggroCalmRemain = 0.f;
        }
      }
      if (e.meleeState == 2 || e.meleeState == 3 || e.meleeState == 4) {
        if (e.staffDead) {
          e.meleeState = 2;
          const glm::vec2 kb = e.staffShoveKnockbackVelXZ;
          const glm::vec2 fwd(std::sin(e.yaw), std::cos(e.yaw));
          const glm::vec2 right(fwd.y, -fwd.x);
          // Convert horizontal impact into a soft angular impulse.
          e.deadRagdollRollVel += glm::dot(kb, right) * 1.55f;
          e.deadRagdollPitchVel += -glm::dot(kb, fwd) * 1.15f;
          e.deadRagdollYawVel += glm::dot(kb, right) * 0.95f + glm::dot(kb, fwd) * 0.35f;
          // Damped spring keeps corpse wobble plausible instead of rigid.
          e.deadRagdollPitchVel += (-e.deadRagdollPitch) * 4.4f * dt;
          e.deadRagdollRollVel += (-e.deadRagdollRoll) * 3.9f * dt;
          const float ragdollDamp = std::exp(-4.6f * dt);
          e.deadRagdollPitchVel *= ragdollDamp;
          e.deadRagdollRollVel *= ragdollDamp;
          e.deadRagdollYawVel *= std::exp(-2.3f * dt);
          e.deadRagdollYaw += e.deadRagdollYawVel * dt;
          e.yaw = e.deadRagdollYaw;
          e.deadRagdollPitch = glm::clamp(e.deadRagdollPitch + e.deadRagdollPitchVel * dt, -1.15f, 1.15f);
          e.deadRagdollRoll = glm::clamp(e.deadRagdollRoll + e.deadRagdollRollVel * dt, -1.42f, 1.42f);
          const float kbLen = glm::length(kb);
          if (!e.deadRagdollJointSimInited && gStaffDeadRagdollSimBoneCount > 0) {
            const int nJ = std::min(gStaffDeadRagdollSimBoneCount, kStaffRagdollSimMaxBones);
            e.deadRagdollJointSimCount = static_cast<uint8_t>(nJ);
            uint32_t jh = static_cast<uint32_t>(e.residentKey ^ (e.residentKey >> 32));
            for (int j = 0; j < nJ; ++j) {
              jh = jh * 1664525u + 1013904223u;
              const uint8_t bkj = gStaffDeadRagdollSimBoneKind[j];
              const bool tightInit = (bkj == 1u || bkj == 4u || bkj == 8u);
              const float eAmp = tightInit ? (0.0045f / 127.5f) : (0.015f / 127.5f);
              const float ezAmp = tightInit ? (0.005f / 127.5f) : (0.02f / 127.5f);
              e.deadRagdollJointEuler[j] = glm::vec3(
                  (static_cast<float>((jh >> 8) & 255u) - 127.5f) * eAmp,
                  (static_cast<float>((jh >> 16) & 255u) - 127.5f) * eAmp,
                  (static_cast<float>((jh >> 24) & 255u) - 127.5f) * ezAmp);
              const float s =
                  (0.002f + static_cast<float>(jh & 255u) * (0.28f / 255.f)) * (tightInit ? 0.35f : 1.f);
              e.deadRagdollJointVel[j] =
                  glm::vec3(glm::dot(kb, right) * s, kbLen * s * 0.4f, glm::dot(kb, fwd) * (-s * 0.85f));
            }
            e.deadRagdollJointSimInited = true;
          }
          constexpr float kJRSpring = 11.f;
          constexpr float kJRKnockImpulse = 0.14f;
          constexpr float kJRLinearDamp = 3.4f;
          for (int j = 0; j < e.deadRagdollJointSimCount; ++j) {
            e.deadRagdollJointVel[j] -= e.deadRagdollJointEuler[j] * kJRSpring * dt;
            if (kbLen > 1e-5f) {
              const glm::vec3 knock(glm::dot(kb, right) * kJRKnockImpulse, kbLen * kJRKnockImpulse * 0.28f,
                                    glm::dot(kb, fwd) * (-kJRKnockImpulse * 0.72f));
              e.deadRagdollJointVel[j] += knock * dt;
            }
            e.deadRagdollJointVel[j] *= std::exp(-kJRLinearDamp * dt);
            e.deadRagdollJointEuler[j] += e.deadRagdollJointVel[j] * dt;
            glm::vec3& eu = e.deadRagdollJointEuler[j];
            eu = glm::clamp(eu, glm::vec3(-1.18f), glm::vec3(1.18f));
            if (gStaffDeadRagdollSimBoneKind[j] == 1u)
              eu = glm::clamp(eu, glm::vec3(-kDeadRagdollNeckEulerClamp),
                              glm::vec3(kDeadRagdollNeckEulerClamp));
            else if (gStaffDeadRagdollSimBoneKind[j] == 8u)
              eu = glm::clamp(eu, glm::vec3(-kDeadRagdollHeadEulerClamp),
                              glm::vec3(kDeadRagdollHeadEulerClamp));
            else if (gStaffDeadRagdollSimBoneKind[j] == 4u)
              eu = glm::clamp(eu, glm::vec3(-kDeadRagdollCoreEulerClamp),
                              glm::vec3(kDeadRagdollCoreEulerClamp));
          }
        }
        e.velXZ = glm::vec2(0.f);
        const float kbSq = glm::dot(e.staffShoveKnockbackVelXZ, e.staffShoveKnockbackVelXZ);
        if (kbSq > 1e-10f) {
          e.posXZ += e.staffShoveKnockbackVelXZ * dt;
          e.staffShoveKnockbackVelXZ *= std::exp(-kStaffShoveKnockbackDecay * dt);
          if (glm::dot(e.staffShoveKnockbackVelXZ, e.staffShoveKnockbackVelXZ) < 1e-8f)
            e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
        }
        continue;
      }
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
      if (staffNpcShouldHoldShrekDancePose(e, playerFeetForStaffSupport)) {
        e.velXZ = glm::vec2(0.f);
        e.lastHorizSpeed = 0.f;
        e.chaseLedgeClimbRem = -1.f;
        e.chaseLedgeClimbTotalDur = 0.f;
        e.staffMantelAnimPhaseSpanSec = 0.f;
        e.staffMantelRunnerChase = 0;
        const glm::vec2 toSh(shrekEggPos.x - e.posXZ.x, shrekEggPos.z - e.posXZ.y);
        if (glm::dot(toSh, toSh) > 1e-8f)
          e.yaw = std::atan2(toSh.x, toSh.y);
        if (e.meleeState == 1) {
          e.meleeState = 0;
          e.meleePhaseSec = 0.0;
          e.meleeAnimBlend = 1.f;
          e.shovePlayerPushDurSec = 0.f;
          e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
        }
        continue;
      }
#endif
      const float playerFeetYVis = camPos.y - eyeHeight;
      if (day) {
        e.staffNightShoveChase = false;
        e.staffNightShoveRevealRemain = 0.f;
        if (e.staffPushAggro && !staffPlayerVisibleInChaseCone(e, distP, toPlayer, playerFeetYVis)) {
          e.staffPushAggro = false;
          e.staffPushAggroCalmRemain = 0.f;
        }
        if (!e.staffPushAggro) {
          e.nightPhase = 0;
          e.meleeState = 0;
          e.meleePhaseSec = 0.0;
          e.meleeAnimBlend = 1.f;
          e.shovePlayerPushDurSec = 0.f;
          e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
          e.meleeKnockdownFeetAnchorY = kGroundY;
          e.nightSpotTimer = 0.f;
          e.nightInvestigateTimer = 0.f;
          e.chaseUnstuckTimer = 0.f;
          e.chaseLedgeClimbRem = -1.f;
          e.chaseLedgeClimbTotalDur = 0.f;
          e.staffMantelAnimPhaseSpanSec = 0.f;
          e.staffMantelRunnerChase = 0;
          e.staffPushAggroCalmRemain = 0.f;
          shelfEmpStepWanderTowardTarget(e, dt, e.residentKey, pXZ, false, playerFeetForStaffSupport);
        } else {
          e.nightPhase = 2;
          e.nightLastKnownPlayerXZ = pXZ;
          shelfEmployeeRunChasePhase2(e, dt, pXZ, distP, toPlayer, playerFeetForStaffSupport, false,
                                        staffSuppressMeleeNearShrekThisFrame);
        }
      } else {
        if (e.nightPhase != 2 || e.meleeState != 0) {
          e.chaseLedgeClimbRem = -1.f;
          e.chaseLedgeClimbTotalDur = 0.f;
          e.staffMantelAnimPhaseSpanSec = 0.f;
          e.staffMantelRunnerChase = 0;
        }
        // Forward in XZ matches draw / movement: yaw = atan2(dir.x, dir.y) for dir toward target.
        const glm::vec2 fwd(std::sin(e.yaw), std::cos(e.yaw));
        glm::vec2 toPn(0.f);
        if (distP > 1e-4f)
          toPn = toPlayer / distP;
        const float dotF = glm::dot(toPn, fwd);
        const bool chaseVision = (e.nightPhase == 2);
        float dotCone = dotF;
        if (chaseVision) {
          const float vCh = glm::length(e.velXZ);
          if (vCh > kNightStaffChaseVisionVelConeMinSpeed)
            dotCone = std::max(dotF, glm::dot(toPn, e.velXZ * (1.f / vCh)));
        }
        const float staffEyeY =
            e.feetWorldY + (kNightStaffVisionEyeY - kGroundY) * e.bodyScale.y;
        const float dyPlayerEyes = camPos.y - staffEyeY;
        const float elevAboveHoriz =
            std::atan2(dyPlayerEyes, std::max(distP, 1e-3f)); // + = player above staff eye level
        const float crouchT = glm::clamp((kEyeHeight - eyeHeight) / (kEyeHeight - kCrouchEyeHeight), 0.f, 1.f);
        const float stealthVisMul = glm::mix(1.f, kCrouchStealthVisionRangeMul, crouchT);
        const float stealthBehindMul = glm::mix(1.f, kCrouchStealthBehindSenseMul, crouchT);
        const float visRange = (chaseVision ? kNightStaffChaseVisionRange : kNightStaffVisionRange) * stealthVisMul;
        const float visCosHalf = chaseVision ? kNightStaffChaseVisionCosHalfFov : kNightStaffVisionCosHalfFov;
        const float behindSenseM = (chaseVision ? kNightStaffChaseBehindSenseM : kNightStaffBehindSenseM) * stealthBehindMul;
        const float maxElevAboveDeg =
            chaseVision ? kNightStaffChaseVisionMaxElevAboveDeg : kNightStaffVisionMaxElevAboveDeg;
        const bool ledgedUpForVision =
            playerFeetYVis >= kGroundY + kNightStaffVisionPlayerLedgedFeetAboveFloorM;
        const float elevCapRad =
            (ledgedUpForVision && distP <= visRange)
                ? glm::radians(kNightStaffVisionLedgedMaxElevAboveDeg)
                : glm::radians(maxElevAboveDeg);
        const bool withinElevSight = elevAboveHoriz <= elevCapRad;
        const bool inFrontCone =
            withinElevSight && distP <= visRange && dotCone >= visCosHalf;
        const bool closeBehind =
            withinElevSight && distP <= behindSenseM && dotF < 0.12f; // rear + “really close”
        const bool detected = inFrontCone || closeBehind;
        const bool pushRevealKnows =
            e.staffNightShoveChase && e.staffNightShoveRevealRemain > 0.f;
        if (pushRevealKnows)
          e.nightLastKnownPlayerXZ = pXZ;

        if ((e.staffPushAggro || e.staffNightShoveChase) && !detected && !pushRevealKnows) {
          e.staffPushAggro = false;
          e.staffPushAggroCalmRemain = 0.f;
          e.staffNightShoveChase = false;
          e.staffNightShoveRevealRemain = 0.f;
        }
        if (detected || pushRevealKnows) {
          if (detected)
            e.nightLastKnownPlayerXZ = pXZ;
          if (e.staffPushAggro || e.staffNightShoveChase)
            e.nightPhase = 2;
          if (e.nightPhase == 3 && detected) {
            // Spotted again while investigating — go back to watching (delay before chase restarts).
            e.nightPhase = 1;
            e.nightSpotTimer = 0.f;
            e.nightInvestigateTimer = 0.f;
          }
          if (e.nightPhase == 2) {
            shelfEmployeeRunChasePhase2(e, dt, pXZ, distP, toPlayer, playerFeetForStaffSupport, true,
                                        staffSuppressMeleeNearShrekThisFrame);
          } else if (!e.staffPushAggro && !e.staffNightShoveChase) {
            if (e.nightPhase == 0)
              audioPlayStaffSpotted();
            e.velXZ = glm::vec2(0.f);
            e.yaw = std::atan2(toPlayer.x, toPlayer.y);
            e.lastHorizSpeed = 0.f;
            e.nightSpotTimer += dt;
            if (e.nightPhase == 0)
              e.nightPhase = 1;
            const float chaseDelay = kNightStaffChaseDelayS * glm::mix(1.f, kCrouchStealthChaseDelayMul, crouchT);
            if (e.nightPhase == 1 && e.nightSpotTimer >= chaseDelay) {
              e.nightPhase = 2;
              e.nightSpotTimer = 0.f;
            }
          }
        } else {
          e.meleeState = 0;
          e.meleeAnimBlend = 1.f;
          e.shovePlayerPushDurSec = 0.f;
          e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
          // Lost sight: if they were watching or chasing, walk toward last known XZ before giving up.
          if (e.nightPhase == 1 || e.nightPhase == 2) {
            e.nightPhase = 3;
            e.nightSpotTimer = 0.f;
            e.nightInvestigateTimer = 0.f;
            e.chaseUnstuckTimer = 0.f;
          }
          if (e.nightPhase == 3) {
            e.nightInvestigateTimer += dt;
            glm::vec2 toL = e.nightLastKnownPlayerXZ - e.posXZ;
            const float dL = glm::length(toL);
            if (dL < kNightStaffInvestigateReachM ||
                e.nightInvestigateTimer >= kNightStaffInvestigateMaxS) {
              e.nightPhase = 0;
              e.nightSpotTimer = 0.f;
              e.nightInvestigateTimer = 0.f;
              e.chaseUnstuckTimer = 0.f;
              {
                const float vlen = glm::length(e.velXZ);
                const float cap = kShelfEmpWalkSpeed * staffSpeedScaleForBody(e) * 1.4f;
                if (vlen > cap)
                  e.velXZ *= cap / std::max(vlen, 1e-6f);
              }
              shelfEmpStepWanderTowardTarget(e, dt, e.residentKey, pXZ, false, playerFeetForStaffSupport);
            } else {
              glm::vec2 inv = toL;
              if (glm::dot(inv, inv) > 1e-8f)
                inv = staffNavAdjustedDesiredXZ(e, inv, distP, 1.22f);
              const float syI = staffNpcFootSupportY(e, playerFeetForStaffSupport);
              const bool airI = !staffNpcIsGroundedLikePlayer(e, syI);
              float acI = kStaffSteerAccelWalk;
              if (airI)
                acI *= kAirAccel / kWalkAccel;
              staffIntegrateSteering(e, dt, inv,
                                     kShelfEmpWalkSpeed * kNightStaffInvestigateSpeedMul *
                                         staffSpeedScaleForBody(e),
                                     acI);
              if (airI)
                e.velXZ *= std::exp(-kAirDrag * dt);
            }
          } else {
            e.nightPhase = 0;
            e.nightSpotTimer = 0.f;
            e.nightInvestigateTimer = 0.f;
            e.chaseUnstuckTimer = 0.f;
            {
              const float vlen = glm::length(e.velXZ);
              const float cap = kShelfEmpWalkSpeed * staffSpeedScaleForBody(e) * 1.4f;
              if (vlen > cap)
                e.velXZ *= cap / std::max(vlen, 1e-6f);
            }
            shelfEmpStepWanderTowardTarget(e, dt, e.residentKey, pXZ, false, playerFeetForStaffSupport);
          }
        }
      }
    }
    }
    if (!day) {
      const float alertR = kStaffChaseNeighborAlertRadiusM;
      const float alertR2 = alertR * alertR;
      const float nearP2 = kStaffNightAlertMaxDistFromPlayer * kStaffNightAlertMaxDistFromPlayer;
      constexpr float kAlertBucketSize = 10.f;
      struct AlertEntry { uint64_t key; ShelfEmployeeNpc* npc; };
      thread_local std::unordered_map<int64_t, std::vector<AlertEntry>> alertGrid;
      alertGrid.clear();
      auto alertBucketKey = [](float x, float z) -> int64_t {
        const int bx = static_cast<int>(std::floor(x / kAlertBucketSize));
        const int bz = static_cast<int>(std::floor(z / kAlertBucketSize));
        return (static_cast<int64_t>(bx) << 32) | static_cast<int64_t>(static_cast<uint32_t>(bz));
      };
      for (uint32_t si2 : shelfEmpActiveSlots) {
        ShelfEmployeeNpc& e2 = shelfEmpPool[si2];
        if (!e2.inited || e2.meleeState >= 2) continue;
        const glm::vec2 dp2 = e2.posXZ - pXZ;
        if (glm::dot(dp2, dp2) > nearP2) continue;
        alertGrid[alertBucketKey(e2.posXZ.x, e2.posXZ.y)].push_back({e2.residentKey, &e2});
      }
      const int bucketR = static_cast<int>(std::ceil(alertR / kAlertBucketSize));
      for (auto& [bkey, entries] : alertGrid) {
        for (auto& ae : entries) {
          const ShelfEmployeeNpc& a = *ae.npc;
          const bool aShoveRadio = a.staffNightShoveChase && a.staffNightShoveRevealRemain > 0.f;
          const bool aChaseUpright = a.meleeState < 2 && a.nightPhase == 2;
          if (!aShoveRadio && !aChaseUpright) continue;
          const int abx = static_cast<int>(std::floor(a.posXZ.x / kAlertBucketSize));
          const int abz = static_cast<int>(std::floor(a.posXZ.y / kAlertBucketSize));
          for (int ndx = -bucketR; ndx <= bucketR; ++ndx) {
            for (int ndz = -bucketR; ndz <= bucketR; ++ndz) {
              const int64_t nk = (static_cast<int64_t>(abx + ndx) << 32) |
                                  static_cast<int64_t>(static_cast<uint32_t>(abz + ndz));
              auto nit = alertGrid.find(nk);
              if (nit == alertGrid.end()) continue;
              for (auto& be : nit->second) {
                if (be.key == ae.key) continue;
                ShelfEmployeeNpc& b = *be.npc;
                if (!b.inited || b.meleeState >= 2) continue;
                const glm::vec2 d = b.posXZ - a.posXZ;
                if (glm::dot(d, d) > alertR2) continue;
                if (aShoveRadio) {
                  if (b.nightPhase == 2) continue;
                  b.nightPhase = 2;
                  b.staffNightShoveChase = true;
                  b.nightLastKnownPlayerXZ = pXZ;
                  b.nightSpotTimer = 0.f;
                  b.nightInvestigateTimer = 0.f;
                  b.chaseUnstuckTimer = 0.f;
                  b.staffNightShoveRevealRemain =
                      std::max(b.staffNightShoveRevealRemain, a.staffNightShoveRevealRemain);
                  continue;
                }
                if (b.nightPhase == 2) continue;
                b.nightPhase = 2;
                b.nightLastKnownPlayerXZ = pXZ;
                b.nightSpotTimer = 0.f;
                b.nightInvestigateTimer = 0.f;
                b.chaseUnstuckTimer = 0.f;
              }
            }
          }
        }
      }
    }
    bool anyStaffChasing = false;
    shelfEmpAnyDayPushChase = false;
    shelfEmpNightPursuitActive = false;
    for (uint32_t siCh : shelfEmpActiveSlots) {
      const ShelfEmployeeNpc& e = shelfEmpPool[siCh];
      if (!e.inited || e.nightPhase != 2 || e.meleeState >= 2)
        continue;
      if (!day)
        shelfEmpNightPursuitActive = true;
      if (!day || e.staffPushAggro) {
        anyStaffChasing = true;
        if (day && e.staffPushAggro)
          shelfEmpAnyDayPushChase = true;
      }
    }
    audioUpdateStaffChaseTaunts(dt, anyStaffChasing);
    separateShelfEmployeesFromEachOther();
    if (nudgeShelfEmployeesFromPlayer())
      separateShelfEmployeesFromEachOther();
    for (uint32_t siW : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& eW = shelfEmpPool[siW];
      if (!eW.inited)
        continue;
      eW.posXZPreResolve = eW.posXZ;
    }
    for (uint32_t siW : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& eW = shelfEmpPool[siW];
      if (!eW.inited)
        continue;
      resolveStaffNpcAgainstWorld(eW);
    }
    for (uint32_t siW : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& eW = shelfEmpPool[siW];
      if (!eW.inited)
        continue;
      staffNpcIntegrateVerticalPhysics(eW, dt, playerFeetForStaffSupport);
    }
    if (staffSkinnedActive && gStaffDeadRagdollSimBoneCount > 0) {
      constexpr float kDeadTiltMul = 0.42f;
      constexpr int kPbdIters = 7;
      constexpr float kPbdGravity = 9.2f;
      constexpr float kPbdKinSpringRoot = 14.f;
      constexpr float kPbdKinSpringChild = 5.1f;
      constexpr float kPbdDampPerSec = 2.55f;
      constexpr float kPbdEulerFb = 1.22f;
      // Hard cap on bone chain stretch (prevents spaghetti limbs / skin spikes when constraints fight).
      constexpr float kPbdMaxStretch = 1.11f;
      constexpr float kPbdEulerDeltaCap = 0.062f;
      constexpr float kPbdGroundPad = 0.048f;
      constexpr float kPbdGroundPadDistal = 0.074f;
      static thread_local std::vector<glm::vec3> corpseEulScratch;
      static thread_local std::vector<glm::vec3> corpseKinW;
      const float dts = glm::clamp(dt, 1e-4f, 0.05f);
      for (uint32_t siW : shelfEmpActiveSlots) {
        ShelfEmployeeNpc& e = shelfEmpPool[siW];
        if (!e.inited || !e.staffDead || e.meleeState < 2)
          continue;
        const int nJ = std::min({gStaffDeadRagdollSimBoneCount, kStaffRagdollSimMaxBones,
                                 static_cast<int>(e.deadRagdollJointSimCount)});
        if (nJ <= 0)
          continue;
        const auto corpseM = [&](float feetY) {
          const float feetSink = staffMeleeDrawFeetSinkY(e);
          const float ragLiftY = kStaffRagdollBindVisualLiftY * e.bodyScale.y;
          glm::mat4 rot = glm::rotate(glm::mat4(1.f), e.yaw, glm::vec3(0.f, 1.f, 0.f));
          rot = rot * glm::rotate(glm::mat4(1.f), e.deadRagdollPitch * kDeadTiltMul,
                                  glm::vec3(1.f, 0.f, 0.f)) *
                glm::rotate(glm::mat4(1.f), e.deadRagdollRoll * kDeadTiltMul, glm::vec3(0.f, 0.f, 1.f));
          return glm::translate(glm::mat4(1.f),
                                glm::vec3(e.posXZ.x, feetY - feetSink + ragLiftY, e.posXZ.y)) *
                 rot * glm::scale(glm::mat4(1.f), e.bodyScale);
        };
        glm::mat4 M = corpseM(e.feetWorldY);
        static const glm::vec3 kGndProbeLocal[] = {
            {0.f, 0.06f, 0.02f},    {0.1f, 0.05f, 0.04f},   {-0.1f, 0.05f, 0.04f},
            {0.f, 0.38f, 0.05f},    {0.f, 0.72f, 0.08f},    {0.12f, 0.52f, -0.06f},
            {-0.12f, 0.52f, -0.06f},
            // Prone arms often reach sideways — extra hull samples reduce hands through the floor.
            {0.36f, 0.2f, 0.14f},   {-0.36f, 0.2f, 0.14f}, {0.28f, 0.14f, -0.2f},
            {-0.28f, 0.14f, -0.2f}, {0.22f, 0.55f, 0.1f},   {-0.22f, 0.55f, 0.1f},
            // Low hip / upper leg when lying on side.
            {0.2f, 0.12f, 0.02f},   {-0.2f, 0.12f, 0.02f}, {0.f, 0.18f, -0.14f}};
        float maxPen = 0.f;
        for (const glm::vec3& pl : kGndProbeLocal) {
          const glm::vec3 ls(pl.x * e.bodyScale.x, pl.y * e.bodyScale.y, pl.z * e.bodyScale.z);
          const glm::vec3 w = glm::vec3(M * glm::vec4(ls, 1.f));
          const float ty = terrainSupportY(w.x, w.z, w.y + 4.f);
          const float pen = ty + kDeadCorpseGroundProbeSkinPad - w.y;
          maxPen = std::max(maxPen, pen);
        }
        // Avoid one-frame spikes (odd shelf samples / tilt) launching the corpse upward.
        maxPen = std::min(maxPen, 0.56f);
        if (maxPen > 0.f)
          e.feetWorldY += maxPen;
        M = corpseM(e.feetWorldY);
        corpseEulScratch.assign(static_cast<size_t>(staffRig.boneCount), glm::vec3(0.f));
        for (int j = 0; j < nJ; ++j)
          corpseEulScratch[static_cast<size_t>(gStaffDeadRagdollSimBoneRigIdx[j])] = e.deadRagdollJointEuler[j];
        glm::mat4 gAnim[staff_skin::kMaxPaletteBones];
        staff_skin::sampleBindBoneGlobalMatricesWithExtras(staffRig, corpseEulScratch.data(), gAnim);
        corpseKinW.resize(static_cast<size_t>(nJ));
        for (int j = 0; j < nJ; ++j) {
          const int bi = gStaffDeadRagdollSimBoneRigIdx[j];
          const glm::vec3 mlocal =
              glm::vec3(staffRig.meshNorm * gAnim[bi] * glm::vec4(0.f, 0.f, 0.f, 1.f));
          corpseKinW[static_cast<size_t>(j)] = glm::vec3(M * glm::vec4(mlocal, 1.f));
        }
        if (!e.deadRagdollPbdInited) {
          for (int j = 0; j < nJ; ++j) {
            e.deadRagdollPbdPosW[j] = corpseKinW[static_cast<size_t>(j)];
            e.deadRagdollPbdVelW[j] = glm::vec3(0.f);
          }
          e.deadRagdollPbdInited = true;
        }
        for (int j = 0; j < nJ; ++j) {
          const int pj = gStaffDeadRagdollSimParentIdx[j];
          const bool rootChain = (pj < 0);
          glm::vec3& v = e.deadRagdollPbdVelW[j];
          glm::vec3& p = e.deadRagdollPbdPosW[j];
          const float spr = rootChain ? kPbdKinSpringRoot : kPbdKinSpringChild;
          v += (corpseKinW[static_cast<size_t>(j)] - p) * spr * dts;
          if (!rootChain)
            v.y -= kPbdGravity * dts;
          v *= std::exp(-kPbdDampPerSec * dts);
        }
        for (int j = 0; j < nJ; ++j)
          e.deadRagdollPbdPosW[j] += e.deadRagdollPbdVelW[j] * dts;
        for (int it = 0; it < kPbdIters; ++it) {
          for (int j = 0; j < nJ; ++j) {
            const int pj = gStaffDeadRagdollSimParentIdx[j];
            if (pj < 0)
              continue;
            const float rest = gStaffDeadRagdollSimRestLen[j] * e.bodyScale.y;
            glm::vec3& pa = e.deadRagdollPbdPosW[pj];
            glm::vec3& pb = e.deadRagdollPbdPosW[j];
            glm::vec3 d = pb - pa;
            const float len = glm::length(d);
            if (len < 1e-6f)
              continue;
            const float err = 0.5f * (len - rest) / len;
            const glm::vec3 corr = d * err;
            const bool parentIsSimRoot = (gStaffDeadRagdollSimParentIdx[pj] < 0);
            const float wp = parentIsSimRoot ? 0.32f : 0.5f;
            pa += corr * wp;
            pb -= corr * (1.f - wp);
          }
          for (int j = 0; j < nJ; ++j) {
            const int pj = gStaffDeadRagdollSimParentIdx[j];
            if (pj < 0)
              continue;
            const float rest = gStaffDeadRagdollSimRestLen[j] * e.bodyScale.y;
            const float maxLen = rest * kPbdMaxStretch;
            glm::vec3& pa = e.deadRagdollPbdPosW[pj];
            glm::vec3& pb = e.deadRagdollPbdPosW[j];
            glm::vec3 d = pb - pa;
            const float len = glm::length(d);
            if (len > maxLen && len > 1e-6f) {
              const float pull = 0.5f * (len - maxLen) / len;
              const glm::vec3 c = d * pull;
              const bool parentIsSimRoot = (gStaffDeadRagdollSimParentIdx[pj] < 0);
              const float wp = parentIsSimRoot ? 0.32f : 0.5f;
              pa += c * wp;
              pb -= c * (1.f - wp);
            }
          }
        }
        for (int j = 0; j < nJ; ++j) {
          glm::vec3& p = e.deadRagdollPbdPosW[j];
          glm::vec3& v = e.deadRagdollPbdVelW[j];
          const uint8_t bk = gStaffDeadRagdollSimBoneKind[j];
          const float pad = (bk == 2u || bk == 3u || bk == 5u || bk == 6u) ? kPbdGroundPadDistal : kPbdGroundPad;
          const float ty = terrainSupportY(p.x, p.z, p.y + 3.f);
          if (p.y < ty + pad) {
            p.y = ty + pad;
            v.y = std::max(0.f, v.y);
            v.x *= 0.92f;
            v.z *= 0.92f;
          }
        }
        // Bone pivots sit mid-segment; extrude toward the free end so mesh doesn’t tunnel (arms/thighs).
        for (int j = 0; j < nJ; ++j) {
          const uint8_t bk = gStaffDeadRagdollSimBoneKind[j];
          if (bk == 1u || bk == 4u)
            continue;
          const int pj = gStaffDeadRagdollSimParentIdx[j];
          if (pj < 0)
            continue;
          glm::vec3& p = e.deadRagdollPbdPosW[j];
          glm::vec3& v = e.deadRagdollPbdVelW[j];
          glm::vec3 d = p - e.deadRagdollPbdPosW[pj];
          const float dl = glm::length(d);
          if (dl < 1e-5f)
            continue;
          d *= 1.f / dl;
          float extM = 0.26f;
          if (bk == 8u)
            extM = 0.30f;
          else if (bk == 2u)
            extM = 0.44f;
          else if (bk == 3u)
            extM = 0.30f;
          else if (bk == 5u)
            extM = 0.54f;
          else if (bk == 6u)
            extM = 0.42f;
          else
            extM = 0.22f;
          const float ext = extM * e.bodyScale.y;
          glm::vec3 tip = p + d * ext;
          const float tyT = terrainSupportY(tip.x, tip.z, tip.y + 3.f);
          const float tipPad = kPbdGroundPadDistal + 0.018f;
          if (tip.y < tyT + tipPad) {
            p.y += (tyT + tipPad) - tip.y;
            v.y = std::max(0.f, v.y);
          }
        }
        for (int it2 = 0; it2 < 2; ++it2) {
          for (int j = 0; j < nJ; ++j) {
            const int pj = gStaffDeadRagdollSimParentIdx[j];
            if (pj < 0)
              continue;
            const float rest = gStaffDeadRagdollSimRestLen[j] * e.bodyScale.y;
            glm::vec3& pa = e.deadRagdollPbdPosW[pj];
            glm::vec3& pb = e.deadRagdollPbdPosW[j];
            glm::vec3 d = pb - pa;
            const float len = glm::length(d);
            if (len < 1e-6f)
              continue;
            const float err = 0.5f * (len - rest) / len;
            const glm::vec3 corr = d * err;
            const bool parentIsSimRoot = (gStaffDeadRagdollSimParentIdx[pj] < 0);
            const float wp = parentIsSimRoot ? 0.32f : 0.5f;
            pa += corr * wp;
            pb -= corr * (1.f - wp);
          }
          for (int j = 0; j < nJ; ++j) {
            const int pj = gStaffDeadRagdollSimParentIdx[j];
            if (pj < 0)
              continue;
            const float rest = gStaffDeadRagdollSimRestLen[j] * e.bodyScale.y;
            const float maxLen = rest * kPbdMaxStretch;
            glm::vec3& pa = e.deadRagdollPbdPosW[pj];
            glm::vec3& pb = e.deadRagdollPbdPosW[j];
            glm::vec3 d = pb - pa;
            const float len = glm::length(d);
            if (len > maxLen && len > 1e-6f) {
              const float pull = 0.5f * (len - maxLen) / len;
              const glm::vec3 c = d * pull;
              const bool parentIsSimRoot = (gStaffDeadRagdollSimParentIdx[pj] < 0);
              const float wp = parentIsSimRoot ? 0.32f : 0.5f;
              pa += c * wp;
              pb -= c * (1.f - wp);
            }
          }
        }
        const glm::mat4 invM = glm::inverse(M);
        for (int j = 0; j < nJ; ++j) {
          if (gStaffDeadRagdollSimParentIdx[j] < 0)
            continue;
          const uint8_t bk = gStaffDeadRagdollSimBoneKind[j];
          float fb = kPbdEulerFb;
          if (bk == 1u)
            fb *= 0.22f;
          else if (bk == 8u)
            fb *= 0.26f;
          else if (bk == 4u)
            fb *= 0.12f;
          else if (bk == 2u)
            fb *= 0.48f;
          else if (bk == 3u)
            fb *= 0.62f;
          else if (bk == 5u || bk == 6u)
            fb *= 0.58f;
          else
            fb *= 0.55f;
          const glm::vec3 mk =
              glm::vec3(invM * glm::vec4(corpseKinW[static_cast<size_t>(j)], 1.f));
          const glm::vec3 ms = glm::vec3(invM * glm::vec4(e.deadRagdollPbdPosW[j], 1.f));
          glm::vec3 delta = ms - mk;
          delta = glm::clamp(delta, glm::vec3(-kPbdEulerDeltaCap), glm::vec3(kPbdEulerDeltaCap));
          e.deadRagdollJointEuler[j] += delta * fb * dts;
          glm::vec3& eu = e.deadRagdollJointEuler[j];
          eu = glm::clamp(eu, glm::vec3(-1.18f), glm::vec3(1.18f));
          if (bk == 1u)
            eu = glm::clamp(eu, glm::vec3(-kDeadRagdollNeckEulerClamp),
                            glm::vec3(kDeadRagdollNeckEulerClamp));
          else if (bk == 8u)
            eu = glm::clamp(eu, glm::vec3(-kDeadRagdollHeadEulerClamp),
                            glm::vec3(kDeadRagdollHeadEulerClamp));
          else if (bk == 4u)
            eu = glm::clamp(eu, glm::vec3(-kDeadRagdollCoreEulerClamp),
                            glm::vec3(kDeadRagdollCoreEulerClamp));
        }
      }
    }
    for (uint32_t siW : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& e = shelfEmpPool[siW];
      if (!e.inited || !e.staffDead || e.meleeState < 2)
        continue;
      const float syG = staffNpcFootSupportY(e, playerFeetForStaffSupport);
      const float kbSqG = glm::dot(e.staffShoveKnockbackVelXZ, e.staffShoveKnockbackVelXZ);
      const float dtf = glm::clamp(dt, 1e-4f, 0.05f);
      // After knockback/gravity settle, pull logical feet down if we’re still visibly hovering.
      if (e.staffVelY <= 0.04f && kbSqG < 0.0036f && e.feetWorldY > syG + 0.045f) {
        const float drop = glm::min(8.5f * dtf, e.feetWorldY - syG - 0.018f);
        e.feetWorldY = glm::max(syG, e.feetWorldY - drop);
      }
    }
    for (uint32_t siW : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& eW = shelfEmpPool[siW];
      if (!eW.inited)
        continue;
      if (eW.staffTallFallKnockdownPending) {
        eW.staffTallFallKnockdownPending = 0;
        applyStaffTallFallRagdoll(eW.residentKey, eW);
      }
    }
    for (uint32_t siW : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& eW = shelfEmpPool[siW];
      if (!eW.inited)
        continue;
      staffNpcUpdateAirFallLoco(eW, dt, playerFeetForStaffSupport);
    }
    for (uint32_t siW : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& e = shelfEmpPool[siW];
      if (!e.inited)
        continue;
      if (e.nightPhase != 2 || e.meleeState != 0)
        continue;
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
      if (staffNpcShouldHoldShrekDancePose(e, playerFeetForStaffSupport)) {
        e.chaseLedgeClimbRem = -1.f;
        e.chaseLedgeClimbTotalDur = 0.f;
        e.staffMantelAnimPhaseSpanSec = 0.f;
        e.staffMantelRunnerChase = 0;
        continue;
      }
#endif
      const glm::vec2 toP = pXZ - e.posXZ;
      const float distPXZ = glm::length(toP);
      if (distPXZ > kStaffChaseClimbMaxHorizToPlayer) {
        e.chaseLedgeClimbRem = -1.f;
        e.chaseLedgeClimbTotalDur = 0.f;
        e.staffMantelAnimPhaseSpanSec = 0.f;
        e.staffMantelRunnerChase = 0;
        continue;
      }
      // updateShelfEmployees runs before advanceLedgeClimb: use mantle landing feet so chasers target the shelf
      // tier you’re pulling onto, not one tick of lag behind the camera lerp.
      const float playerFeetNow = camPos.y - eyeHeight;
      const float playerFeetGoal = (ledgeClimbT >= 0.f)   ? (ledgeClimbEndCam.y - eyeHeight)
                                   : ladderClimbActive ? (ladderClimbEndCam.y - eyeHeight)
                                                       : playerFeetNow;
      const float playerFeet = std::max(playerFeetNow, playerFeetGoal);
      if (playerFeet <= e.feetWorldY + kStaffChaseClimbPlayerFeetMinAbove) {
        e.chaseLedgeClimbRem = -1.f;
        e.chaseLedgeClimbTotalDur = 0.f;
        e.staffMantelAnimPhaseSpanSec = 0.f;
        e.staffMantelRunnerChase = 0;
        continue;
      }
      const float probeFeet =
          std::min(playerFeet + 0.72f, e.feetWorldY + kStaffChaseClimbProbeMaxAboveFeetM);
      // Deck under staff (aisle) is often floor; deck under player is the shelf you’re on — climb toward that.
      const float deckAtStaff = staffChaseClimbSupportY(e.posXZ.x, e.posXZ.y, probeFeet);
      const float deckAtPlayer = staffChaseClimbSupportY(pXZ.x, pXZ.y, probeFeet);
      const float targetFull =
          std::max(deckAtStaff, std::min(std::max(deckAtPlayer, deckAtStaff), playerFeet + 0.11f));
      float rise = targetFull - e.feetWorldY;
      if (rise < 0.055f) {
        e.chaseLedgeClimbRem = -1.f;
        e.chaseLedgeClimbTotalDur = 0.f;
        e.staffMantelAnimPhaseSpanSec = 0.f;
        e.staffMantelRunnerChase = 0;
        continue;
      }
      // One mantel per clip: at most one shelf tier (no multi-level snap in a single pull-up).
      rise = std::min(rise, kStaffChaseLedgeClimbMaxSingleStepRiseM);
      const float targetY = e.feetWorldY + rise;
      const float horizSp = glm::length(e.velXZ);
      const float runGauge = std::max(horizSp, e.lastHorizSpeed);
      const glm::vec2 toN = distPXZ > 1e-4f ? toP * (1.f / distPXZ) : glm::vec2(0.f, 1.f);
      const float vToward =
          horizSp > 0.07f ? glm::dot(e.velXZ * (1.f / horizSp), toN) : 0.f;
      const bool runnerApproach =
          e.nightPhase == 2 && runGauge >= kStaffChaseRunnerGrabMinHorizSpeed &&
          vToward >= kStaffChaseRunnerVelTowardPlayerMin * 0.85f;
      const bool playerPullingUp = ledgeClimbT >= 0.f || ladderClimbActive;
      const bool chaseVertUrgent =
          playerFeet > e.feetWorldY + 0.42f && distPXZ < 44.f;
      const bool grabRelaxedGate = playerPullingUp || chaseVertUrgent;
      const float syGrab = staffNpcFootSupportY(e, playerFeetForStaffSupport);
      const bool grabGrounded = staffNpcIsGroundedLikePlayer(e, syGrab);
      const bool grabVelOk =
          grabRelaxedGate ? (e.staffVelY > -6.1f) : (e.staffVelY > kStaffChaseLedgeGrabMaxStartVelY);
      const float multiCap =
          (runnerApproach ? kStaffChaseRunnerLedgeGrabMultiSampleMaxM : kStaffChaseLedgeGrabMultiSampleMaxM) *
          (grabRelaxedGate ? kStaffChaseRelaxedGrabMultiSampleMul : 1.f);
      const bool grabAlignedLoose = grabRelaxedGate && staffChaseLedgeGrabMultiSampleSupport(
                                      e, targetY, toP, distPXZ, multiCap);
      const bool grabAlignedStrict =
          runnerApproach ? staffChaseLedgeGrabAlignedRunner(e, targetY, toP, distPXZ)
                         : staffChaseLedgeGrabAligned(e, targetY, toP, distPXZ);
      const bool grabAligned = grabAlignedStrict || grabAlignedLoose;
      const bool grabOrientOk =
          grabRelaxedGate ||
          (runnerApproach ? staffChaseLedgeGrabOrientRunnerOk(e, toP, distPXZ, vToward)
                          : staffChaseLedgeGrabOrientOk(e, toP, distPXZ));
      const bool canStartMantel =
          rise >= 0.055f && e.staffChaseMantelCooldownRem <= 0.f && grabGrounded && grabVelOk &&
          grabAligned && grabOrientOk;
      if (e.chaseLedgeClimbRem <= 0.f && canStartMantel) {
        e.staffMantelRunnerChase = runnerApproach ? 1u : 0u;
        e.chaseLedgeClimbY0 = e.feetWorldY;
        e.chaseLedgeClimbY1 = targetY;
        e.staffLastMantelTargetY = targetY;
        const float durSec = staffChaseLedgePullUpDurationSec();
        e.chaseLedgeClimbTotalDur = durSec;
        e.chaseLedgeClimbRem = durSec;
        e.staffMantelAnimPhaseSpanSec = 0.f;
        if (avClipLedgeClimb >= 0 && static_cast<size_t>(avClipLedgeClimb) < staffRig.clips.size()) {
          const double durLc = staff_skin::clipDuration(staffRig, avClipLedgeClimb);
          if (rise < kStaffChaseLedgeClimbMinRiseM * 1.65f)
            e.staffMantelAnimPhaseSpanSec =
                std::max(0.35f, std::min(static_cast<float>(durLc), rise * 11.f + 0.38f));
        }
      }
      if (e.chaseLedgeClimbRem > 0.f) {
        const float td = e.chaseLedgeClimbTotalDur > 1e-4f ? e.chaseLedgeClimbTotalDur
                                                             : kStaffChaseLedgeClimbDurationS;
        const float uPre = glm::clamp(1.f - e.chaseLedgeClimbRem / td, 0.f, 1.f);
        if (uPre < kStaffChaseLedgeGrabAbortEarlyU &&
            !staffChaseHasSupportNearHeight(e.posXZ.x, e.posXZ.y, e.chaseLedgeClimbY1)) {
          e.feetWorldY = e.chaseLedgeClimbY0;
          e.staffVelY = 0.f;
          e.chaseLedgeClimbRem = -1.f;
          e.chaseLedgeClimbTotalDur = 0.f;
          e.staffMantelAnimPhaseSpanSec = 0.f;
          e.staffMantelRunnerChase = 0;
        } else {
          const float sProgDrift = playerLedgeGrabHeightS(uPre);
          const float driftMul = glm::clamp(1.f - sProgDrift * 1.06f, 0.22f, 1.f);
          const float driftRun = e.staffMantelRunnerChase != 0 ? kStaffChaseRunnerDriftMul : 1.f;
          if (distPXZ > 0.18f)
            e.posXZ += (toP / distPXZ) *
                       std::min(distPXZ, dt * kStaffChaseClimbDriftToPlayerMps * driftMul * driftRun);
          e.chaseLedgeClimbRem = std::max(0.f, e.chaseLedgeClimbRem - dt);
          const float uFeet = glm::clamp(1.f - e.chaseLedgeClimbRem / td, 0.f, 1.f);
          e.feetWorldY = staffChaseMantelFeetWorldY(e.chaseLedgeClimbY0, e.chaseLedgeClimbY1, uFeet);
          if (e.chaseLedgeClimbRem <= 0.f) {
            e.chaseLedgeClimbRem = -1.f;
            e.chaseLedgeClimbTotalDur = 0.f;
            e.staffMantelAnimPhaseSpanSec = 0.f;
            e.staffMantelRunnerChase = 0;
            float coolMul = 1.f;
            if (playerFeet > e.feetWorldY + kStaffChaseMantelChainPlayerFeetGapM)
              coolMul = kStaffChaseMantelCooldownChainMul;
            e.staffChaseMantelCooldownRem = kStaffChaseMantelCooldownSec * coolMul;
          }
        }
      }
      const float headTop = e.feetWorldY + kEmployeeVisualHeight * e.bodyScale.y;
      if (headTop > kCeilingY - 0.18f) {
        e.feetWorldY = kCeilingY - 0.18f - kEmployeeVisualHeight * e.bodyScale.y;
        e.staffVelY = std::min(0.f, e.staffVelY);
      }
      // Post-mantel: cap support snap only during cooldown — unless they fell off the tier (then re-climb allowed).
      if (e.chaseLedgeClimbRem <= 0.f && e.staffChaseMantelCooldownRem > 0.f) {
        const float syRaw = staffNpcFootSupportY(e, playerFeetForStaffSupport);
        const bool droppedFarBelowDeck =
            e.feetWorldY < e.staffLastMantelTargetY - kStaffPostMantelFallOffDropM;
        const bool justSteppedOffLip = e.feetWorldY < e.staffLastMantelTargetY - 0.055f &&
                                       e.staffVelY < kStaffPostMantelCancelSnapFallVelY;
        if (droppedFarBelowDeck || justSteppedOffLip) {
          e.staffChaseMantelCooldownRem = 0.f;
          e.staffLastMantelTargetY = syRaw;
        } else {
          e.feetWorldY = std::min(syRaw, e.staffLastMantelTargetY + kStaffChasePostMantelSnapSlopM);
          e.staffVelY = 0.f;
        }
      }
    }
    for (uint32_t siW : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& e = shelfEmpPool[siW];
      if (!e.inited)
        continue;
      if (e.meleeState >= 2) {
        e.stuckRefXZ = e.posXZ;
        e.stuckTimer = 0.f;
        continue;
      }
      if (e.nightPhase == 1) {
        e.stuckRefXZ = e.posXZ;
        e.stuckTimer = 0.f;
        continue;
      }
      const float moved = glm::length(e.posXZ - e.stuckRefXZ);
      if (moved >= kShelfEmpStuckMoveEps) {
        e.stuckRefXZ = e.posXZ;
        e.stuckTimer = 0.f;
      } else {
        e.stuckTimer += dt;
        const float stuckNeed =
            (!day && e.nightPhase == 2) ? kShelfEmpStuckWindowNightChaseS : kShelfEmpStuckWindowS;
        if (e.stuckTimer >= stuckNeed) {
          if (e.nightPhase == 3) {
            glm::vec2 toL = e.nightLastKnownPlayerXZ - e.posXZ;
            const float len = glm::length(toL);
            if (len > 0.35f) {
              glm::vec2 side(-toL.y, toL.x);
              side *= 1.15f / std::max(glm::length(side), 1e-4f);
              const int ka = static_cast<int>(static_cast<uint32_t>(e.residentKey >> 32));
              const int kb = static_cast<int>(static_cast<uint32_t>(e.residentKey & 0xffffffffull));
              const uint32_t h = scp3008ShelfHash(ka, kb, static_cast<int>(e.wanderSalt ^ 0x1E3517A8u));
              const float sign = (h & 1u) ? 1.f : -1.f;
              e.nightLastKnownPlayerXZ += side * sign;
            }
          } else {
            shelfEmpPickWanderStoreWide(e, e.residentKey, pXZ);
            shelfEmpEnsureWanderTargetClear(e, e.residentKey, false, pXZ);
          }
          if (e.nightPhase == 2)
            e.chaseUnstuckTimer = !day ? kShelfEmpChaseUnstuckNavSNight : kShelfEmpChaseUnstuckNavS;
          e.stuckRefXZ = e.posXZ;
          e.stuckTimer = 0.f;
        }
      }
    }
    for (uint32_t siW : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& e = shelfEmpPool[siW];
      if (!e.inited)
        continue;
      if (e.meleeState == 1) {
        const int curAtkClip = (e.meleeAttackPick == 1 && staffClipMeleeKick >= 0)
                                   ? staffClipMeleeKick : staffClipMeleePunch;
        const double atkDur = curAtkClip >= 0 ? staff_skin::clipDuration(staffRig, curAtkClip) : 1.0;
        const double prevPhase = e.meleePhaseSec;
        e.meleePhaseSec += static_cast<double>(dt);
        if (atkDur > 1e-6 && std::floor(prevPhase / atkDur) < std::floor(e.meleePhaseSec / atkDur)) {
          e.meleeAttackPick = static_cast<uint8_t>((e.meleeAttackPick + 1) % 2);
          if (e.meleeAttackPick == 1 && staffClipMeleeKick < 0)
            e.meleeAttackPick = 0;
          if (e.meleeAttackPick == 0 && staffClipMeleePunch < 0)
            e.meleeAttackPick = 1;
          e.meleePhaseSec = 0.0;
        }
        continue;
      }
      if (e.meleeState == 4) {
        if (staffClipShoveHair < 0) {
          e.meleeState = staffClipMeleeFall >= 0 ? 2 : 0;
          e.meleePhaseSec = 0.0;
          e.meleeAnimBlend = 1.f;
          e.shovePlayerPushDurSec = 0.f;
          if (e.meleeState == 2) {
            e.meleeKnockdownFeetAnchorY = e.feetWorldY;
            const glm::vec2 toP = pXZ - e.posXZ;
            if (glm::dot(toP, toP) > 1e-8f)
              e.yaw = std::atan2(toP.x, toP.y);
          }
          continue;
        }
        const double pushD = std::max(static_cast<double>(e.shovePlayerPushDurSec), 1e-4);
        const double dHair = staff_skin::clipDuration(staffRig, staffClipShoveHair);
        e.meleePhaseSec += static_cast<double>(dt) * (dHair / pushD);
        if (e.meleePhaseSec >= dHair) {
          if (staffClipMeleeFall >= 0) {
            e.meleeAnimFromClip = staffClipShoveHair;
            e.meleeAnimFromPhase = dHair;
            e.meleeAnimFromLoop = 0;
            e.meleeAnimBlend = 1.f;
            const glm::vec2 toP = pXZ - e.posXZ;
            if (glm::dot(toP, toP) > 1e-8f)
              e.yaw = std::atan2(toP.x, toP.y);
            e.meleeKnockdownFeetAnchorY = e.feetWorldY;
            e.meleeState = 2;
            e.meleePhaseSec = 0.0;
            e.shovePlayerPushDurSec = 0.f;
          } else {
            e.meleeKnockdownFeetAnchorY = kGroundY;
            e.meleeState = 0;
            e.meleePhaseSec = 0.0;
            e.meleeAnimBlend = 1.f;
            e.shovePlayerPushDurSec = 0.f;
            e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
          }
        }
        continue;
      }
      if (e.meleeState == 2) {
        if (playerPushAnimRemain <= 1e-3f)
          e.meleePhaseSec += static_cast<double>(dt);
        if (staffClipMeleeFall < 0) {
          if (e.staffDead) {
            e.meleeState = 2;
            e.meleePhaseSec = 0.0;
            e.shovePlayerPushDurSec = 0.f;
          } else {
            e.meleeKnockdownFeetAnchorY = kGroundY;
            e.meleeState = 0;
            e.meleePhaseSec = 0.0;
            e.meleeAnimBlend = 1.f;
            e.shovePlayerPushDurSec = 0.f;
            e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
          }
          continue;
        }
        const double dFall = staff_skin::clipDuration(staffRig, staffClipMeleeFall);
        if (e.staffDead) {
          if (dFall > 1e-6)
            e.meleePhaseSec = std::min(e.meleePhaseSec, dFall);
          else
            e.meleePhaseSec = 0.0;
          continue;
        }
        if (e.meleePhaseSec >= dFall) {
          if (staffClipMeleeStand >= 0) {
            e.meleeAnimFromClip = staffClipMeleeFall;
            e.meleeAnimFromPhase = dFall;
            e.meleeAnimFromLoop = 0;
            e.meleeAnimBlend = 0.f;
            const glm::vec2 toP = pXZ - e.posXZ;
            if (glm::dot(toP, toP) > 1e-8f)
              e.yaw = std::atan2(toP.x, toP.y);
            e.meleeState = 3;
            e.meleePhaseSec = 0.0;
          } else {
            e.meleeKnockdownFeetAnchorY = kGroundY;
            e.meleeState = 0;
            e.meleePhaseSec = 0.0;
            e.meleeAnimBlend = 1.f;
            e.shovePlayerPushDurSec = 0.f;
            e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
          }
        }
        continue;
      }
      if (e.meleeState == 3) {
        e.meleePhaseSec += static_cast<double>(dt);
        if (staffClipMeleeStand < 0) {
          e.meleeKnockdownFeetAnchorY = kGroundY;
          e.meleeState = 0;
          e.meleePhaseSec = 0.0;
          e.meleeAnimBlend = 1.f;
          e.shovePlayerPushDurSec = 0.f;
          e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
          continue;
        }
        const double dUp = staff_skin::clipDuration(staffRig, staffClipMeleeStand);
        if (e.meleePhaseSec >= dUp) {
          e.meleeKnockdownFeetAnchorY = kGroundY;
          e.meleeState = 0;
          e.meleePhaseSec = 0.0;
          e.meleeAnimBlend = 1.f;
          e.shovePlayerPushDurSec = 0.f;
          e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
        }
      }
    }
    if (staffSkinnedActive && playerHealth > 0.f && !staffSuppressMeleeNearShrekThisFrame) {
      auto applyStaffHitToPlayer = [&](float dmg) {
        if (playerStaffMeleeInvulnRem > 0.f || playerHealth <= 0.f)
          return;
        playerHealth = std::max(0.f, playerHealth - dmg);
        playerStaffMeleeInvulnRem = kStaffMeleePlayerInvulnSec;
        playerScreenDamagePulse =
            std::max(playerScreenDamagePulse,
                     glm::clamp(dmg / std::max(1e-4f, kPlayerScreenDamagePulseRefDmg), 0.14f, 1.f));
        refreshWindowTitleWithHealth();
        audioPlayStaffMeleeImpact();
        if (playerHealth <= 0.f && !playerDeathActive)
          beginPlayerDeath();
      };
      const AABB pBox = playerCollisionBox();
      if ((staffClipMeleePunch >= 0 || staffClipMeleeKick >= 0) && playerStaffMeleeInvulnRem <= 0.f) {
        for (uint32_t siW : shelfEmpActiveSlots) {
          ShelfEmployeeNpc& e = shelfEmpPool[siW];
          if (playerStaffMeleeInvulnRem > 0.f)
            break;
          if (!e.inited || e.meleeState != 1)
            continue;
          const int atkClip = (e.meleeAttackPick == 1 && staffClipMeleeKick >= 0) ? staffClipMeleeKick
                                                                                   : staffClipMeleePunch;
          if (atkClip < 0)
            continue;
          const double dP = staff_skin::clipDuration(staffRig, atkClip);
          if (dP <= 1e-6)
            continue;
          const double ph = std::fmod(e.meleePhaseSec, dP);
          const double u = ph / dP;
          if (u < kStaffMeleeDamagePhaseU0 || u > kStaffMeleeDamagePhaseU1)
            continue;
          if (!aabbOverlap(pBox, staffNpcMeleeDamageHitbox(e)))
            continue;
          applyStaffHitToPlayer(kStaffMeleePlayerDamage);
          break;
        }
      }
      if (playerStaffMeleeInvulnRem <= 0.f && playerHealth > 0.f) {
        for (uint32_t siW : shelfEmpActiveSlots) {
          ShelfEmployeeNpc& e = shelfEmpPool[siW];
          if (playerStaffMeleeInvulnRem > 0.f)
            break;
          if (!e.inited || e.meleeState >= 2)
            continue;
          const bool hostile = (!day && e.nightPhase == 2) || (day && e.staffPushAggro);
          if (!hostile)
            continue;
          const float distHit = glm::length(pXZ - e.posXZ);
          const bool impactLike =
              e.meleeState == 1 || e.lastHorizSpeed >= kStaffContactHitMinHorizSpeed ||
              distHit <= kStaffMeleePunchHoldRange + 0.45f;
          if (!impactLike)
            continue;
          if (!aabbOverlap(pBox, staffNpcMeleeDamageHitbox(e)))
            continue;
          applyStaffHitToPlayer(kStaffContactPlayerDamage);
          break;
        }
      }
    }
    int staffFootstepsPlayed = 0;
    for (uint32_t siW : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& e = shelfEmpPool[siW];
      if (!e.inited)
        continue;
      if (!e.staffFootstepHavePrev) {
        e.staffFootstepPrevXZ = e.posXZ;
        e.staffFootstepHavePrev = true;
        e.staffFootstepAccum = 0.f;
        continue;
      }
      const glm::vec2 dMoved = e.posXZ - e.staffFootstepPrevXZ;
      e.staffFootstepPrevXZ = e.posXZ;
      const bool canStep = e.meleeState == 0 && e.chaseLedgeClimbRem <= 0.f;
      if (!canStep || e.lastHorizSpeed < kStaffNpcFootstepMinMoveSpeed) {
        e.staffFootstepAccum = 0.f;
        continue;
      }
      const float denom =
          std::max(kShelfEmpNightChaseSpeed * 0.92f - kShelfEmpWalkSpeed * 0.75f, 0.08f);
      const float runT = glm::clamp(
          (e.lastHorizSpeed - kShelfEmpWalkSpeed * 0.75f) / denom, 0.f, 1.f);
      const float strideM = glm::mix(kStaffNpcFootstepStrideWalkM, kStaffNpcFootstepStrideRunM, runT);
      e.staffFootstepAccum += glm::length(dMoved) / std::max(strideM, 0.1f);
      const glm::vec2 toP = pXZ - e.posXZ;
      const float distH = glm::length(toP);
      const bool nightPursuitSteps = !day && e.nightPhase == 2;
      const float hearR =
          kStaffNpcFootstepHearRadiusM *
          (nightPursuitSteps ? kStaffNightPursuitFootstepHearRadiusMul : 1.f);
      const float distMul = glm::clamp(1.f - distH / std::max(hearR, 1.f), 0.f, 1.f);
      const float speedMul =
          glm::clamp(0.34f + e.lastHorizSpeed / std::max(kShelfEmpWalkSpeed * 2.2f, 0.05f), 0.22f, 1.12f);
      float volMul = kStaffNpcFootstepBaseVolMul * distMul * speedMul;
      if (nightPursuitSteps)
        volMul *= kStaffNightPursuitFootstepVolMul;
      while (e.staffFootstepAccum >= 1.f && staffFootstepsPlayed < kStaffNpcFootstepsMaxPerFrame) {
        e.staffFootstepAccum -= 1.f;
        audioPlayFootstep(false, volMul);
        staffFootstepsPlayed++;
      }
    }
    for (size_t ai = 0; ai < shelfEmpActiveSlots.size();) {
      const uint32_t si = shelfEmpActiveSlots[ai];
      ShelfEmployeeNpc& ex = shelfEmpPool[si];
      if (!ex.inited) {
        ++ai;
        continue;
      }
      const glm::vec2 d = pXZ - ex.posXZ;
      if (glm::dot(d, d) > pruneSq) {
        const uint64_t rk = ex.residentKey;
        shelfEmpKeyToSlot.erase(rk);
        shelfEmpActiveSlots[ai] = shelfEmpActiveSlots.back();
        shelfEmpActiveSlots.pop_back();
        shelfEmpPool[si] = ShelfEmployeeNpc{};
        shelfEmpFreeSlots.push_back(si);
      } else
        ++ai;
    }
  }

  void separateShelfEmployeesFromEachOther() {
    const glm::vec2 pXZ(camPos.x, camPos.z);
    const float sepR2 = kStaffSepMaxDistFromPlayer * kStaffSepMaxDistFromPlayer;
    shelfSepEmpScratch.clear();
    if (shelfSepEmpScratch.capacity() < shelfEmpActiveSlots.size())
      shelfSepEmpScratch.reserve(shelfEmpActiveSlots.size());
    for (uint32_t si : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& se = shelfEmpPool[si];
      if (!se.inited)
        continue;
      const glm::vec2 d = se.posXZ - pXZ;
      if (glm::dot(d, d) > sepR2)
        continue;
      shelfSepEmpScratch.push_back(&se);
    }
    auto& emps = shelfSepEmpScratch;
    if (emps.size() < 2)
      return;
    if (shelfSepFootRScratch.size() < emps.size())
      shelfSepFootRScratch.resize(emps.size());
    for (size_t i = 0; i < emps.size(); ++i)
      shelfSepFootRScratch[i] = staffNpcFootprintRadiusXZ(emps[i]->bodyScale);
    // XZ: skip pair when feet farther than sum of footprint radii + slop (scales with bodyScale).
    constexpr float kStaffPairSepSlopM = 0.32f;
    for (size_t i = 0; i < emps.size(); ++i) {
      ShelfEmployeeNpc& a = *emps[i];
      for (size_t j = i + 1; j < emps.size(); ++j) {
        ShelfEmployeeNpc& b = *emps[j];
        const glm::vec2 dXZ = a.posXZ - b.posXZ;
        const float pairThresh = shelfSepFootRScratch[i] + shelfSepFootRScratch[j] + kStaffPairSepSlopM;
        if (glm::dot(dXZ, dXZ) > pairThresh * pairThresh)
          continue;
        if (a.meleeState >= 2 || b.meleeState >= 2)
          continue;
        const AABB ba =
            staffNpcWorldHitbox(a.posXZ.x, a.posXZ.y, a.yaw, a.feetWorldY, a.bodyScale);
        const AABB bb =
            staffNpcWorldHitbox(b.posXZ.x, b.posXZ.y, b.yaw, b.feetWorldY, b.bodyScale);
        if (!aabbOverlap(ba, bb))
          continue;
        // Off-screen pairs that are idle and not walking into each other: skip micro-separation
        // (treat as merged / resting). Still resolve when either moves, or near the player.
        constexpr float kStaffSepNearPlayerSq = 38.f * 38.f;
        constexpr float kStaffSepApproachDotThr = 0.055f;
        constexpr float kStaffSepBusySpeedSq = 0.11f * 0.11f;
        const bool nearCam =
            glm::dot(a.posXZ - pXZ, a.posXZ - pXZ) < kStaffSepNearPlayerSq ||
            glm::dot(b.posXZ - pXZ, b.posXZ - pXZ) < kStaffSepNearPlayerSq;
        if (!nearCam && glm::dot(a.velXZ, a.velXZ) < kStaffSepBusySpeedSq &&
            glm::dot(b.velXZ, b.velXZ) < kStaffSepBusySpeedSq) {
          const glm::vec2 ab = b.posXZ - a.posXZ;
          const float abLen2 = glm::dot(ab, ab);
          if (abLen2 > 1e-6f) {
            const glm::vec2 abn = ab * (1.f / std::sqrt(abLen2));
            const bool aTowardB = glm::dot(a.velXZ, abn) > kStaffSepApproachDotThr;
            const bool bTowardA = glm::dot(b.velXZ, -abn) > kStaffSepApproachDotThr;
            if (!aTowardB && !bTowardA)
              continue;
          } else
            continue;
        }
        const float ax = 0.5f * (ba.min.x + ba.max.x);
        const float az = 0.5f * (ba.min.z + ba.max.z);
        const float bx = 0.5f * (bb.min.x + bb.max.x);
        const float bz = 0.5f * (bb.min.z + bb.max.z);
        float sx = ax - bx;
        float sz = az - bz;
        float len = std::sqrt(sx * sx + sz * sz);
        if (len < 1e-5f) {
          sx = 1.f;
          sz = 0.f;
          len = 1.f;
        } else {
          sx /= len;
          sz /= len;
        }
        const float ovx = std::min(ba.max.x, bb.max.x) - std::max(ba.min.x, bb.min.x);
        const float ovz = std::min(ba.max.z, bb.max.z) - std::max(ba.min.z, bb.min.z);
        const float pen = std::max(0.f, std::min(ovx, ovz));
        const float push = 0.5f * pen + 0.05f;
        a.posXZ.x += sx * push;
        a.posXZ.y += sz * push;
        b.posXZ.x -= sx * push;
        b.posXZ.y -= sz * push;
      }
    }
  }

  bool nudgeShelfEmployeesFromPlayer() {
    bool any = false;
    const AABB pBox = playerCollisionBox();
    const glm::vec2 pc(camPos.x, camPos.z);
    for (uint32_t si : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& e = shelfEmpPool[si];
      if (!e.inited)
        continue;
      if (e.meleeState >= 2)
        continue;
      const glm::vec2 d = e.posXZ - pc;
      const float cullR = kPlayerHalfXZ + staffNpcFootprintRadiusXZ(e.bodyScale) + kStaffPlayerCollisionPadM;
      if (glm::dot(d, d) > cullR * cullR)
        continue;
      AABB sBox = staffNpcWorldHitbox(e.posXZ.x, e.posXZ.y, e.yaw, e.feetWorldY, e.bodyScale);
      if (!aabbOverlap(pBox, sBox))
        continue;
      const float px = 0.5f * (pBox.min.x + pBox.max.x);
      const float pz = 0.5f * (pBox.min.z + pBox.max.z);
      const float sx = 0.5f * (sBox.min.x + sBox.max.x);
      const float sz = 0.5f * (sBox.min.z + sBox.max.z);
      const float ovx = std::min(pBox.max.x, sBox.max.x) - std::max(pBox.min.x, sBox.min.x);
      const float ovz = std::min(pBox.max.z, sBox.max.z) - std::max(pBox.min.z, sBox.min.z);
      float dx = sx - px;
      float dz = sz - pz;
      float len = std::sqrt(dx * dx + dz * dz);
      if (len < 1e-5f) {
        dx = 1.f;
        dz = 0.f;
        len = 1.f;
      } else {
        dx /= len;
        dz /= len;
      }
      const float move = (ovx < ovz ? ovx : ovz) * 0.55f + 0.06f;
      if (move > 0.f) {
        e.posXZ.x += dx * move;
        e.posXZ.y += dz * move;
        any = true;
      }
    }
    return any;
  }

  void syncInputGrab() {
    if (showControlsOverlay) {
      SDL_SetRelativeMouseMode(SDL_FALSE);
      SDL_ShowCursor(SDL_ENABLE);
      applyYellowMenuCursorIfNeeded();
      if (std::getenv("VULKAN_GAME_MOUSE_CAPTURE"))
        SDL_CaptureMouse(SDL_FALSE);
      return;
    }
    if (inLoadingScreen) {
      SDL_SetRelativeMouseMode(SDL_FALSE);
      SDL_ShowCursor(SDL_DISABLE);
      if (std::getenv("VULKAN_GAME_MOUSE_CAPTURE"))
        SDL_CaptureMouse(SDL_FALSE);
      return;
    }
    if (inIntroSplash) {
      SDL_SetRelativeMouseMode(SDL_FALSE);
      SDL_ShowCursor(SDL_ENABLE);
      applyYellowMenuCursorIfNeeded();
      if (std::getenv("VULKAN_GAME_MOUSE_CAPTURE"))
        SDL_CaptureMouse(SDL_FALSE);
      return;
    }
    if (inTitleMenu || showPauseMenu || showInventoryMenu || playerDeathShowMenu) {
      SDL_SetRelativeMouseMode(SDL_FALSE);
      SDL_ShowCursor(SDL_ENABLE);
      applyYellowMenuCursorIfNeeded();
      if (std::getenv("VULKAN_GAME_MOUSE_CAPTURE"))
        SDL_CaptureMouse(SDL_FALSE);
      return;
    }
    SDL_SetRelativeMouseMode(mouseGrab ? SDL_TRUE : SDL_FALSE);
    SDL_ShowCursor(mouseGrab ? SDL_DISABLE : SDL_ENABLE);
    SDL_SetCursor(SDL_GetDefaultCursor());
    if (std::getenv("VULKAN_GAME_MOUSE_CAPTURE"))
      SDL_CaptureMouse(mouseGrab ? SDL_TRUE : SDL_FALSE);
  }

  void initWindow() {
    if (!std::getenv("VULKAN_GAME_MOUSE_WARP_OFF"))
      SDL_SetHint(SDL_HINT_MOUSE_RELATIVE_MODE_WARP, "1");
    SDL_SetHint(SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS, "0");
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0)
      throw std::runtime_error(std::string("SDL_Init: ") + SDL_GetError());
    constexpr int kImgWant = IMG_INIT_PNG | IMG_INIT_JPG;
    if ((IMG_Init(kImgWant) & kImgWant) != kImgWant)
      throw std::runtime_error(std::string("IMG_Init (PNG+JPG): ") + IMG_GetError());
    Uint32 winFlags = SDL_WINDOW_VULKAN | SDL_WINDOW_FULLSCREEN_DESKTOP;
    if (std::getenv("VULKAN_GAME_WINDOWED"))
      winFlags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;
    window = SDL_CreateWindow(
        "retro ikea v" VULKAN_GAME_VERSION_STRING
        " — hall — WASD | Shift sprint | crouch | slide | Space jump / ledge hang | Shift+wall run | "
        "wall+Space jump/climb | speed vault",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, winW, winH, winFlags);
    if (!window)
      throw std::runtime_error(std::string("SDL_CreateWindow: ") + SDL_GetError());
    refreshWindowTitleWithHealth();
    if (std::getenv("VULKAN_GAME_WINDOWED"))
      std::cerr << "[vulkan_game] Windowed mode (VULKAN_GAME_WINDOWED). Omit for fullscreen desktop.\n";
    SDL_Vulkan_GetDrawableSize(window, &winW, &winH);
    SDL_RaiseWindow(window);
    migrateLegacySaveIfNeeded();
    refreshTitleMenuContinueState();
    syncInputGrab();
  }

  void initVulkan() {
    loadGamePerfFromEnv();
    gPerfTrace.init();
    createInstance();
    if (!SDL_Vulkan_CreateSurface(window, instance, &surface))
      throw std::runtime_error("SDL_Vulkan_CreateSurface");
    pickPhysicalDevice();
    createLogicalDevice();
    createPipelineCache();
    createSwapchain();
    createImageViews();
    createSceneColorResources();
    createDepthResources();
    createRenderPass();
    createDescriptorAndPipelineLayout();
    createPostProcessResources();
    createGraphicsPipeline();
    createPostPipeline();
    createFramebuffers();
    createCommandPool();
    createSceneTextureResources();
    createSignTextureResources();
    createShelfRackTextureResources();
    createCrateTextureResources();
    createPalletTextureResources();
    createDeliMetalTextureResources();
    createDeliBaseTextureResources();
    createExtraTextureResources();
    createHudFontTextureResources();
    createTitleIkeaLogoTextureResources();
    createVertexBuffers();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createPostDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
    uboCachedExtraBlend = 0;
    if (const char* eb = std::getenv("VULKAN_GAME_EXTRA_BLEND"))
      uboCachedExtraBlend = std::clamp(std::atoi(eb), 0, 255);
    uboCachedStaffTexBlend = 255;
    if (const char* sb = std::getenv("VULKAN_GAME_STAFF_TEX_BLEND"))
      uboCachedStaffTexBlend = std::clamp(std::atoi(sb), 0, 255);
  }

  void createInstance() {
    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "retro ikea";
    ai.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    ai.pEngineName = "No Engine";
    ai.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    ai.apiVersion = VK_API_VERSION_1_2;

    unsigned int sdlExtCount = 0;
    if (!SDL_Vulkan_GetInstanceExtensions(nullptr, &sdlExtCount, nullptr))
      throw std::runtime_error("SDL_Vulkan_GetInstanceExtensions");
    std::vector<const char*> extNames(sdlExtCount);
    if (!SDL_Vulkan_GetInstanceExtensions(nullptr, &sdlExtCount, extNames.data()))
      throw std::runtime_error("SDL_Vulkan_GetInstanceExtensions");

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &ai;
    ci.enabledExtensionCount = sdlExtCount;
    ci.ppEnabledExtensionNames = extNames.data();
    ci.enabledLayerCount = 0;
    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance), "vkCreateInstance");
  }

  void pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (!count)
      throw std::runtime_error("no Vulkan GPU");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    for (auto d : devices) {
      if (!checkDeviceExtensionSupport(d))
        continue;
      auto q = findQueueFamilies(d, surface);
      auto sw = querySwapchainSupport(d, surface);
      if (q.complete() && !sw.formats.empty() && !sw.presentModes.empty()) {
        physicalDevice = d;
        return;
      }
    }
    throw std::runtime_error("no suitable GPU");
  }

  void createLogicalDevice() {
    auto q = findQueueFamilies(physicalDevice, surface);
    std::set<uint32_t> unique{*q.graphicsFamily, *q.presentFamily};
    std::vector<VkDeviceQueueCreateInfo> qcis;
    float prio = 1.0f;
    for (uint32_t fam : unique) {
      VkDeviceQueueCreateInfo qci{};
      qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      qci.queueFamilyIndex = fam;
      qci.queueCount = 1;
      qci.pQueuePriorities = &prio;
      qcis.push_back(qci);
    }

    VkPhysicalDeviceFeatures feats{};
    const char* ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
    ci.pQueueCreateInfos = qcis.data();
    ci.pEnabledFeatures = &feats;
    ci.enabledExtensionCount = 1;
    ci.ppEnabledExtensionNames = &ext;
    VK_CHECK(vkCreateDevice(physicalDevice, &ci, nullptr, &device), "vkCreateDevice");
    vkGetDeviceQueue(device, *q.graphicsFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, *q.presentFamily, 0, &presentQueue);
  }

  void createPipelineCache() {
    const char* env = std::getenv("VULKAN_GAME_PIPELINE_CACHE");
    pipelineCachePath = (env && env[0]) ? std::string(env) : std::string("vulkan_game_pipeline.cache");
    std::vector<char> initial;
    std::ifstream inf(pipelineCachePath, std::ios::ate | std::ios::binary);
    if (inf) {
      const auto sz = static_cast<size_t>(inf.tellg());
      if (sz > 0) {
        initial.resize(sz);
        inf.seekg(0);
        inf.read(initial.data(), static_cast<std::streamsize>(sz));
      }
    }
    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    if (!initial.empty()) {
      ci.pInitialData = initial.data();
      ci.initialDataSize = initial.size();
    }
    VkResult r = vkCreatePipelineCache(device, &ci, nullptr, &pipelineCache);
    if (r != VK_SUCCESS) {
      ci.pInitialData = nullptr;
      ci.initialDataSize = 0;
      VK_CHECK(vkCreatePipelineCache(device, &ci, nullptr, &pipelineCache), "vkCreatePipelineCache");
    }
  }

  void savePipelineCacheToDisk() {
    if (pipelineCache == VK_NULL_HANDLE || pipelineCachePath.empty())
      return;
    size_t sz = 0;
    VkResult r = vkGetPipelineCacheData(device, pipelineCache, &sz, nullptr);
    if (r != VK_SUCCESS || sz == 0)
      return;
    std::vector<char> data(sz);
    r = vkGetPipelineCacheData(device, pipelineCache, &sz, data.data());
    if (r != VK_SUCCESS)
      return;
    std::ofstream out(pipelineCachePath.c_str(), std::ios::binary);
    out.write(data.data(), static_cast<std::streamsize>(sz));
  }

  void createSwapchain() {
    auto details = querySwapchainSupport(physicalDevice, surface);
    VkSurfaceFormatKHR fmt = chooseSwapSurfaceFormat(details.formats);
    VkPresentModeKHR present = chooseSwapPresentMode(details.presentModes);
    VkExtent2D extent = chooseSwapExtent(details.capabilities, winW, winH);

    // Extra swap images reduce vkAcquireNextImageKHR stalls when the GPU is slightly behind the display.
    const uint32_t minImg = details.capabilities.minImageCount;
    const uint32_t maxImg = details.capabilities.maxImageCount;
    uint32_t want = minImg + 2u;
    uint32_t imageCount = want;
    if (maxImg > 0)
      imageCount = std::clamp(want, minImg, maxImg);
    else
      imageCount = std::max(minImg, want);

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = surface;
    ci.minImageCount = imageCount;
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    auto q = findQueueFamilies(physicalDevice, surface);
    uint32_t fams[] = {*q.graphicsFamily, *q.presentFamily};
    if (*q.graphicsFamily != *q.presentFamily) {
      ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      ci.queueFamilyIndexCount = 2;
      ci.pQueueFamilyIndices = fams;
    } else {
      ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    ci.preTransform = details.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = present;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain), "vkCreateSwapchainKHR");

    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());
    swapchainImageFormat = fmt.format;
    swapchainExtent = extent;
    // Low internal res → chunky PS1-style pixels when nearest-upscaled in post (tunable via gGamePerf).
    {
      const uint32_t pct = static_cast<uint32_t>(std::clamp(gGamePerf.sceneScalePct, 20, 95));
      uint32_t sw = std::max(240u, swapchainExtent.width * pct / 100u);
      uint32_t sh = std::max(136u, swapchainExtent.height * pct / 100u);
      sw &= ~1u;
      sh &= ~1u;
      sceneExtent = {sw, sh};
    }
  }

  void createImageViews() {
    swapchainImageViews.resize(swapchainImages.size());
    for (size_t i = 0; i < swapchainImages.size(); ++i) {
      VkImageViewCreateInfo ci{};
      ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      ci.image = swapchainImages[i];
      ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      ci.format = swapchainImageFormat;
      ci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      ci.subresourceRange.levelCount = 1;
      ci.subresourceRange.layerCount = 1;
      VK_CHECK(vkCreateImageView(device, &ci, nullptr, &swapchainImageViews[i]),
               "vkCreateImageView");
    }
  }

  void createSceneColorResources() {
    sceneColorImages.resize(static_cast<size_t>(kMaxFramesInFlight));
    sceneColorMemories.resize(static_cast<size_t>(kMaxFramesInFlight));
    sceneColorViews.resize(static_cast<size_t>(kMaxFramesInFlight));
    sceneColorWasSampled.assign(static_cast<size_t>(kMaxFramesInFlight), false);
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      VkImageCreateInfo ii{};
      ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      ii.imageType = VK_IMAGE_TYPE_2D;
      ii.extent.width = sceneExtent.width;
      ii.extent.height = sceneExtent.height;
      ii.extent.depth = 1;
      ii.mipLevels = 1;
      ii.arrayLayers = 1;
      ii.format = swapchainImageFormat;
      ii.tiling = VK_IMAGE_TILING_OPTIMAL;
      ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
      ii.samples = VK_SAMPLE_COUNT_1_BIT;
      VK_CHECK(vkCreateImage(device, &ii, nullptr, &sceneColorImages[static_cast<size_t>(i)]),
               "scene color image");

      VkMemoryRequirements req{};
      vkGetImageMemoryRequirements(device, sceneColorImages[static_cast<size_t>(i)], &req);
      VkMemoryAllocateInfo ai{};
      ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      ai.allocationSize = req.size;
      ai.memoryTypeIndex =
          findMemoryType(physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &sceneColorMemories[static_cast<size_t>(i)]),
               "scene color mem");
      vkBindImageMemory(device, sceneColorImages[static_cast<size_t>(i)],
                        sceneColorMemories[static_cast<size_t>(i)], 0);

      VkImageViewCreateInfo vi{};
      vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      vi.image = sceneColorImages[static_cast<size_t>(i)];
      vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vi.format = swapchainImageFormat;
      vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      vi.subresourceRange.levelCount = 1;
      vi.subresourceRange.layerCount = 1;
      VK_CHECK(vkCreateImageView(device, &vi, nullptr, &sceneColorViews[static_cast<size_t>(i)]),
               "scene color view");
    }
    if (sceneRenderSampler == VK_NULL_HANDLE) {
      VkSamplerCreateInfo si{};
      si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      si.magFilter = VK_FILTER_NEAREST;
      si.minFilter = VK_FILTER_NEAREST;
      si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.anisotropyEnable = VK_FALSE;
      si.maxAnisotropy = 1.f;
      si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
      si.unnormalizedCoordinates = VK_FALSE;
      si.compareEnable = VK_FALSE;
      si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      si.minLod = 0.f;
      si.maxLod = 0.f;
      VK_CHECK(vkCreateSampler(device, &si, nullptr, &sceneRenderSampler), "sceneRenderSampler");
    }
  }

  void createDepthResources() {
    const VkFormat depthFormat = findDepthFormat(physicalDevice);
    depthImages.resize(static_cast<size_t>(kMaxFramesInFlight));
    depthMemories.resize(static_cast<size_t>(kMaxFramesInFlight));
    depthViews.resize(static_cast<size_t>(kMaxFramesInFlight));
    depthGpuReady.assign(static_cast<size_t>(kMaxFramesInFlight), false);
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      VkImageCreateInfo ii{};
      ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      ii.imageType = VK_IMAGE_TYPE_2D;
      ii.extent.width = sceneExtent.width;
      ii.extent.height = sceneExtent.height;
      ii.extent.depth = 1;
      ii.mipLevels = 1;
      ii.arrayLayers = 1;
      ii.format = depthFormat;
      ii.tiling = VK_IMAGE_TILING_OPTIMAL;
      ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
      ii.samples = VK_SAMPLE_COUNT_1_BIT;
      VK_CHECK(vkCreateImage(device, &ii, nullptr, &depthImages[static_cast<size_t>(i)]),
               "depth image");

      VkMemoryRequirements req{};
      vkGetImageMemoryRequirements(device, depthImages[static_cast<size_t>(i)], &req);
      VkMemoryAllocateInfo ai{};
      ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      ai.allocationSize = req.size;
      ai.memoryTypeIndex =
          findMemoryType(physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &depthMemories[static_cast<size_t>(i)]),
               "depth mem");
      vkBindImageMemory(device, depthImages[static_cast<size_t>(i)], depthMemories[static_cast<size_t>(i)],
                        0);

      VkImageViewCreateInfo vi{};
      vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      vi.image = depthImages[static_cast<size_t>(i)];
      vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vi.format = depthFormat;
      vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      vi.subresourceRange.levelCount = 1;
      vi.subresourceRange.layerCount = 1;
      VK_CHECK(vkCreateImageView(device, &vi, nullptr, &depthViews[static_cast<size_t>(i)]),
               "depth view");
    }
  }

  void createTextureResourcesFromMemory(const unsigned char* bytes, size_t size, VkImage& outImage,
                                        VkDeviceMemory& outMemory, VkImageView& outView,
                                        VkSampler& outSampler) {
    if (!bytes || size == 0 || size > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::runtime_error("invalid embedded texture bytes");
    SDL_RWops* rw = SDL_RWFromConstMem(bytes, static_cast<int>(size));
    if (!rw)
      throw std::runtime_error(std::string("SDL_RWFromConstMem: ") + SDL_GetError());
    SDL_Surface* loaded = IMG_Load_RW(rw, 1);
    if (!loaded)
      throw std::runtime_error(std::string("IMG_Load_RW: ") + IMG_GetError());
    SDL_Surface* rgba = SDL_ConvertSurfaceFormat(loaded, SDL_PIXELFORMAT_RGBA32, 0);
    SDL_FreeSurface(loaded);
    if (!rgba)
      throw std::runtime_error("SDL_ConvertSurfaceFormat failed");

    const uint32_t texW = static_cast<uint32_t>(rgba->w);
    const uint32_t texH = static_cast<uint32_t>(rgba->h);
    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(texW) * texH * 4;

    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging,
                 stagingMem);
    void* mapped = nullptr;
    vkMapMemory(device, stagingMem, 0, imageSize, 0, &mapped);
    if (SDL_MUSTLOCK(rgba))
      SDL_LockSurface(rgba);
    copySdlRgbaSurfaceRowsToBuffer(rgba, mapped, texW, texH);
    if (SDL_MUSTLOCK(rgba))
      SDL_UnlockSurface(rgba);
    vkUnmapMemory(device, stagingMem);
    SDL_FreeSurface(rgba);

    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.extent = {texW, texH, 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateImage(device, &ii, nullptr, &outImage), "texture image");

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, outImage, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex =
        findMemoryType(physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &outMemory), "texture memory");
    vkBindImageMemory(device, outImage, outMemory, 0);

    transitionImageLayout(device, commandPool, graphicsQueue, outImage, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(device, commandPool, graphicsQueue, staging, outImage, texW, texH);
    transitionImageLayout(device, commandPool, graphicsQueue, outImage,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = outImage;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device, &vi, nullptr, &outView), "texture view");

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.anisotropyEnable = VK_FALSE;
    si.maxAnisotropy = 1.0f;
    si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    si.unnormalizedCoordinates = VK_FALSE;
    si.compareEnable = VK_FALSE;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.minLod = 0.f;
    si.maxLod = 0.f;
    VK_CHECK(vkCreateSampler(device, &si, nullptr, &outSampler), "texture sampler");
  }

  void createTextureResources(const char* path, VkImage& outImage, VkDeviceMemory& outMemory,
                              VkImageView& outView, VkSampler& outSampler) {
    SDL_Surface* loaded = IMG_Load(path);
    if (!loaded)
      throw std::runtime_error(std::string("IMG_Load: ") + IMG_GetError());
    SDL_Surface* rgba = SDL_ConvertSurfaceFormat(loaded, SDL_PIXELFORMAT_RGBA32, 0);
    SDL_FreeSurface(loaded);
    if (!rgba)
      throw std::runtime_error("SDL_ConvertSurfaceFormat failed");

    const uint32_t texW = static_cast<uint32_t>(rgba->w);
    const uint32_t texH = static_cast<uint32_t>(rgba->h);
    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(texW) * texH * 4;

    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging,
                 stagingMem);
    void* mapped = nullptr;
    vkMapMemory(device, stagingMem, 0, imageSize, 0, &mapped);
    if (SDL_MUSTLOCK(rgba))
      SDL_LockSurface(rgba);
    copySdlRgbaSurfaceRowsToBuffer(rgba, mapped, texW, texH);
    if (SDL_MUSTLOCK(rgba))
      SDL_UnlockSurface(rgba);
    vkUnmapMemory(device, stagingMem);
    SDL_FreeSurface(rgba);

    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.extent = {texW, texH, 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateImage(device, &ii, nullptr, &outImage), "texture image");

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, outImage, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex =
        findMemoryType(physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &outMemory), "texture memory");
    vkBindImageMemory(device, outImage, outMemory, 0);

    transitionImageLayout(device, commandPool, graphicsQueue, outImage, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(device, commandPool, graphicsQueue, staging, outImage, texW, texH);
    transitionImageLayout(device, commandPool, graphicsQueue, outImage,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = outImage;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device, &vi, nullptr, &outView), "texture view");

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.anisotropyEnable = VK_FALSE;
    si.maxAnisotropy = 1.0f;
    si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    si.unnormalizedCoordinates = VK_FALSE;
    si.compareEnable = VK_FALSE;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.minLod = 0.f;
    si.maxLod = 0.f;
    VK_CHECK(vkCreateSampler(device, &si, nullptr, &outSampler), "texture sampler");
  }

  // Tries compile-time absolute assets dir first, then cwd-relative assets/ (e.g. running from build/).
  bool tryLoadTextureFile(const char* primaryPath, const char* relativeFallback, VkImage& outImage,
                          VkDeviceMemory& outMemory, VkImageView& outView, VkSampler& outSampler) {
    try {
      createTextureResources(primaryPath, outImage, outMemory, outView, outSampler);
      return true;
    } catch (const std::exception& ex) {
      std::cerr << "[tex] " << primaryPath << ": " << ex.what() << '\n';
    }
    if (relativeFallback && relativeFallback[0] != '\0') {
      try {
        createTextureResources(relativeFallback, outImage, outMemory, outView, outSampler);
        return true;
      } catch (const std::exception& ex) {
        std::cerr << "[tex] " << relativeFallback << ": " << ex.what() << '\n';
      }
    }
    return false;
  }

  void createTextureResourcesFromRgbaLinear(const uint8_t* rgba, uint32_t texW, uint32_t texH,
                                            VkImage& outImage, VkDeviceMemory& outMemory,
                                            VkImageView& outView, VkSampler& outSampler) {
    if (!rgba || texW < 1 || texH < 1 || texW > 16384 || texH > 16384)
      throw std::runtime_error("createTextureResourcesFromRgbaLinear: bad size");
    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(texW) * texH * 4;

    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging,
                 stagingMem);
    void* mapped = nullptr;
    vkMapMemory(device, stagingMem, 0, imageSize, 0, &mapped);
    std::memcpy(mapped, rgba, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingMem);

    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.extent = {texW, texH, 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateImage(device, &ii, nullptr, &outImage), "staff glb texture image");

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, outImage, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex =
        findMemoryType(physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &outMemory), "staff glb texture memory");
    vkBindImageMemory(device, outImage, outMemory, 0);

    transitionImageLayout(device, commandPool, graphicsQueue, outImage, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(device, commandPool, graphicsQueue, staging, outImage, texW, texH);
    transitionImageLayout(device, commandPool, graphicsQueue, outImage,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = outImage;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device, &vi, nullptr, &outView), "staff glb texture view");

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.anisotropyEnable = VK_FALSE;
    si.maxAnisotropy = 1.0f;
    si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    si.unnormalizedCoordinates = VK_FALSE;
    si.compareEnable = VK_FALSE;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.minLod = 0.f;
    si.maxLod = 0.f;
    VK_CHECK(vkCreateSampler(device, &si, nullptr, &outSampler), "staff glb texture sampler");
  }

  void createSceneTextureResources() {
    if (!tryLoadTextureFile(VULKAN_GAME_ASSETS_DIR "/textures/wall_concrete_gray.png",
                            "assets/textures/wall_concrete_gray.png", sceneTextureImage,
                            sceneTextureMemory, sceneTextureView, sceneTextureSampler)) {
      std::cerr << "[tex] scene fallback: solid neutral grey\n";
      createSolidColorTexture2D(148, 148, 150, 255, sceneTextureImage, sceneTextureMemory,
                                sceneTextureView, sceneTextureSampler);
    }
  }

  void createSignTextureResources() {
    if (!tryLoadTextureFile(VULKAN_GAME_ASSETS_DIR "/textures/sign.png", "assets/textures/sign.png",
                            signTextureImage, signTextureMemory, signTextureView, signTextureSampler)) {
      std::cerr << "[tex] sign fallback: solid white\n";
      createSolidColorTexture2D(230, 230, 230, 255, signTextureImage, signTextureMemory,
                                signTextureView, signTextureSampler);
    }
  }

  void createShelfRackTextureResources() {
    if (!tryLoadTextureFile(VULKAN_GAME_ASSETS_DIR "/textures/shelf_rack_metal_osb.jpg",
                            "assets/textures/shelf_rack_metal_osb.jpg", shelfRackTextureImage,
                            shelfRackTextureMemory, shelfRackTextureView, shelfRackTextureSampler)) {
      std::cerr << "[tex] shelf rack fallback: muted steel\n";
      createSolidColorTexture2D(118, 123, 130, 255, shelfRackTextureImage, shelfRackTextureMemory,
                                shelfRackTextureView, shelfRackTextureSampler);
    }
  }

  void createCrateTextureResources() {
    if (!tryLoadTextureFile(VULKAN_GAME_ASSETS_DIR "/textures/crate_cardboard.jpeg",
                            "assets/textures/crate_cardboard.jpeg", crateTextureImage, crateTextureMemory,
                            crateTextureView, crateTextureSampler)) {
      std::cerr << "[tex] crate fallback: cardboard brown\n";
      createSolidColorTexture2D(160, 122, 78, 255, crateTextureImage, crateTextureMemory,
                                crateTextureView, crateTextureSampler);
    }
  }

  void createPalletTextureResources() {
    if (!tryLoadTextureFile(VULKAN_GAME_ASSETS_DIR "/textures/shelf_pallet_wood.png",
                            "assets/textures/shelf_pallet_wood.png", palletTextureImage,
                            palletTextureMemory, palletTextureView, palletTextureSampler)) {
      std::cerr << "[tex] pallet fallback: warm wood\n";
      createSolidColorTexture2D(154, 118, 72, 255, palletTextureImage, palletTextureMemory,
                                palletTextureView, palletTextureSampler);
    }
  }

  void createDeliMetalTextureResources() {
    if (!tryLoadTextureFile(VULKAN_GAME_DELI_METAL_TEXTURE, "assets/textures/deli_metal_brushed.jpg",
                            deliMetalTextureImage, deliMetalTextureMemory, deliMetalTextureView,
                            deliMetalTextureSampler)) {
      std::cerr << "[tex] deli metal fallback: brushed grey\n";
      createSolidColorTexture2D(132, 136, 141, 255, deliMetalTextureImage, deliMetalTextureMemory,
                                deliMetalTextureView, deliMetalTextureSampler);
    }
  }

  void createDeliBaseTextureResources() {
    if (!tryLoadTextureFile(VULKAN_GAME_ASSETS_DIR "/textures/deli_base_white_leather.png",
                            "assets/textures/deli_base_white_leather.png", deliBaseTextureImage,
                            deliBaseTextureMemory, deliBaseTextureView, deliBaseTextureSampler)) {
      std::cerr << "[tex] deli base fallback: off-white\n";
      createSolidColorTexture2D(230, 230, 224, 255, deliBaseTextureImage, deliBaseTextureMemory,
                                deliBaseTextureView, deliBaseTextureSampler);
    }
  }

  void createSolidColorTexture2D(uint8_t r, uint8_t g, uint8_t b, uint8_t a, VkImage& outImage,
                                 VkDeviceMemory& outMemory, VkImageView& outView,
                                 VkSampler& outSampler) {
    constexpr uint32_t texW = 1;
    constexpr uint32_t texH = 1;
    const VkDeviceSize imageSize = 4;
    uint8_t pixel[4] = {r, g, b, a};

    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging,
                 stagingMem);
    void* mapped = nullptr;
    vkMapMemory(device, stagingMem, 0, imageSize, 0, &mapped);
    std::memcpy(mapped, pixel, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingMem);

    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.extent = {texW, texH, 1};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateImage(device, &ii, nullptr, &outImage), "solid texture image");

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, outImage, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex =
        findMemoryType(physicalDevice, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &outMemory), "solid texture memory");
    vkBindImageMemory(device, outImage, outMemory, 0);

    transitionImageLayout(device, commandPool, graphicsQueue, outImage, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(device, commandPool, graphicsQueue, staging, outImage, texW, texH);
    transitionImageLayout(device, commandPool, graphicsQueue, outImage,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = outImage;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_R8G8B8A8_UNORM;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device, &vi, nullptr, &outView), "solid texture view");

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.anisotropyEnable = VK_FALSE;
    si.maxAnisotropy = 1.0f;
    si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    si.unnormalizedCoordinates = VK_FALSE;
    si.compareEnable = VK_FALSE;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.minLod = 0.f;
    si.maxLod = 0.f;
    VK_CHECK(vkCreateSampler(device, &si, nullptr, &outSampler), "solid texture sampler");
  }

  void createExtraTextureResources() {
    auto nukem = [this](ExtraTexSlot& s) {
      if (s.sampler != VK_NULL_HANDLE)
        vkDestroySampler(device, s.sampler, nullptr);
      if (s.view != VK_NULL_HANDLE)
        vkDestroyImageView(device, s.view, nullptr);
      if (s.image != VK_NULL_HANDLE)
        vkDestroyImage(device, s.image, nullptr);
      if (s.memory != VK_NULL_HANDLE)
        vkFreeMemory(device, s.memory, nullptr);
      s = {};
    };
    const std::vector<std::string> paths = parseExtraTexturePathsFromEnv();
    extraTexturesLoadedCount = 0;
    for (uint32_t i = 0; i < kMaxExtraTextures; ++i) {
      ExtraTexSlot& slot = extraTexSlots[i];
      bool loaded = false;
      if (i < paths.size()) {
        try {
          createTextureResources(paths[i].c_str(), slot.image, slot.memory, slot.view, slot.sampler);
          loaded = true;
          ++extraTexturesLoadedCount;
        } catch (const std::exception& ex) {
          std::cerr << "[textures] extra[" << i << "] \"" << paths[i] << "\": " << ex.what() << '\n';
          nukem(slot);
        }
      }
      if (!loaded)
        createSolidColorTexture2D(0, 0, 0, 255, slot.image, slot.memory, slot.view, slot.sampler);
    }
    if (!paths.empty())
      std::cerr << "[textures] loaded " << extraTexturesLoadedCount << " / " << paths.size()
                << " extra image(s); slot count " << kMaxExtraTextures << '\n';
  }

  void createHudFontTextureResources() {
    gHudUiFontReady = false;
    constexpr uint8_t kBlankPx[4] = {0, 0, 0, 0};
    auto ensureFallback = [&]() {
      if (hudFontTextureView == VK_NULL_HANDLE)
        createTextureResourcesFromRgbaLinear(kBlankPx, 1, 1, hudFontTextureImage, hudFontTextureMemory,
                                             hudFontTextureView, hudFontTextureSampler);
    };
#if defined(VULKAN_GAME_UI_FONT_PATH)
    try {
      std::ifstream f(VULKAN_GAME_UI_FONT_PATH, std::ios::binary | std::ios::ate);
      if (!f)
        throw std::runtime_error("open font");
      const std::streamoff sz = f.tellg();
      if (sz <= 0 || sz > 12 * 1024 * 1024)
        throw std::runtime_error("bad font size");
      f.seekg(0);
      std::vector<unsigned char> ttf(static_cast<size_t>(sz));
      if (!f.read(reinterpret_cast<char*>(ttf.data()), sz))
        throw std::runtime_error("read font");
      std::vector<unsigned char> bitmap(static_cast<size_t>(gHudUiFontAtlasW * gHudUiFontAtlasH));
      stbtt_pack_context pc{};
      if (!stbtt_PackBegin(&pc, bitmap.data(), gHudUiFontAtlasW, gHudUiFontAtlasH, 0, 2, nullptr))
        throw std::runtime_error("PackBegin");
      stbtt_PackSetOversampling(&pc, 2, 2);
      if (!stbtt_PackFontRange(&pc, ttf.data(), 0, gHudUiFontSizePx, kHudFontFirstChar, kHudFontCharCount,
                               gHudUiFontPacked))
        throw std::runtime_error("PackFontRange");
      stbtt_PackEnd(&pc);
      stbtt_fontinfo finfo{};
      if (!stbtt_InitFont(&finfo, ttf.data(), stbtt_GetFontOffsetForIndex(ttf.data(), 0)))
        throw std::runtime_error("InitFont");
      int asc = 0, desc = 0, lg = 0;
      stbtt_GetFontVMetrics(&finfo, &asc, &desc, &lg);
      const float scale = stbtt_ScaleForPixelHeight(&finfo, gHudUiFontSizePx);
      gHudUiFontLineSkipPx = static_cast<float>(asc - desc + lg) * scale * 1.06f;
      std::vector<uint8_t> rgba(static_cast<size_t>(gHudUiFontAtlasW * gHudUiFontAtlasH * 4));
      for (size_t i = 0; i < bitmap.size(); ++i) {
        const uint8_t a = bitmap[i];
        rgba[i * 4 + 0] = 255;
        rgba[i * 4 + 1] = 255;
        rgba[i * 4 + 2] = 255;
        rgba[i * 4 + 3] = a;
      }
      createTextureResourcesFromRgbaLinear(rgba.data(), static_cast<uint32_t>(gHudUiFontAtlasW),
                                           static_cast<uint32_t>(gHudUiFontAtlasH), hudFontTextureImage,
                                           hudFontTextureMemory, hudFontTextureView, hudFontTextureSampler);
      vkDestroySampler(device, hudFontTextureSampler, nullptr);
      VkSamplerCreateInfo si{};
      si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      si.magFilter = VK_FILTER_LINEAR;
      si.minFilter = VK_FILTER_LINEAR;
      si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.anisotropyEnable = VK_FALSE;
      si.maxAnisotropy = 1.f;
      si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
      si.unnormalizedCoordinates = VK_FALSE;
      si.compareEnable = VK_FALSE;
      si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      si.minLod = 0.f;
      si.maxLod = 0.f;
      VK_CHECK(vkCreateSampler(device, &si, nullptr, &hudFontTextureSampler), "hudFont sampler");
      gHudUiFontReady = true;
      std::cerr << "[hud] TrueType UI font OK (" VULKAN_GAME_UI_FONT_PATH ")\n";
    } catch (const std::exception& ex) {
      std::cerr << "[hud] UI font disabled (" << ex.what() << ") — using stb_easy_font fallback\n";
      ensureFallback();
    }
#else
    ensureFallback();
#endif
  }

  void createTitleIkeaLogoTextureResources() {
    auto replaceSamplerClampLinear = [this] {
      if (titleIkeaLogoSampler != VK_NULL_HANDLE)
        vkDestroySampler(device, titleIkeaLogoSampler, nullptr);
      VkSamplerCreateInfo si{};
      si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      si.magFilter = VK_FILTER_NEAREST;
      si.minFilter = VK_FILTER_NEAREST;
      si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      si.anisotropyEnable = VK_FALSE;
      si.maxAnisotropy = 1.f;
      si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
      si.unnormalizedCoordinates = VK_FALSE;
      si.compareEnable = VK_FALSE;
      si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      si.minLod = 0.f;
      si.maxLod = 0.f;
      VK_CHECK(vkCreateSampler(device, &si, nullptr, &titleIkeaLogoSampler), "title ikea logo sampler");
    };
    bool loaded = false;
    try {
      createTextureResources(VULKAN_GAME_ASSETS_DIR "/ui/title_ikea_logo.png", titleIkeaLogoImage,
                             titleIkeaLogoMemory, titleIkeaLogoView, titleIkeaLogoSampler);
      loaded = true;
    } catch (const std::exception& ex) {
      std::cerr << "[ui] title_ikea_logo (" VULKAN_GAME_ASSETS_DIR "/ui/title_ikea_logo.png): " << ex.what()
                << '\n';
    }
    if (!loaded) {
      try {
        createTextureResources("assets/ui/title_ikea_logo.png", titleIkeaLogoImage, titleIkeaLogoMemory,
                               titleIkeaLogoView, titleIkeaLogoSampler);
        loaded = true;
      } catch (const std::exception& ex) {
        std::cerr << "[ui] title_ikea_logo (assets/ui/title_ikea_logo.png): " << ex.what() << '\n';
      }
    }
    if (!loaded) {
      createSolidColorTexture2D(0, 81, 186, 255, titleIkeaLogoImage, titleIkeaLogoMemory, titleIkeaLogoView,
                              titleIkeaLogoSampler);
    }
    replaceSamplerClampLinear();
  }

  void createRenderPass() {
    VkAttachmentDescription color{};
    color.format = swapchainImageFormat;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentDescription depth{};
    depth.format = findDepthFormat(physicalDevice);
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    // Reused each frame with clear; keep layout stable across frames (no UNDEFINED ping-pong).
    depth.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depthRef{};
    depthRef.attachment = 1;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> atts{color, depth};
    VkRenderPassCreateInfo ri{};
    ri.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ri.attachmentCount = static_cast<uint32_t>(atts.size());
    ri.pAttachments = atts.data();
    ri.subpassCount = 1;
    ri.pSubpasses = &sub;
    ri.dependencyCount = 1;
    ri.pDependencies = &dep;
    VK_CHECK(vkCreateRenderPass(device, &ri, nullptr, &renderPass), "vkCreateRenderPass");

    VkAttachmentDescription pColor{};
    pColor.format = swapchainImageFormat;
    pColor.samples = VK_SAMPLE_COUNT_1_BIT;
    pColor.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    pColor.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    pColor.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    pColor.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    pColor.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    pColor.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference pColorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription pSub{};
    pSub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    pSub.colorAttachmentCount = 1;
    pSub.pColorAttachments = &pColorRef;
    VkSubpassDependency pDep{};
    pDep.srcSubpass = VK_SUBPASS_EXTERNAL;
    pDep.dstSubpass = 0;
    pDep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    pDep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    pDep.srcAccessMask = 0;
    pDep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    VkRenderPassCreateInfo pri{};
    pri.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    pri.attachmentCount = 1;
    pri.pAttachments = &pColor;
    pri.subpassCount = 1;
    pri.pSubpasses = &pSub;
    pri.dependencyCount = 1;
    pri.pDependencies = &pDep;
    VK_CHECK(vkCreateRenderPass(device, &pri, nullptr, &presentRenderPass), "presentRenderPass");
  }

  void createDescriptorAndPipelineLayout() {
    VkDescriptorSetLayoutBinding ubo{};
    ubo.binding = 0;
    ubo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ubo.descriptorCount = 1;
    ubo.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding tex{};
    tex.binding = 1;
    tex.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    tex.descriptorCount = 1;
    tex.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding signTex{};
    signTex.binding = 2;
    signTex.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    signTex.descriptorCount = 1;
    signTex.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding shelfTex{};
    shelfTex.binding = 3;
    shelfTex.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shelfTex.descriptorCount = 1;
    shelfTex.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding crateTex{};
    crateTex.binding = 4;
    crateTex.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    crateTex.descriptorCount = 1;
    crateTex.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding extraTexArr{};
    extraTexArr.binding = 5;
    extraTexArr.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    extraTexArr.descriptorCount = kMaxExtraTextures;
    extraTexArr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding staffGlb{};
    staffGlb.binding = 6;
    staffGlb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    staffGlb.descriptorCount = 1;
    staffGlb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding staffBones{};
    staffBones.binding = 7;
    staffBones.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    staffBones.descriptorCount = 1;
    staffBones.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutBinding shrekEggTexBind{};
    shrekEggTexBind.binding = 8;
    shrekEggTexBind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shrekEggTexBind.descriptorCount = 1;
    shrekEggTexBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding hudFontTexBind{};
    hudFontTexBind.binding = 9;
    hudFontTexBind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    hudFontTexBind.descriptorCount = 1;
    hudFontTexBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding titleIkeaLogoTexBind{};
    titleIkeaLogoTexBind.binding = 10;
    titleIkeaLogoTexBind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    titleIkeaLogoTexBind.descriptorCount = 1;
    titleIkeaLogoTexBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding boxCutterTexBind{};
    boxCutterTexBind.binding = 11;
    boxCutterTexBind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    boxCutterTexBind.descriptorCount = 1;
    boxCutterTexBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding rustyPipeTexBind{};
    rustyPipeTexBind.binding = 12;
    rustyPipeTexBind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    rustyPipeTexBind.descriptorCount = 1;
    rustyPipeTexBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding palletTexBind{};
    palletTexBind.binding = 13;
    palletTexBind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    palletTexBind.descriptorCount = 1;
    palletTexBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding deliMetalTexBind{};
    deliMetalTexBind.binding = 14;
    deliMetalTexBind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    deliMetalTexBind.descriptorCount = 1;
    deliMetalTexBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding deliBaseTexBind{};
    deliBaseTexBind.binding = 15;
    deliBaseTexBind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    deliBaseTexBind.descriptorCount = 1;
    deliBaseTexBind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    std::array<VkDescriptorSetLayoutBinding, 16> bindings{
        ubo,        tex,        signTex,       shelfTex,     crateTex,    extraTexArr,
        staffGlb,   staffBones, shrekEggTexBind, hudFontTexBind, titleIkeaLogoTexBind, boxCutterTexBind,
        rustyPipeTexBind, palletTexBind, deliMetalTexBind, deliBaseTexBind};

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ci.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device, &ci, nullptr, &descriptorSetLayout),
             "descriptorSetLayout");

    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &descriptorSetLayout;
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PushModel);
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device, &pli, nullptr, &pipelineLayout), "pipelineLayout");
  }

  void createGraphicsPipeline() {
    const std::string base = std::string(SHADER_DIR);
    auto vertCode = readFile(base + "/shader.vert.spv");
    auto fragCode = readFile(base + "/shader.frag.spv");
    VkShaderModule vert = createShaderModule(device, vertCode);
    VkShaderModule frag = createShaderModule(device, fragCode);

    VkPipelineShaderStageCreateInfo vs{};
    vs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vs.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vs.module = vert;
    vs.pName = "main";
    VkPipelineShaderStageCreateInfo fs{};
    fs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fs.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fs.module = frag;
    fs.pName = "main";
    VkPipelineShaderStageCreateInfo stages[] = {vs, fs};

    std::array<VkVertexInputBindingDescription, 2> binds{};
    binds[0].binding = 0;
    binds[0].stride = sizeof(Vertex);
    binds[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    binds[1].binding = 1;
    binds[1].stride = sizeof(glm::mat4);
    binds[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    std::array<VkVertexInputAttributeDescription, 8> attrs{};
    attrs[0].location = 0;
    attrs[0].binding = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset = offsetof(Vertex, pos);
    attrs[1].location = 1;
    attrs[1].binding = 0;
    attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset = offsetof(Vertex, normal);
    attrs[2].location = 2;
    attrs[2].binding = 0;
    attrs[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[2].offset = offsetof(Vertex, color);
    attrs[3].location = 7;
    attrs[3].binding = 0;
    attrs[3].format = VK_FORMAT_R32G32_SFLOAT;
    attrs[3].offset = offsetof(Vertex, uv);
    for (int c = 0; c < 4; ++c) {
      attrs[4 + static_cast<size_t>(c)].location = 3 + c;
      attrs[4 + static_cast<size_t>(c)].binding = 1;
      attrs[4 + static_cast<size_t>(c)].format = VK_FORMAT_R32G32B32A32_SFLOAT;
      attrs[4 + static_cast<size_t>(c)].offset = static_cast<uint32_t>(sizeof(glm::vec4) * c);
    }

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = static_cast<uint32_t>(binds.size());
    vi.pVertexBindingDescriptions = binds.data();
    vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vi.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{};
    vp.width = static_cast<float>(sceneExtent.width);
    vp.height = static_cast<float>(sceneExtent.height);
    vp.minDepth = 0;
    vp.maxDepth = 1;
    VkRect2D sc{};
    sc.extent = sceneExtent;

    VkPipelineViewportStateCreateInfo vpi{};
    vpi.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpi.viewportCount = 1;
    vpi.pViewports = &vp;
    vpi.scissorCount = 1;
    vpi.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    // No culling: mixed mesh winding (floor/ceiling vs pillars) + Y-flipped proj makes back-face
    // culling drop whole faces; drawing both sides keeps pillars and hall geometry solid.
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;
    // Polygon offset was pushing grazing shelf/crate faces backward in depth so the rack behind
    // could win the depth test (see-through boxes when climbing). Prefer occasional Z-fight over that.
    rs.depthBiasEnable = VK_FALSE;
    rs.depthBiasConstantFactor = 0.f;
    rs.depthBiasClamp = 0.0f;
    rs.depthBiasSlopeFactor = 0.f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendAttachmentState cbaUi = cba;
    cbaUi.blendEnable = VK_TRUE;
    cbaUi.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cbaUi.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cbaUi.colorBlendOp = VK_BLEND_OP_ADD;
    cbaUi.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cbaUi.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cbaUi.alphaBlendOp = VK_BLEND_OP_ADD;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;
    VkPipelineColorBlendStateCreateInfo cbUi = cb;
    cbUi.pAttachments = &cbaUi;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.stageCount = 2;
    pi.pStages = stages;
    pi.pVertexInputState = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vpi;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &dyn;
    pi.layout = pipelineLayout;
    pi.renderPass = renderPass;
    pi.subpass = 0;
    VK_CHECK(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pi, nullptr, &graphicsPipeline),
             "graphicsPipeline");

    VkPipelineDepthStencilStateCreateInfo dsUi = ds;
    dsUi.depthTestEnable = VK_FALSE;
    dsUi.depthWriteEnable = VK_FALSE;
    pi.pDepthStencilState = &dsUi;
    pi.pColorBlendState = &cbUi;
    VK_CHECK(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pi, nullptr, &uiPipeline), "uiPipeline");

    pi.renderPass = presentRenderPass;
    VK_CHECK(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pi, nullptr, &uiPresentPipeline),
             "uiPresentPipeline");

    vkDestroyShaderModule(device, frag, nullptr);
    vkDestroyShaderModule(device, vert, nullptr);
  }

  void createPostProcessResources() {
    if (postDescriptorSetLayout != VK_NULL_HANDLE)
      return;
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo dsl{};
    dsl.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl.bindingCount = 1;
    dsl.pBindings = &b;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dsl, nullptr, &postDescriptorSetLayout), "post DSL");
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PushPost);
    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &postDescriptorSetLayout;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device, &pl, nullptr, &postPipelineLayout), "post PL");
  }

  void createPostPipeline() {
    if (postProcessPipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(device, postProcessPipeline, nullptr);
      postProcessPipeline = VK_NULL_HANDLE;
    }
    const std::string base = std::string(SHADER_DIR);
    auto vcode = readFile(base + "/post.vert.spv");
    auto fcode = readFile(base + "/post.frag.spv");
    VkShaderModule vm = createShaderModule(device, vcode);
    VkShaderModule fm = createShaderModule(device, fcode);
    VkPipelineShaderStageCreateInfo vs{};
    vs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vs.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vs.module = vm;
    vs.pName = "main";
    VkPipelineShaderStageCreateInfo fs{};
    fs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fs.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fs.module = fm;
    fs.pName = "main";
    VkPipelineShaderStageCreateInfo st[] = {vs, fs};

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport ppvp{};
    ppvp.width = static_cast<float>(swapchainExtent.width);
    ppvp.height = static_cast<float>(swapchainExtent.height);
    ppvp.minDepth = 0;
    ppvp.maxDepth = 1;
    VkRect2D ppsc{};
    ppsc.extent = swapchainExtent;
    VkPipelineViewportStateCreateInfo vpi{};
    vpi.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpi.viewportCount = 1;
    vpi.pViewports = &ppvp;
    vpi.scissorCount = 1;
    vpi.pScissors = &ppsc;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                         VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;
    VkDynamicState pDyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = pDyn;
    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.stageCount = 2;
    pi.pStages = st;
    pi.pVertexInputState = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vpi;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &dyn;
    pi.layout = postPipelineLayout;
    pi.renderPass = presentRenderPass;
    pi.subpass = 0;
    VK_CHECK(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pi, nullptr, &postProcessPipeline),
             "postProcessPipeline");
    vkDestroyShaderModule(device, fm, nullptr);
    vkDestroyShaderModule(device, vm, nullptr);
  }

  void createStaffSkinnedPipeline() {
    if (!staffSkinnedActive)
      return;
    if (graphicsPipelineStaffSkinned != VK_NULL_HANDLE) {
      vkDestroyPipeline(device, graphicsPipelineStaffSkinned, nullptr);
      graphicsPipelineStaffSkinned = VK_NULL_HANDLE;
    }
    const std::string base = std::string(SHADER_DIR);
    auto staffVertCode = readFile(base + "/shader_staff.vert.spv");
    auto fragCode = readFile(base + "/shader.frag.spv");
    VkShaderModule svert = createShaderModule(device, staffVertCode);
    VkShaderModule frag = createShaderModule(device, fragCode);

    VkPipelineShaderStageCreateInfo vs{};
    vs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vs.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vs.module = svert;
    vs.pName = "main";
    VkPipelineShaderStageCreateInfo fs{};
    fs.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fs.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fs.module = frag;
    fs.pName = "main";
    VkPipelineShaderStageCreateInfo stages[] = {vs, fs};

    std::array<VkVertexInputBindingDescription, 2> binds{};
    binds[0].binding = 0;
    binds[0].stride = sizeof(staff_skin::SkinnedVertex);
    binds[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    binds[1].binding = 1;
    binds[1].stride = sizeof(glm::mat4);
    binds[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
    std::array<VkVertexInputAttributeDescription, 10> attrs{};
    attrs[0].location = 0;
    attrs[0].binding = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset = offsetof(staff_skin::SkinnedVertex, pos);
    attrs[1].location = 1;
    attrs[1].binding = 0;
    attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset = offsetof(staff_skin::SkinnedVertex, normal);
    attrs[2].location = 2;
    attrs[2].binding = 0;
    attrs[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[2].offset = offsetof(staff_skin::SkinnedVertex, color);
    attrs[3].location = 7;
    attrs[3].binding = 0;
    attrs[3].format = VK_FORMAT_R32G32_SFLOAT;
    attrs[3].offset = offsetof(staff_skin::SkinnedVertex, uv);
    for (int c = 0; c < 4; ++c) {
      attrs[4 + static_cast<size_t>(c)].location = 3 + c;
      attrs[4 + static_cast<size_t>(c)].binding = 1;
      attrs[4 + static_cast<size_t>(c)].format = VK_FORMAT_R32G32B32A32_SFLOAT;
      attrs[4 + static_cast<size_t>(c)].offset = static_cast<uint32_t>(sizeof(glm::vec4) * c);
    }
    attrs[8].location = 8;
    attrs[8].binding = 0;
    attrs[8].format = VK_FORMAT_R32G32B32A32_SINT;
    attrs[8].offset = offsetof(staff_skin::SkinnedVertex, boneIds);
    attrs[9].location = 9;
    attrs[9].binding = 0;
    attrs[9].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[9].offset = offsetof(staff_skin::SkinnedVertex, boneWts);

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = static_cast<uint32_t>(binds.size());
    vi.pVertexBindingDescriptions = binds.data();
    vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vi.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{};
    vp.width = static_cast<float>(sceneExtent.width);
    vp.height = static_cast<float>(sceneExtent.height);
    vp.minDepth = 0;
    vp.maxDepth = 1;
    VkRect2D sc{};
    sc.extent = sceneExtent;

    VkPipelineViewportStateCreateInfo vpi{};
    vpi.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpi.viewportCount = 1;
    vpi.pViewports = &vp;
    vpi.scissorCount = 1;
    vpi.pScissors = &sc;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rs.lineWidth = 1.0f;
    rs.depthBiasEnable = VK_FALSE;
    rs.depthBiasConstantFactor = 0.f;
    rs.depthBiasClamp = 0.0f;
    rs.depthBiasSlopeFactor = 0.f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState staffDyn2[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo staffDynInfo2{};
    staffDynInfo2.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    staffDynInfo2.dynamicStateCount = 2;
    staffDynInfo2.pDynamicStates = staffDyn2;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.stageCount = 2;
    pi.pStages = stages;
    pi.pVertexInputState = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vpi;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &staffDynInfo2;
    pi.layout = pipelineLayout;
    pi.renderPass = renderPass;
    pi.subpass = 0;
    VK_CHECK(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pi, nullptr,
                                       &graphicsPipelineStaffSkinned),
             "graphicsPipelineStaffSkinned");

    vkDestroyShaderModule(device, frag, nullptr);
    vkDestroyShaderModule(device, svert, nullptr);
  }

  void createFramebuffers() {
    sceneFramebuffers.resize(static_cast<size_t>(kMaxFramesInFlight));
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      std::array<VkImageView, 2> atts{sceneColorViews[static_cast<size_t>(i)],
                                     depthViews[static_cast<size_t>(i)]};
      VkFramebufferCreateInfo fi{};
      fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      fi.renderPass = renderPass;
      fi.attachmentCount = static_cast<uint32_t>(atts.size());
      fi.pAttachments = atts.data();
      fi.width = sceneExtent.width;
      fi.height = sceneExtent.height;
      fi.layers = 1;
      VK_CHECK(vkCreateFramebuffer(device, &fi, nullptr, &sceneFramebuffers[static_cast<size_t>(i)]),
               "sceneFramebuffer");
    }
    framebuffers.resize(swapchainImageViews.size());
    for (size_t i = 0; i < swapchainImageViews.size(); ++i) {
      std::array<VkImageView, 1> atts{swapchainImageViews[i]};
      VkFramebufferCreateInfo fi{};
      fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      fi.renderPass = presentRenderPass;
      fi.attachmentCount = static_cast<uint32_t>(atts.size());
      fi.pAttachments = atts.data();
      fi.width = swapchainExtent.width;
      fi.height = swapchainExtent.height;
      fi.layers = 1;
      VK_CHECK(vkCreateFramebuffer(device, &fi, nullptr, &framebuffers[i]), "swapchain framebuffer");
    }
  }

  void createCommandPool() {
    auto q = findQueueFamilies(physicalDevice, surface);
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = *q.graphicsFamily;
    VK_CHECK(vkCreateCommandPool(device, &ci, nullptr, &commandPool), "commandPool");
  }

  void createVertexBuffers() {
    const VkDeviceSize terrainMax = sizeof(Vertex) * kMaxTerrainVerts();

    createBuffer(physicalDevice, device, terrainMax, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 groundVertexBuffer, groundVertexBufferMemory);
    vkMapMemory(device, groundVertexBufferMemory, 0, terrainMax, 0, &groundMapped);

    createBuffer(physicalDevice, device, terrainMax, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 ceilingVertexBuffer, ceilingVertexBufferMemory);
    vkMapMemory(device, ceilingVertexBufferMemory, 0, terrainMax, 0, &ceilingMapped);

    auto pillar = makePillarMesh();
    pillarVertexCount = static_cast<uint32_t>(pillar.size());
    const VkDeviceSize pillarSize = sizeof(Vertex) * pillar.size();
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, pillarSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);
    void* data = nullptr;
    vkMapMemory(device, stagingMem, 0, pillarSize, 0, &data);
    std::memcpy(data, pillar.data(), static_cast<size_t>(pillarSize));
    vkUnmapMemory(device, stagingMem);
    createBuffer(physicalDevice, device, pillarSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, pillarVertexBuffer, pillarVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, staging, pillarVertexBuffer, pillarSize);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    const VkDeviceSize pillarInstBufSize = sizeof(glm::mat4) * kMaxPillarInstances;
    createBuffer(physicalDevice, device, pillarInstBufSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 pillarInstanceBuffer, pillarInstanceBufferMemory);
    vkMapMemory(device, pillarInstanceBufferMemory, 0, pillarInstBufSize, 0, &pillarInstanceMapped);
    pillarInstanceScratch.reserve(kMaxPillarInstances);

    const VkDeviceSize signInstBufSize = sizeof(glm::mat4) * kMaxSignInstances;
    createBuffer(physicalDevice, device, signInstBufSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 signInstanceBuffer, signInstanceBufferMemory);
    vkMapMemory(device, signInstanceBufferMemory, 0, signInstBufSize, 0, &signInstanceMapped);
    signBoardInstanceScratch.reserve(kMaxSignInstances / 2);
    signStringInstanceScratch.reserve(kMaxSignInstances);

    auto crosshair = makeCrosshairQuadMesh();
    crosshairVertexCount = static_cast<uint32_t>(crosshair.size());
    const VkDeviceSize chSize = sizeof(Vertex) * crosshair.size();
    VkBuffer chStaging = VK_NULL_HANDLE;
    VkDeviceMemory chStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, chSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 chStaging, chStagingMem);
    void* chData = nullptr;
    vkMapMemory(device, chStagingMem, 0, chSize, 0, &chData);
    std::memcpy(chData, crosshair.data(), static_cast<size_t>(chSize));
    vkUnmapMemory(device, chStagingMem);
    createBuffer(physicalDevice, device, chSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, crosshairVertexBuffer, crosshairVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, chStaging, crosshairVertexBuffer, chSize);
    vkDestroyBuffer(device, chStaging, nullptr);
    vkFreeMemory(device, chStagingMem, nullptr);

    auto helpVerts = buildControlsHelpOverlayVertices();
    controlsHelpVertexCount = static_cast<uint32_t>(helpVerts.size());
    const VkDeviceSize helpSize = sizeof(Vertex) * helpVerts.size();
    VkBuffer helpStaging = VK_NULL_HANDLE;
    VkDeviceMemory helpStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, helpSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 helpStaging, helpStagingMem);
    void* helpData = nullptr;
    vkMapMemory(device, helpStagingMem, 0, helpSize, 0, &helpData);
    std::memcpy(helpData, helpVerts.data(), static_cast<size_t>(helpSize));
    vkUnmapMemory(device, helpStagingMem);
    createBuffer(physicalDevice, device, helpSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, controlsHelpVertexBuffer,
                 controlsHelpVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, helpStaging, controlsHelpVertexBuffer, helpSize);
    vkDestroyBuffer(device, helpStaging, nullptr);
    vkFreeMemory(device, helpStagingMem, nullptr);

    auto deathMenuVerts = buildDeathMenuOverlayVertices();
    deathMenuVertexCount = static_cast<uint32_t>(deathMenuVerts.size());
    const VkDeviceSize deathMenuSize = sizeof(Vertex) * deathMenuVerts.size();
    VkBuffer deathMenuStaging = VK_NULL_HANDLE;
    VkDeviceMemory deathMenuStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, deathMenuSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 deathMenuStaging, deathMenuStagingMem);
    void* deathMenuData = nullptr;
    vkMapMemory(device, deathMenuStagingMem, 0, deathMenuSize, 0, &deathMenuData);
    std::memcpy(deathMenuData, deathMenuVerts.data(), static_cast<size_t>(deathMenuSize));
    vkUnmapMemory(device, deathMenuStagingMem);
    createBuffer(physicalDevice, device, deathMenuSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, deathMenuVertexBuffer, deathMenuVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, deathMenuStaging, deathMenuVertexBuffer, deathMenuSize);
    vkDestroyBuffer(device, deathMenuStaging, nullptr);
    vkFreeMemory(device, deathMenuStagingMem, nullptr);

    auto pauseMenuVerts = buildPauseMenuOverlayVertices();
    pauseMenuVertexCount = static_cast<uint32_t>(pauseMenuVerts.size());
    const VkDeviceSize pauseMenuSize = sizeof(Vertex) * pauseMenuVerts.size();
    VkBuffer pauseMenuStaging = VK_NULL_HANDLE;
    VkDeviceMemory pauseMenuStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, pauseMenuSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 pauseMenuStaging, pauseMenuStagingMem);
    void* pauseMenuData = nullptr;
    vkMapMemory(device, pauseMenuStagingMem, 0, pauseMenuSize, 0, &pauseMenuData);
    std::memcpy(pauseMenuData, pauseMenuVerts.data(), static_cast<size_t>(pauseMenuSize));
    vkUnmapMemory(device, pauseMenuStagingMem);
    createBuffer(physicalDevice, device, pauseMenuSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, pauseMenuVertexBuffer, pauseMenuVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, pauseMenuStaging, pauseMenuVertexBuffer, pauseMenuSize);
    vkDestroyBuffer(device, pauseMenuStaging, nullptr);
    vkFreeMemory(device, pauseMenuStagingMem, nullptr);

    constexpr VkDeviceSize kInventoryMenuVbMaxBytes = sizeof(Vertex) * 8192u;
    inventoryMenuVertexBufferBytes = kInventoryMenuVbMaxBytes;
    createBuffer(physicalDevice, device, kInventoryMenuVbMaxBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 inventoryMenuVertexBuffer, inventoryMenuVertexBufferMemory);
    VK_CHECK(vkMapMemory(device, inventoryMenuVertexBufferMemory, 0, kInventoryMenuVbMaxBytes, 0,
                         &inventoryMenuVertexMapped),
             "inventoryMenu vb map");

    recreateTitleMenuMainGpuMesh();
    recreateTitleMenuSlotGpuMesh();

    uploadUiMeshToGpu(buildIntroSplashVertices(), introSplashVertexBuffer,
                      introSplashVertexBufferMemory, introSplashVertexCount);
    uploadUiMeshToGpu(buildLoadingScreenVertices(), loadingScreenVertexBuffer,
                      loadingScreenVertexBufferMemory, loadingScreenVertexCount);

    constexpr VkDeviceSize kHealthHudVbMaxBytes = sizeof(Vertex) * 4096u;
    healthHudVertexBufferBytes = kHealthHudVbMaxBytes;
    createBuffer(physicalDevice, device, kHealthHudVbMaxBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 healthHudVertexBuffer, healthHudVertexBufferMemory);
    VK_CHECK(vkMapMemory(device, healthHudVertexBufferMemory, 0, kHealthHudVbMaxBytes, 0,
                         &healthHudVertexMapped),
             "healthHud vb map");

    auto sign = makeHandQuadMesh(glm::vec3(0.0f, 0.0f, 1.0f));
    signVertexCount = static_cast<uint32_t>(sign.size());
    const VkDeviceSize signSize = sizeof(Vertex) * sign.size();
    VkBuffer signStaging = VK_NULL_HANDLE;
    VkDeviceMemory signStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, signSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 signStaging, signStagingMem);
    void* signData = nullptr;
    vkMapMemory(device, signStagingMem, 0, signSize, 0, &signData);
    std::memcpy(signData, sign.data(), static_cast<size_t>(signSize));
    vkUnmapMemory(device, signStagingMem);
    createBuffer(physicalDevice, device, signSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, signVertexBuffer, signVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, signStaging, signVertexBuffer, signSize);
    vkDestroyBuffer(device, signStaging, nullptr);
    vkFreeMemory(device, signStagingMem, nullptr);

    auto signString = makeHandQuadMesh(glm::vec3(1.0f, 1.0f, 0.0f));
    signStringVertexCount = static_cast<uint32_t>(signString.size());
    const VkDeviceSize signStringSize = sizeof(Vertex) * signString.size();
    VkBuffer signStringStaging = VK_NULL_HANDLE;
    VkDeviceMemory signStringStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, signStringSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 signStringStaging, signStringStagingMem);
    void* signStringData = nullptr;
    vkMapMemory(device, signStringStagingMem, 0, signStringSize, 0, &signStringData);
    std::memcpy(signStringData, signString.data(), static_cast<size_t>(signStringSize));
    vkUnmapMemory(device, signStringStagingMem);
    createBuffer(physicalDevice, device, signStringSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, signStringVertexBuffer,
                 signStringVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, signStringStaging, signStringVertexBuffer,
               signStringSize);
    vkDestroyBuffer(device, signStringStaging, nullptr);
    vkFreeMemory(device, signStringStagingMem, nullptr);

    auto shelfMesh = makeWarehouseShelfMesh();
    shelfVertexCount = static_cast<uint32_t>(shelfMesh.size());
    const VkDeviceSize shelfSize = sizeof(Vertex) * shelfMesh.size();
    VkBuffer shelfStaging = VK_NULL_HANDLE;
    VkDeviceMemory shelfStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, shelfSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 shelfStaging, shelfStagingMem);
    void* shelfData = nullptr;
    vkMapMemory(device, shelfStagingMem, 0, shelfSize, 0, &shelfData);
    std::memcpy(shelfData, shelfMesh.data(), static_cast<size_t>(shelfSize));
    vkUnmapMemory(device, shelfStagingMem);
    createBuffer(physicalDevice, device, shelfSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shelfVertexBuffer, shelfVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, shelfStaging, shelfVertexBuffer, shelfSize);
    vkDestroyBuffer(device, shelfStaging, nullptr);
    vkFreeMemory(device, shelfStagingMem, nullptr);

    const VkDeviceSize idMatSize = sizeof(glm::mat4);
    glm::mat4 identityM(1.0f);
    VkBuffer idStaging = VK_NULL_HANDLE;
    VkDeviceMemory idStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, idMatSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, idStaging,
                 idStagingMem);
    void* idPtr = nullptr;
    vkMapMemory(device, idStagingMem, 0, idMatSize, 0, &idPtr);
    std::memcpy(idPtr, &identityM, static_cast<size_t>(idMatSize));
    vkUnmapMemory(device, idStagingMem);
    createBuffer(physicalDevice, device, idMatSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, identityInstanceBuffer,
                 identityInstanceBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, idStaging, identityInstanceBuffer, idMatSize);
    vkDestroyBuffer(device, idStaging, nullptr);
    vkFreeMemory(device, idStagingMem, nullptr);

    const VkDeviceSize shelfInstBufSize = sizeof(glm::mat4) * kMaxShelfInstances;
    createBuffer(physicalDevice, device, shelfInstBufSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 shelfInstanceBuffer, shelfInstanceBufferMemory);
    vkMapMemory(device, shelfInstanceBufferMemory, 0, shelfInstBufSize, 0, &shelfInstanceMapped);
    shelfInstanceScratch.reserve(kMaxShelfInstances);

    auto crateMesh = makeShelfCrateUnitMesh();
    shelfCrateVertexCount = static_cast<uint32_t>(crateMesh.size());
    const VkDeviceSize crateVSize = sizeof(Vertex) * crateMesh.size();
    VkBuffer crateStaging = VK_NULL_HANDLE;
    VkDeviceMemory crateStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, crateVSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 crateStaging, crateStagingMem);
    void* cratePtr = nullptr;
    vkMapMemory(device, crateStagingMem, 0, crateVSize, 0, &cratePtr);
    std::memcpy(cratePtr, crateMesh.data(), static_cast<size_t>(crateVSize));
    vkUnmapMemory(device, crateStagingMem);
    createBuffer(physicalDevice, device, crateVSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shelfCrateVertexBuffer, shelfCrateVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, crateStaging, shelfCrateVertexBuffer, crateVSize);
    vkDestroyBuffer(device, crateStaging, nullptr);
    vkFreeMemory(device, crateStagingMem, nullptr);

    const VkDeviceSize crateInstBufSize = sizeof(glm::mat4) * kMaxShelfCrates;
    createBuffer(physicalDevice, device, crateInstBufSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 shelfCrateInstanceBuffer, shelfCrateInstanceBufferMemory);
    vkMapMemory(device, shelfCrateInstanceBufferMemory, 0, crateInstBufSize, 0, &shelfCrateInstanceMapped);
    shelfCrateInstanceScratch.reserve(kMaxShelfCrates);

    auto palletMesh = makeShelfPalletUnitMesh();
    shelfPalletVertexCount = static_cast<uint32_t>(palletMesh.size());
    if (shelfPalletVertexCount != shelfCrateVertexCount)
      throw std::runtime_error("shelf pallet unit mesh must match crate vertex count");
    const VkDeviceSize palletVSize = sizeof(Vertex) * palletMesh.size();
    VkBuffer palletStaging = VK_NULL_HANDLE;
    VkDeviceMemory palletStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, palletVSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, palletStaging,
                 palletStagingMem);
    void* palletPtr = nullptr;
    vkMapMemory(device, palletStagingMem, 0, palletVSize, 0, &palletPtr);
    std::memcpy(palletPtr, palletMesh.data(), static_cast<size_t>(palletVSize));
    vkUnmapMemory(device, palletStagingMem);
    createBuffer(physicalDevice, device, palletVSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shelfPalletVertexBuffer, shelfPalletVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, palletStaging, shelfPalletVertexBuffer, palletVSize);
    vkDestroyBuffer(device, palletStaging, nullptr);
    vkFreeMemory(device, palletStagingMem, nullptr);

    const VkDeviceSize palletInstBufSize = sizeof(glm::mat4) * kMaxShelfCrates;
    createBuffer(physicalDevice, device, palletInstBufSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 shelfPalletInstanceBuffer, shelfPalletInstanceBufferMemory);
    vkMapMemory(device, shelfPalletInstanceBufferMemory, 0, palletInstBufSize, 0, &shelfPalletInstanceMapped);
    shelfPalletInstanceScratch.reserve(kMaxShelfCrates);

    auto ladderMesh = makeShelfLadderUnitMesh();
    shelfLadderVertexCount = static_cast<uint32_t>(ladderMesh.size());
    const VkDeviceSize ladderVSize = sizeof(Vertex) * ladderMesh.size();
    VkBuffer ladderStaging = VK_NULL_HANDLE;
    VkDeviceMemory ladderStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, ladderVSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 ladderStaging, ladderStagingMem);
    void* ladderPtr = nullptr;
    vkMapMemory(device, ladderStagingMem, 0, ladderVSize, 0, &ladderPtr);
    std::memcpy(ladderPtr, ladderMesh.data(), static_cast<size_t>(ladderVSize));
    vkUnmapMemory(device, ladderStagingMem);
    createBuffer(physicalDevice, device, ladderVSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shelfLadderVertexBuffer, shelfLadderVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, ladderStaging, shelfLadderVertexBuffer, ladderVSize);
    vkDestroyBuffer(device, ladderStaging, nullptr);
    vkFreeMemory(device, ladderStagingMem, nullptr);

    const VkDeviceSize ladderInstBufSize = sizeof(glm::mat4) * kMaxShelfLadders;
    createBuffer(physicalDevice, device, ladderInstBufSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 shelfLadderInstanceBuffer, shelfLadderInstanceBufferMemory);
    vkMapMemory(device, shelfLadderInstanceBufferMemory, 0, ladderInstBufSize, 0, &shelfLadderInstanceMapped);
    shelfLadderInstanceScratch.reserve(kMaxShelfLadders);

#if defined(VULKAN_GAME_SHELF_BOX_CUTTER_GLB)
    {
      std::vector<emp_mesh::LoadedVertex> bcLoad;
      std::vector<uint8_t> bcRgba;
      uint32_t bcW = 0, bcH = 0;
      std::string bcErr;
      constexpr float kBoxCutterTargetH = 0.08f;
      if (emp_mesh::loadFbx(VULKAN_GAME_SHELF_BOX_CUTTER_GLB, kBoxCutterTargetH, bcLoad, bcErr, &bcRgba,
                             &bcW, &bcH, gGamePerf.propMeshMaxVerts) &&
          !bcLoad.empty()) {
        const glm::vec4 bcTag = vrgb(shelfBoxCutterVertexColor());
        for (auto& v : bcLoad)
          v.color = bcTag;
        shelfBoxCutterVertexCount = static_cast<uint32_t>(bcLoad.size());
        const VkDeviceSize bcVSize = sizeof(Vertex) * bcLoad.size();
        VkBuffer bcStaging = VK_NULL_HANDLE;
        VkDeviceMemory bcStagingMem = VK_NULL_HANDLE;
        createBuffer(physicalDevice, device, bcVSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, bcStaging,
                     bcStagingMem);
        void* bcPtr = nullptr;
        vkMapMemory(device, bcStagingMem, 0, bcVSize, 0, &bcPtr);
        std::memcpy(bcPtr, bcLoad.data(), static_cast<size_t>(bcVSize));
        vkUnmapMemory(device, bcStagingMem);
        createBuffer(physicalDevice, device, bcVSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shelfBoxCutterVertexBuffer,
                     shelfBoxCutterVertexBufferMemory);
        copyBuffer(device, commandPool, graphicsQueue, bcStaging, shelfBoxCutterVertexBuffer, bcVSize);
        vkDestroyBuffer(device, bcStaging, nullptr);
        vkFreeMemory(device, bcStagingMem, nullptr);

        const VkDeviceSize bcInstBufSize = sizeof(glm::mat4) * kMaxShelfBoxCutters;
        createBuffer(physicalDevice, device, bcInstBufSize,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     shelfBoxCutterInstanceBuffer, shelfBoxCutterInstanceBufferMemory);
        vkMapMemory(device, shelfBoxCutterInstanceBufferMemory, 0, bcInstBufSize, 0,
                    &shelfBoxCutterInstanceMapped);
        shelfBoxCutterInstanceScratch.reserve(kMaxShelfBoxCutters);
        if (!bcRgba.empty() && bcW > 0 && bcH > 0)
          createTextureResourcesFromRgbaLinear(bcRgba.data(), bcW, bcH, shelfBoxCutterTextureImage,
                                               shelfBoxCutterTextureMemory, shelfBoxCutterTextureView,
                                               shelfBoxCutterTextureSampler);
        gShelfBoxCutterMeshLoaded = true;
        std::cout << "[shelf] box cutter GLB verts=" << shelfBoxCutterVertexCount << "\n";
      } else if (!bcErr.empty()) {
        std::cerr << "[shelf] box cutter GLB: " << bcErr << "\n";
      }
    }
#endif
    if (shelfBoxCutterTextureView == VK_NULL_HANDLE) {
      static const uint8_t kBoxCutterWhite[4] = {255, 255, 255, 255};
      createTextureResourcesFromRgbaLinear(kBoxCutterWhite, 1, 1, shelfBoxCutterTextureImage,
                                           shelfBoxCutterTextureMemory, shelfBoxCutterTextureView,
                                           shelfBoxCutterTextureSampler);
    }

#if defined(VULKAN_GAME_SHELF_RUSTY_PIPE_GLB)
    {
      std::vector<emp_mesh::LoadedVertex> rpLoad;
      std::vector<uint8_t> rpRgba;
      uint32_t rpW = 0, rpH = 0;
      std::string rpErr;
      constexpr float kRustyPipeTargetH = 0.35f;
      if (emp_mesh::loadFbx(VULKAN_GAME_SHELF_RUSTY_PIPE_GLB, kRustyPipeTargetH, rpLoad, rpErr, &rpRgba,
                            &rpW, &rpH, gGamePerf.propMeshMaxVerts) &&
          !rpLoad.empty()) {
        const glm::vec4 rpTag = vrgb(shelfRustyPipeVertexColor());
        for (auto& v : rpLoad)
          v.color = rpTag;
        shelfRustyPipeVertexCount = static_cast<uint32_t>(rpLoad.size());
        const VkDeviceSize rpVSize = sizeof(Vertex) * rpLoad.size();
        VkBuffer rpStaging = VK_NULL_HANDLE;
        VkDeviceMemory rpStagingMem = VK_NULL_HANDLE;
        createBuffer(physicalDevice, device, rpVSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, rpStaging,
                     rpStagingMem);
        void* rpPtr = nullptr;
        vkMapMemory(device, rpStagingMem, 0, rpVSize, 0, &rpPtr);
        std::memcpy(rpPtr, rpLoad.data(), static_cast<size_t>(rpVSize));
        vkUnmapMemory(device, rpStagingMem);
        createBuffer(physicalDevice, device, rpVSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shelfRustyPipeVertexBuffer,
                     shelfRustyPipeVertexBufferMemory);
        copyBuffer(device, commandPool, graphicsQueue, rpStaging, shelfRustyPipeVertexBuffer, rpVSize);
        vkDestroyBuffer(device, rpStaging, nullptr);
        vkFreeMemory(device, rpStagingMem, nullptr);

        const VkDeviceSize rpInstBufSize = sizeof(glm::mat4) * kMaxShelfRustyPipes;
        createBuffer(physicalDevice, device, rpInstBufSize,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     shelfRustyPipeInstanceBuffer, shelfRustyPipeInstanceBufferMemory);
        vkMapMemory(device, shelfRustyPipeInstanceBufferMemory, 0, rpInstBufSize, 0,
                    &shelfRustyPipeInstanceMapped);
        shelfRustyPipeInstanceScratch.reserve(kMaxShelfRustyPipes);
        if (!rpRgba.empty() && rpW > 0 && rpH > 0)
          createTextureResourcesFromRgbaLinear(rpRgba.data(), rpW, rpH, shelfRustyPipeTextureImage,
                                               shelfRustyPipeTextureMemory, shelfRustyPipeTextureView,
                                               shelfRustyPipeTextureSampler);
        gShelfRustyPipeMeshLoaded = true;
        std::cout << "[shelf] rusty pipe GLB verts=" << shelfRustyPipeVertexCount << "\n";
      } else if (!rpErr.empty()) {
        std::cerr << "[shelf] rusty pipe GLB: " << rpErr << "\n";
      }
    }
#endif
    if (shelfRustyPipeTextureView == VK_NULL_HANDLE) {
      static const uint8_t kRustyPipeWhite[4] = {255, 255, 255, 255};
      createTextureResourcesFromRgbaLinear(kRustyPipeWhite, 1, 1, shelfRustyPipeTextureImage,
                                           shelfRustyPipeTextureMemory, shelfRustyPipeTextureView,
                                           shelfRustyPipeTextureSampler);
    }

    auto marketMesh = makeDeliCounterMesh();
    marketVertexCount = static_cast<uint32_t>(marketMesh.size());
    const VkDeviceSize marketVSize = sizeof(Vertex) * marketMesh.size();
    VkBuffer marketStaging = VK_NULL_HANDLE;
    VkDeviceMemory marketStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, marketVSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, marketStaging,
                 marketStagingMem);
    void* marketPtr = nullptr;
    vkMapMemory(device, marketStagingMem, 0, marketVSize, 0, &marketPtr);
    std::memcpy(marketPtr, marketMesh.data(), static_cast<size_t>(marketVSize));
    vkUnmapMemory(device, marketStagingMem);
    createBuffer(physicalDevice, device, marketVSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, marketVertexBuffer, marketVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, marketStaging, marketVertexBuffer, marketVSize);
    vkDestroyBuffer(device, marketStaging, nullptr);
    vkFreeMemory(device, marketStagingMem, nullptr);

    const VkDeviceSize marketInstBufSize = sizeof(glm::mat4) * kMaxMarketInstances;
    createBuffer(physicalDevice, device, marketInstBufSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, marketInstanceBuffer,
                 marketInstanceBufferMemory);
    vkMapMemory(device, marketInstanceBufferMemory, 0, marketInstBufSize, 0, &marketInstanceMapped);
    marketInstanceScratch.reserve(kMaxMarketInstances);

#if defined(VULKAN_GAME_PIZZA_SLICE_GLB)
    {
      std::vector<Vertex> pzLoad = makeDeliPizzaSliceMesh();
      if (!pzLoad.empty()) {
        deliPizzaVertexCount = static_cast<uint32_t>(pzLoad.size());
        const VkDeviceSize pzVSize = sizeof(Vertex) * pzLoad.size();
        VkBuffer pzStaging = VK_NULL_HANDLE;
        VkDeviceMemory pzStagingMem = VK_NULL_HANDLE;
        createBuffer(physicalDevice, device, pzVSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, pzStaging,
                     pzStagingMem);
        void* pzPtr = nullptr;
        vkMapMemory(device, pzStagingMem, 0, pzVSize, 0, &pzPtr);
        std::memcpy(pzPtr, pzLoad.data(), static_cast<size_t>(pzVSize));
        vkUnmapMemory(device, pzStagingMem);
        createBuffer(physicalDevice, device, pzVSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, deliPizzaVertexBuffer,
                     deliPizzaVertexBufferMemory);
        copyBuffer(device, commandPool, graphicsQueue, pzStaging, deliPizzaVertexBuffer, pzVSize);
        vkDestroyBuffer(device, pzStaging, nullptr);
        vkFreeMemory(device, pzStagingMem, nullptr);
        const VkDeviceSize pzInstBufSize = sizeof(glm::mat4) * kMaxDeliPizzaInstances;
        createBuffer(physicalDevice, device, pzInstBufSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     deliPizzaInstanceBuffer, deliPizzaInstanceBufferMemory);
        vkMapMemory(device, deliPizzaInstanceBufferMemory, 0, pzInstBufSize, 0, &deliPizzaInstanceMapped);
        deliPizzaInstanceScratch.reserve(kMaxDeliPizzaInstances);
        gDeliPizzaMeshLoaded = true;
        std::cout << "[deli] pizza procedural verts=" << deliPizzaVertexCount << "\n";
      }
    }
#endif
    {
      std::vector<Vertex> mbLoad = makeDeliMeatballMesh();
      if (!mbLoad.empty()) {
        deliMeatballVertexCount = static_cast<uint32_t>(mbLoad.size());
        const VkDeviceSize mbVSize = sizeof(Vertex) * mbLoad.size();
        VkBuffer mbStaging = VK_NULL_HANDLE;
        VkDeviceMemory mbStagingMem = VK_NULL_HANDLE;
        createBuffer(physicalDevice, device, mbVSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, mbStaging,
                     mbStagingMem);
        void* mbPtr = nullptr;
        vkMapMemory(device, mbStagingMem, 0, mbVSize, 0, &mbPtr);
        std::memcpy(mbPtr, mbLoad.data(), static_cast<size_t>(mbVSize));
        vkUnmapMemory(device, mbStagingMem);
        createBuffer(physicalDevice, device, mbVSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, deliMeatballVertexBuffer,
                     deliMeatballVertexBufferMemory);
        copyBuffer(device, commandPool, graphicsQueue, mbStaging, deliMeatballVertexBuffer, mbVSize);
        vkDestroyBuffer(device, mbStaging, nullptr);
        vkFreeMemory(device, mbStagingMem, nullptr);
        const VkDeviceSize mbInstBufSize = sizeof(glm::mat4) * kMaxDeliMeatballInstances;
        createBuffer(physicalDevice, device, mbInstBufSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     deliMeatballInstanceBuffer, deliMeatballInstanceBufferMemory);
        vkMapMemory(device, deliMeatballInstanceBufferMemory, 0, mbInstBufSize, 0, &deliMeatballInstanceMapped);
        deliMeatballInstanceScratch.reserve(kMaxDeliMeatballInstances);
        gDeliMeatballMeshLoaded = true;
        std::cout << "[deli] meatball procedural verts=" << deliMeatballVertexCount << "\n";
      }
    }

    auto fluorescentMesh = makeFluorescentFixtureMesh();
    fluorescentVertexCount = static_cast<uint32_t>(fluorescentMesh.size());
    const VkDeviceSize fluorescentSize = sizeof(Vertex) * fluorescentMesh.size();
    VkBuffer fluorescentStaging = VK_NULL_HANDLE;
    VkDeviceMemory fluorescentStagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, fluorescentSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 fluorescentStaging, fluorescentStagingMem);
    void* fluorescentData = nullptr;
    vkMapMemory(device, fluorescentStagingMem, 0, fluorescentSize, 0, &fluorescentData);
    std::memcpy(fluorescentData, fluorescentMesh.data(), static_cast<size_t>(fluorescentSize));
    vkUnmapMemory(device, fluorescentStagingMem);
    createBuffer(physicalDevice, device, fluorescentSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, fluorescentVertexBuffer,
                 fluorescentVertexBufferMemory);
    copyBuffer(device, commandPool, graphicsQueue, fluorescentStaging, fluorescentVertexBuffer,
               fluorescentSize);
    vkDestroyBuffer(device, fluorescentStaging, nullptr);
    vkFreeMemory(device, fluorescentStagingMem, nullptr);

    const VkDeviceSize fluorescentInstBufSize = sizeof(glm::mat4) * kMaxFluorescentInstances;
    createBuffer(physicalDevice, device, fluorescentInstBufSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 fluorescentInstanceBuffer, fluorescentInstanceBufferMemory);
    vkMapMemory(device, fluorescentInstanceBufferMemory, 0, fluorescentInstBufSize, 0,
                &fluorescentInstanceMapped);
    fluorescentInstanceScratch.reserve(kMaxFluorescentInstances);

    const VkDeviceSize staffBoneBufTotal =
        sizeof(glm::mat4) * kStaffPaletteBoneCount * kStaffSkinnedInstanceSlots;
    createBuffer(physicalDevice, device, staffBoneBufTotal,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staffBoneSsbBuffer, staffBoneSsbMemory);
    vkMapMemory(device, staffBoneSsbMemory, 0, staffBoneBufTotal, 0, &staffBoneSsbMapped);
    std::memset(staffBoneSsbMapped, 0, static_cast<size_t>(staffBoneBufTotal));

    std::vector<emp_mesh::LoadedVertex> empLoad;
    std::vector<staff_skin::SkinnedVertex> skinLoad;
    std::string empErr;
    std::vector<uint8_t> staffDiffuseRgba;
    uint32_t staffDiffuseW = 0, staffDiffuseH = 0;

    std::string empPathStr(VULKAN_GAME_EMPLOYEE_FBX);
    const bool trySkinnedGlb =
        empPathStr.size() >= 4 &&
        (empPathStr.compare(empPathStr.size() - 4, 4, ".glb") == 0 ||
         empPathStr.compare(empPathStr.size() - 4, 4, ".GLB") == 0);

    staffSkinnedActive = false;
    staffRigBoneCount = 0;
    gStaffDeadRagdollSimBoneCount = 0;

    if (trySkinnedGlb &&
        staff_skin::loadSkinnedIdleGlb(VULKAN_GAME_EMPLOYEE_FBX, kEmployeeVisualHeight, skinLoad, staffRig,
                                       empErr, &staffDiffuseRgba, &staffDiffuseW, &staffDiffuseH)) {
      std::string clipErr;
      if (std::strcmp(VULKAN_GAME_MESHY_WALK_GLB, VULKAN_GAME_EMPLOYEE_FBX) != 0) {
        if (staff_skin::appendAnimationFromGlb(VULKAN_GAME_MESHY_WALK_GLB, staffRig, clipErr))
          std::cout << "[staff] walk clip loaded\n";
        else
          std::cerr << "[staff] walk clip: " << clipErr << "\n";
        clipErr.clear();
      }
      if (std::strcmp(VULKAN_GAME_MESHY_RUN_GLB, VULKAN_GAME_EMPLOYEE_FBX) != 0 &&
          std::strcmp(VULKAN_GAME_MESHY_RUN_GLB, VULKAN_GAME_MESHY_WALK_GLB) != 0) {
        if (staff_skin::appendAnimationFromGlb(VULKAN_GAME_MESHY_RUN_GLB, staffRig, clipErr))
          std::cout << "[staff] lean sprint clip loaded (clip index 2 for chase / sprint)\n";
        else
          std::cerr << "[staff] lean sprint clip: " << clipErr << "\n";
      }
      // Strip baked root motion from walk/sprint clips: pin root and hips/pelvis pos-keys
      // to bind pose so the animation never drifts the mesh away from the physics hitbox.
      {
        auto stripRootMotionKeys = [&](size_t clipIdx) {
          if (clipIdx >= staffRig.clips.size()) return;
          auto& cl = staffRig.clips[clipIdx];
          for (auto& ch : cl.channels) {
            std::string lower = ch.nodeName;
            for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            const bool isRoot = (ch.nodeName == staffRig.rootName);
            const bool isHips = (lower.find("hips") != std::string::npos ||
                                 lower.find("pelvis") != std::string::npos);
            if (isRoot || isHips) {
              auto it = staffRig.nodes.find(ch.nodeName);
              if (it != staffRig.nodes.end()) {
                const glm::vec3 bindT(it->second.bindLocal[3]);
                ch.posKeys.clear();
                ch.posKeys.push_back({0.0, bindT});
              }
            }
          }
        };
        for (size_t ci = 0; ci < staffRig.clips.size(); ++ci)
          stripRootMotionKeys(ci);
      }
      // Find the head bone for FP camera tracking.
      fpHeadBoneIdx = -1;
      for (int i = 0; i < staffRig.boneCount; ++i) {
        std::string lower = staffRig.boneNames[static_cast<size_t>(i)];
        for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (lower.find("head") != std::string::npos && lower.find("top") == std::string::npos) {
          fpHeadBoneIdx = i;
          fpHeadBindPos = glm::vec3(glm::inverse(staffRig.invBindTweaked[static_cast<size_t>(i)]) *
                                    glm::vec4(0.f, 0.f, 0.f, 1.f));
          std::fprintf(stderr, "[fp head] bone %d '%s' bindPos=(%.3f,%.3f,%.3f)\n",
                       i, staffRig.boneNames[static_cast<size_t>(i)].c_str(),
                       fpHeadBindPos.x, fpHeadBindPos.y, fpHeadBindPos.z);
          break;
        }
      }
      // Find neck bone for head-tilt pivot.
      fpNeckBoneIdx = -1;
      for (int i = 0; i < staffRig.boneCount; ++i) {
        std::string lower = staffRig.boneNames[static_cast<size_t>(i)];
        for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (lower.find("neck") != std::string::npos) {
          fpNeckBoneIdx = i;
          fpNeckBindPos = glm::vec3(glm::inverse(staffRig.invBindTweaked[static_cast<size_t>(i)]) *
                                    glm::vec4(0.f, 0.f, 0.f, 1.f));
          break;
        }
      }
      // Collect all bones from neck (or head) downward for head-only tilt.
      fpHeadTiltBoneIndices.clear();
      {
        std::string startName;
        if (fpNeckBoneIdx >= 0)
          startName = staffRig.boneNames[static_cast<size_t>(fpNeckBoneIdx)];
        else if (fpHeadBoneIdx >= 0)
          startName = staffRig.boneNames[static_cast<size_t>(fpHeadBoneIdx)];
        if (!startName.empty()) {
          std::vector<std::string> stk;
          stk.push_back(startName);
          while (!stk.empty()) {
            std::string cur = stk.back(); stk.pop_back();
            auto it = staffRig.boneNameToIndex.find(cur);
            if (it != staffRig.boneNameToIndex.end())
              fpHeadTiltBoneIndices.push_back(it->second);
            auto nit = staffRig.nodes.find(cur);
            if (nit != staffRig.nodes.end())
              for (const auto& ch : nit->second.children)
                stk.push_back(ch);
          }
          std::fprintf(stderr, "[head tilt] %zu bones from '%s'\n",
                       fpHeadTiltBoneIndices.size(), startName.c_str());
        }
      }
      avClipIdle = 0;
      avClipWalk = staffRig.clips.size() > 1 ? 1 : -1;
      avClipSprint = staffRig.clips.size() > 2 ? 2 : -1;
      avClipSlideRight = -1;
      avClipCrouchLeft = -1;
      avClipCrouchFwd = -1;
      avClipCrouchBack = -1;
      avClipCrouchRight = -1;
      avClipSlideLight = -1;
      avClipStepPush = -1;
      avClipCrouchIdleBow = -1;
      avClipLedgeClimb = -1;
      avClipJump = -1;
      avClipJumpRun = -1;
      avClipLand = -1;
      {
        auto tryAppendPlayerAnim = [&](const char* path, int& outIdx) {
          if (!path || !path[0])
            return;
          if (std::strcmp(path, VULKAN_GAME_EMPLOYEE_FBX) == 0)
            return;
          clipErr.clear();
          const int idx = static_cast<int>(staffRig.clips.size());
          if (staff_skin::appendAnimationFromGlb(path, staffRig, clipErr)) {
            outIdx = idx;
            const auto& cl = staffRig.clips[static_cast<size_t>(idx)];
            std::fprintf(stderr, "[player anim] clip %d loaded  channels=%zu  dur=%.3fs  path=%s\n",
                         idx, cl.channels.size(),
                         cl.ticksPerSecond > 1e-6 ? cl.duration / cl.ticksPerSecond : 0.0,
                         path);
          } else
            std::cerr << "[player anim] FAILED: " << clipErr << "  path=" << path << "\n";
        };
        auto tryAppendPlayerAnimRetargetFallback = [&](const char* path, int& outIdx, const char* label) {
          tryAppendPlayerAnim(path, outIdx);
          if (outIdx >= 0)
            return;
          if (!path || !path[0])
            return;
          clipErr.clear();
          int rtIdx = -1;
          if (staff_skin::appendLongestRetargetedClipFromGlb(path, staffRig, false, rtIdx, clipErr)) {
            outIdx = rtIdx;
            const auto& cl = staffRig.clips[static_cast<size_t>(rtIdx)];
            std::fprintf(stderr,
                         "[player anim] %s retarget fallback loaded  clip=%d  channels=%zu  dur=%.3fs  path=%s\n",
                         label ? label : "anim", rtIdx, cl.channels.size(),
                         cl.ticksPerSecond > 1e-6 ? cl.duration / cl.ticksPerSecond : 0.0, path);
          } else {
            std::cerr << "[player anim] " << (label ? label : "anim")
                      << " retarget fallback FAILED: " << clipErr
                      << "  path=" << path << "\n";
          }
        };
        auto tryAppendPlayerAnimLongestRetarget = [&](const char* path, int& outIdx, const char* label,
                                                     bool freeRootMotion = true) {
          if (!path || !path[0])
            return;
          if (std::strcmp(path, VULKAN_GAME_EMPLOYEE_FBX) == 0)
            return;
          clipErr.clear();
          int rtIdx = -1;
          // Prefer longest clip for shimmy assets because some GLBs put a static/preview take first.
          if (staff_skin::appendLongestRetargetedClipFromGlb(path, staffRig, freeRootMotion, rtIdx, clipErr)) {
            outIdx = rtIdx;
            const auto& cl = staffRig.clips[static_cast<size_t>(rtIdx)];
            std::fprintf(stderr,
                         "[player anim] %s longest retarget loaded  clip=%d  channels=%zu  dur=%.3fs  path=%s\n",
                         label ? label : "anim", rtIdx, cl.channels.size(),
                         cl.ticksPerSecond > 1e-6 ? cl.duration / cl.ticksPerSecond : 0.0, path);
            return;
          }
          // Fallback to normal + fallback chain for maximum compatibility.
          tryAppendPlayerAnimRetargetFallback(path, outIdx, label);
        };
        tryAppendPlayerAnim(VULKAN_GAME_MESHY_SLIDE_RIGHT_GLB, avClipSlideRight);
        tryAppendPlayerAnim(VULKAN_GAME_MESHY_CROUCH_WALK_LEFT_GLB, avClipCrouchLeft);
        tryAppendPlayerAnim(VULKAN_GAME_MESHY_CAUTIOUS_CROUCH_FWD_GLB, avClipCrouchFwd);
        tryAppendPlayerAnim(VULKAN_GAME_MESHY_CAUTIOUS_CROUCH_BACK_GLB, avClipCrouchBack);
        tryAppendPlayerAnim(VULKAN_GAME_MESHY_CROUCH_WALK_RIGHT_GLB, avClipCrouchRight);
        tryAppendPlayerAnim(VULKAN_GAME_MESHY_SLIDE_LIGHT_GLB, avClipSlideLight);
        tryAppendPlayerAnim(VULKAN_GAME_MESHY_STEP_PUSH_GLB, avClipStepPush);
        tryAppendPlayerAnim(VULKAN_GAME_MESHY_CROUCH_IDLE_BOW_GLB, avClipCrouchIdleBow);
        tryAppendPlayerAnimLongestRetarget(VULKAN_GAME_MESHY_SLOW_LADDER_CLIMB_GLB, avClipLedgeClimb,
                                           "slow-ladder");
        tryAppendPlayerAnimLongestRetarget(VULKAN_GAME_MESHY_LEDGE_GRAB_WALL_GLB, avClipLedgeGrab,
                                           "ledge-grab", false);
        tryAppendPlayerAnimLongestRetarget(VULKAN_GAME_MESHY_SHIMMY_LEFT_GLB, avClipShimmyLeft,
                                           "shimmy-left", false);
        tryAppendPlayerAnimLongestRetarget(VULKAN_GAME_MESHY_SHIMMY_RIGHT_GLB, avClipShimmyRight,
                                           "shimmy-right", false);
        tryAppendPlayerAnim(VULKAN_GAME_JUMPING_FBX, avClipJump);
#if defined(VULKAN_GAME_JUMP_RUN_GLB)
        tryAppendPlayerAnim(VULKAN_GAME_JUMP_RUN_GLB, avClipJumpRun);
#endif
        tryAppendPlayerAnim(VULKAN_GAME_MESHY_LAND_GLB, avClipLand);
      }
      staffClipMeleePunch = -1;
      staffClipMeleeKick = -1;
      staffClipMeleeFall = -1;
      staffClipMeleeStand = -1;
      staffClipShoveHair = -1;
      {
        auto tryAppendStaffAnim = [&](const char* envName, const char* compilePath, const char* logLabel,
                                      int& outClipIndex) {
          const char* p = (envName && envName[0]) ? std::getenv(envName) : nullptr;
          if (!p || !p[0])
            p = compilePath;
          if (!p || !p[0])
            return;
          if (std::strcmp(p, VULKAN_GAME_EMPLOYEE_FBX) == 0)
            return;
          clipErr.clear();
          const int idx = static_cast<int>(staffRig.clips.size());
          if (staff_skin::appendAnimationFromGlb(p, staffRig, clipErr)) {
            outClipIndex = idx;
            std::cout << "[staff] " << logLabel << " clip loaded\n";
          } else
            std::cerr << "[staff] " << logLabel << " clip: " << clipErr << "\n";
        };
#if defined(VULKAN_GAME_MESHY_PUNCH_GLB)
        tryAppendStaffAnim("VULKAN_GAME_STAFF_PUNCH_GLB", VULKAN_GAME_MESHY_PUNCH_GLB, "punch",
                           staffClipMeleePunch);
#endif
#if defined(VULKAN_GAME_MESHY_KICK_GLB)
        tryAppendStaffAnim("VULKAN_GAME_STAFF_KICK_GLB", VULKAN_GAME_MESHY_KICK_GLB, "kick",
                           staffClipMeleeKick);
#endif
#if defined(VULKAN_GAME_MESHY_FALL_GLB)
        tryAppendStaffAnim("VULKAN_GAME_STAFF_FALL_GLB", VULKAN_GAME_MESHY_FALL_GLB, "fall",
                           staffClipMeleeFall);
#endif
#if defined(VULKAN_GAME_MESHY_STANDUP_GLB)
        tryAppendStaffAnim("VULKAN_GAME_STAFF_STANDUP_GLB", VULKAN_GAME_MESHY_STANDUP_GLB, "stand up",
                           staffClipMeleeStand);
#endif
#if defined(VULKAN_GAME_MESHY_HAIR_GLB)
        tryAppendStaffAnim("VULKAN_GAME_STAFF_HAIR_GLB", VULKAN_GAME_MESHY_HAIR_GLB, "hair shove",
                           staffClipShoveHair);
#endif
#if defined(VULKAN_GAME_STAFF_SHREK_PROXIMITY_DANCE_GLB) || defined(VULKAN_GAME_SHREK_EGG_GLB)
        {
          std::string shrekDanceErr;
          int danceIdx = -1;
          const char* danceGlb = nullptr;
#if defined(VULKAN_GAME_STAFF_SHREK_PROXIMITY_DANCE_GLB)
          danceGlb = std::getenv("VULKAN_GAME_STAFF_SHREK_DANCE_GLB");
          if (!danceGlb || !danceGlb[0])
            danceGlb = VULKAN_GAME_STAFF_SHREK_PROXIMITY_DANCE_GLB;
#elif defined(VULKAN_GAME_SHREK_EGG_GLB)
          danceGlb = VULKAN_GAME_SHREK_EGG_GLB;
#endif
          if (danceGlb && staff_skin::appendLongestRetargetedClipFromGlb(danceGlb, staffRig, true, danceIdx,
                                                                        shrekDanceErr)) {
            staffClipShrekProximityDance = danceIdx;
            std::cout << "[staff] Shrek proximity dance clip index " << staffClipShrekProximityDance << " ("
                      << danceGlb << ")\n";
          } else {
            staffClipShrekProximityDance = -1;
            if (!shrekDanceErr.empty())
              std::cerr << "[staff] Shrek proximity dance clip: " << shrekDanceErr << "\n";
          }
        }
#endif
      }
      staff_skin::optimizeRigClips(staffRig);
      staffSkinnedActive = true;
      staffRigBoneCount = staffRig.boneCount;
      staffRebuildDeadRagdollSimBoneMap(staffRig);
      employeeBounds = computeEmployeeBoundsFromSkinnedMesh(skinLoad);
      employeeVertexCount = static_cast<uint32_t>(skinLoad.size());
      const VkDeviceSize empVSize = sizeof(staff_skin::SkinnedVertex) * skinLoad.size();
      VkBuffer empStaging = VK_NULL_HANDLE;
      VkDeviceMemory empStagingMem = VK_NULL_HANDLE;
      createBuffer(physicalDevice, device, empVSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   empStaging, empStagingMem);
      void* empPtr = nullptr;
      vkMapMemory(device, empStagingMem, 0, empVSize, 0, &empPtr);
      std::memcpy(empPtr, skinLoad.data(), static_cast<size_t>(empVSize));
      vkUnmapMemory(device, empStagingMem);
      createBuffer(physicalDevice, device, empVSize,
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, employeeVertexBuffer,
                   employeeVertexBufferMemory);
      copyBuffer(device, commandPool, graphicsQueue, empStaging, employeeVertexBuffer, empVSize);
      vkDestroyBuffer(device, empStaging, nullptr);
      vkFreeMemory(device, empStagingMem, nullptr);

      const VkDeviceSize empInstSize = sizeof(glm::mat4) * kStaffSkinnedInstanceSlots;
      createBuffer(physicalDevice, device, empInstSize,
                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   employeeInstanceBuffer, employeeInstanceBufferMemory);
      vkMapMemory(device, employeeInstanceBufferMemory, 0, empInstSize, 0, &employeeInstanceMapped);
      staffNpcDrawBuild.reserve(kMaxEmployees);
      shelfSepEmpScratch.reserve(256);
      shelfSepFootRScratch.reserve(256);
      createStaffSkinnedPipeline();
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
      {
        std::vector<staff_skin::SkinnedVertex> shrekVerts;
        std::vector<uint8_t> shrekDiffuseRgba;
        uint32_t shrekTexW = 0, shrekTexH = 0;
        std::string shrekErr;
        if (staff_skin::loadSkinnedIdleGlb(VULKAN_GAME_SHREK_EGG_GLB, 1.78f, shrekVerts, shrekEggRig, shrekErr,
                                           &shrekDiffuseRgba, &shrekTexW, &shrekTexH, true) &&
            shrekEggRig.boneCount > 0 &&
            shrekEggRig.boneCount <= staff_skin::kMaxPaletteBones && !shrekVerts.empty() &&
            !shrekEggRig.clips.empty()) {
          for (auto& c : shrekEggRig.clips) {
            c.lockLocomotionRoot = false;
            c.lockRootTranslationToBind = false;
            c.lockHipsTranslationToBind = false;
          }
          shrekEggAnimClipIndex = 0;
          double bestDur = 0.0;
          for (int ci = 0; ci < static_cast<int>(shrekEggRig.clips.size()); ++ci) {
            const double d = staff_skin::clipDuration(shrekEggRig, ci);
            if (d > bestDur) {
              bestDur = d;
              shrekEggAnimClipIndex = ci;
            }
          }
          shrekEggVertexCount = static_cast<uint32_t>(shrekVerts.size());
          const VkDeviceSize shrekVSize = sizeof(staff_skin::SkinnedVertex) * shrekVerts.size();
          VkBuffer shrekStaging = VK_NULL_HANDLE;
          VkDeviceMemory shrekStagingMem = VK_NULL_HANDLE;
          createBuffer(physicalDevice, device, shrekVSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       shrekStaging, shrekStagingMem);
          void* shrekPtr = nullptr;
          vkMapMemory(device, shrekStagingMem, 0, shrekVSize, 0, &shrekPtr);
          std::memcpy(shrekPtr, shrekVerts.data(), static_cast<size_t>(shrekVSize));
          vkUnmapMemory(device, shrekStagingMem);
          createBuffer(physicalDevice, device, shrekVSize,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shrekEggVertexBuffer,
                       shrekEggVertexBufferMemory);
          copyBuffer(device, commandPool, graphicsQueue, shrekStaging, shrekEggVertexBuffer, shrekVSize);
          vkDestroyBuffer(device, shrekStaging, nullptr);
          vkFreeMemory(device, shrekStagingMem, nullptr);
          shrekEggAssetLoaded = true;
          if (!shrekDiffuseRgba.empty() && shrekTexW > 0 && shrekTexH > 0) {
            createTextureResourcesFromRgbaLinear(shrekDiffuseRgba.data(), shrekTexW, shrekTexH,
                                                 shrekEggDiffuseImage, shrekEggDiffuseMemory,
                                                 shrekEggDiffuseView, shrekEggDiffuseSampler);
            shrekEggDiffuseLoaded = true;
          }
          std::cout << "[easter] Shrek GLB: mesh clips=" << shrekEggRig.clips.size()
                    << " animClip=" << shrekEggAnimClipIndex
                    << (shrekEggDiffuseLoaded ? " diffuse OK\n" : " (no embedded diffuse — grey)\n");
        } else if (!shrekErr.empty())
          std::cerr << "[easter] Shrek GLB: " << shrekErr << "\n";
      }
#endif
      std::cout << "[staff] skinned mesh " << VULKAN_GAME_EMPLOYEE_FBX << " bones=" << staffRigBoneCount
                << " clips=" << staffRig.clips.size() << "\n";
      if (!staffDiffuseRgba.empty() && staffDiffuseW > 0 && staffDiffuseH > 0) {
        createTextureResourcesFromRgbaLinear(staffDiffuseRgba.data(), staffDiffuseW, staffDiffuseH,
                                             staffGlbDiffuseImage, staffGlbDiffuseMemory,
                                             staffGlbDiffuseView, staffGlbDiffuseSampler);
        staffGlbDiffuseActive = 1;
        std::cout << "[staff] embedded diffuse " << staffDiffuseW << "x" << staffDiffuseH << "\n";
      }
    } else {
      if (trySkinnedGlb)
        std::cerr << "[staff] skinned load failed, trying static mesh: " << empErr << "\n";
      empErr.clear();
      if (emp_mesh::loadFbx(VULKAN_GAME_EMPLOYEE_FBX, kEmployeeVisualHeight, empLoad, empErr,
                            &staffDiffuseRgba, &staffDiffuseW, &staffDiffuseH,
                            gGamePerf.employeeMeshMaxVerts)) {
        employeeBounds = computeEmployeeBoundsFromMesh(empLoad);
        employeeVertexCount = static_cast<uint32_t>(empLoad.size());
        const VkDeviceSize empVSize = sizeof(Vertex) * empLoad.size();
        VkBuffer empStaging = VK_NULL_HANDLE;
        VkDeviceMemory empStagingMem = VK_NULL_HANDLE;
        createBuffer(physicalDevice, device, empVSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     empStaging, empStagingMem);
        void* empPtr = nullptr;
        vkMapMemory(device, empStagingMem, 0, empVSize, 0, &empPtr);
        std::memcpy(empPtr, empLoad.data(), static_cast<size_t>(empVSize));
        vkUnmapMemory(device, empStagingMem);
        createBuffer(physicalDevice, device, empVSize,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, employeeVertexBuffer,
                     employeeVertexBufferMemory);
        copyBuffer(device, commandPool, graphicsQueue, empStaging, employeeVertexBuffer, empVSize);
        vkDestroyBuffer(device, empStaging, nullptr);
        vkFreeMemory(device, empStagingMem, nullptr);

        const VkDeviceSize empInstSize = sizeof(glm::mat4) * kMaxEmployees;
        createBuffer(physicalDevice, device, empInstSize,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     employeeInstanceBuffer, employeeInstanceBufferMemory);
        vkMapMemory(device, employeeInstanceBufferMemory, 0, empInstSize, 0, &employeeInstanceMapped);
        staffNpcDrawBuild.reserve(kMaxEmployees);
        std::cout << "[staff] mesh (static) " << VULKAN_GAME_EMPLOYEE_FBX << "\n";
        if (!staffDiffuseRgba.empty() && staffDiffuseW > 0 && staffDiffuseH > 0) {
          createTextureResourcesFromRgbaLinear(staffDiffuseRgba.data(), staffDiffuseW, staffDiffuseH,
                                               staffGlbDiffuseImage, staffGlbDiffuseMemory,
                                               staffGlbDiffuseView, staffGlbDiffuseSampler);
          staffGlbDiffuseActive = 1;
          std::cout << "[staff] embedded diffuse " << staffDiffuseW << "x" << staffDiffuseH << "\n";
        }
      } else {
        std::cerr << "[employee] mesh load failed: " << empErr << " (" << VULKAN_GAME_EMPLOYEE_FBX
                  << ")\n";
        employeeVertexCount = 0;
        employeeBounds = glm::vec4(0.88f, 1.39f, 0.90f, 0.24f);
      }
    }
    if (staffGlbDiffuseView == VK_NULL_HANDLE) {
      static const uint8_t kWhitePx[4] = {255, 255, 255, 255};
      createTextureResourcesFromRgbaLinear(kWhitePx, 1, 1, staffGlbDiffuseImage, staffGlbDiffuseMemory,
                                           staffGlbDiffuseView, staffGlbDiffuseSampler);
    }

    const float cw = static_cast<float>(kChunkCellCount) * kCellSize;
    int pcx = static_cast<int>(std::floor(camPos.x / cw));
    int pcz = static_cast<int>(std::floor(camPos.z / cw));
    std::vector<Vertex> initTerrain;
    buildTerrainMesh(pcx, pcz, initTerrain);
    groundVertexCount = static_cast<uint32_t>(initTerrain.size());
    std::memcpy(groundMapped, initTerrain.data(), initTerrain.size() * sizeof(Vertex));

    std::vector<Vertex> initCeiling;
    buildCeilingMesh(pcx, pcz, initCeiling);
    ceilingVertexCount = static_cast<uint32_t>(initCeiling.size());
    std::memcpy(ceilingMapped, initCeiling.data(), initCeiling.size() * sizeof(Vertex));

    lastTerrainChunkX = pcx;
    lastTerrainChunkZ = pcz;
  }

  void createUniformBuffers() {
    uniformBuffers.resize(kMaxFramesInFlight);
    uniformBuffersMemory.resize(kMaxFramesInFlight);
    uniformBuffersMapped.resize(kMaxFramesInFlight);
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      createBuffer(physicalDevice, device, sizeof(UniformBufferObject),
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   uniformBuffers[i], uniformBuffersMemory[i]);
      vkMapMemory(device, uniformBuffersMemory[i], 0, sizeof(UniformBufferObject), 0,
                  &uniformBuffersMapped[i]);
    }
  }

  void createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 3> ps{};
    ps[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ps[0].descriptorCount = static_cast<uint32_t>(kMaxFramesInFlight);
    ps[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps[1].descriptorCount =
        static_cast<uint32_t>(kMaxFramesInFlight) * (5u + kMaxExtraTextures + 1u + 1u + 1u + 1u + 1u + 1u + 1u + 1u + 1u);
    ps[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps[2].descriptorCount = static_cast<uint32_t>(kMaxFramesInFlight);
    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = static_cast<uint32_t>(kMaxFramesInFlight) * 2u;
    ci.poolSizeCount = static_cast<uint32_t>(ps.size());
    ci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(device, &ci, nullptr, &descriptorPool), "descriptorPool");
  }

  void createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(kMaxFramesInFlight, descriptorSetLayout);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descriptorPool;
    ai.descriptorSetCount = static_cast<uint32_t>(kMaxFramesInFlight);
    ai.pSetLayouts = layouts.data();
    descriptorSets.resize(kMaxFramesInFlight);
    VK_CHECK(vkAllocateDescriptorSets(device, &ai, descriptorSets.data()), "allocateDescriptorSets");

    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      VkDescriptorBufferInfo bi{};
      bi.buffer = uniformBuffers[i];
      bi.offset = 0;
      bi.range = sizeof(UniformBufferObject);
      VkDescriptorImageInfo ii{};
      ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      ii.imageView = sceneTextureView;
      ii.sampler = sceneTextureSampler;
      VkDescriptorImageInfo si{};
      si.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      si.imageView = signTextureView;
      si.sampler = signTextureSampler;
      VkDescriptorImageInfo sh{};
      sh.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      sh.imageView = shelfRackTextureView;
      sh.sampler = shelfRackTextureSampler;
      VkDescriptorImageInfo cr{};
      cr.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      cr.imageView = crateTextureView;
      cr.sampler = crateTextureSampler;
      VkDescriptorImageInfo pl{};
      pl.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      pl.imageView = palletTextureView;
      pl.sampler = palletTextureSampler;
      std::array<VkWriteDescriptorSet, 6> w{};
      w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[0].dstSet = descriptorSets[i];
      w[0].dstBinding = 0;
      w[0].descriptorCount = 1;
      w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      w[0].pBufferInfo = &bi;
      w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[1].dstSet = descriptorSets[i];
      w[1].dstBinding = 1;
      w[1].descriptorCount = 1;
      w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w[1].pImageInfo = &ii;
      w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[2].dstSet = descriptorSets[i];
      w[2].dstBinding = 2;
      w[2].descriptorCount = 1;
      w[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w[2].pImageInfo = &si;
      w[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[3].dstSet = descriptorSets[i];
      w[3].dstBinding = 3;
      w[3].descriptorCount = 1;
      w[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w[3].pImageInfo = &sh;
      w[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[4].dstSet = descriptorSets[i];
      w[4].dstBinding = 4;
      w[4].descriptorCount = 1;
      w[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w[4].pImageInfo = &cr;
      std::array<VkDescriptorImageInfo, kMaxExtraTextures> extraInfos{};
      for (uint32_t ti = 0; ti < kMaxExtraTextures; ++ti) {
        extraInfos[ti].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        extraInfos[ti].imageView = extraTexSlots[ti].view;
        extraInfos[ti].sampler = extraTexSlots[ti].sampler;
      }
      w[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[5].dstSet = descriptorSets[i];
      w[5].dstBinding = 5;
      w[5].descriptorCount = kMaxExtraTextures;
      w[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w[5].pImageInfo = extraInfos.data();
      VkDescriptorImageInfo stGlb{};
      stGlb.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      stGlb.imageView = staffGlbDiffuseView;
      stGlb.sampler = staffGlbDiffuseSampler;
      std::array<VkWriteDescriptorSet, 7> w7{};
      for (int j = 0; j < 6; ++j)
        w7[static_cast<size_t>(j)] = w[static_cast<size_t>(j)];
      w7[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w7[6].dstSet = descriptorSets[i];
      w7[6].dstBinding = 6;
      w7[6].descriptorCount = 1;
      w7[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w7[6].pImageInfo = &stGlb;
      VkDescriptorImageInfo shrekEggDesc{};
      shrekEggDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
      if (shrekEggDiffuseLoaded && shrekEggDiffuseView != VK_NULL_HANDLE) {
        shrekEggDesc.imageView = shrekEggDiffuseView;
        shrekEggDesc.sampler = shrekEggDiffuseSampler;
      } else
#endif
      {
        shrekEggDesc.imageView = stGlb.imageView;
        shrekEggDesc.sampler = stGlb.sampler;
      }
      VkDescriptorBufferInfo boneBi{};
      boneBi.buffer = staffBoneSsbBuffer;
      boneBi.offset = 0;
      boneBi.range = sizeof(glm::mat4) * kStaffPaletteBoneCount * kStaffSkinnedInstanceSlots;
      VkDescriptorImageInfo hudFontDesc{};
      hudFontDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      hudFontDesc.imageView = hudFontTextureView;
      hudFontDesc.sampler = hudFontTextureSampler;
      VkDescriptorImageInfo titleIkeaLogoDesc{};
      titleIkeaLogoDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      titleIkeaLogoDesc.imageView = titleIkeaLogoView;
      titleIkeaLogoDesc.sampler = titleIkeaLogoSampler;
      VkDescriptorImageInfo boxCutterDesc{};
      boxCutterDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      boxCutterDesc.imageView = shelfBoxCutterTextureView;
      boxCutterDesc.sampler = shelfBoxCutterTextureSampler;
      VkDescriptorImageInfo rustyPipeDesc{};
      rustyPipeDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      rustyPipeDesc.imageView = shelfRustyPipeTextureView;
      rustyPipeDesc.sampler = shelfRustyPipeTextureSampler;
      VkDescriptorImageInfo deliMetalDesc{};
      deliMetalDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      deliMetalDesc.imageView = deliMetalTextureView;
      deliMetalDesc.sampler = deliMetalTextureSampler;
      VkDescriptorImageInfo deliBaseDesc{};
      deliBaseDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      deliBaseDesc.imageView = deliBaseTextureView;
      deliBaseDesc.sampler = deliBaseTextureSampler;
      std::array<VkWriteDescriptorSet, 16> w14{};
      for (int j = 0; j < 7; ++j)
        w14[static_cast<size_t>(j)] = w7[static_cast<size_t>(j)];
      w14[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w14[7].dstSet = descriptorSets[i];
      w14[7].dstBinding = 7;
      w14[7].descriptorCount = 1;
      w14[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w14[7].pBufferInfo = &boneBi;
      w14[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w14[8].dstSet = descriptorSets[i];
      w14[8].dstBinding = 8;
      w14[8].descriptorCount = 1;
      w14[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w14[8].pImageInfo = &shrekEggDesc;
      w14[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w14[9].dstSet = descriptorSets[i];
      w14[9].dstBinding = 9;
      w14[9].descriptorCount = 1;
      w14[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w14[9].pImageInfo = &hudFontDesc;
      w14[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w14[10].dstSet = descriptorSets[i];
      w14[10].dstBinding = 10;
      w14[10].descriptorCount = 1;
      w14[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w14[10].pImageInfo = &titleIkeaLogoDesc;
      w14[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w14[11].dstSet = descriptorSets[i];
      w14[11].dstBinding = 11;
      w14[11].descriptorCount = 1;
      w14[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w14[11].pImageInfo = &boxCutterDesc;
      w14[12].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w14[12].dstSet = descriptorSets[i];
      w14[12].dstBinding = 12;
      w14[12].descriptorCount = 1;
      w14[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w14[12].pImageInfo = &rustyPipeDesc;
      w14[13].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w14[13].dstSet = descriptorSets[i];
      w14[13].dstBinding = 13;
      w14[13].descriptorCount = 1;
      w14[13].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w14[13].pImageInfo = &pl;
      w14[14].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w14[14].dstSet = descriptorSets[i];
      w14[14].dstBinding = 14;
      w14[14].descriptorCount = 1;
      w14[14].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w14[14].pImageInfo = &deliMetalDesc;
      w14[15].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w14[15].dstSet = descriptorSets[i];
      w14[15].dstBinding = 15;
      w14[15].descriptorCount = 1;
      w14[15].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w14[15].pImageInfo = &deliBaseDesc;
      vkUpdateDescriptorSets(device, static_cast<uint32_t>(w14.size()), w14.data(), 0, nullptr);
    }
  }

  void createPostDescriptorSets() {
    if (postDescriptorSets.empty()) {
      postDescriptorSets.resize(static_cast<size_t>(kMaxFramesInFlight));
      std::vector<VkDescriptorSetLayout> lays(static_cast<size_t>(kMaxFramesInFlight),
                                              postDescriptorSetLayout);
      VkDescriptorSetAllocateInfo ai{};
      ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      ai.descriptorPool = descriptorPool;
      ai.descriptorSetCount = static_cast<uint32_t>(kMaxFramesInFlight);
      ai.pSetLayouts = lays.data();
      VK_CHECK(vkAllocateDescriptorSets(device, &ai, postDescriptorSets.data()), "postDescriptorSets");
    }
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      VkDescriptorImageInfo ii{};
      ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      ii.imageView = sceneColorViews[static_cast<size_t>(i)];
      ii.sampler = sceneRenderSampler;
      VkWriteDescriptorSet w{};
      w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w.dstSet = postDescriptorSets[static_cast<size_t>(i)];
      w.dstBinding = 0;
      w.descriptorCount = 1;
      w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w.pImageInfo = &ii;
      vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }
  }

  void createCommandBuffers() {
    commandBuffers.resize(framebuffers.size());
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    VK_CHECK(vkAllocateCommandBuffers(device, &ai, commandBuffers.data()), "commandBuffers");
  }

  void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex, uint32_t flightIdx, bool storeLit,
                           const glm::vec3& sceneViewEye, const glm::vec3& sceneViewFwd) {
    PerfScopeTimer _recT(&gPerfTrace.accumRecordCmd, gPerfTrace.enabled);
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &bi);

    VkImageMemoryBarrier sceneColBar{};
    sceneColBar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    sceneColBar.srcAccessMask =
        sceneColorWasSampled[static_cast<size_t>(flightIdx)] ? VK_ACCESS_SHADER_READ_BIT : 0;
    sceneColBar.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    sceneColBar.oldLayout = sceneColorWasSampled[static_cast<size_t>(flightIdx)]
                                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                : VK_IMAGE_LAYOUT_UNDEFINED;
    sceneColBar.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    sceneColBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sceneColBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    sceneColBar.image = sceneColorImages[static_cast<size_t>(flightIdx)];
    sceneColBar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    sceneColBar.subresourceRange.levelCount = 1;
    sceneColBar.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(
        cmd,
        sceneColorWasSampled[static_cast<size_t>(flightIdx)]
            ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &sceneColBar);

    if (!depthGpuReady[static_cast<size_t>(flightIdx)]) {
      VkImageMemoryBarrier dbar{};
      dbar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      dbar.srcAccessMask = 0;
      dbar.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      dbar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      dbar.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      dbar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      dbar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      dbar.image = depthImages[static_cast<size_t>(flightIdx)];
      dbar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      dbar.subresourceRange.levelCount = 1;
      dbar.subresourceRange.layerCount = 1;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                           0, 0, nullptr, 0, nullptr, 1, &dbar);
      depthGpuReady[static_cast<size_t>(flightIdx)] = true;
    }

    std::array<VkClearValue, 2> clears{};
    if (inLoadingScreen)
      clears[0].color = {{0.15f, 0.15f, 0.16f, 1.0f}};
    else if (storeLit)
      clears[0].color = {{0.78f, 0.79f, 0.82f, 1.0f}};
    else
      clears[0].color = {{0.018f, 0.021f, 0.034f, 1.0f}};
    clears[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = renderPass;
    rp.framebuffer = sceneFramebuffers[static_cast<size_t>(flightIdx)];
    rp.renderArea.offset = {0, 0};
    rp.renderArea.extent = sceneExtent;
    rp.clearValueCount = static_cast<uint32_t>(clears.size());
    rp.pClearValues = clears.data();
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport sceneVp{};
    sceneVp.x = 0;
    sceneVp.y = 0;
    sceneVp.width = static_cast<float>(sceneExtent.width);
    sceneVp.height = static_cast<float>(sceneExtent.height);
    sceneVp.minDepth = 0;
    sceneVp.maxDepth = 1;
    VkRect2D sceneSc{};
    sceneSc.offset = {0, 0};
    sceneSc.extent = sceneExtent;
    vkCmdSetViewport(cmd, 0, 1, &sceneVp);
    vkCmdSetScissor(cmd, 0, 1, &sceneSc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                            &descriptorSets[flightIdx], 0, nullptr);

    // View eye + forward for distance LOD (same as drawFrame lookAt). Full frustum is still skipped
    // (avoids lateral popping); we only drop instances clearly behind the camera on CPU.
    const float sceneFocusX = sceneViewEye.x;
    const float sceneFocusY = sceneViewEye.y;
    const float sceneFocusZ = sceneViewEye.z;
    const glm::vec2 lodHFwd = lodHorizViewDir(sceneViewFwd);

    PushModel push{};
    push.model = glm::mat4(1.0f);

    if (inLoadingScreen)
      goto skipSceneGeometry;

    {
    const VkDeviceSize bindOffs[2] = {0, 0};
    VkBuffer vbGround[2] = {groundVertexBuffer, identityInstanceBuffer};
    vkCmdBindVertexBuffers(cmd, 0, 2, vbGround, bindOffs);
    vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel), &push);
    {
      // LOD: single active terrain chunk — draw whenever streamed in (no view-frustum test).
      const bool drawGround = (lastTerrainChunkX != INT_MAX);
      if (drawGround && groundVertexCount > 0)
        vkCmdDraw(cmd, groundVertexCount, 1, 0, 0);
    }

    VkBuffer vbCeil[2] = {ceilingVertexBuffer, identityInstanceBuffer};
    vkCmdBindVertexBuffers(cmd, 0, 2, vbCeil, bindOffs);
    vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel), &push);
    {
      const bool drawCeil = (lastTerrainChunkX != INT_MAX);
      if (drawCeil && ceilingVertexCount > 0)
        vkCmdDraw(cmd, ceilingVertexCount, 1, 0, 0);
    }

    if (storeLit) {
      fluorescentInstanceScratch.clear();
      const int flx = static_cast<int>(std::floor(sceneFocusX / kFluorescentGridCell));
      const int flz = static_cast<int>(std::floor(sceneFocusZ / kFluorescentGridCell));
      const int flRad = gGamePerf.fluorescentGridRadius;
      const float ex = sceneFocusX;
      const float ey = sceneFocusY;
      const float ez = sceneFocusZ;
      const float flLod = gGamePerf.shelfCullHardDist * 1.08f;
      const float flLodSq = flLod * flLod;
      for (int fdx = -flRad; fdx <= flRad; ++fdx) {
        for (int fdz = -flRad; fdz <= flRad; ++fdz) {
          const int Fix = flx + fdx;
          const int Fiz = flz + fdz;
          const float wx = (static_cast<float>(Fix) + 0.5f) * kFluorescentGridCell;
          const float wz = (static_cast<float>(Fiz) + 0.5f) * kFluorescentGridCell;
          const float fdxw = wx - ex;
          const float fdy = kFluorescentCullCenterY - ey;
          const float fdzw = wz - ez;
          if (lodBehindCamera3D(sceneViewEye, glm::vec3(wx, kFluorescentCullCenterY, wz), sceneViewFwd,
                                kLodBehindMarginFluor))
            continue;
          if (fdxw * fdxw + fdy * fdy + fdzw * fdzw > flLodSq)
            continue;
          if (fluorescentInstanceScratch.size() >= kMaxFluorescentInstances)
            break;
          fluorescentInstanceScratch.push_back(
              glm::translate(glm::mat4(1.f), glm::vec3(wx, kCeilingY, wz)));
        }
        if (fluorescentInstanceScratch.size() >= kMaxFluorescentInstances)
          break;
      }
      if (!fluorescentInstanceScratch.empty() && fluorescentInstanceMapped != nullptr) {
        const size_t nFl = fluorescentInstanceScratch.size();
        std::memcpy(fluorescentInstanceMapped, fluorescentInstanceScratch.data(),
                    sizeof(glm::mat4) * nFl);
        VkBuffer vbFl[2] = {fluorescentVertexBuffer, fluorescentInstanceBuffer};
        vkCmdBindVertexBuffers(cmd, 0, 2, vbFl, bindOffs);
        push.model = glm::mat4(1.f);
        vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                           &push);
        vkCmdDraw(cmd, fluorescentVertexCount, static_cast<uint32_t>(nFl), 0, 0);
      }
    }

    pillarInstanceScratch.clear();
    const int pcx = static_cast<int>(std::floor(sceneFocusX / kPillarSpacing));
    const int pcz = static_cast<int>(std::floor(sceneFocusZ / kPillarSpacing));
    const int pillRad = gGamePerf.pillarDrawGridRadius;
    const float pillarLodSq = gGamePerf.shelfCullHardDist * gGamePerf.shelfCullHardDist * 1.02f;
    for (int dx = -pillRad; dx <= pillRad; ++dx) {
      for (int dz = -pillRad; dz <= pillRad; ++dz) {
        const float px = static_cast<float>(pcx + dx) * kPillarSpacing;
        const float pz = static_cast<float>(pcz + dz) * kPillarSpacing;
        const float pdx = px - sceneFocusX;
        const float pdz = pz - sceneFocusZ;
        if (lodBehindCameraXZ(sceneFocusX, sceneFocusZ, px, pz, lodHFwd, kLodBehindMarginPillar))
          continue;
        if (pdx * pdx + pdz * pdz > pillarLodSq)
          continue;
        if (pillarInstanceScratch.size() >= kMaxPillarInstances)
          break;
        pillarInstanceScratch.push_back(glm::translate(glm::mat4(1.f), glm::vec3(px, kGroundY, pz)));
      }
      if (pillarInstanceScratch.size() >= kMaxPillarInstances)
        break;
    }
    if (!pillarInstanceScratch.empty() && pillarInstanceMapped != nullptr) {
      const size_t nPill = pillarInstanceScratch.size();
      std::memcpy(pillarInstanceMapped, pillarInstanceScratch.data(), sizeof(glm::mat4) * nPill);
      VkBuffer vbPill[2] = {pillarVertexBuffer, pillarInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbPill, bindOffs);
      push.model = glm::mat4(1.f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                         &push);
      vkCmdDraw(cmd, pillarVertexCount, static_cast<uint32_t>(nPill), 0, 0);
    }

    // Metal warehouse shelving — instanced: one draw per mesh type. CPU uses distance LOD +
    // behind-camera cull; shelfDetailCullDist trims props (3D); staff palette LOD / freeze in palette loop.
    {
    PerfScopeTimer _shelfCpu(&gPerfTrace.accumShelfBuild, gPerfTrace.enabled);
    shelfInstanceScratch.clear();
    shelfCrateInstanceScratch.clear();
    shelfPalletInstanceScratch.clear();
    shelfLadderInstanceScratch.clear();
    shelfBoxCutterInstanceScratch.clear();
    shelfRustyPipeInstanceScratch.clear();
    marketInstanceScratch.clear();
    deliPizzaInstanceScratch.clear();
    deliMeatballInstanceScratch.clear();
    staffNpcDrawBuild.clear();
    const bool storeLitForStaffAnim = storeLit;
    int waMin, waMax, wlMin, wlMax;
    // CPU: shelfSlotOccupied is cached (pillar/AABB); outer loop still scales with grid area.
    // Shelf scan range is runtime-tunable for low-end machines.
    const float shelfGridRangeM = std::min(gGamePerf.shelfCullHardDist, gGamePerf.shelfGridCpuScanRangeM);
    shelfGridWindowForRange(sceneFocusX, sceneFocusZ, shelfGridRangeM, waMin, waMax, wlMin, wlMax);
    const float shelfHardSq = gGamePerf.shelfCullHardDist * gGamePerf.shelfCullHardDist;
    const float shelfDetailSq = gGamePerf.shelfDetailCullDist * gGamePerf.shelfDetailCullDist;
    for (int wa = waMin; wa <= waMax && marketInstanceScratch.size() < kMaxMarketInstances; ++wa) {
      const float aisleCX = (static_cast<float>(wa) + 0.5f) * kShelfAisleModulePitch;
      const float nearDx = std::abs(aisleCX - sceneFocusX);
      if (nearDx * nearDx > shelfHardSq) continue;
      const float maxDz = std::sqrt(std::max(0.f, shelfHardSq - nearDx * nearDx));
      const int alMin = std::max(wlMin, static_cast<int>(std::floor((sceneFocusZ - maxDz) / kShelfAlongAislePitch)));
      const int alMax = std::min(wlMax, static_cast<int>(std::ceil((sceneFocusZ + maxDz) / kShelfAlongAislePitch)));
      for (int wl = alMin; wl <= alMax && marketInstanceScratch.size() < kMaxMarketInstances; ++wl) {
        if (!deliBarSlotOccupied(wa, wl)) continue;
        const float cx = (static_cast<float>(wa) + 0.5f) * kShelfAisleModulePitch;
        const float cz = (static_cast<float>(wl) + 0.5f) * kShelfAlongAislePitch;
        const float ddx = cx - sceneFocusX;
        const float ddz = cz - sceneFocusZ;
        if (ddx * ddx + ddz * ddz > shelfHardSq) continue;
        marketInstanceScratch.push_back(glm::translate(glm::mat4(1.f), {cx, kGroundY, cz}));
        const float foodDistSq = ddx * ddx + ddz * ddz;
        if (foodDistSq > kDeliFoodRenderDist * kDeliFoodRenderDist)
          continue; // Prevent distant edge-on food strips from showing at spawn.
        const bool meatballCounter = deliCounterUsesMeatballs(wa, wl);
        if (!meatballCounter) {
          const int slicesRem = deliPizzaSlicesRemaining(wa, wl);
          if (gDeliPizzaMeshLoaded && slicesRem > 0 &&
              deliPizzaInstanceScratch.size() + static_cast<size_t>(slicesRem) <= kMaxDeliPizzaInstances) {
            const float pizzaY = kGroundY + kDeliCounterH + kDeliCounterTopH + 0.010f;
            for (int si = 0; si < slicesRem; ++si) {
              const float angle =
                  static_cast<float>(si) * glm::two_pi<float>() / static_cast<float>(kDeliPizzaSlicesPerCounter);
              glm::mat4 m(1.f);
              m = glm::translate(m, {cx, pizzaY, cz});
              m = glm::rotate(m, angle, {0.f, 1.f, 0.f});
              deliPizzaInstanceScratch.push_back(m);
            }
          }
        } else {
          const int meatballsRem = deliMeatballsRemaining(wa, wl);
          if (gDeliMeatballMeshLoaded && meatballsRem > 0 &&
              deliMeatballInstanceScratch.size() + static_cast<size_t>(meatballsRem) <= kMaxDeliMeatballInstances) {
            const float mbY = kGroundY + kDeliCounterH + kDeliCounterTopH + 0.070f;
            constexpr float kRingR = 0.14f;
            for (int mi = 0; mi < meatballsRem; ++mi) {
              const float angle =
                  static_cast<float>(mi) * glm::two_pi<float>() / static_cast<float>(kDeliMeatballsPerCounter);
              glm::mat4 m(1.f);
              m = glm::translate(m, {cx, mbY, cz});
              m = glm::rotate(m, angle, {0.f, 1.f, 0.f});
              m = glm::translate(m, {kRingR, 0.f, 0.f});
              deliMeatballInstanceScratch.push_back(m);
            }
          }
        }
      }
    }
    for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
      const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
      const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
      const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
      // Per-aisle X distance → clip the Z range that could be within shelfHardDist.
      // Aisles far in X get a narrow Z band; aisles near the camera iterate fully.
      const float nearDx = std::min(std::abs(cxLeft - sceneFocusX), std::abs(cxRight - sceneFocusX));
      if (nearDx * nearDx > shelfHardSq)
        continue;
      const float maxDz = std::sqrt(std::max(0.f, shelfHardSq - nearDx * nearDx));
      const int wlMinA = std::max(wlMin, static_cast<int>(std::floor((sceneFocusZ - maxDz) / kShelfAlongAislePitch)));
      const int wlMaxA = std::min(wlMax, static_cast<int>(std::ceil((sceneFocusZ + maxDz) / kShelfAlongAislePitch)));
      for (int worldAlong = wlMinA; worldAlong <= wlMaxA; ++worldAlong) {
        const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
        {
          const float dz0 = cz - sceneFocusZ;
          const float dxL = cxLeft - sceneFocusX;
          const float dxR = cxRight - sceneFocusX;
          if (dxL * dxL + dz0 * dz0 > shelfHardSq && dxR * dxR + dz0 * dz0 > shelfHardSq)
            continue;
        }
        // Early reject whole row when both shelf sides sit behind the camera half-space.
        if (lodBehindCameraXZ(sceneFocusX, sceneFocusZ, cxLeft, cz, lodHFwd, kLodBehindMarginShelf) &&
            lodBehindCameraXZ(sceneFocusX, sceneFocusZ, cxRight, cz, lodHFwd, kLodBehindMarginShelf))
          continue;
        for (int side = 0; side < 2; ++side) {
          const float cx = side ? cxRight : cxLeft;
          if (!shelfSlotOccupied(worldAisle, worldAlong, side))
            continue;
          const float dxs = cx - sceneFocusX;
          const float dzs = cz - sceneFocusZ;
          {
            if (dxs * dxs + dzs * dzs > shelfHardSq)
              continue;
            if (lodBehindCameraXZ(sceneFocusX, sceneFocusZ, cx, cz, lodHFwd, kLodBehindMarginShelf))
              continue;
          }
          const float yawDeg = side ? -90.0f : 90.0f;
          {
            const float shelfCY = kGroundY + kShelfMeshHeight * 0.5f;
            const float ddy = shelfCY - sceneFocusY;
            const float dist3Sq = dxs * dxs + dzs * dzs + ddy * ddy;
            if (dist3Sq > shelfHardSq)
              continue;
            const float rim =
                gGamePerf.shelfCullNominalDist + shelfCullRimJitter(worldAisle, worldAlong) * kShelfCullRimJitterM;
            if (dist3Sq > rim * rim)
              continue;
          }
          const glm::mat4 shelfModel = glm::translate(glm::mat4(1.0f), glm::vec3(cx, kGroundY, cz)) *
                                       glm::rotate(glm::mat4(1.0f), glm::radians(yawDeg),
                                                   glm::vec3(0.f, 1.f, 0.f));
          shelfInstanceScratch.push_back(shelfModel);
          // Detail payload (crates/ladders/props/NPC seeds) is expensive; trim for low-end profiles.
          // Use 3D distance (matches shelf hard cull) so vertical camera offset affects LOD smoothly.
          {
            const float shelfCY = kGroundY + kShelfMeshHeight * 0.5f;
            const float ddyDet = shelfCY - sceneFocusY;
            if (dxs * dxs + dzs * dzs + ddyDet * ddyDet > shelfDetailSq)
              continue;
          }
          float lx, lz, yDeck, hx, hy, hz;
          const bool hasCrate = shelfCrateLocalLayout(worldAisle, worldAlong, side, lx, lz, yDeck, hx, hy, hz);
          if (shelfCrateInstanceScratch.size() + shelfPalletInstanceScratch.size() < kMaxShelfCrates &&
              hasCrate) {
            const float shelfYawRad = glm::radians(yawDeg);
            const glm::mat4 crateModel =
                glm::translate(glm::mat4(1.f), glm::vec3(cx, kGroundY, cz)) *
                glm::rotate(glm::mat4(1.f), shelfYawRad, glm::vec3(0.f, 1.f, 0.f)) *
                glm::translate(glm::mat4(1.f), glm::vec3(lx, yDeck + hy, lz)) *
                glm::scale(glm::mat4(1.f), glm::vec3(2.f * hx, 2.f * hy, 2.f * hz));
            shelfCrateInstanceScratch.push_back(crateModel);
          }
          float plx, plz, pyDeck, phx, phy, phz, yaw0, yaw1, offX, offZ;
          int stackCount = 0;
          if (shelfCrateInstanceScratch.size() + shelfPalletInstanceScratch.size() < kMaxShelfCrates &&
              shelfPalletLocalLayout(worldAisle, worldAlong, side, hasCrate, plx, plz, pyDeck, phx, phy, phz,
                                     stackCount, yaw0, yaw1, offX, offZ)) {
            const float shelfYawRad = glm::radians(yawDeg);
            const glm::mat4 shelfRoot =
                glm::translate(glm::mat4(1.f), glm::vec3(cx, kGroundY, cz)) *
                glm::rotate(glm::mat4(1.f), shelfYawRad, glm::vec3(0.f, 1.f, 0.f));
            for (int si = 0;
                 si < stackCount &&
                 shelfCrateInstanceScratch.size() + shelfPalletInstanceScratch.size() < kMaxShelfCrates;
                 ++si) {
              const float ly = pyDeck + phy + static_cast<float>(si) * (2.f * phy + 0.015f);
              const float yawLocal = (si == 0) ? yaw0 : (yaw0 + yaw1);
              const float ox = (si == 0) ? 0.f : offX;
              const float oz = (si == 0) ? 0.f : offZ;
              const glm::mat4 palletModel = shelfRoot *
                                            glm::translate(glm::mat4(1.f), glm::vec3(plx + ox, ly, plz + oz)) *
                                            glm::rotate(glm::mat4(1.f), yawLocal, glm::vec3(0.f, 1.f, 0.f)) *
                                            glm::scale(glm::mat4(1.f), glm::vec3(2.f * phx, 2.f * phy, 2.f * phz));
              shelfPalletInstanceScratch.push_back(palletModel);
            }
          }
          float llx, llz, leanR;
          glm::vec3 ladderScl;
          if (shelfLadderInstanceScratch.size() < kMaxShelfLadders &&
              shelfLadderLocalLayout(worldAisle, worldAlong, side, llx, llz, leanR, ladderScl)) {
            const glm::mat4 ladderModel =
                shelfLadderWorldModel(cx, cz, yawDeg, llx, llz, leanR, ladderScl);
            shelfLadderInstanceScratch.push_back(ladderModel);
          }
          float bcLx, bcLz, bcYDeck, bcYaw, bcScl;
          if (shelfBoxCutterVertexCount > 0 && shelfBoxCutterVertexBuffer != VK_NULL_HANDLE &&
              shelfBoxCutterInstanceMapped != nullptr &&
              shelfBoxCutterInstanceScratch.size() < kMaxShelfBoxCutters &&
              shelfBoxCutterLocalLayout(worldAisle, worldAlong, side, bcLx, bcLz, bcYDeck, bcYaw, bcScl)) {
            const float shelfYawRad = glm::radians(yawDeg);
            const glm::mat4 shelfRoot =
                glm::translate(glm::mat4(1.f), glm::vec3(cx, kGroundY, cz)) *
                glm::rotate(glm::mat4(1.f), shelfYawRad, glm::vec3(0.f, 1.f, 0.f));
            const glm::mat4 cutterM =
                shelfRoot * glm::translate(glm::mat4(1.f),
                                           glm::vec3(bcLx, bcYDeck + kShelfDeckSmallPropLiftY, bcLz)) *
                glm::rotate(glm::mat4(1.f), bcYaw, glm::vec3(0.f, 1.f, 0.f)) *
                glm::scale(glm::mat4(1.f), glm::vec3(bcScl));
            shelfBoxCutterInstanceScratch.push_back(cutterM);
          }
          float rpLx, rpLz, rpYDeck, rpYaw, rpScl;
          if (shelfRustyPipeVertexCount > 0 && shelfRustyPipeVertexBuffer != VK_NULL_HANDLE &&
              shelfRustyPipeInstanceMapped != nullptr &&
              shelfRustyPipeInstanceScratch.size() < kMaxShelfRustyPipes &&
              shelfRustyPipeLocalLayout(worldAisle, worldAlong, side, rpLx, rpLz, rpYDeck, rpYaw, rpScl)) {
            const float shelfYawRad = glm::radians(yawDeg);
            const glm::mat4 shelfRoot =
                glm::translate(glm::mat4(1.f), glm::vec3(cx, kGroundY, cz)) *
                glm::rotate(glm::mat4(1.f), shelfYawRad, glm::vec3(0.f, 1.f, 0.f));
            const glm::mat4 pipeM =
                shelfRoot * glm::translate(glm::mat4(1.f),
                                           glm::vec3(rpLx, rpYDeck + kShelfDeckSmallPropLiftY, rpLz)) *
                glm::rotate(glm::mat4(1.f), rpYaw, glm::vec3(0.f, 1.f, 0.f)) *
                glm::scale(glm::mat4(1.f), glm::vec3(rpScl));
            shelfRustyPipeInstanceScratch.push_back(pipeM);
          }
        }
        if (employeeVertexCount > 0 && staffNpcDrawBuild.size() < kMaxEmployees &&
            std::max(std::abs(worldAisle), std::abs(worldAlong)) > 3) {
          const uint32_t eh = scp3008ShelfHash(worldAisle, worldAlong, 0xE3910EE5u);
          if ((eh % kShelfEmpSpawnModulus) == 0u) {
            const float aisleHalf = kStoreAisleWidth * 0.5f;
            const float jx =
                (static_cast<float>(static_cast<int>(eh >> 4) % 31) / 15.5f - 1.f) * (aisleHalf * 0.88f);
            const float jz =
                (static_cast<float>(static_cast<int>(eh >> 10) % 17) / 8.5f - 1.f) * 1.2f;
            const glm::vec3 ep(aisleCX + jx, kGroundY, cz + jz);
            const int pic = static_cast<int>(std::floor(ep.x / kPillarSpacing));
            const int piz = static_cast<int>(std::floor(ep.z / kPillarSpacing));
            const glm::vec2 pc(static_cast<float>(pic) * kPillarSpacing,
                               static_cast<float>(piz) * kPillarSpacing);
            if (glm::length(glm::vec2(ep.x, ep.z) - pc) > kPillarHalfW + 0.45f) {
              const uint64_t empKey =
                  (static_cast<uint64_t>(static_cast<uint32_t>(worldAisle)) << 32) |
                  static_cast<uint64_t>(static_cast<uint32_t>(worldAlong));
              ShelfEmployeeNpc& npc = shelfEmpAcquire(empKey);
              if (!npc.inited) {
                npc.aisleCenterX = aisleCX;
                npc.aisleCenterZ = cz;
                npc.roamHalfX = aisleHalf * 0.96f;
                npc.roamHalfZ = 2.45f;
                npc.posXZ = glm::vec2(ep.x, ep.z);
                npc.feetWorldY =
                    terrainSupportY(npc.posXZ.x, npc.posXZ.y, kGroundY + kStaffTerrainStepProbe);
                npc.yaw = glm::radians(static_cast<float>(eh % 628u) * 0.573f);
                const glm::vec2 spawnXZ(ep.x, ep.z);
                shelfEmpPickWanderStoreWide(npc, empKey, spawnXZ);
                shelfEmpEnsureWanderTargetClear(npc, empKey, false, spawnXZ);
                npc.stuckRefXZ = npc.posXZ;
                npc.stuckTimer = 0.f;
                npc.velXZ = glm::vec2(0.f);
                npc.staffVelY = 0.f;
                npc.bodyScale = staffBodyScaleFromKey(empKey);
                npc.staffClassArchetype = staffClassArchetypeFromKey(empKey);
                npc.staffHpMax = staffClassMaxHp(npc.staffClassArchetype);
                npc.staffHp = npc.staffHpMax;
                npc.staffDead = false;
                npc.deadRagdollPitch = 0.f;
                npc.deadRagdollRoll = 0.f;
                npc.deadRagdollPitchVel = 0.f;
                npc.deadRagdollRollVel = 0.f;
                npc.deadRagdollYaw = npc.yaw;
                npc.deadRagdollYawVel = 0.f;
                npc.inited = true;
              }
            }
          }
        }
      }
    }
    // Draw staff from sim state at their *current* XZ — not only when their home bay is inside the
    // shelf grid window (wander/night chase left them invisible when the camera followed them).
    for (uint32_t siR : shelfEmpActiveSlots) {
      if (employeeVertexCount == 0 || staffNpcDrawBuild.size() >= kMaxEmployees)
        break;
      ShelfEmployeeNpc& npc = shelfEmpPool[siR];
      if (!npc.inited)
        continue;
      const uint64_t rk = npc.residentKey;
      const int wa = static_cast<int>(static_cast<uint32_t>(rk >> 32));
      const int wl = static_cast<int>(static_cast<uint32_t>(rk & 0xffffffffull));
      // Alive staff: match shelf spawn density (1/83 hash + skip central aisles). Dead staff must skip
      // these — otherwise debug `\` corpses (~random keys) almost never enter staffNpcDrawBuild, so RP3D
      // palettes never run and the mesh stays upright bind pose.
      if (!npc.staffDead) {
        if (std::max(std::abs(wa), std::abs(wl)) <= 3)
          continue;
        const uint32_t ehAnim = scp3008ShelfHash(wa, wl, 0xE3910EE5u);
        if ((ehAnim % kShelfEmpSpawnModulus) != 0u)
          continue;
      }

      const float wx = npc.posXZ.x;
      const float wz = npc.posXZ.y;
      const float dx = wx - sceneFocusX;
      const float dz = wz - sceneFocusZ;
      const float distSqXZ = dx * dx + dz * dz;
      const float staffListSq = gGamePerf.staffCpuListDist * gGamePerf.staffCpuListDist;
      if (distSqXZ > staffListSq)
        continue;
      if (lodBehindCameraXZ(sceneFocusX, sceneFocusZ, wx, wz, lodHFwd, kLodBehindMarginStaff))
        continue;
      const float feetSink = staffMeleeDrawFeetSinkY(npc);
      const float ragLiftY =
          npc.staffDead ? kStaffRagdollBindVisualLiftY * npc.bodyScale.y : 0.f;
      glm::mat4 rot = glm::rotate(glm::mat4(1.f), npc.yaw, glm::vec3(0.f, 1.f, 0.f));
      if (npc.staffDead) {
        // Bind + ragdoll extras are upright in model space; damp world tilt so instance pitch/roll doesn’t stack into a stiff “card”.
        constexpr float kDeadCorpseWorldTiltMul = 0.42f;
        rot = rot * glm::rotate(glm::mat4(1.f), npc.deadRagdollPitch * kDeadCorpseWorldTiltMul,
                               glm::vec3(1.f, 0.f, 0.f)) *
              glm::rotate(glm::mat4(1.f), npc.deadRagdollRoll * kDeadCorpseWorldTiltMul,
                          glm::vec3(0.f, 0.f, 1.f));
      }
      const glm::mat4 M =
          glm::translate(glm::mat4(1.f), glm::vec3(wx, npc.feetWorldY - feetSink + ragLiftY, wz)) *
                          rot * glm::scale(glm::mat4(1.f), npc.bodyScale);
      StaffNpcDrawSlot slot{};
      slot.model = M;
      if (staffSkinnedActive) {
        if (npc.staffDead) {
          slot.bindPoseOnly = 1u;
          slot.ragdollAngVelForSkin =
              glm::vec3(npc.deadRagdollPitchVel, npc.deadRagdollYawVel, npc.deadRagdollRollVel);
          slot.ragdollLooseSeed =
              static_cast<uint32_t>(npc.residentKey ^ (npc.residentKey >> 32));
          slot.deadRagdollJointCount = npc.deadRagdollJointSimCount;
          for (int j = 0; j < npc.deadRagdollJointSimCount && j < kStaffRagdollSimMaxBones; ++j)
            slot.deadRagdollJointEuler[j] = npc.deadRagdollJointEuler[j];
        } else {
          int clipIdx = 0;
          double ph = 0.0;
          bool loopClip = true;
          staffNpcResolveDrawAnim(npc, npc.residentKey, storeLitForStaffAnim, clipIdx, ph, loopClip);
          slot.clipIdx = clipIdx;
          slot.phase = ph;
          slot.animLoop = loopClip ? 1u : 0u;
          slot.meleeBlend = npc.meleeAnimBlend;
          slot.meleeFromClip = npc.meleeAnimFromClip;
          slot.meleeFromPhase = npc.meleeAnimFromPhase;
          slot.meleeFromLoop = npc.meleeAnimFromLoop;
        }
      }
      staffNpcDrawBuild.push_back(slot);
    }
    gPerfTrace.totalVisibleStaff += static_cast<uint32_t>(staffNpcDrawBuild.size());
    }
    // Static meshes / static batching / “dynamic” buffers — how that maps here (vs Unity naming):
    // - **Static geometry**: rack/crate/ladder/etc. topology lives in DEVICE_LOCAL vertex buffers uploaded
    //   at load; it does not move in model space. Per-frame motion is only in instance matrices.
    // - **Batching**: Unity “static batching” merges different meshes into one draw. We instead use
    //   **instancing** — one vkCmdDraw per mesh *type* with instanceCount = visible copies (same idea:
    //   amortize CPU/GPU draw setup over many identical meshes). Different topologies still need separate
    //   draws unless we add a much heavier path (atlas VB + drawId, indirect, etc.).
    // - **Dynamic updates**: there is no Mesh.MarkDynamic — “dynamic” data is the HOST_VISIBLE
    //   instance buffers (and similar) memcpy’d each frame; marking would mean choosing that allocation
    //   path, not a separate API.
    // - **Baked lighting**: still runtime lit in the fragment shader; true “static light” would be
    //   offline/baked into textures or vertex colors for those rails.
    // - **CPU visibility**: distance / grid LOD from the view eye (`shelfCullHardDist`, `shelfDetailCullDist`,
    //   etc.) plus conservative behind-camera rejects; GPU clip still handles lateral off-screen geometry.
    gPerfTrace.totalShelfInst += static_cast<uint32_t>(shelfInstanceScratch.size());
    gPerfTrace.totalCrateInst += static_cast<uint32_t>(shelfCrateInstanceScratch.size() +
                                                       shelfPalletInstanceScratch.size());
    if (!shelfInstanceScratch.empty()) {
      if (shelfInstanceScratch.size() > kMaxShelfInstances)
        shelfInstanceScratch.resize(kMaxShelfInstances);
      std::memcpy(shelfInstanceMapped, shelfInstanceScratch.data(),
                  sizeof(glm::mat4) * shelfInstanceScratch.size());
      VkBuffer vbSh[2] = {shelfVertexBuffer, shelfInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbSh, bindOffs);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel), &push);
      vkCmdDraw(cmd, shelfVertexCount, static_cast<uint32_t>(shelfInstanceScratch.size()), 0, 0);
    }
    // Kitchen/food market block removed.
    if (!shelfCrateInstanceScratch.empty()) {
      std::memcpy(shelfCrateInstanceMapped, shelfCrateInstanceScratch.data(),
                  sizeof(glm::mat4) * shelfCrateInstanceScratch.size());
      VkBuffer vbCr[2] = {shelfCrateVertexBuffer, shelfCrateInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbCr, bindOffs);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel), &push);
      vkCmdDraw(cmd, shelfCrateVertexCount, static_cast<uint32_t>(shelfCrateInstanceScratch.size()), 0, 0);
    }
    if (!shelfPalletInstanceScratch.empty()) {
      std::memcpy(shelfPalletInstanceMapped, shelfPalletInstanceScratch.data(),
                  sizeof(glm::mat4) * shelfPalletInstanceScratch.size());
      VkBuffer vbPl[2] = {shelfPalletVertexBuffer, shelfPalletInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbPl, bindOffs);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel), &push);
      vkCmdDraw(cmd, shelfPalletVertexCount, static_cast<uint32_t>(shelfPalletInstanceScratch.size()), 0, 0);
    }
    if (!shelfLadderInstanceScratch.empty()) {
      std::memcpy(shelfLadderInstanceMapped, shelfLadderInstanceScratch.data(),
                  sizeof(glm::mat4) * shelfLadderInstanceScratch.size());
      VkBuffer vbLd[2] = {shelfLadderVertexBuffer, shelfLadderInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbLd, bindOffs);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel), &push);
      vkCmdDraw(cmd, shelfLadderVertexCount, static_cast<uint32_t>(shelfLadderInstanceScratch.size()), 0, 0);
    }
    if (shelfBoxCutterVertexCount > 0 && shelfBoxCutterVertexBuffer != VK_NULL_HANDLE &&
        shelfBoxCutterInstanceMapped != nullptr && !shelfBoxCutterInstanceScratch.empty()) {
      std::memcpy(shelfBoxCutterInstanceMapped, shelfBoxCutterInstanceScratch.data(),
                  sizeof(glm::mat4) * shelfBoxCutterInstanceScratch.size());
      VkBuffer vbBc[2] = {shelfBoxCutterVertexBuffer, shelfBoxCutterInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbBc, bindOffs);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel), &push);
      vkCmdDraw(cmd, shelfBoxCutterVertexCount, static_cast<uint32_t>(shelfBoxCutterInstanceScratch.size()), 0,
                0);
    }
    if (shelfRustyPipeVertexCount > 0 && shelfRustyPipeVertexBuffer != VK_NULL_HANDLE &&
        shelfRustyPipeInstanceMapped != nullptr && !shelfRustyPipeInstanceScratch.empty()) {
      std::memcpy(shelfRustyPipeInstanceMapped, shelfRustyPipeInstanceScratch.data(),
                  sizeof(glm::mat4) * shelfRustyPipeInstanceScratch.size());
      VkBuffer vbRp[2] = {shelfRustyPipeVertexBuffer, shelfRustyPipeInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbRp, bindOffs);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel), &push);
      vkCmdDraw(cmd, shelfRustyPipeVertexCount, static_cast<uint32_t>(shelfRustyPipeInstanceScratch.size()), 0,
                0);
    }
    if (!marketInstanceScratch.empty() && marketVertexBuffer != VK_NULL_HANDLE) {
      std::memcpy(marketInstanceMapped, marketInstanceScratch.data(),
                  sizeof(glm::mat4) * marketInstanceScratch.size());
      VkBuffer vbMk[2] = {marketVertexBuffer, marketInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbMk, bindOffs);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel), &push);
      vkCmdDraw(cmd, marketVertexCount, static_cast<uint32_t>(marketInstanceScratch.size()), 0, 0);
    }
    if (!deliPizzaInstanceScratch.empty() && deliPizzaVertexBuffer != VK_NULL_HANDLE) {
      std::memcpy(deliPizzaInstanceMapped, deliPizzaInstanceScratch.data(),
                  sizeof(glm::mat4) * deliPizzaInstanceScratch.size());
      VkBuffer vbPz[2] = {deliPizzaVertexBuffer, deliPizzaInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbPz, bindOffs);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel), &push);
      vkCmdDraw(cmd, deliPizzaVertexCount, static_cast<uint32_t>(deliPizzaInstanceScratch.size()), 0, 0);
    }
    if (!deliMeatballInstanceScratch.empty() && deliMeatballVertexBuffer != VK_NULL_HANDLE) {
      std::memcpy(deliMeatballInstanceMapped, deliMeatballInstanceScratch.data(),
                  sizeof(glm::mat4) * deliMeatballInstanceScratch.size());
      VkBuffer vbMb[2] = {deliMeatballVertexBuffer, deliMeatballInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbMb, bindOffs);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel), &push);
      vkCmdDraw(cmd, deliMeatballVertexCount, static_cast<uint32_t>(deliMeatballInstanceScratch.size()), 0, 0);
    }
    if (employeeVertexCount > 0 && staffSkinnedActive && staffBoneSsbMapped &&
        !staffRig.clips.empty()) {
      auto* palettes = reinterpret_cast<glm::mat4*>(staffBoneSsbMapped);
      const size_t nNpc = staffNpcDrawBuild.size();
      if (nNpc > 0) {
        {
          auto* empDst = static_cast<glm::mat4*>(employeeInstanceMapped);
          for (size_t ii = 0; ii < nNpc; ++ii)
            empDst[ii] = staffNpcDrawBuild[ii].model;
        }
        {
        PerfScopeTimer _palT(&gPerfTrace.accumSkinPalette, gPerfTrace.enabled);
        constexpr float kStaffPaletteLodFarSq = 45.f * 45.f;
        // Very far NPCs: reuse last frame's palette (identity-init on first encounter).
        // Saves ~70% of computePalette calls at distance where animation isn't perceptible.
        constexpr float kStaffPaletteFreezeSq = 72.f * 72.f;
        static uint32_t palFrameCount = 0;
        ++palFrameCount;
        for (size_t ii = 0; ii < nNpc; ++ii) {
          const StaffNpcDrawSlot& slEarly = staffNpcDrawBuild[ii];
          if (slEarly.bindPoseOnly != 0u) {
            // Dead: bind pose + joint sim only (no fall-clip sampling).
            if (staffRig.boneCount > 0) {
              if (slEarly.deadRagdollJointCount > 0u && gStaffDeadRagdollSimBoneCount > 0) {
                static thread_local std::vector<glm::vec3> staffRagExtraScratch;
                const int nb = staffRig.boneCount;
                if (static_cast<int>(staffRagExtraScratch.size()) < nb)
                  staffRagExtraScratch.resize(static_cast<size_t>(nb));
                std::fill(staffRagExtraScratch.begin(), staffRagExtraScratch.begin() + nb, glm::vec3(0.f));
                for (int j = 0; j < slEarly.deadRagdollJointCount && j < kStaffRagdollSimMaxBones; ++j)
                  staffRagExtraScratch[static_cast<size_t>(gStaffDeadRagdollSimBoneRigIdx[j])] =
                      slEarly.deadRagdollJointEuler[j];
                staff_skin::computeBindPosePaletteWithRagdollExtras(staffRig, staffRagExtraScratch.data(),
                                                                    palettes + ii * kStaffPaletteBoneCount);
              } else {
                staff_skin::computeBindPosePalette(staffRig, palettes + ii * kStaffPaletteBoneCount);
              }
            } else {
              staff_skin::computeLooseBindPosePalette(staffRig, palettes + ii * kStaffPaletteBoneCount,
                                                      slEarly.ragdollAngVelForSkin, staffSimTime,
                                                      slEarly.ragdollLooseSeed);
            }
            continue;
          }
          const glm::mat4& staffM = staffNpcDrawBuild[ii].model;
          const float sdx = staffM[3][0] - sceneFocusX;
          const float sdz = staffM[3][2] - sceneFocusZ;
          const float staffDistSq = sdx * sdx + sdz * sdz;
          // Freeze tier: update palette every 8th frame (staggered by instance) for
          // NPCs beyond 72m — at that range animation detail is invisible.
          if (staffDistSq > kStaffPaletteFreezeSq) {
            if (((palFrameCount + static_cast<uint32_t>(ii)) & 7u) != 0u)
              continue;
          }
          const StaffNpcDrawSlot& sl = staffNpcDrawBuild[ii];
          const float bl = sl.meleeBlend;
          const double ph = ps1QuantizeClipPhase(sl.phase, kPs1NpcClipPhaseStrength);
          const bool farStaff = staffDistSq > kStaffPaletteLodFarSq;
          if (farStaff || bl >= 1.f - 1e-5f) {
            staff_skin::computePalette(staffRig, sl.clipIdx, ph, palettes + ii * kStaffPaletteBoneCount,
                                       sl.animLoop != 0);
          } else {
            const double ph0 = ps1QuantizeClipPhase(sl.meleeFromPhase, kPs1NpcClipPhaseStrength);
            staff_skin::computePaletteLerp(staffRig, sl.meleeFromClip, ph0, sl.meleeFromLoop != 0,
                                           sl.clipIdx, ph, sl.animLoop != 0, bl,
                                           palettes + ii * kStaffPaletteBoneCount);
          }
        }
        }
      }
      const size_t avSlot = nNpc;
      if (!inTitleMenu && !inIntroSplash && !inLoadingScreen && avSlot < static_cast<size_t>(kStaffSkinnedInstanceSlots)) {
        const float avCamY = viewStepSmoothYInited ? viewStepSmoothY : camPos.y;
        float feetY = avCamY - eyeHeight;
        // Lerp clip sink across crossfade so stand-up returns to ground level; max() kept us sunk in crouch pose.
        const auto clipFeetSink = [&](int c) -> float {
          if (c < 0)
            return 0.f;
          if ((avClipSlideRight >= 0 && c == avClipSlideRight) ||
              (avClipSlideLight >= 0 && c == avClipSlideLight))
            return kAvatarSlideFeetVisualDown;
          if (avClipCrouchIdleBow >= 0 && c == avClipCrouchIdleBow)
            return kAvatarCrouchIdleBowFeetVisualDown;
          if ((avClipCrouchFwd >= 0 && c == avClipCrouchFwd) ||
              (avClipCrouchBack >= 0 && c == avClipCrouchBack) ||
              (avClipCrouchLeft >= 0 && c == avClipCrouchLeft) ||
              (avClipCrouchRight >= 0 && c == avClipCrouchRight))
            return kAvatarCrouchWalkFeetVisualDown;
          if (avClipLedgeClimb >= 0 && c == avClipLedgeClimb)
            return kAvatarLedgeClimbFeetVisualDown;
          if ((avClipShimmyLeft >= 0 && c == avClipShimmyLeft) ||
              (avClipShimmyRight >= 0 && c == avClipShimmyRight))
            return 0.f;
          // Death knockdown: same feet lerp as NPC melee fall so soles meet the ground as the clip progresses.
          if (playerDeathActive && playerDeathClipIndex >= 0 && c == playerDeathClipIndex) {
            if (playerDeathPlayingFallClip) {
              const double dur = staff_skin::clipDuration(staffRig, playerDeathClipIndex);
              const double endT = dur * static_cast<double>(playerDeathClipFracEnd);
              const float u =
                  endT > 1e-6
                      ? glm::clamp(static_cast<float>(playerDeathAnimTime / endT), 0.f, 1.f)
                      : 1.f;
              return glm::mix(kStaffMeleeFallFeetSinkMax, kStaffMeleeFallFeetSinkEnd, u) +
                     kStaffMeleeFallFeetSinkWorldBias;
            }
            return kStaffMeleeFallFeetSinkEnd + kStaffMeleeFallFeetSinkWorldBias;
          }
          return 0.f;
        };
        float sinkClip = glm::mix(clipFeetSink(playerAvatarBlendFromClip), clipFeetSink(playerAvatarClip),
                                  playerAvatarClipBlend);
        // Ledge-hang has its own explicit vertical placement below; do not apply clip sink
        // (shimmy may use fallback ledge-climb clip, which otherwise lowers the mesh).
        if (ledgeHangActive)
          sinkClip = 0.f;
        const float sinkSlideState =
            (slideActive || slideClearClipNextFrame) ? kAvatarSlideFeetVisualDown : 0.f;
        // Takeoff still grounded: don’t sink full jump amount — only while descending or in post-land hold.
        float sinkJumpPhase = 0.f;
        if ((playerJumpAnimRemain > 1e-4f || playerPreFallAnimRemain > 1e-4f ||
             playerAvatarJumpFallMidPose()) &&
            (avClipJump >= 0 || avClipJumpRun >= 0)) {
          if (velY < -0.16f)
            sinkJumpPhase = kAvatarJumpFallFeetVisualDown;
        } else if (playerJumpPostLandRemain > 1e-4f) {
          float landSink = kAvatarJumpLandFeetVisualDown;
          if (playerJumpPostLandDurationInit > 1e-6f)
            landSink *= playerJumpPostLandRemain / playerJumpPostLandDurationInit;
          else {
            const float refHold =
                (avClipJumpRun >= 0 && playerJumpPostLandClipIndex == avClipJumpRun) ? kJumpLandPoseHoldSecRunJump
                                                                                      : kJumpLandPoseHoldSec;
            landSink *= glm::clamp(playerJumpPostLandRemain / std::max(refHold, 0.05f), 0.f, 1.f);
          }
          sinkJumpPhase = landSink;
        }
        feetY -= std::max(sinkClip, std::max(sinkSlideState, sinkJumpPhase));
        if (ledgeHangActive) {
          if (ledgeHangGrabTimer <= 0.f && !ledgeHangFeetYCaptured) {
            ledgeHangSettledFeetY = feetY - kAvatarLedgeHangFeetVisualDown;
            ledgeHangFeetYCaptured = true;
          }
          if (ledgeHangFeetYCaptured)
            feetY = ledgeHangSettledFeetY;
          else
            feetY -= kAvatarLedgeHangFeetVisualDown;
        }
        // Skinned staff mesh matches NPC convention: horizontal facing uses (sin φ, cos φ) in XZ, i.e.
        // φ = atan2(vx, vz). Player camera uses (cos yaw, sin yaw) for forward — rotating the model by
        // raw yaw misaligns run/walk clips; use velocity when moving, else camera look (atan2(cos,sin)).
        const float spXZ = glm::length(horizVel);
        constexpr float kAvatarYawVelEps = 0.07f;
        float avYaw;
        const Uint8* keysAv = SDL_GetKeyboardState(nullptr);
        const bool backKeyHeld =
            (keysAv[SDL_SCANCODE_S] != 0) || scancodeDown[static_cast<size_t>(SDL_SCANCODE_S)] ||
            (keysAv[SDL_SCANCODE_DOWN] != 0) || scancodeDown[static_cast<size_t>(SDL_SCANCODE_DOWN)];
        const bool strafeLeftHeld =
            (keysAv[SDL_SCANCODE_A] != 0) || scancodeDown[static_cast<size_t>(SDL_SCANCODE_A)] ||
            (keysAv[SDL_SCANCODE_LEFT] != 0) || scancodeDown[static_cast<size_t>(SDL_SCANCODE_LEFT)];
        const bool strafeRightHeld =
            (keysAv[SDL_SCANCODE_D] != 0) || scancodeDown[static_cast<size_t>(SDL_SCANCODE_D)] ||
            (keysAv[SDL_SCANCODE_RIGHT] != 0) || scancodeDown[static_cast<size_t>(SDL_SCANCODE_RIGHT)];
        const bool strafeKeyHeld = strafeLeftHeld || strafeRightHeld;
        // S / A / D (and arrows, Q): keep torso facing camera-forward; strafe clips move sideways on that heading.
        if (dropKickActive && glm::length(dropKickDir) > 1e-4f)
          avYaw = std::atan2(dropKickDir.x, dropKickDir.y);
        else if (!thirdPersonTestMode && slideActive)
          avYaw = std::atan2(std::cos(yaw), std::sin(yaw));
        else if (ledgeHangActive)
          avYaw = std::atan2(std::cos(ledgeHangTargetYaw), std::sin(ledgeHangTargetYaw));
        else if (ledgeClimbT >= 0.f && glm::length(ledgeClimbExitHoriz) > 1e-4f)
          avYaw = std::atan2(-ledgeClimbExitHoriz.x, -ledgeClimbExitHoriz.y);
        else if (ladderClimbActive && glm::length(ladderClimbExitHoriz) > 1e-4f)
          avYaw = std::atan2(ladderClimbExitHoriz.x, ladderClimbExitHoriz.y);
        else if (wallClimbActive && glm::length(glm::vec2(wallClimbNormal.x, wallClimbNormal.z)) > 1e-4f)
          avYaw = std::atan2(-wallClimbNormal.x, -wallClimbNormal.z);
        else if (backKeyHeld || strafeKeyHeld)
          avYaw = std::atan2(std::cos(yaw), std::sin(yaw));
        else if (spXZ > kAvatarYawVelEps)
          avYaw = std::atan2(horizVel.x, horizVel.y);
        else
          avYaw = std::atan2(std::cos(yaw), std::sin(yaw));
        float avPitch = std::clamp(-pitch * kFpBodyPitchFollow, -kFpBodyPitchMaxTilt, kFpBodyPitchMaxTilt);
        if (ledgeHangActive || slideActive || dropKickActive)
          avPitch = 0.f;
        float avYawDraw = avYaw;
        if (thirdPersonTestMode)
          fpAvatarYawSmooth = avYaw;
        else {
          const bool climbFaceLock =
              ledgeHangActive || ledgeClimbT >= 0.f || ladderClimbActive || wallClimbActive;
          // First-person: free-look yaw for normal locomotion; lock to climb/ledge face while climbing.
          avYawDraw = climbFaceLock ? avYaw : std::atan2(std::cos(yaw), std::sin(yaw));
          fpAvatarYawSmooth = avYawDraw;
        }
        const float avRoll = 0.f;
        // Disable authored shimmy roll bias to avoid visible side-tilt while hanging.
        const float avShimmyLean = 0.f;
        const glm::mat4 avM =
            glm::translate(glm::mat4(1.f), glm::vec3(camPos.x, feetY, camPos.z)) *
            glm::rotate(glm::mat4(1.f), avYawDraw, glm::vec3(0.f, 1.f, 0.f)) *
            glm::rotate(glm::mat4(1.f), avRoll + avShimmyLean, glm::vec3(0.f, 0.f, 1.f));
        *reinterpret_cast<glm::mat4*>(static_cast<char*>(employeeInstanceMapped) +
                                        sizeof(glm::mat4) * avSlot) = avM;
        int avClip = playerAvatarClip;
        // Safety override: when ledge shimmy input is active, force shimmy clip at render-time.
        // This guards against edge cases where gameplay state can transiently keep ledge-grab clip selected.
        if (ledgeHangActive) {
          // A / +dir → ledge_shimmy_left.glb (Meshy biped Climb_Left_inplace); D → Climb_Right.
          if (ledgeHangShimmyDir > 0 && avClipShimmyLeft >= 0)
            avClip = avClipShimmyLeft;
          else if (ledgeHangShimmyDir < 0 && avClipShimmyRight >= 0)
            avClip = avClipShimmyRight;
        }
        if (avClip < 0 || static_cast<size_t>(avClip) >= staffRig.clips.size())
          avClip = 0;
        int blendFrom = playerAvatarBlendFromClip;
        if (blendFrom < 0 || static_cast<size_t>(blendFrom) >= staffRig.clips.size())
          blendFrom = 0;
        double phFrom = 0.0;
        double phTo = 0.0;
        bool loopFrom = true;
        bool loopTo = true;
        resolvePlayerAvatarPhase(blendFrom, phFrom, loopFrom);
        resolvePlayerAvatarPhase(avClip, phTo, loopTo);
        const float pq =
            glm::clamp(parkourPs1PresentMix * kPs1PlayerPhaseFromParkourMul, 0.f, 1.f);
        const double phFromQ = ps1QuantizeClipPhase(phFrom, pq);
        const double phToQ = ps1QuantizeClipPhase(phTo, pq);
        const float avBlend = playerAvatarClipBlend;
        if (ledgeHangActive && ledgeHangShimmyDir != 0) {
          static int renderDbg = 0;
          if (renderDbg < 15) {
            std::fprintf(stderr, "[shimmy render] avClip=%d phTo=%.4f phToQ=%.4f blend=%.2f loop=%d\n",
                         avClip, phTo, phToQ, avBlend, loopTo ? 1 : 0);
            ++renderDbg;
          }
        }
        if (avBlend < 1.f - 1e-4f) {
          staff_skin::computePaletteLerp(staffRig, blendFrom, phFromQ, loopFrom, avClip, phToQ, loopTo,
                                         avBlend, palettes + avSlot * kStaffPaletteBoneCount);
        } else {
          staff_skin::computePalette(staffRig, avClip, phToQ, palettes + avSlot * kStaffPaletteBoneCount,
                                     loopTo);
        }
        // Apply look-pitch tilt only to head/neck bones (not the whole body).
        if (!fpHeadTiltBoneIndices.empty() && std::abs(avPitch) > 1e-5f) {
          const int pivotIdx = fpNeckBoneIdx >= 0 ? fpNeckBoneIdx : fpHeadBoneIdx;
          const glm::vec3 pivotBind = (pivotIdx == fpNeckBoneIdx) ? fpNeckBindPos : fpHeadBindPos;
          const glm::mat4& pivotPal =
              palettes[avSlot * kStaffPaletteBoneCount + static_cast<size_t>(pivotIdx)];
          const glm::vec3 pivotPos = glm::vec3(pivotPal * glm::vec4(pivotBind, 1.f));
          const glm::mat4 tiltM =
              glm::translate(glm::mat4(1.f), pivotPos) *
              glm::rotate(glm::mat4(1.f), avPitch, glm::vec3(1.f, 0.f, 0.f)) *
              glm::translate(glm::mat4(1.f), -pivotPos);
          for (int idx : fpHeadTiltBoneIndices) {
            auto& pal = palettes[avSlot * kStaffPaletteBoneCount + static_cast<size_t>(idx)];
            pal = tiltM * pal;
          }
        }
        // Extract head bone world position from the palette for FP camera tracking.
        if (fpHeadBoneIdx >= 0 && !thirdPersonTestMode) {
          const glm::mat4& headPal =
              palettes[avSlot * kStaffPaletteBoneCount + static_cast<size_t>(fpHeadBoneIdx)];
          glm::vec3 headModel = glm::vec3(headPal * glm::vec4(fpHeadBindPos, 1.f));
          glm::vec3 headWorld = glm::vec3(avM * glm::vec4(headModel, 1.f));
          fpHeadAnimOffset = headWorld - camPos;
        } else {
          fpHeadAnimOffset = glm::vec3(0.f);
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineStaffSkinned);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                                &descriptorSets[flightIdx], 0, nullptr);
        VkBuffer vbEmp[2] = {employeeVertexBuffer, employeeInstanceBuffer};
        vkCmdBindVertexBuffers(cmd, 0, 2, vbEmp, bindOffs);
        PushModel pushStaff{};
        pushStaff.model = glm::mat4(1.f);
        pushStaff.staffShade = glm::vec4(0.f);
        const VkShaderStageFlags pushVF =
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        if (nNpc > 0) {
          vkCmdPushConstants(cmd, pipelineLayout, pushVF, 0, sizeof(PushModel), &pushStaff);
          vkCmdDraw(cmd, employeeVertexCount, static_cast<uint32_t>(nNpc), 0, 0);
        }
        {
          pushStaff.staffShade.x = 0.f;
          pushStaff.staffShade.y = 0.f;
          pushStaff.staffShade.w = 1.f;
          vkCmdPushConstants(cmd, pipelineLayout, pushVF, 0, sizeof(PushModel), &pushStaff);
          vkCmdDraw(cmd, employeeVertexCount, 1, 0, static_cast<uint32_t>(avSlot));
        }
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
        if (shrekEggAssetLoaded && shrekEggActive && shrekEggVertexCount > 0u &&
            kShrekEggStaffSlotIndex < kStaffSkinnedInstanceSlots) {
          int shClip = shrekEggAnimClipIndex;
          if (shClip < 0 || static_cast<size_t>(shClip) >= shrekEggRig.clips.size())
            shClip = 0;
          staff_skin::computePalette(
              shrekEggRig, shClip,
              ps1QuantizeClipPhase(static_cast<double>(shrekEggAnimPhase), 0.26f),
              palettes + static_cast<size_t>(kShrekEggStaffSlotIndex) * kStaffPaletteBoneCount, true);
          const glm::mat4 eggM =
              glm::translate(glm::mat4(1.f), shrekEggPos) *
              glm::rotate(glm::mat4(1.f), shrekEggYaw, glm::vec3(0.f, 1.f, 0.f));
          *reinterpret_cast<glm::mat4*>(static_cast<char*>(employeeInstanceMapped) +
                                          sizeof(glm::mat4) * kShrekEggStaffSlotIndex) = eggM;
          VkBuffer vbShrek[2] = {shrekEggVertexBuffer, employeeInstanceBuffer};
          vkCmdBindVertexBuffers(cmd, 0, 2, vbShrek, bindOffs);
          // w=2: fragment samples binding 8 shrekEggTex (set at init). w=1: grey (no embedded tex).
          PushModel pushShrek{};
          pushShrek.staffShade.x = 0.f;
          pushShrek.staffShade.w = shrekEggDiffuseLoaded ? 2.f : 1.f;
          vkCmdPushConstants(cmd, pipelineLayout, pushVF, 0, sizeof(PushModel), &pushShrek);
          vkCmdDraw(cmd, shrekEggVertexCount, 1u, 0u, kShrekEggStaffSlotIndex);
          VkBuffer vbEmpRestore[2] = {employeeVertexBuffer, employeeInstanceBuffer};
          vkCmdBindVertexBuffers(cmd, 0, 2, vbEmpRestore, bindOffs);
        }
#endif
      }
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
    } else if (employeeVertexCount > 0 && !staffNpcDrawBuild.empty()) {
      const size_t nEmp = staffNpcDrawBuild.size();
      {
        auto* empDst = static_cast<glm::mat4*>(employeeInstanceMapped);
        for (size_t ii = 0; ii < nEmp; ++ii)
          empDst[ii] = staffNpcDrawBuild[ii].model;
      }
      VkBuffer vbEmp[2] = {employeeVertexBuffer, employeeInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbEmp, bindOffs);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                           &push);
      vkCmdDraw(cmd, employeeVertexCount, static_cast<uint32_t>(nEmp), 0, 0);
    }

    // Hanging signs between neighboring pillars — batched into instanced draws.
    {
    constexpr bool kDrawHangingSigns = false;
    constexpr bool kDrawHangingSignStrings = false;
    const glm::mat4 signScale =
        glm::scale(glm::mat4(1.0f), glm::vec3(kPillarSpacing * 0.22f, 4.5f, 1.0f));
    const float signY = kCeilingY - 8.2f;
    const float stringTopY = kCeilingY - 0.35f;
    const float stringBottomY = signY + 4.85f;
    const float stringLen = std::max(0.2f, stringTopY - stringBottomY);
    const glm::mat4 stringScale = glm::scale(glm::mat4(1.0f), glm::vec3(0.16f, stringLen, 1.0f));
    const float signHalfW = kPillarSpacing * 0.22f;
    signBoardInstanceScratch.clear();
    signStringInstanceScratch.clear();
    static const glm::mat4 zRot = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 1, 0));
    static const glm::mat4 zRotSignScale = zRot * signScale;
    static const glm::mat4 zRotStringScale = zRot * stringScale;
    const int signRad = kDrawHangingSigns ? gGamePerf.signDrawGridRadius : 0;
    const float signCullSq = gGamePerf.shelfCullHardDist * gGamePerf.shelfCullHardDist;
    for (int dx = -signRad; dx < signRad; ++dx) {
      for (int dz = -signRad; dz <= signRad; ++dz) {
        const float px = static_cast<float>(pcx + dx) * kPillarSpacing;
        const float pz = static_cast<float>(pcz + dz) * kPillarSpacing;
        const glm::vec3 center(px + 0.5f * kPillarSpacing, signY, pz);
        const float sdx = center.x - sceneFocusX;
        const float sdz = center.z - sceneFocusZ;
        if (sdx * sdx + sdz * sdz > signCullSq)
          continue;
        if (lodBehindCameraXZ(sceneFocusX, sceneFocusZ, center.x, center.z, lodHFwd, kLodBehindMarginSign))
          continue;
        const glm::vec3 sL(center.x - signHalfW * 0.78f, stringBottomY + stringLen * 0.5f, center.z);
        const glm::vec3 sR(center.x + signHalfW * 0.78f, stringBottomY + stringLen * 0.5f, center.z);
        signBoardInstanceScratch.push_back(glm::translate(glm::mat4(1.0f), center) * signScale);
        if (kDrawHangingSignStrings && signStringInstanceScratch.size() + 2 <= kMaxSignInstances) {
          signStringInstanceScratch.push_back(glm::translate(glm::mat4(1.0f), sL) * stringScale);
          signStringInstanceScratch.push_back(glm::translate(glm::mat4(1.0f), sR) * stringScale);
        }
      }
    }
    for (int dx = -signRad; dx <= signRad; ++dx) {
      for (int dz = -signRad; dz < signRad; ++dz) {
        const float px = static_cast<float>(pcx + dx) * kPillarSpacing;
        const float pz = static_cast<float>(pcz + dz) * kPillarSpacing;
        const glm::vec3 center(px, signY, pz + 0.5f * kPillarSpacing);
        const float sdx = center.x - sceneFocusX;
        const float sdz = center.z - sceneFocusZ;
        if (sdx * sdx + sdz * sdz > signCullSq)
          continue;
        if (lodBehindCameraXZ(sceneFocusX, sceneFocusZ, center.x, center.z, lodHFwd, kLodBehindMarginSign))
          continue;
        const glm::vec3 sL(center.x, stringBottomY + stringLen * 0.5f, center.z - signHalfW * 0.78f);
        const glm::vec3 sR(center.x, stringBottomY + stringLen * 0.5f, center.z + signHalfW * 0.78f);
        signBoardInstanceScratch.push_back(glm::translate(glm::mat4(1.0f), center) * zRotSignScale);
        if (kDrawHangingSignStrings && signStringInstanceScratch.size() + 2 <= kMaxSignInstances) {
          signStringInstanceScratch.push_back(glm::translate(glm::mat4(1.0f), sL) * zRotStringScale);
          signStringInstanceScratch.push_back(glm::translate(glm::mat4(1.0f), sR) * zRotStringScale);
        }
      }
    }
    if (signInstanceMapped) {
      auto* dst = static_cast<glm::mat4*>(signInstanceMapped);
      const size_t nBoards = std::min(signBoardInstanceScratch.size(), static_cast<size_t>(kMaxSignInstances / 2));
      const size_t nStrings = std::min(signStringInstanceScratch.size(), static_cast<size_t>(kMaxSignInstances - nBoards));
      if (nBoards > 0)
        std::memcpy(dst, signBoardInstanceScratch.data(), sizeof(glm::mat4) * nBoards);
      if (nStrings > 0)
        std::memcpy(dst + nBoards, signStringInstanceScratch.data(), sizeof(glm::mat4) * nStrings);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel), &push);
      if (kDrawHangingSigns && nBoards > 0 && signVertexCount > 0) {
        VkBuffer vbSignB[2] = {signVertexBuffer, signInstanceBuffer};
        vkCmdBindVertexBuffers(cmd, 0, 2, vbSignB, bindOffs);
        vkCmdDraw(cmd, signVertexCount, static_cast<uint32_t>(nBoards), 0, 0);
      }
      if (kDrawHangingSignStrings && nStrings > 0 && signStringVertexCount > 0) {
        VkBuffer vbSignS[2] = {signStringVertexBuffer, signInstanceBuffer};
        const VkDeviceSize strOff[2] = {0, sizeof(glm::mat4) * nBoards};
        vkCmdBindVertexBuffers(cmd, 0, 2, vbSignS, strOff);
        vkCmdDraw(cmd, signStringVertexCount, static_cast<uint32_t>(nStrings), 0, 0);
      }
    }
    }

    }
    skipSceneGeometry:

    vkCmdEndRenderPass(cmd);
    sceneColorWasSampled[static_cast<size_t>(flightIdx)] = true;

    if (!inTitleMenu && healthHudVertexMapped != nullptr && healthHudVertexBuffer != VK_NULL_HANDLE) {
      constexpr float kHudHpCacheEps = 0.01f;
      constexpr float kHudYawCacheEps = 0.0025f;
      const int curDayCount = audioGetDayCount();
      const bool canShowInteractHint =
          !showPauseMenu && !showInventoryMenu && !showControlsOverlay && !playerDeathActive && !inTitleMenu &&
          canPickupNearbyDeliFood();
      const bool hudDirty =
          healthHudCachedVertexCount == 0 ||
          std::fabs(playerHealth - healthHudCacheHp) > kHudHpCacheEps ||
          std::fabs(kPlayerHealthMax - healthHudCacheHpMax) > kHudHpCacheEps ||
          std::fabs(playerHunger - healthHudCacheHunger) > kHudHpCacheEps ||
          std::fabs(kPlayerHungerMax - healthHudCacheHungerMax) > kHudHpCacheEps ||
          std::fabs(yaw - healthHudCacheYaw) > kHudYawCacheEps ||
          curDayCount != healthHudCacheDayCount ||
          canShowInteractHint != healthHudCacheInteractHint;
      if (hudDirty) {
        buildHealthHudOverlayVertices(playerHealth, kPlayerHealthMax, playerHunger, kPlayerHungerMax, yaw,
                                      curDayCount, canShowInteractHint, healthHudVertexCache);
        healthHudCacheHp = playerHealth;
        healthHudCacheHpMax = kPlayerHealthMax;
        healthHudCacheHunger = playerHunger;
        healthHudCacheHungerMax = kPlayerHungerMax;
        healthHudCacheYaw = yaw;
        healthHudCacheDayCount = curDayCount;
        healthHudCacheInteractHint = canShowInteractHint;
        healthHudCachedVertexCount = static_cast<uint32_t>(healthHudVertexCache.size());
        const VkDeviceSize hudBytes =
            sizeof(Vertex) * static_cast<VkDeviceSize>(healthHudCachedVertexCount);
        if (hudBytes > 0 && hudBytes <= healthHudVertexBufferBytes)
          std::memcpy(healthHudVertexMapped, healthHudVertexCache.data(), static_cast<size_t>(hudBytes));
      }
    }
    if (showInventoryMenu && inventoryMenuVertexMapped != nullptr &&
        inventoryMenuVertexBuffer != VK_NULL_HANDLE) {
      const bool invDirty = inventoryMenuVertexCount == 0 ||
                            inventoryMenuCacheRevision != inventoryRevision ||
                            inventoryMenuCacheScroll != inventoryScrollRow;
      if (invDirty) {
        inventoryMenuVertexCache = buildInventoryOverlayVertices(inventoryItems, inventoryScrollRow);
        inventoryMenuVertexCount = static_cast<uint32_t>(inventoryMenuVertexCache.size());
        const VkDeviceSize invBytes = sizeof(Vertex) * static_cast<VkDeviceSize>(inventoryMenuVertexCount);
        if (invBytes > 0 && invBytes <= inventoryMenuVertexBufferBytes)
          std::memcpy(inventoryMenuVertexMapped, inventoryMenuVertexCache.data(),
                      static_cast<size_t>(invBytes));
        inventoryMenuCacheRevision = inventoryRevision;
        inventoryMenuCacheScroll = inventoryScrollRow;
      }
    }

    VkRenderPassBeginInfo pr{};
    pr.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pr.renderPass = presentRenderPass;
    pr.framebuffer = framebuffers[imageIndex];
    pr.renderArea.offset = {0, 0};
    pr.renderArea.extent = swapchainExtent;
    pr.clearValueCount = 0;
    pr.pClearValues = nullptr;
    vkCmdBeginRenderPass(cmd, &pr, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport swVp{};
    swVp.x = 0;
    swVp.y = 0;
    swVp.width = static_cast<float>(swapchainExtent.width);
    swVp.height = static_cast<float>(swapchainExtent.height);
    swVp.minDepth = 0;
    swVp.maxDepth = 1;
    VkRect2D swSc{};
    swSc.offset = {0, 0};
    swSc.extent = swapchainExtent;
    vkCmdSetViewport(cmd, 0, 1, &swVp);
    vkCmdSetScissor(cmd, 0, 1, &swSc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, postProcessPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipelineLayout, 0, 1,
                            &postDescriptorSets[static_cast<size_t>(flightIdx)], 0, nullptr);
    PushPost pp{};
    pp.g.x = horrorPresentTime;
    pp.g.y = postNightHorrorWeight;
    pp.g.z = static_cast<float>(swapchainExtent.width);
    pp.g.w = static_cast<float>(swapchainExtent.height);
    pp.v.x = glm::clamp(playerScreenDamagePulse, 0.f, 1.f);
    pp.v.w = postNightPursuitMix;
    if (playerHealth <= kPlayerHealthScreenEdgeCritical) {
      const float t =
          glm::clamp((kPlayerHealthScreenEdgeCritical - playerHealth) / kPlayerHealthScreenEdgeCritical,
                     0.f, 1.f);
      pp.v.y = glm::clamp(0.38f + 0.62f * t, 0.f, 1.f);
    } else
      pp.v.y = 0.f;
    pp.v.z = effectivePs1HorrorMix(parkourPs1PresentMix, postNightHorrorWeight);
    vkCmdPushConstants(cmd, postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushPost), &pp);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    const VkDeviceSize bindOffsUi[2] = {0, 0};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, uiPresentPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
                            &descriptorSets[flightIdx], 0, nullptr);
    if (!inTitleMenu && healthHudCachedVertexCount > 0 && healthHudVertexBuffer != VK_NULL_HANDLE) {
      const VkDeviceSize hudBytes =
          sizeof(Vertex) * static_cast<VkDeviceSize>(healthHudCachedVertexCount);
      if (hudBytes <= healthHudVertexBufferBytes) {
        VkBuffer vbHud[2] = {healthHudVertexBuffer, identityInstanceBuffer};
        vkCmdBindVertexBuffers(cmd, 0, 2, vbHud, bindOffsUi);
        push.model = glm::mat4(1.0f);
        vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                           &push);
        vkCmdDraw(cmd, healthHudCachedVertexCount, 1, 0, 0);
      }
    }
    if (inIntroSplash && introSplashVertexCount > 0 && introSplashVertexBuffer != VK_NULL_HANDLE) {
      VkBuffer vbIntro[2] = {introSplashVertexBuffer, identityInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbIntro, bindOffsUi);

      constexpr float kFadeInDur = 2.0f;
      constexpr float kFadeOutStart = 14.5f;
      constexpr float kFadeOutDur = 2.5f;
      constexpr float kSlideInDur = 1.8f;
      auto easeOutExpo = [](float x) { return x >= 1.f ? 1.f : 1.f - std::pow(2.f, -10.f * x); };

      float fadeAlpha = 1.f;
      if (introSplashTime < kFadeInDur)
        fadeAlpha = glm::clamp(introSplashTime / kFadeInDur, 0.f, 1.f);
      else if (introSplashTime > kFadeOutStart)
        fadeAlpha = 1.f - glm::clamp((introSplashTime - kFadeOutStart) / kFadeOutDur, 0.f, 1.f);

      push.model = glm::mat4(1.0f);
      push.staffShade.z = 1.f;
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                         &push);
      vkCmdDraw(cmd, std::min(introSplashVertexCount, 6u), 1, 0, 0);

      if (introSplashVertexCount > 6) {
        float yOff = 0.f;
        if (introSplashTime < kSlideInDur) {
          float t = glm::clamp(introSplashTime / kSlideInDur, 0.f, 1.f);
          yOff = -3.0f * (1.f - easeOutExpo(t));
        }
        push.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.f, yOff, 0.f));
        push.staffShade.z = fadeAlpha;
        vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                           &push);
        vkCmdDraw(cmd, introSplashVertexCount - 6, 1, 6, 0);
      }
    } else if (inLoadingScreen && loadingScreenVertexCount > 0 && loadingScreenVertexBuffer != VK_NULL_HANDLE) {
      VkBuffer vbLoad[2] = {loadingScreenVertexBuffer, identityInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbLoad, bindOffsUi);

      constexpr float kLoadFadeInDur = 2.2f;
      constexpr float kLoadFadeOutDur = 2.8f;
      const float kLoadShowDur = loadingScreenIsNewSave ? 14.0f : 8.0f;
      float loadAlpha = 1.f;
      if (loadingScreenTime < kLoadFadeInDur) {
        float t = glm::clamp(loadingScreenTime / kLoadFadeInDur, 0.f, 1.f);
        loadAlpha = t * t * (3.f - 2.f * t);
      } else if (loadingScreenTime > kLoadShowDur - kLoadFadeOutDur) {
        float t = glm::clamp((loadingScreenTime - (kLoadShowDur - kLoadFadeOutDur)) / kLoadFadeOutDur, 0.f, 1.f);
        loadAlpha = 1.f - t * t * (3.f - 2.f * t);
      }

      push.model = glm::mat4(1.0f);
      push.staffShade.z = loadAlpha;
      push.staffShade.w = loadingScreenTime;
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                         &push);
      vkCmdDraw(cmd, std::min(loadingScreenVertexCount, 6u), 1, 0, 0);

      if (loadingScreenVertexCount > 6) {
        float bob = 0.008f * sin(loadingScreenTime * 0.5f * 6.2832f);
        float drift = 0.004f * sin(loadingScreenTime * 0.3f * 6.2832f);
        push.model = glm::translate(glm::mat4(1.0f), glm::vec3(drift, bob, 0.f));
        push.staffShade.z = loadAlpha;
        push.staffShade.w = loadingScreenTime;
        vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                           &push);
        vkCmdDraw(cmd, loadingScreenVertexCount - 6, 1, 6, 0);
      }
    } else if (playerDeathShowMenu && deathMenuVertexCount > 0 && deathMenuVertexBuffer != VK_NULL_HANDLE) {
      VkBuffer vbDeath[2] = {deathMenuVertexBuffer, identityInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbDeath, bindOffsUi);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                         &push);
      vkCmdDraw(cmd, deathMenuVertexCount, 1, 0, 0);
    } else if (inTitleMenu && titleMenuPickSlot && titleMenuSlotVertexCount > 0 &&
               titleMenuSlotVertexBuffer != VK_NULL_HANDLE) {
      VkBuffer vbSlot[2] = {titleMenuSlotVertexBuffer, identityInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbSlot, bindOffsUi);
      constexpr float kSlotSlideDur = 0.9f;
      constexpr float kSlotStartOff = 3.0f;
      constexpr float kBobAmp = 0.004f;
      constexpr float kBobHz = 0.6f;
      auto easeOutExpo = [](float x) { return x >= 1.f ? 1.f : 1.f - std::pow(2.f, -10.f * x); };

      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                         &push);
      vkCmdDraw(cmd, std::min(titleMenuSlotVertexCount, 6u), 1, 0, 0);

      if (titleMenuSlotVertexCount > 6) {
        float st = glm::clamp(titleMenuSlideTime / kSlotSlideDur, 0.f, 1.f);
        float slotOff = kSlotStartOff * (1.f - easeOutExpo(st));
        float bob = kBobAmp * std::sin(titleMenuSlideTime * kBobHz * 6.2832f);
        push.model = glm::translate(glm::mat4(1.0f), glm::vec3(slotOff, bob, 0.f));
        vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                           &push);
        vkCmdDraw(cmd, titleMenuSlotVertexCount - 6, 1, 6, 0);
      }
    } else if (inTitleMenu && !titleMenuPickSlot && titleMenuMainVertexCount > 0 &&
               titleMenuMainVertexBuffer != VK_NULL_HANDLE) {
      VkBuffer vbTit[2] = {titleMenuMainVertexBuffer, identityInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbTit, bindOffsUi);
      constexpr float kSlideDur = 1.1f;
      constexpr float kLogoDelay = 0.08f;
      constexpr float kOptsDelay = 0.35f;
      constexpr float kLogoStartOff = 3.0f;
      constexpr float kOptsStartOff = -3.0f;
      constexpr float kLogoBobAmp = 0.005f;
      constexpr float kLogoBobHz = 0.45f;
      constexpr float kOptsBobAmp = 0.004f;
      constexpr float kOptsBobHz = 0.55f;
      auto easeOutExpo = [](float x) { return x >= 1.f ? 1.f : 1.f - std::pow(2.f, -10.f * x); };

      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                         &push);
      vkCmdDraw(cmd, std::min(titleMenuMainVertexCount, 6u), 1, 0, 0);

      if (titleMenuMainVertexCount > 6) {
        float lt = glm::clamp((titleMenuSlideTime - kLogoDelay) / kSlideDur, 0.f, 1.f);
        float logoSlide = kLogoStartOff * (1.f - easeOutExpo(lt));
        float logoBob = kLogoBobAmp * std::sin(titleMenuSlideTime * kLogoBobHz * 6.2832f);
        push.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.f, logoSlide + logoBob, 0.f));
        vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                           &push);
        vkCmdDraw(cmd, std::min(titleMenuMainVertexCount - 6, 6u), 1, 6, 0);
      }
      if (titleMenuMainVertexCount > 12) {
        float ot = glm::clamp((titleMenuSlideTime - kOptsDelay) / kSlideDur, 0.f, 1.f);
        float optsSlide = kOptsStartOff * (1.f - easeOutExpo(ot));
        float optsBob = kOptsBobAmp * std::sin((titleMenuSlideTime + 0.7f) * kOptsBobHz * 6.2832f);
        push.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.f, optsSlide + optsBob, 0.f));
        vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                           &push);
        vkCmdDraw(cmd, titleMenuMainVertexCount - 12, 1, 12, 0);
      }
    } else if (showPauseMenu && pauseMenuVertexCount > 0 && pauseMenuVertexBuffer != VK_NULL_HANDLE) {
      VkBuffer vbPause[2] = {pauseMenuVertexBuffer, identityInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbPause, bindOffsUi);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                         &push);
      vkCmdDraw(cmd, pauseMenuVertexCount, 1, 0, 0);
    } else if (showInventoryMenu && inventoryMenuVertexCount > 0 &&
               inventoryMenuVertexBuffer != VK_NULL_HANDLE) {
      VkBuffer vbInv[2] = {inventoryMenuVertexBuffer, identityInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbInv, bindOffsUi);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                         &push);
      vkCmdDraw(cmd, inventoryMenuVertexCount, 1, 0, 0);
    }
    if (showControlsOverlay) {
      VkBuffer vbHelp[2] = {controlsHelpVertexBuffer, identityInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbHelp, bindOffsUi);
      push.model = glm::mat4(1.0f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                         &push);
      vkCmdDraw(cmd, controlsHelpVertexCount, 1, 0, 0);
    } else if (!thirdPersonTestMode && !playerDeathShowMenu && !showPauseMenu && !showInventoryMenu &&
               !inTitleMenu && !inIntroSplash && !inLoadingScreen) {
      VkBuffer vbCh[2] = {crosshairVertexBuffer, identityInstanceBuffer};
      vkCmdBindVertexBuffers(cmd, 0, 2, vbCh, bindOffsUi);
      float chScale = kCrosshairBaseScale;
      float chRot = 0.f;
      float chBright = 1.f;
      if (crosshairShoveAnimRemain > 0.f) {
        const float t = 1.f - crosshairShoveAnimRemain / kCrosshairShoveAnimDur;
        const float bump = std::exp(-5.f * t) * (1.f - 0.22f * t);
        chScale = kCrosshairBaseScale * (1.f + 0.42f * bump);
        chRot = 0.13f * bump * std::sin(t * 22.f);
        chBright = 1.f + 0.4f * bump;
      }
      push.model = glm::mat4(1.0f);
      push.staffShade = glm::vec4(chScale, chRot, chBright, 0.f);
      vkCmdPushConstants(cmd, pipelineLayout,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushModel),
                         &push);
      vkCmdDraw(cmd, crosshairVertexCount, 1, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
  }

  void createSyncObjects() {
    imageAvailableSemaphores.resize(kMaxFramesInFlight);
    renderFinishedSemaphores.resize(kMaxFramesInFlight);
    inFlightFences.resize(kMaxFramesInFlight);
    VkSemaphoreCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      vkCreateSemaphore(device, &si, nullptr, &imageAvailableSemaphores[i]);
      vkCreateSemaphore(device, &si, nullptr, &renderFinishedSemaphores[i]);
      vkCreateFence(device, &fi, nullptr, &inFlightFences[i]);
    }
  }

  void cleanupSwapchain() {
    for (auto fb : sceneFramebuffers) {
      if (fb != VK_NULL_HANDLE)
        vkDestroyFramebuffer(device, fb, nullptr);
    }
    sceneFramebuffers.clear();
    for (auto fb : framebuffers) {
      if (fb != VK_NULL_HANDLE)
        vkDestroyFramebuffer(device, fb, nullptr);
    }
    framebuffers.clear();
    if (postProcessPipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(device, postProcessPipeline, nullptr);
      postProcessPipeline = VK_NULL_HANDLE;
    }
    vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()),
                         commandBuffers.data());
    commandBuffers.clear();
    vkDestroyPipeline(device, uiPresentPipeline, nullptr);
    uiPresentPipeline = VK_NULL_HANDLE;
    vkDestroyPipeline(device, uiPipeline, nullptr);
    uiPipeline = VK_NULL_HANDLE;
    vkDestroyPipeline(device, graphicsPipelineStaffSkinned, nullptr);
    graphicsPipelineStaffSkinned = VK_NULL_HANDLE;
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    graphicsPipeline = VK_NULL_HANDLE;
    if (presentRenderPass != VK_NULL_HANDLE) {
      vkDestroyRenderPass(device, presentRenderPass, nullptr);
      presentRenderPass = VK_NULL_HANDLE;
    }
    if (renderPass != VK_NULL_HANDLE) {
      vkDestroyRenderPass(device, renderPass, nullptr);
      renderPass = VK_NULL_HANDLE;
    }
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      const size_t j = static_cast<size_t>(i);
      if (j < sceneColorViews.size() && sceneColorViews[j] != VK_NULL_HANDLE)
        vkDestroyImageView(device, sceneColorViews[j], nullptr);
      if (j < sceneColorImages.size() && sceneColorImages[j] != VK_NULL_HANDLE)
        vkDestroyImage(device, sceneColorImages[j], nullptr);
      if (j < sceneColorMemories.size() && sceneColorMemories[j] != VK_NULL_HANDLE)
        vkFreeMemory(device, sceneColorMemories[j], nullptr);
      if (j < depthViews.size() && depthViews[j] != VK_NULL_HANDLE)
        vkDestroyImageView(device, depthViews[j], nullptr);
      if (j < depthImages.size() && depthImages[j] != VK_NULL_HANDLE)
        vkDestroyImage(device, depthImages[j], nullptr);
      if (j < depthMemories.size() && depthMemories[j] != VK_NULL_HANDLE)
        vkFreeMemory(device, depthMemories[j], nullptr);
    }
    sceneColorViews.clear();
    sceneColorImages.clear();
    sceneColorMemories.clear();
    depthViews.clear();
    depthImages.clear();
    depthMemories.clear();
    sceneColorWasSampled.clear();
    depthGpuReady.clear();
    for (auto v : swapchainImageViews) {
      if (v != VK_NULL_HANDLE)
        vkDestroyImageView(device, v, nullptr);
    }
    swapchainImageViews.clear();
    if (swapchain != VK_NULL_HANDLE) {
      vkDestroySwapchainKHR(device, swapchain, nullptr);
      swapchain = VK_NULL_HANDLE;
    }
  }

  void recreateSwapchain() {
    int w = 0, h = 0;
    SDL_Vulkan_GetDrawableSize(window, &w, &h);
    while (w == 0 || h == 0) {
      SDL_Vulkan_GetDrawableSize(window, &w, &h);
      SDL_WaitEvent(nullptr);
    }
    vkDeviceWaitIdle(device);
    cleanupSwapchain();
    winW = w;
    winH = h;
    createSwapchain();
    createImageViews();
    createSceneColorResources();
    createDepthResources();
    createRenderPass();
    createPostProcessResources();
    createGraphicsPipeline();
    createPostPipeline();
    if (staffSkinnedActive)
      createStaffSkinnedPipeline();
    createFramebuffers();
    createCommandBuffers();
    createPostDescriptorSets();
  }

  void drawFrame(float dt) {
    lastDrawFrameDt = std::max(dt, 1e-4f);
    {
      float target = kPs1ParkourBaselineMix;
      if (ledgeClimbT >= 0.f || ladderClimbActive)
        target = 0.98f;
      else if (wallClimbActive)
        target = std::max(target, 0.94f);
      else if (wallRunActive)
        target = std::max(target, 0.9f);
      else if (speedVaultActive)
        target = std::max(target, 0.88f);
      else if (slideActive)
        target = std::max(target, 0.8f);
      else if (playerJumpAnimRemain > 1e-4f || playerPreFallAnimRemain > 1e-4f ||
               playerJumpPostLandRemain > 1e-4f)
        target = std::max(target, 0.86f);
      else if (wasGrounded && runAnimBlend > 0.4f && glm::length(horizVel) > kMaxSpeed * 0.48f)
        target = std::max(target, 0.62f);
      parkourPs1PresentMix =
          glm::mix(parkourPs1PresentMix, target, 1.f - std::exp(-dt * kPs1ParkourPresentSmoothHz));
    }
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex = 0;
    VkResult acq = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                         imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE,
                                         &imageIndex);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
      recreateSwapchain();
      return;
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR)
      throw std::runtime_error("acquire");

    vkResetFences(device, 1, &inFlightFences[currentFrame]);

    horrorPresentTime += dt;
    glm::mat4 proj = glm::perspective(
        glm::radians(70.0f),
        static_cast<float>(sceneExtent.width) / std::max(1u, sceneExtent.height), 0.09f,
        kProjFarPlane);
    proj[1][1] *= -1.0f;
    {
      if (!viewStepSmoothYInited) {
        viewStepSmoothY = camPos.y;
        viewStepSmoothYInited = true;
      }
      const float delta = camPos.y - viewStepSmoothY;
      constexpr float kStepSmoothMaxDelta = 0.35f;
      constexpr float kStepSmoothRate = 38.f;
      if (std::abs(delta) > kStepSmoothMaxDelta || dt > 0.15f)
        viewStepSmoothY = camPos.y;
      else
        viewStepSmoothY += delta * (1.f - std::exp(-dt * kStepSmoothRate));
    }
    glm::vec3 eye, fwd, right, up;
    getRenderViewBasis(eye, fwd, right, up);
    glm::mat4 view = glm::lookAt(eye, eye + fwd, up);
    UniformBufferObject ubo{};
    ubo.viewProj = proj * view;
    ubo.cameraPos = glm::vec4(eye, 0.0f);
    const bool storeLit = audioAreStoreFluorescentsOn();
    postNightHorrorWeight = storeLit ? 0.18f : 0.72f;
    {
      const float pursuitTarget = shelfEmpNightPursuitActive ? 1.f : 0.f;
      postNightPursuitMix =
          glm::mix(postNightPursuitMix, pursuitTarget, 1.f - std::exp(-dt * 3.4f));
    }
    const float storeLightMul = storeLit ? 1.f : kStoreLightMulBlackout;
    const float fogA = storeLit ? kViewFogStart : kViewFogStartBlackout;
    const float fogB = storeLit ? kViewFogEnd : kViewFogEndBlackout;
    ubo.fogParams = glm::vec4(fogA, fogB, storeLightMul, storeLit ? 0.f : 1.f);
    const bool menuCam = inTitleMenu || inIntroSplash || inLoadingScreen;
    ubo.shadowParams =
        menuCam ? glm::vec4(eye.x, eye.z, lastShadowGroundUnderY, 0.58f)
                : glm::vec4(camPos.x, camPos.z, lastShadowGroundUnderY, 0.58f);
    ubo.employeeFadeH = glm::vec4(kEmployeeFadeInnerH, kEmployeeFadeOuterH,
                                  menuCam ? titleMenuSceneTime : 0.f,
                                  thirdPersonTestMode ? 0.f : 1.f);
    ubo.employeeBounds = employeeBounds;
    ubo.extraTexInfo = glm::ivec4(static_cast<int>(extraTexturesLoadedCount), uboCachedExtraBlend,
                                  uboCachedStaffTexBlend,
                                  staffGlbDiffuseActive);
    const float staffGaitY =
        menuCam ? 1.08f
                    : (storeLit ? (shelfEmpAnyDayPushChase ? 2.85f : 1.25f)
                                : glm::mix(2.85f, 3.24f, postNightPursuitMix));
    const float ps1MixForHorrorGfx = effectivePs1HorrorMix(parkourPs1PresentMix, postNightHorrorWeight);
    ubo.staffAnim =
        glm::vec4(staffSimTime, staffGaitY, ps1MixForHorrorGfx,
                  staffSkinnedActive ? static_cast<float>(staffRigBoneCount) : 0.f);
    std::memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(UniformBufferObject));

    recordCommandBuffer(commandBuffers[imageIndex], imageIndex, currentFrame, storeLit, eye, fwd);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &imageAvailableSemaphores[currentFrame];
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &commandBuffers[imageIndex];
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];
    VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &si, inFlightFences[currentFrame]), "submit");

    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &renderFinishedSemaphores[currentFrame];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain;
    pi.pImageIndices = &imageIndex;
    VkResult pr = vkQueuePresentKHR(presentQueue, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || framebufferResized) {
      framebufferResized = false;
      recreateSwapchain();
    } else if (pr != VK_SUCCESS) {
      throw std::runtime_error("present");
    }

    currentFrame = (currentFrame + 1) % kMaxFramesInFlight;
    (void)dt;
  }

  bool tryHandleMenuClick(float ndcX, float ndcY, uint8_t mouseButton = SDL_BUTTON_LEFT) {
    if (showControlsOverlay)
      return false;
    const float padY = 0.014f;
    const auto rowHit = [&](const UiMenuClickLayout& L, int rowIndex) -> bool {
      if (ndcX < -L.panelHalfW || ndcX > L.panelHalfW)
        return false;
      const float step = std::max(L.lineSkipNdc, 1e-4f);
      const float rel = (L.line2Y - ndcY) / step;
      const int nearest = static_cast<int>(std::lround(rel));
      // Keep each clickable band non-overlapping; only accept clicks near a row center.
      const float tol = 0.43f + padY / step;
      if (std::fabs(rel - static_cast<float>(nearest)) > tol)
        return false;
      return nearest == rowIndex;
    };
    if (inTitleMenu) {
      if (titleMenuPickSlot) {
        std::array<bool, 4> used{};
        for (int i = 0; i < kGameSaveSlotCount; ++i)
          used[static_cast<size_t>(i)] = saveSlotFileLooksValid(gameSaveSlotPath(i));
        const UiMenuClickLayout L = computeTitleMenuSlotPickerClickLayout(used);
        if (rowHit(L, 6) && mouseButton == SDL_BUTTON_LEFT) {
          titleMenuPickSlot = false;
          recreateTitleMenuMainGpuMesh();
          return true;
        }
        for (int s = 0; s < kGameSaveSlotCount; ++s) {
          if (rowHit(L, 1 + s)) {
            if (mouseButton == SDL_BUTTON_RIGHT) {
              deleteSaveSlot(s);
              recreateTitleMenuMainGpuMesh();
              recreateTitleMenuSlotGpuMesh();
            } else if (mouseButton == SDL_BUTTON_LEFT) {
              beginGameFromSaveSlot(s);
            } else
              return false;
            return true;
          }
        }
        return false;
      }
      const UiMenuClickLayout L = computeTitleMenuMainClickLayout(titleMenuHasContinue);
      if (titleMenuHasContinue) {
        if (rowHit(L, 0)) {
          continueFromLastSave();
          return true;
        }
        if (rowHit(L, 1)) {
          titleMenuPickSlot = true;
          recreateTitleMenuSlotGpuMesh();
          return true;
        }
        if (rowHit(L, 2)) {
          running = false;
          return true;
        }
      } else {
        if (rowHit(L, 0)) {
          titleMenuPickSlot = true;
          recreateTitleMenuSlotGpuMesh();
          return true;
        }
        if (rowHit(L, 1)) {
          running = false;
          return true;
        }
      }
      return false;
    }
    if (playerDeathShowMenu) {
      const UiMenuClickLayout L = computeDeathMenuClickLayout();
      if (rowHit(L, 0)) {
        respawnPlayerAfterDeath();
        return true;
      }
      if (rowHit(L, 1)) {
        returnToTitleMenuFromGame();
        return true;
      }
      return false;
    }
    if (showPauseMenu) {
      const UiMenuClickLayout L = computePauseMenuClickLayout();
      if (rowHit(L, 0)) {
        showPauseMenu = false;
        audioSetStoreDayNightCyclePaused(false);
        mouseGrab = true;
        syncInputGrab();
        return true;
      }
      if (rowHit(L, 1)) {
        gameSaveWrite();
        returnToTitleMenuFromGame();
        return true;
      }
      return false;
    }
    if (showInventoryMenu) {
      if (mouseButton != SDL_BUTTON_LEFT)
        return true;
      constexpr float listHalfW = 0.64f;
      constexpr float listTopY = 0.42f;
      constexpr float listBotY = -0.50f;
      constexpr int kVisibleRows = 8;
      if (ndcX < -listHalfW || ndcX > listHalfW || ndcY > listTopY || ndcY < listBotY)
        return true;
      const auto stacks = buildInventoryStacks(inventoryItems);
      if (stacks.empty())
        return true;
      const int maxScroll = std::max(0, static_cast<int>(stacks.size()) - kVisibleRows);
      const int start = std::clamp(inventoryScrollRow, 0, maxScroll);
      const float rowStep = (listTopY - listBotY - 0.12f) / static_cast<float>(kVisibleRows);
      const float rowY0 = listTopY - 0.09f;
      const int row = static_cast<int>(std::floor((rowY0 - ndcY + rowStep * 0.5f) / rowStep));
      if (row < 0 || row >= kVisibleRows)
        return true;
      const int idx = start + row;
      if (idx < 0 || idx >= static_cast<int>(stacks.size()))
        return true;
      const std::string& item = stacks[static_cast<size_t>(idx)].first;
      if (item == "PIZZA SLICE" || item == "MEATBALL") {
        if (playerHunger >= (kPlayerHungerMax - 1e-3f))
          return true;
        auto it = std::find(inventoryItems.begin(), inventoryItems.end(), item);
        if (it != inventoryItems.end()) {
          inventoryItems.erase(it);
          playerHunger = std::min(kPlayerHungerMax, playerHunger + kPlayerHungerPizzaGain);
          ++inventoryRevision;
          const int newMaxScroll = std::max(0, static_cast<int>(buildInventoryStacks(inventoryItems).size()) -
                                                   kVisibleRows);
          inventoryScrollRow = std::clamp(inventoryScrollRow, 0, newMaxScroll);
          inventoryMenuCacheScroll = -1;
        }
      }
      return true;
    }
    return false;
  }

  static uint64_t deliPizzaSlotKey(int worldAisleI, int worldAlongI) {
    const uint64_t a = static_cast<uint64_t>(static_cast<uint32_t>(worldAisleI));
    const uint64_t l = static_cast<uint64_t>(static_cast<uint32_t>(worldAlongI));
    return (a << 32) | l;
  }

static bool deliCounterUsesMeatballs(int worldAisleI, int worldAlongI) {
  if (!gDeliMeatballMeshLoaded)
    return false;
  const int ca = shelfBiomeClusterCoord(worldAisleI, kDeliBiomeClusterSpan);
  const int cl = shelfBiomeClusterCoord(worldAlongI, kDeliBiomeClusterSpan);
  int baseA, baseL;
  if (!deliClusterBase(ca, cl, baseA, baseL))
    return false;
  const int da = worldAisleI - baseA;
  const int dl = worldAlongI - baseL;
  if (da < 0 || da > 1 || dl < 0 || dl > 1)
    return false;
  // Deterministic 2x2 split per deli cluster: two pizza counters, two meatball counters.
  return ((da + dl) & 1) == 0;
}

  int deliPizzaSlicesRemaining(int worldAisleI, int worldAlongI) {
    const uint64_t k = deliPizzaSlotKey(worldAisleI, worldAlongI);
    auto it = deliPizzaSlicesBySlot.find(k);
    if (it == deliPizzaSlicesBySlot.end()) {
      auto rt = deliPizzaReplenishTimerBySlot.find(k);
      if (rt != deliPizzaReplenishTimerBySlot.end() && rt->second > 0.f) {
        deliPizzaSlicesBySlot.emplace(k, static_cast<uint8_t>(0));
        return 0;
      }
      deliPizzaSlicesBySlot.emplace(k, static_cast<uint8_t>(kDeliPizzaSlicesPerCounter));
      return kDeliPizzaSlicesPerCounter;
    }
    return static_cast<int>(it->second);
  }

  int deliMeatballsRemaining(int worldAisleI, int worldAlongI) {
    const uint64_t k = deliPizzaSlotKey(worldAisleI, worldAlongI);
    auto it = deliMeatballsBySlot.find(k);
    if (it == deliMeatballsBySlot.end()) {
      auto rt = deliMeatballReplenishTimerBySlot.find(k);
      if (rt != deliMeatballReplenishTimerBySlot.end() && rt->second > 0.f) {
        deliMeatballsBySlot.emplace(k, static_cast<uint8_t>(0));
        return 0;
      }
      deliMeatballsBySlot.emplace(k, static_cast<uint8_t>(kDeliMeatballsPerCounter));
      return kDeliMeatballsPerCounter;
    }
    return static_cast<int>(it->second);
  }

  void tickDeliPizzaReplenish(float dt) {
    if (!std::isfinite(dt) || dt <= 0.f || deliPizzaReplenishTimerBySlot.empty())
      dt = 0.f;
    if (dt > 0.f) {
      for (auto it = deliPizzaReplenishTimerBySlot.begin(); it != deliPizzaReplenishTimerBySlot.end();) {
        float t = std::max(0.f, it->second - dt);
        if (t <= 0.f) {
          deliPizzaSlicesBySlot[it->first] = static_cast<uint8_t>(kDeliPizzaSlicesPerCounter);
          it = deliPizzaReplenishTimerBySlot.erase(it);
        } else {
          it->second = t;
          ++it;
        }
      }
      for (auto it = deliMeatballReplenishTimerBySlot.begin(); it != deliMeatballReplenishTimerBySlot.end();) {
        float t = std::max(0.f, it->second - dt);
        if (t <= 0.f) {
          deliMeatballsBySlot[it->first] = static_cast<uint8_t>(kDeliMeatballsPerCounter);
          it = deliMeatballReplenishTimerBySlot.erase(it);
        } else {
          it->second = t;
          ++it;
        }
      }
    }
  }

  int inventoryStackRowCount() const {
    return static_cast<int>(buildInventoryStacks(inventoryItems).size());
  }

  bool tryPickupNearestDeliPizzaSlice() {
    constexpr float kPickupRadius = kDeliFoodPickupRadius;
    int waMin, waMax, wlMin, wlMax;
    shelfGridWindowForRange(camPos.x, camPos.z, kPickupRadius + 1.0f, waMin, waMax, wlMin, wlMax);
    bool found = false;
    float bestD2 = kPickupRadius * kPickupRadius;
    int bestWa = 0;
    int bestWl = 0;
    for (int wa = waMin; wa <= waMax; ++wa) {
      for (int wl = wlMin; wl <= wlMax; ++wl) {
        if (!deliBarSlotOccupied(wa, wl))
          continue;
        const bool meatballCounter = deliCounterUsesMeatballs(wa, wl);
        const int pizzaRem = meatballCounter ? 0 : deliPizzaSlicesRemaining(wa, wl);
        const int meatRem = meatballCounter ? deliMeatballsRemaining(wa, wl) : 0;
        if (pizzaRem <= 0 && meatRem <= 0)
          continue;
        const float cx = (static_cast<float>(wa) + 0.5f) * kShelfAisleModulePitch;
        const float cz = (static_cast<float>(wl) + 0.5f) * kShelfAlongAislePitch;
        const float dx = cx - camPos.x;
        const float dz = cz - camPos.z;
        const float d2 = dx * dx + dz * dz;
        if (d2 <= bestD2) {
          bestD2 = d2;
          bestWa = wa;
          bestWl = wl;
          found = true;
        }
      }
    }
    if (!found)
      return false;

    const uint64_t k = deliPizzaSlotKey(bestWa, bestWl);
    const bool meatballCounter = deliCounterUsesMeatballs(bestWa, bestWl);
    int pizzaRem = 0;
    auto itPz = deliPizzaSlicesBySlot.find(k);
    if (itPz != deliPizzaSlicesBySlot.end())
      pizzaRem = static_cast<int>(itPz->second);
    int meatRem = 0;
    auto itMb = deliMeatballsBySlot.find(k);
    if (itMb != deliMeatballsBySlot.end())
      meatRem = static_cast<int>(itMb->second);
    const bool canPickupPizza = !meatballCounter && pizzaRem > 0;
    const bool canPickupMeatball = meatballCounter && gDeliMeatballMeshLoaded && meatRem > 0;
    if (!canPickupPizza && !canPickupMeatball)
      return false;
    if (canPickupPizza && itPz != deliPizzaSlicesBySlot.end()) {
      --itPz->second;
      deliPizzaReplenishTimerBySlot[k] = kDeliPizzaReplenishSec;
      inventoryItems.emplace_back("PIZZA SLICE");
    } else if (canPickupMeatball && itMb != deliMeatballsBySlot.end()) {
      --itMb->second;
      deliMeatballReplenishTimerBySlot[k] = kDeliPizzaReplenishSec;
      inventoryItems.emplace_back("MEATBALL");
    } else {
      return false;
    }
    ++inventoryRevision;
    const int maxScroll = std::max(0, inventoryStackRowCount() - 8);
    inventoryScrollRow = std::clamp(inventoryScrollRow, 0, maxScroll);
    return true;
  }

  bool canPickupNearbyDeliFood() {
    constexpr float kPickupRadius = kDeliFoodPickupRadius;
    int waMin, waMax, wlMin, wlMax;
    shelfGridWindowForRange(camPos.x, camPos.z, kPickupRadius + 1.0f, waMin, waMax, wlMin, wlMax);
    float bestD2 = kPickupRadius * kPickupRadius;
    for (int wa = waMin; wa <= waMax; ++wa) {
      for (int wl = wlMin; wl <= wlMax; ++wl) {
        if (!deliBarSlotOccupied(wa, wl))
          continue;
        const bool meatballCounter = deliCounterUsesMeatballs(wa, wl);
        const uint64_t k = deliPizzaSlotKey(wa, wl);
        int pizzaRem = 0;
        if (!meatballCounter) {
          pizzaRem = kDeliPizzaSlicesPerCounter;
          auto pz = deliPizzaSlicesBySlot.find(k);
          if (pz != deliPizzaSlicesBySlot.end()) {
            pizzaRem = static_cast<int>(pz->second);
          } else {
            auto pt = deliPizzaReplenishTimerBySlot.find(k);
            if (pt != deliPizzaReplenishTimerBySlot.end() && pt->second > 0.f)
              pizzaRem = 0;
          }
        }
        int meatRem = 0;
        if (meatballCounter && gDeliMeatballMeshLoaded) {
          meatRem = kDeliMeatballsPerCounter;
          auto mb = deliMeatballsBySlot.find(k);
          if (mb != deliMeatballsBySlot.end()) {
            meatRem = static_cast<int>(mb->second);
          } else {
            auto mt = deliMeatballReplenishTimerBySlot.find(k);
            if (mt != deliMeatballReplenishTimerBySlot.end() && mt->second > 0.f)
              meatRem = 0;
          }
        }
        if (pizzaRem <= 0 && meatRem <= 0)
          continue;
        const float cx = (static_cast<float>(wa) + 0.5f) * kShelfAisleModulePitch;
        const float cz = (static_cast<float>(wl) + 0.5f) * kShelfAlongAislePitch;
        const float dx = cx - camPos.x;
        const float dz = cz - camPos.z;
        const float d2 = dx * dx + dz * dz;
        if (d2 <= bestD2)
          return true;
      }
    }
    return false;
  }

  void handleEvent(const SDL_Event& e) {
    if (e.type == SDL_QUIT)
      running = false;
    if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_RESIZED)
      framebufferResized = true;
    if (inLoadingScreen)
      return;
    if (inIntroSplash && (e.type == SDL_KEYDOWN || e.type == SDL_MOUSEBUTTONDOWN)) {
      inIntroSplash = false;
      inTitleMenu = true;
      titleMenuSceneTime = 0.f;
      titleMenuSlideTime = 0.f;
      titleMenuSlideWasSlot = false;
      return;
    }
    if (inIntroSplash)
      return;
    if (e.type == SDL_MOUSEBUTTONDOWN && showControlsOverlay) {
      showControlsOverlay = false;
      audioSetStoreDayNightCyclePaused(false);
      mouseGrab = true;
      syncInputGrab();
      return;
    }
    if (e.type == SDL_MOUSEBUTTONDOWN &&
        (e.button.button == SDL_BUTTON_LEFT || e.button.button == SDL_BUTTON_RIGHT)) {
      const int mx = static_cast<int>(e.button.x);
      const int my = static_cast<int>(e.button.y);
      float ndcX = 0.f, ndcY = 0.f;
      sdlWindowMouseToUiNdc(window, mx, my, swapchainExtent.width, swapchainExtent.height, &ndcX, &ndcY);
      if (tryHandleMenuClick(ndcX, ndcY, e.button.button))
        return;
    }
    if (e.type == SDL_KEYDOWN) {
      const int sc = static_cast<int>(e.key.keysym.scancode);
      if (sc >= 0 && sc < SDL_NUM_SCANCODES)
        scancodeDown[static_cast<size_t>(sc)] = true;
      if (e.key.repeat == 0 && sc == SDL_SCANCODE_C && !playerDeathActive && !showPauseMenu &&
          !showInventoryMenu && !inTitleMenu)
        pendingSlideCrouchEdge = true;
    }
    if (e.type == SDL_KEYUP) {
      const int sc = static_cast<int>(e.key.keysym.scancode);
      if (sc >= 0 && sc < SDL_NUM_SCANCODES)
        scancodeDown[static_cast<size_t>(sc)] = false;
    }
    if (e.type == SDL_MOUSEWHEEL && showInventoryMenu) {
      inventoryScrollRow = std::max(0, inventoryScrollRow - e.wheel.y);
      {
        const int maxScroll = std::max(0, inventoryStackRowCount() - 8);
        inventoryScrollRow = std::clamp(inventoryScrollRow, 0, maxScroll);
      }
      inventoryMenuCacheScroll = -1;
      return;
    }
    if (e.type == SDL_MOUSEWHEEL && !playerDeathActive && !showPauseMenu && !showInventoryMenu &&
        !inTitleMenu) {
      constexpr float kTpZoomMin = 1.15f;
      constexpr float kTpZoomMax = 14.f;
      // When zoomed all the way in, switch to true first-person; scrolling out re-enters third-person orbit.
      constexpr float kTpToFpCutover = kTpZoomMin + 0.06f;
      constexpr float kTpZoomWheelStep = 0.50f;
      thirdPersonCamDist = glm::clamp(
          thirdPersonCamDist - static_cast<float>(e.wheel.y) * kTpZoomWheelStep, kTpZoomMin, kTpZoomMax);
      thirdPersonTestMode = thirdPersonCamDist > kTpToFpCutover;
    }
    if (e.type == SDL_WINDOWEVENT && e.window.event == SDL_WINDOWEVENT_FOCUS_GAINED &&
        (mouseGrab || inTitleMenu || showPauseMenu || showInventoryMenu || playerDeathShowMenu))
      syncInputGrab();
    if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT && !showControlsOverlay &&
        !mouseGrab && !inTitleMenu && !showPauseMenu && !showInventoryMenu && !playerDeathShowMenu) {
      mouseGrab = true;
      syncInputGrab();
    }
    if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT && !showControlsOverlay &&
        mouseGrab && !playerDeathActive && !showPauseMenu && !showInventoryMenu && !inTitleMenu)
      pendingStaffShoveLmb = true;
    if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_f && e.key.repeat == 0 &&
        !showControlsOverlay && mouseGrab && !playerDeathActive && !showPauseMenu &&
        !showInventoryMenu && !inTitleMenu)
      pendingPlayerKick = true;
    if (e.type == SDL_KEYDOWN && e.key.repeat == 0 && e.key.keysym.sym == SDLK_e &&
        !showControlsOverlay && !showPauseMenu && !showInventoryMenu && !playerDeathActive &&
        !inTitleMenu) {
      if (tryPickupNearestDeliPizzaSlice())
        return;
    }
    if (e.type == SDL_KEYDOWN && e.key.repeat == 0) {
      const SDL_Keycode sym = e.key.keysym.sym;
      if (inTitleMenu) {
        if (titleMenuPickSlot) {
          if (sym == SDLK_ESCAPE) {
            titleMenuPickSlot = false;
            return;
          }
          int slotChoice = -1;
          if (sym >= SDLK_1 && sym <= SDLK_4)
            slotChoice = sym - SDLK_1;
          else if (sym >= SDLK_KP_1 && sym <= SDLK_KP_4)
            slotChoice = sym - SDLK_KP_1;
          if (slotChoice >= 0) {
            beginGameFromSaveSlot(slotChoice);
            return;
          }
        } else {
          if (sym == SDLK_ESCAPE) {
            running = false;
            return;
          }
          if (sym == SDLK_RETURN || sym == SDLK_KP_ENTER) {
            if (titleMenuHasContinue)
              continueFromLastSave();
            else {
              titleMenuPickSlot = true;
              recreateTitleMenuSlotGpuMesh();
            }
            return;
          }
          if (sym == SDLK_c && titleMenuHasContinue) {
            continueFromLastSave();
            return;
          }
          if (sym == SDLK_s) {
            titleMenuPickSlot = true;
            recreateTitleMenuSlotGpuMesh();
            return;
          }
        }
      }
      if (playerDeathShowMenu) {
        if (sym == SDLK_r || sym == SDLK_RETURN || sym == SDLK_KP_ENTER)
          respawnPlayerAfterDeath();
        else if (sym == SDLK_ESCAPE)
          returnToTitleMenuFromGame();
        return;
      }
      if (showPauseMenu) {
        if (sym == SDLK_RETURN || sym == SDLK_KP_ENTER ||
            (sym == SDLK_ESCAPE && e.key.repeat == 0)) {
          showPauseMenu = false;
          audioSetStoreDayNightCyclePaused(false);
          mouseGrab = true;
          syncInputGrab();
        }
        return;
      }
      if (showInventoryMenu) {
        if (sym == SDLK_UP || sym == SDLK_w)
          inventoryScrollRow = std::max(0, inventoryScrollRow - 1);
        else if (sym == SDLK_DOWN || sym == SDLK_s)
          ++inventoryScrollRow;
        else if (sym == SDLK_PAGEUP)
          inventoryScrollRow = std::max(0, inventoryScrollRow - 6);
        else if (sym == SDLK_PAGEDOWN)
          inventoryScrollRow += 6;
        else if (sym == SDLK_TAB || sym == SDLK_ESCAPE) {
          showInventoryMenu = false;
          audioSetStoreDayNightCyclePaused(false);
          mouseGrab = true;
          syncInputGrab();
        }
        {
          const int maxScroll = std::max(0, inventoryStackRowCount() - 8);
          inventoryScrollRow = std::clamp(inventoryScrollRow, 0, maxScroll);
        }
        inventoryMenuCacheScroll = -1;
        return;
      }
      if (sym == SDLK_F1) {
        showControlsOverlay = !showControlsOverlay;
        audioSetStoreDayNightCyclePaused(showControlsOverlay);
        mouseGrab = !showControlsOverlay;
        syncInputGrab();
      } else if (sym == SDLK_TAB && !inTitleMenu && !showControlsOverlay && !playerDeathActive &&
                 !showPauseMenu && e.key.repeat == 0) {
        showInventoryMenu = !showInventoryMenu;
        if (showInventoryMenu) {
          audioSetStoreDayNightCyclePaused(true);
          mouseGrab = false;
        } else {
          audioSetStoreDayNightCyclePaused(false);
          mouseGrab = true;
        }
        syncInputGrab();
      } else if (!inTitleMenu && !showControlsOverlay &&
                 (e.key.keysym.scancode == SDL_SCANCODE_RIGHTBRACKET ||
                  e.key.keysym.scancode == SDL_SCANCODE_F10)) {
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
        spawnShrekEggNearPlayer();
#else
        std::fprintf(stderr,
                     "[easter] Shrek easter egg disabled at build time: add vulkan_game/assets/models/"
                     "character/proximity_dance.glb (or set VULKAN_GAME_SHREK_EGG_GLB_PATH), reconfigure, rebuild.\n");
#endif
      } else if (!inTitleMenu && !showControlsOverlay && !playerDeathActive && !showInventoryMenu &&
                 !showPauseMenu &&
                 (e.key.keysym.scancode == SDL_SCANCODE_BACKSLASH || sym == SDLK_BACKSLASH)) {
        spawnDeadRagdollStaffNearPlayer();
      } else if (showControlsOverlay &&
                 (sym == SDLK_w || sym == SDLK_a || sym == SDLK_s || sym == SDLK_d || sym == SDLK_z ||
                  sym == SDLK_SPACE || sym == SDLK_RETURN || sym == SDLK_KP_ENTER)) {
        showControlsOverlay = false;
        audioSetStoreDayNightCyclePaused(false);
        mouseGrab = true;
        syncInputGrab();
      } else if (sym == SDLK_ESCAPE) {
        if (showControlsOverlay) {
          showControlsOverlay = false;
          audioSetStoreDayNightCyclePaused(false);
          mouseGrab = true;
          syncInputGrab();
        } else if (!playerDeathActive && !inTitleMenu && e.key.repeat == 0) {
          showPauseMenu = !showPauseMenu;
          if (showPauseMenu) {
            audioSetStoreDayNightCyclePaused(true);
            mouseGrab = false;
            syncInputGrab();
            gameSaveWrite();
          } else {
            audioSetStoreDayNightCyclePaused(false);
            mouseGrab = true;
            syncInputGrab();
          }
        }
      } else if (!inTitleMenu && !showControlsOverlay && !playerDeathActive && !showInventoryMenu &&
                 e.key.keysym.scancode == SDL_SCANCODE_H && staffClipShrekProximityDance >= 0 &&
                 static_cast<size_t>(staffClipShrekProximityDance) < staffRig.clips.size() &&
                 ledgeClimbT < 0.f && !ledgeHangActive && !ladderClimbActive) {
        playerDanceEmoteActive = true;
        playerDanceEmoteStopGraceRemain = 0.22f;
        slideActive = false;
        slideAnimClip = -1;
        slideAnimElapsed = 0.f;
        slideAnimDurSec = 0.f;
        slideStartSpeed = 0.f;
        slideClearClipNextFrame = false;
        playerPushAnimRemain = 0.f;
        playerKickAnimRemain = 0.f;
        dropKickActive = false;
        dropKickTimer = 0.f;
      }
    }
  }

  AABB playerCollisionBox() const {
    const float feet = camPos.y - eyeHeight;
    return AABB{{camPos.x - kPlayerHalfXZ, feet, camPos.z - kPlayerHalfXZ},
                {camPos.x + kPlayerHalfXZ, feet + eyeHeight + 0.12f, camPos.z + kPlayerHalfXZ}};
  }

  // First-person eye and axes (mantle rays, staff shove cone, crosshair pick). Not used for draw when
  // thirdPersonTestMode is on.
  void getFirstPersonViewBasis(glm::vec3& outEye, glm::vec3& outFwd, glm::vec3& outRight,
                               glm::vec3& outUp) const {
    const glm::vec3 side(-std::sin(yaw), 0.f, std::cos(yaw));
    const float eyeY = viewStepSmoothYInited ? viewStepSmoothY : camPos.y;
    const glm::vec3 eyeBase = glm::vec3(camPos.x, eyeY, camPos.z) + glm::vec3(0.f, bobOffsetY + idleBobY + randomSwayBobY, 0.f);
    glm::vec3 eye = eyeBase + side * (bobSideOffset + idleSide + randomSwaySide + runSideSway);
    if (cameraOffsetHitsPillar(eyeBase, eye))
      eye = eyeBase;
    resolveEyeAgainstPillars(eye);
    resolveCameraEyeSupportAndCeiling(eye);
    // Follow head bone animation offset (run bob, idle sway, etc.).
    if (!thirdPersonTestMode && fpHeadBoneIdx >= 0)
      eye += fpHeadAnimOffset;
    const float pitchCam =
        pitch + swayPitch + walkPitchOsc + idlePitch + landingPitchOfs + randomSwayPitch;
    const float rollTotal = swayRoll + idleRoll + randomSwayRoll;
    const glm::vec3 lookForward{std::cos(yaw) * std::cos(pitchCam), std::sin(pitchCam),
                                std::sin(yaw) * std::cos(pitchCam)};
    glm::vec3 fwd = glm::normalize(lookForward);
    glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0.f, 1.f, 0.f), fwd));
    if (glm::length(right) < 1e-4f)
      right = glm::vec3(1.f, 0.f, 0.f);
    glm::vec3 up = glm::normalize(glm::cross(fwd, right));
    right = glm::normalize(right * std::cos(rollTotal) + up * std::sin(rollTotal));
    up = glm::normalize(glm::cross(fwd, right));
    outEye = eye;
    outFwd = fwd;
    outRight = right;
    outUp = up;
  }

  // Title screen: Dying Light–style perch on top deck, slow drift, looking down the aisle at staff below.
  void syncTitleMenuSceneAnchor() {
    glm::vec3 e, f, r, u;
    getTitleMenuViewBasis(e, f, r, u);
    titleMenuSceneAnchor = e;
  }

  void getTitleMenuViewBasis(glm::vec3& outEye, glm::vec3& outFwd, glm::vec3& outRight,
                             glm::vec3& outUp) const {
    constexpr int kTitleAisle = 0;
    constexpr int kTitleAlong = 2;
    const float aisleCX = (static_cast<float>(kTitleAisle) + 0.5f) * kShelfAisleModulePitch;
    const float cz = (static_cast<float>(kTitleAlong) + 0.5f) * kShelfAlongAislePitch;
    const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
    const float t = titleMenuSceneTime;
    outEye = glm::vec3(cxRight - 0.62f + 0.07f * std::sin(t * 0.29f),
                       kTopShelfDeckSurfaceY + kEyeHeight - 0.14f + 0.05f * std::sin(t * 0.35f + 0.4f),
                       cz + 0.42f * std::sin(t * 0.17f));
    const glm::vec3 target(aisleCX + 0.35f * std::sin(t * 0.21f), kGroundY + 1.35f,
                           cz + 2.4f + 0.55f * std::cos(t * 0.19f));
    glm::vec3 fwd = glm::normalize(target - outEye);
    glm::vec3 right = glm::normalize(glm::cross(fwd, glm::vec3(0.f, 1.f, 0.f)));
    if (glm::length(right) < 1e-4f)
      right = glm::vec3(1.f, 0.f, 0.f);
    glm::vec3 up = glm::normalize(glm::cross(right, fwd));
    outFwd = fwd;
    outRight = right;
    outUp = up;
  }

  // View used by drawFrame: third-person orbit when testing animations, else same as first person.
  void getRenderViewBasis(glm::vec3& outEye, glm::vec3& outFwd, glm::vec3& outRight,
                          glm::vec3& outUp) const {
    if (inTitleMenu) {
      getTitleMenuViewBasis(outEye, outFwd, outRight, outUp);
      return;
    }
    if (!thirdPersonTestMode) {
      getFirstPersonViewBasis(outEye, outFwd, outRight, outUp);
      return;
    }
    // Orbit around eye/head height (same vertical base as first person before horizontal bob).
    const float pivotY = camPos.y + bobOffsetY + idleBobY + randomSwayBobY;
    const glm::vec3 pivot(camPos.x, pivotY, camPos.z);
    const float pitchCam =
        pitch + swayPitch + walkPitchOsc + idlePitch + landingPitchOfs + randomSwayPitch;
    const glm::vec3 lookForward{std::cos(yaw) * std::cos(pitchCam), std::sin(pitchCam),
                                std::sin(yaw) * std::cos(pitchCam)};
    glm::vec3 orbitDir = lookForward;
    const float oLen = glm::length(orbitDir);
    if (oLen > 1e-5f)
      orbitDir *= 1.f / oLen;
    else
      orbitDir = glm::vec3(std::cos(yaw), 0.f, std::sin(yaw));
    // Full orbit: camera sits on the look ray behind the pivot (mouse look = free aim).
    // Shorten distance when pivot→eye cuts through a pillar (endpoints can both be outside).
    constexpr float kTpCamPullBackMinDist = 1.15f;
    float distUse = thirdPersonCamDist;
    glm::vec3 eye = pivot - orbitDir * distUse;
    for (int i = 0; i < 28 && distUse > kTpCamPullBackMinDist * 0.98f; ++i) {
      eye = pivot - orbitDir * distUse;
      if (!cameraOffsetHitsPillar(pivot, eye))
        break;
      distUse *= 0.91f;
    }
    resolveEyeAgainstPillars(eye);
    resolveThirdPersonEyeAboveFloor(eye);
    glm::vec3 fwd = glm::normalize(pivot - eye);
    glm::vec3 right = glm::normalize(glm::cross(glm::vec3(0.f, 1.f, 0.f), fwd));
    if (glm::length(right) < 1e-4f)
      right = glm::vec3(1.f, 0.f, 0.f);
    glm::vec3 up = glm::normalize(glm::cross(fwd, right));
    const float rollTotal = swayRoll + idleRoll + randomSwayRoll;
    right = glm::normalize(right * std::cos(rollTotal) + up * std::sin(rollTotal));
    up = glm::normalize(glm::cross(fwd, right));
    outEye = eye;
    outFwd = fwd;
    outRight = right;
    outUp = up;
  }

  // Facing a shelf crate ~box-sized: Space jump gets forward + up boost; uses normal jump / fall / land clips.
  bool computeVaultCrateJumpAssist(const glm::vec2& wish, bool hasWishInput, float& outFwdBoost,
                                   float& outVyBoost) const {
    outFwdBoost = 0.f;
    outVyBoost = 0.f;
    const float feetY = camPos.y - eyeHeight;
    const glm::vec2 p{camPos.x, camPos.z};
    const glm::vec2 f{std::cos(yaw), std::sin(yaw)};
    glm::vec2 moveDir = f;
    if (hasWishInput && glm::length(wish) > 1e-4f)
      moveDir = wish;
    else if (glm::length(horizVel) > 0.065f)
      moveDir = glm::normalize(horizVel);
    if (glm::dot(moveDir, f) < kVaultMoveAlignDotMin)
      return false;

    float bestT = 1e30f;
    bool found = false;
    constexpr float kCullR2 = 18.f * 18.f;
    constexpr float kGridRangeM = 20.f;
    int waMin, waMax, wlMin, wlMax;
    shelfGridWindowForRange(camPos.x, camPos.z, kGridRangeM, waMin, waMax, wlMin, wlMax);
    for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
      const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
      const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
      const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
      for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
        const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
        for (int side = 0; side < 2; ++side) {
          if (!shelfSlotOccupied(worldAisle, worldAlong, side))
            continue;
          const float cx = side ? cxRight : cxLeft;
          const float rdx = camPos.x - cx;
          const float rdz = camPos.z - cz;
          if (rdx * rdx + rdz * rdz > kCullR2)
            continue;
          float clx, clz, yDeck, chx, chy, chz;
          if (!shelfCrateLocalLayout(worldAisle, worldAlong, side, clx, clz, yDeck, chx, chy, chz))
            continue;
          if (chy < kVaultMinCrateHalfY || chy > kVaultMaxCrateHalfY)
            continue;
          if (chx < kVaultMinCrateHalfXZ || chz < kVaultMinCrateHalfXZ)
            continue;
          if (chx > kVaultMaxCrateHalfXZ || chz > kVaultMaxCrateHalfXZ)
            continue;
          const glm::vec3 shelfPos{cx, kGroundY, cz};
          const float shelfYawRad = glm::radians(side ? -90.0f : 90.0f);
          const AABB crate = shelfLocalBoxWorldAABB(
              shelfPos, shelfYawRad, {clx - chx, yDeck, clz - chz},
              {clx + chx, yDeck + 2.f * chy, clz + chz});
          const float topRise = crate.max.y - feetY;
          if (topRise < kVaultMinClearanceAboveFeet || topRise > kVaultMaxClearanceAboveFeet)
            continue;
          if (std::abs(feetY - crate.min.y) > kVaultFeetToCrateBaseMaxD)
            continue;
          if (crate.max.y < kGroundY + 0.65f)
            continue;
          if (crate.max.y + eyeHeight + 0.14f >= kCeilingY)
            continue;
          float tEn = 0.f, tEx = 0.f;
          if (!rayXZHitAabbPositiveT(p, f, crate.min.x, crate.max.x, crate.min.z, crate.max.z, tEn, tEx))
            continue;
          const float depth = tEx - tEn;
          if (tEn < kVaultMinForwardM || tEn > kVaultMaxForwardM)
            continue;
          if (depth < kVaultMinDepthAlongRayM)
            continue;
          if (tEn < bestT) {
            bestT = tEn;
            found = true;
          }
        }
      }
    }
    if (!found)
      return false;
    const float u = 1.f - glm::clamp(bestT / kVaultMaxForwardM, 0.f, 1.f);
    outFwdBoost = kVaultForwardImpulseMin + u * kVaultForwardImpulseExtra;
    outVyBoost = kVaultVertImpulseBonus;
    return true;
  }

  bool findSpeedVaultTarget(const glm::vec2& moveDirIn, SpeedVaultTarget& out) const {
    out = {};
    glm::vec2 moveDir = moveDirIn;
    if (glm::length(moveDir) < 1e-4f)
      moveDir = glm::vec2(std::cos(yaw), std::sin(yaw));
    else
      moveDir = glm::normalize(moveDir);
    const float feetY = camPos.y - eyeHeight;
    const glm::vec2 p{camPos.x, camPos.z};
    constexpr float kCullR2 = 18.f * 18.f;
    constexpr float kGridRangeM = 20.f;
    int waMin, waMax, wlMin, wlMax;
    shelfGridWindowForRange(camPos.x, camPos.z, kGridRangeM, waMin, waMax, wlMin, wlMax);
    for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
      const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
      const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
      const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
      for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
        const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
        for (int side = 0; side < 2; ++side) {
          if (!shelfSlotOccupied(worldAisle, worldAlong, side))
            continue;
          const float cx = side ? cxRight : cxLeft;
          const float rdx = camPos.x - cx;
          const float rdz = camPos.z - cz;
          if (rdx * rdx + rdz * rdz > kCullR2)
            continue;
          float clx, clz, yDeck, chx, chy, chz;
          if (!shelfCrateLocalLayout(worldAisle, worldAlong, side, clx, clz, yDeck, chx, chy, chz))
            continue;
          const glm::vec3 shelfPos{cx, kGroundY, cz};
          const float shelfYawRad = glm::radians(side ? -90.0f : 90.0f);
          const AABB crate = shelfLocalBoxWorldAABB(
              shelfPos, shelfYawRad, {clx - chx, yDeck, clz - chz},
              {clx + chx, yDeck + 2.f * chy, clz + chz});
          const float topRise = crate.max.y - feetY;
          if (topRise < 0.22f || topRise > kSpeedVaultMaxHeight)
            continue;
          if (std::abs(feetY - crate.min.y) > kVaultFeetToCrateBaseMaxD)
            continue;
          if (crate.max.y < kGroundY + 0.65f)
            continue;
          if (crate.max.y + eyeHeight + 0.14f >= kCeilingY)
            continue;
          float tEn = 0.f, tEx = 0.f;
          if (!rayXZHitAabbPositiveT(p, moveDir, crate.min.x, crate.max.x, crate.min.z, crate.max.z, tEn, tEx))
            continue;
          const float depth = tEx - tEn;
          if (tEn < 0.02f || tEn > 0.95f)
            continue;
          if (depth < 0.18f)
            continue;
          if (!out.hit || tEn < out.t) {
            out.hit = true;
            out.t = tEn;
            out.box = crate;
            out.dir = moveDir;
            out.topY = crate.max.y;
          }
        }
      }
    }
    return out.hit;
  }

  bool findNearbyWallSurface(const glm::vec2& moveDirIn, float maxDist, ParkourWallHit& out) const {
    out = {};
    glm::vec2 moveDir = moveDirIn;
    if (glm::length(moveDir) < 1e-4f)
      moveDir = glm::vec2(std::cos(yaw), std::sin(yaw));
    else
      moveDir = glm::normalize(moveDir);
    const glm::vec2 right(std::cos(yaw + glm::half_pi<float>()), std::sin(yaw + glm::half_pi<float>()));
    const glm::vec3 ro(camPos.x, camPos.y - eyeHeight + eyeHeight * kWallProbeChestHeightFrac, camPos.z);
    auto testBox = [&](const AABB& box, bool pillar, bool shelf) {
      if (ro.y <= box.min.y + 0.05f || ro.y >= box.max.y - 0.05f)
        return;
      const glm::vec2 dirs[3] = {moveDir, right, -right};
      for (const glm::vec2& dirXZ : dirs) {
        if (glm::length(dirXZ) < 1e-4f)
          continue;
        const glm::vec3 rd(dirXZ.x, 0.f, dirXZ.y);
        float tHit = 0.f;
        if (!rayAABBFirstHit(ro, rd, box, tHit))
          continue;
        if (tHit <= 1e-4f || tHit > maxDist)
          continue;
        const glm::vec3 hitPoint = ro + rd * tHit;
        const glm::vec3 normal = outwardNormalFromAabbHitPointXZ(hitPoint, box);
        const glm::vec2 toWall = glm::normalize(glm::vec2(-normal.x, -normal.z));
        const float sideDot = glm::dot(toWall, right);
        const int side = sideDot >= 0.f ? 1 : -1;
        if (!out.hit || tHit < out.t) {
          out.hit = true;
          out.pillar = pillar;
          out.shelf = shelf;
          out.t = tHit;
          out.side = side;
          out.box = box;
          out.normal = normal;
          out.point = hitPoint;
        }
      }
    };

    const int gcx = static_cast<int>(std::floor(camPos.x / kPillarSpacing));
    const int gcz = static_cast<int>(std::floor(camPos.z / kPillarSpacing));
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dz = -1; dz <= 1; ++dz) {
        const float px = static_cast<float>(gcx + dx) * kPillarSpacing;
        const float pz = static_cast<float>(gcz + dz) * kPillarSpacing;
        testBox(pillarCollisionAABB(px, pz), true, false);
      }
    }

    constexpr float kCullR2 = 20.f * 20.f;
    constexpr float kGridRangeM = 22.f;
    int waMin, waMax, wlMin, wlMax;
    shelfGridWindowForRange(camPos.x, camPos.z, kGridRangeM, waMin, waMax, wlMin, wlMax);
    for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
      const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
      const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
      const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
      for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
        const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
        for (int side = 0; side < 2; ++side) {
          if (!shelfSlotOccupied(worldAisle, worldAlong, side))
            continue;
          const float cx = side ? cxRight : cxLeft;
          const float dx = camPos.x - cx;
          const float dz = camPos.z - cz;
          if (dx * dx + dz * dz > kCullR2)
            continue;
          const glm::vec3 shelfPos{cx, kGroundY, cz};
          const float shelfYawRad = glm::radians(side ? -90.0f : 90.0f);
          const AABB shelfBox = shelfLocalBoxWorldAABB(
              shelfPos, shelfYawRad, {-kShelfMeshHalfW, 0.f, -kShelfMeshHalfD},
              {kShelfMeshHalfW, kShelfMeshHeight, kShelfMeshHalfD});
          testBox(shelfBox, false, true);
          const float yawDeg = side ? -90.0f : 90.0f;
          float llx, llz, leanR;
          glm::vec3 ladderScl;
          if (shelfLadderLocalLayout(worldAisle, worldAlong, side, llx, llz, leanR, ladderScl)) {
            const glm::mat4 ladderM = shelfLadderWorldModel(cx, cz, yawDeg, llx, llz, leanR, ladderScl);
            testBox(shelfLadderCollisionAABB(ladderM), false, true);
          }
        }
      }
    }
    return out.hit;
  }

  bool findLedgeMantleTarget(glm::vec3& outEndCam, AABB* outDeck = nullptr) const {
    glm::vec3 ro, rdCenter, camRight, camUp;
    // Always aim mantle rays from the player's eye, not the orbit camera.
    getFirstPersonViewBasis(ro, rdCenter, camRight, camUp);
    const float feet = camPos.y - eyeHeight;
    const float runSp = glm::length(horizVel);
    float moveBoost = 0.f, extraFall = 0.f;
    mantleLedgeMovementAid(horizVel, isGrounded(), moveBoost, extraFall);
    const float runTEff = glm::min(1.f, mantleRunT(horizVel) + moveBoost);
    const float reachXZ = mantleReachXZFromView(rdCenter, horizVel, moveBoost);
    const float reachUse = reachXZ * kLedgeGrabReachLeniency;
    const float reach2 = reachUse * reachUse;
    const float maxVelUp = mantleMaxVelUp(runTEff);
    const float minFallGrab = kLedgeGrabMaxFallVelY - extraFall;
    float bestTHit = 1e30f;
    glm::vec3 bestEnd(0.f);
    bool have = false;
    AABB bestDeck{};
    const MantleProbeParams mantleMp{feet,      velY,      maxVelUp, minFallGrab, reach2, camPos, ro,
                                       rdCenter,  camRight,  camUp,    horizVel,    runSp,  runTEff,
                                       eyeHeight};

    constexpr float kCullR2 = 20.f * 20.f;
    constexpr float kGridRangeM = 22.f;
    int waMin, waMax, wlMin, wlMax;
    shelfGridWindowForRange(camPos.x, camPos.z, kGridRangeM, waMin, waMax, wlMin, wlMax);
    for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
      const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
      const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
      const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
      for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
        const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
        for (int side = 0; side < 2; ++side) {
          if (!shelfSlotOccupied(worldAisle, worldAlong, side))
            continue;
          const float cx = side ? cxRight : cxLeft;
          const float rdx = camPos.x - cx;
          const float rdz = camPos.z - cz;
          if (rdx * rdx + rdz * rdz > kCullR2)
            continue;
          const glm::vec3 shelfPos{cx, kGroundY, cz};
          const float yawDeg = side ? -90.0f : 90.0f;
          const float shelfYawRad = glm::radians(yawDeg);
          const float hw = kShelfMeshHalfW;
          const float hd = kShelfMeshHalfD;
          const float shelfT = kShelfDeckThickness;
          const int numShelves = kShelfDeckCount;
          constexpr float yBase = 0.12f;
          const float yStep = kShelfGapBetweenLevels + shelfT;
          for (int si = 1; si < numShelves; ++si) {
            const float y0 = yBase + static_cast<float>(si) * yStep;
            const float y1 = y0 + shelfT;
            const AABB deck =
                shelfLocalBoxWorldAABB(shelfPos, shelfYawRad,
                                       {-hw + kShelfDeckInset, y0, -hd + kShelfDeckInset},
                                       {hw - kShelfDeckInset, y1, hd - kShelfDeckInset});
            mantleConsiderHorizontalLedge(mantleMp, deck, aisleCX, cx, bestTHit, bestEnd, have, bestDeck);
          }
          float clx, clz, yDeckTop, chx, chy, chz;
          if (shelfCrateLocalLayout(worldAisle, worldAlong, side, clx, clz, yDeckTop, chx, chy, chz)) {
            const float topLocal = yDeckTop + 2.f * chy;
            constexpr float kCrateTopInset = 0.04f;
            const AABB crateTopSlab = shelfLocalBoxWorldAABB(
                shelfPos, shelfYawRad,
                {clx - chx + kCrateTopInset, topLocal - 0.09f, clz - chz + kCrateTopInset},
                {clx + chx - kCrateTopInset, topLocal, clz + chz - kCrateTopInset});
            const float y0crate = yDeckTop - shelfT;
            const int deckIdxC =
                static_cast<int>(std::floor((y0crate - yBase) / yStep + 0.001f));
            float nextDeckBottomY = 1e30f;
            if (deckIdxC >= 0 && deckIdxC < numShelves - 1)
              nextDeckBottomY = kGroundY + yBase + static_cast<float>(deckIdxC + 1) * yStep;
            mantleConsiderHorizontalLedge(mantleMp, crateTopSlab, aisleCX, cx, bestTHit, bestEnd, have,
                                          bestDeck, true, nextDeckBottomY);
          }
        }
      }
    }
    if (!have)
      return false;
    outEndCam = bestEnd;
    if (outDeck)
      *outDeck = bestDeck;
    return true;
  }

  // Same geometry as air pull-up: “far” = ground jump needs a short squat charge before takeoff.
  bool playerJumpSquatTargetIsFar(glm::vec3& outEndCam) const {
    if (!findLedgeMantleTarget(outEndCam))
      return false;
    const float feetNow = camPos.y - eyeHeight;
    const float feetTarget = outEndCam.y - eyeHeight;
    const float dY = feetTarget - feetNow;
    const float hxz = glm::length(glm::vec2(outEndCam.x - camPos.x, outEndCam.z - camPos.z));
    const bool close = dY <= kJumpSquatCloseMaxDeltaY && hxz <= kJumpSquatCloseMaxHoriz;
    return !close;
  }

  bool tryStartLedgeClimb() {
    glm::vec3 end{};
    if (!findLedgeMantleTarget(end))
      return false;
    if (end.y - eyeHeight < kGroundY + 0.65f)
      return false;
    wallRunActive = false;
    wallRunTimer = 0.f;
    wallClimbActive = false;
    wallClimbTimer = 0.f;
    speedVaultActive = false;
    speedVaultTimer = 0.f;
    wallRunCameraTilt = 0.f;
    slideActive = false;
    slideAnimClip = -1;
    slideAnimElapsed = 0.f;
    slideAnimDurSec = 0.f;
    slideStartSpeed = 0.f;
    slideClearClipNextFrame = false;
    playerJumpAnimRemain = 0.f;
    playerJumpArchActive = false;
    playerJumpAirTimeTargetSec = 0.f;
    playerJumpPostLandRemain = 0.f;
    playerJumpPostLandDurationInit = 0.f;
    playerJumpPostLandSecondHalfScrub = false;
    playerJumpPostLandClipIndex = -1;
    playerJumpRunTailActive = false;
    playerPreFallAnimRemain = 0.f;
    playerPreFallFeetLockY = 0.f;
    playerPreFallUseRunClip = false;
    playerFallAnimTime = 0.f;
    playerJumpAwaitPreLandSecondHalf = false;
    playerJumpLedgeSecondHalfAir = false;
    playerVaultCrateJumpActive = false;
    ledgeClimbApproachVel = horizVel;
    velY = 0.f;
    horizVel = glm::vec2(0.f);
    ledgeClimbStartCam = camPos;
    ledgeClimbEndCam = end;
    const glm::vec2 pullXZ{end.x - camPos.x, end.z - camPos.z};
    const float pullLen = glm::length(pullXZ);
    ledgeClimbExitHoriz = pullLen > 1e-4f ? pullXZ * (1.f / pullLen) : glm::vec2(0.f);
    ledgeClimbT = 0.f;
    ledgeClimbVisPhase = 0.f;
    footstepDistAccum = 0.f;
    playerFallChainMaxFeetY = std::numeric_limits<float>::quiet_NaN();
    return true;
  }

  bool tryStartLedgeHang() {
    glm::vec3 end{};
    if (!findLedgeMantleTarget(end, &ledgeHangDeckAABB))
      return false;
    const float rise = end.y - camPos.y;
    if (rise < kLedgeHangMinRise)
      return false;
    {
      const glm::vec2 viewH(std::cos(yaw), std::sin(yaw));
      const glm::vec2 toDeck(end.x - camPos.x, end.z - camPos.z);
      const float tdLen = glm::length(toDeck);
      if (tdLen > 0.05f && glm::dot(viewH, toDeck / tdLen) < kLedgeHangMinLookToward)
        return false;
    }
    wallRunActive = false;
    wallRunTimer = 0.f;
    wallClimbActive = false;
    wallClimbTimer = 0.f;
    speedVaultActive = false;
    speedVaultTimer = 0.f;
    wallRunCameraTilt = 0.f;
    slideActive = false;
    slideAnimClip = -1;
    slideAnimElapsed = 0.f;
    slideAnimDurSec = 0.f;
    slideStartSpeed = 0.f;
    slideClearClipNextFrame = false;
    playerJumpAnimRemain = 0.f;
    playerJumpArchActive = false;
    playerJumpAirTimeTargetSec = 0.f;
    playerJumpPostLandRemain = 0.f;
    playerJumpPostLandDurationInit = 0.f;
    playerJumpPostLandSecondHalfScrub = false;
    playerJumpPostLandClipIndex = -1;
    playerJumpRunTailActive = false;
    playerPreFallAnimRemain = 0.f;
    playerPreFallFeetLockY = 0.f;
    playerPreFallUseRunClip = false;
    playerFallAnimTime = 0.f;
    playerJumpAwaitPreLandSecondHalf = false;
    playerJumpLedgeSecondHalfAir = false;
    playerVaultCrateJumpActive = false;
    ledgeHangApproachVel = horizVel;
    velY = 0.f;
    horizVel = glm::vec2(0.f);
    ledgeHangEndCam = end;
    const glm::vec2 pullXZ{end.x - camPos.x, end.z - camPos.z};
    const float pullLen = glm::length(pullXZ);
    // Keep hang/shimmy aligned to the player-facing deck face so shimmy runs along the outside ledge.
    glm::vec2 edgeOut(0.f);
    if (std::isfinite(ledgeHangDeckAABB.min.x) && std::isfinite(ledgeHangDeckAABB.max.x) &&
        std::isfinite(ledgeHangDeckAABB.min.z) && std::isfinite(ledgeHangDeckAABB.max.z)) {
      const glm::vec2 toPlayer(camPos.x - end.x, camPos.z - end.z);
      if (std::abs(toPlayer.x) >= std::abs(toPlayer.y))
        edgeOut = glm::vec2(toPlayer.x >= 0.f ? 1.f : -1.f, 0.f);
      else
        edgeOut = glm::vec2(0.f, toPlayer.y >= 0.f ? 1.f : -1.f);
    }
    ledgeHangExitHoriz = glm::length(edgeOut) > 0.5f
        ? edgeOut
        : (pullLen > 1e-4f ? pullXZ * (1.f / pullLen) : glm::vec2(0.f));
    // Face into the ledge while hanging/shimmying (exit vector points outward from shelf face).
    ledgeHangTargetYaw = std::atan2(-ledgeHangExitHoriz.y, -ledgeHangExitHoriz.x);
    ledgeHangStartYaw = yaw;
    ledgeHangStartCam = camPos;
    // Lock the hang anchor to the active deck face so shimmy stays aligned to the ledge edge.
    {
      const float mx = kPlayerHalfXZ + 0.1f;
      if (ledgeHangExitHoriz.x > 0.5f) {
        ledgeHangEndCam.x = ledgeHangDeckAABB.max.x - mx;
        ledgeHangEndCam.z = std::clamp(ledgeHangEndCam.z, ledgeHangDeckAABB.min.z + mx, ledgeHangDeckAABB.max.z - mx);
      } else if (ledgeHangExitHoriz.x < -0.5f) {
        ledgeHangEndCam.x = ledgeHangDeckAABB.min.x + mx;
        ledgeHangEndCam.z = std::clamp(ledgeHangEndCam.z, ledgeHangDeckAABB.min.z + mx, ledgeHangDeckAABB.max.z - mx);
      } else if (ledgeHangExitHoriz.y > 0.5f) {
        ledgeHangEndCam.z = ledgeHangDeckAABB.max.z - mx;
        ledgeHangEndCam.x = std::clamp(ledgeHangEndCam.x, ledgeHangDeckAABB.min.x + mx, ledgeHangDeckAABB.max.x - mx);
      } else if (ledgeHangExitHoriz.y < -0.5f) {
        ledgeHangEndCam.z = ledgeHangDeckAABB.min.z + mx;
        ledgeHangEndCam.x = std::clamp(ledgeHangEndCam.x, ledgeHangDeckAABB.min.x + mx, ledgeHangDeckAABB.max.x - mx);
      }
      ledgeHangCamPos = glm::vec3(
          ledgeHangEndCam.x + ledgeHangExitHoriz.x * kLedgeHangPullBack,
          end.y - eyeHeight - 0.1f,
          ledgeHangEndCam.z + ledgeHangExitHoriz.y * kLedgeHangPullBack);
      if (ledgeHangExitHoriz.x > 0.5f)
        ledgeHangCamPos.x = std::max(ledgeHangCamPos.x, ledgeHangDeckAABB.max.x + kLedgeHangOutsideClearance);
      else if (ledgeHangExitHoriz.x < -0.5f)
        ledgeHangCamPos.x = std::min(ledgeHangCamPos.x, ledgeHangDeckAABB.min.x - kLedgeHangOutsideClearance);
      if (ledgeHangExitHoriz.y > 0.5f)
        ledgeHangCamPos.z = std::max(ledgeHangCamPos.z, ledgeHangDeckAABB.max.z + kLedgeHangOutsideClearance);
      else if (ledgeHangExitHoriz.y < -0.5f)
        ledgeHangCamPos.z = std::min(ledgeHangCamPos.z, ledgeHangDeckAABB.min.z - kLedgeHangOutsideClearance);
    }
    ledgeHangActive = true;
    ledgeHangGrabTimer = kLedgeHangGrabDuration;
    ledgeHangShimmyPhase = 0.f;
    ledgeHangShimmyDir = 0;
    ledgeHangFeetYCaptured = false;
    footstepDistAccum = 0.f;
    playerFallChainMaxFeetY = std::numeric_limits<float>::quiet_NaN();
    return true;
  }

  bool advanceLedgeClimb(float dt) {
    if (ledgeClimbT < 0.f)
      return false;
    if (!std::isfinite(ledgeClimbT)) {
      ledgeClimbT = -1.f;
      return false;
    }
    ledgeClimbT += dt / kLedgeGrabDuration;
    ledgeClimbVisPhase += dt * 24.f;
    const float t = std::min(1.f, ledgeClimbT);
    float s;
    if (t <= kLedgeGrabAnimLiftPhaseEnd) {
      const float u = t / kLedgeGrabAnimLiftPhaseEnd;
      s = (1.f - std::pow(1.f - u, kLedgeGrabEaseOutPow)) * kLedgeGrabAnimLiftHeightFrac;
    } else {
      const float u = (t - kLedgeGrabAnimLiftPhaseEnd) / (1.f - kLedgeGrabAnimLiftPhaseEnd);
      const float settle = u * u * (3.f - 2.f * u);
      s = kLedgeGrabAnimLiftHeightFrac + (1.f - kLedgeGrabAnimLiftHeightFrac) * settle;
    }
    camPos = glm::mix(ledgeClimbStartCam, ledgeClimbEndCam, s);
    velY = 0.f;
    horizVel = glm::vec2(0.f);
    slideActive = false;
    slideAnimClip = -1;
    slideAnimElapsed = 0.f;
    slideAnimDurSec = 0.f;
    slideStartSpeed = 0.f;
    slideClearClipNextFrame = false;
    jumpBuffer = 0.f;
    coyoteTime = 0.f;
    if (ledgeClimbT >= 1.f || ledgeClimbT > 4.f) {
      ledgeClimbT = -1.f;
      ledgeClimbVisPhase = 0.f;
      camPos = ledgeClimbEndCam;
      resolvePillarCollisions();
      const glm::vec2 appr = ledgeClimbApproachVel;
      ledgeClimbApproachVel = glm::vec2(0.f);
      if (glm::length(ledgeClimbExitHoriz) > 1e-4f) {
        const glm::vec2 exitN = glm::normalize(ledgeClimbExitHoriz);
        const float fwdIn = glm::dot(appr, exitN);
        const glm::vec2 side = appr - exitN * fwdIn;
        const float fwdSp = glm::max(
            kLedgeGrabExitSpeed * kLedgeMantleExitMinForward,
            kLedgeGrabExitSpeed + std::max(0.f, fwdIn) * kLedgeMantleExitFwdCarry);
        horizVel = exitN * fwdSp + side * kLedgeMantleExitSideCarry;
      } else
        horizVel = glm::vec2(std::cos(yaw), std::sin(yaw)) * kLedgeGrabExitSpeed;
      ledgeClimbExitHoriz = glm::vec2(0.f);
      // Same second-half landing scrub as space / walk-off. Prefer standing jump clip for touchdown so the
      // outro doesn’t read as “sprinting on impact” (run-jump second half looks like a stride).
      if (staffSkinnedActive && !staffRig.clips.empty()) {
        const int landC = (avClipJump >= 0) ? avClipJump : avClipJumpRun;
        if (landC >= 0 && static_cast<size_t>(landC) < staffRig.clips.size()) {
          const double durJ = staff_skin::clipDuration(staffRig, landC);
          const double tailDur = durJ * (1.0 - static_cast<double>(kJumpClipLedgeFirstHalfFrac));
          const float wall = std::min(static_cast<float>(tailDur), kPlayerLandClipMaxWallSec);
          playerJumpAnimRemain = 0.f;
          playerJumpPostLandClipIndex = landC;
          playerJumpPostLandDurationInit = wall;
          playerJumpPostLandRemain = wall;
          playerJumpPostLandSecondHalfScrub = true;
          playerJumpRunTailActive = (landC == avClipJumpRun);
          playerJumpArchActive = false;
        }
      }
    }
    return true;
  }

  bool findLadderClimbTarget(glm::vec3& outEndCam) const {
    outEndCam = {};
    if (avClipLedgeClimb < 0)
      return false;
    const float feet = camPos.y - eyeHeight;
    const glm::vec2 fwd(std::cos(yaw), std::sin(yaw));
    constexpr float kGridRangeM = 22.f;
    constexpr float yBaseDeck = 0.12f;
    const float shelfT = kShelfDeckThickness;
    const float yStep = kShelfGapBetweenLevels + shelfT;
    const int numShelves = kShelfDeckCount;

    int waMin, waMax, wlMin, wlMax;
    shelfGridWindowForRange(camPos.x, camPos.z, kGridRangeM, waMin, waMax, wlMin, wlMax);
    float bestD2 = 1e30f;
    bool have = false;
    glm::vec3 bestEnd(0.f);
    const float r2 = kLadderClimbInteractRadiusXZ * kLadderClimbInteractRadiusXZ;

    for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
      const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
      const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
      const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
      for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
        const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
        for (int side = 0; side < 2; ++side) {
          if (!shelfSlotOccupied(worldAisle, worldAlong, side))
            continue;
          float llx, llz, leanR;
          glm::vec3 ladderScl;
          if (!shelfLadderLocalLayout(worldAisle, worldAlong, side, llx, llz, leanR, ladderScl))
            continue;
          const float cx = side ? cxRight : cxLeft;
          const float yawDeg = side ? -90.0f : 90.0f;
          const glm::mat4 ladderM = shelfLadderWorldModel(cx, cz, yawDeg, llx, llz, leanR, ladderScl);
          const AABB lbox = shelfLadderCollisionAABB(ladderM);
          const float tcx = 0.5f * (lbox.min.x + lbox.max.x);
          const float tcz = 0.5f * (lbox.min.z + lbox.max.z);
          const float qx = glm::clamp(camPos.x, lbox.min.x, lbox.max.x);
          const float qz = glm::clamp(camPos.z, lbox.min.z, lbox.max.z);
          const float dx = qx - camPos.x;
          const float dz = qz - camPos.z;
          const float d2 = dx * dx + dz * dz;
          if (d2 > r2)
            continue;
          glm::vec2 toL(tcx - camPos.x, tcz - camPos.z);
          const float tl = glm::length(toL);
          if (tl > 1e-3f) {
            const glm::vec2 toN = toL * (1.f / tl);
            if (glm::dot(fwd, toN) < kLadderClimbMinLookDot)
              continue;
          }
          const glm::vec3 shelfPos{cx, kGroundY, cz};
          const float shelfYawRad = glm::radians(yawDeg);
          const float hw = kShelfMeshHalfW;
          const float hd = kShelfMeshHalfD;
          constexpr int si = 1;
          if (si >= numShelves)
            continue;
          const float y0 = yBaseDeck + static_cast<float>(si) * yStep;
          const float y1 = y0 + shelfT;
          const AABB deck =
              shelfLocalBoxWorldAABB(shelfPos, shelfYawRad,
                                     {-hw + kShelfDeckInset, y0, -hd + kShelfDeckInset},
                                     {hw - kShelfDeckInset, y1, hd - kShelfDeckInset});
          const float rise = deck.max.y - feet;
          if (rise < 0.9f || rise > 5.5f)
            continue;
          if (deck.max.y + kLadderClimbFeetAboveDeck + eyeHeight + 0.10f >= kCeilingY - 0.05f)
            continue;
          const float mx = kPlayerHalfXZ + 0.1f;
          if (deck.max.x - deck.min.x < mx * 2.f + 0.05f || deck.max.z - deck.min.z < mx * 2.f + 0.05f)
            continue;
          const float dcx = 0.5f * (deck.min.x + deck.max.x);
          const float dcz = 0.5f * (deck.min.z + deck.max.z);
          glm::vec2 pullH(dcx - camPos.x, dcz - camPos.z);
          if (glm::length(pullH) > 1e-4f)
            pullH = glm::normalize(pullH);
          float endX = camPos.x + pullH.x * kLedgeGrabFwdPull;
          float endZ = camPos.z + pullH.y * kLedgeGrabFwdPull;
          endX = std::clamp(endX, deck.min.x + mx, deck.max.x - mx);
          endZ = std::clamp(endZ, deck.min.z + mx, deck.max.z - mx);
          const glm::vec3 cand(endX, deck.max.y + kLadderClimbFeetAboveDeck + eyeHeight, endZ);
          if (!have || d2 < bestD2) {
            bestD2 = d2;
            bestEnd = cand;
            have = true;
          }
        }
      }
    }
    if (have)
      outEndCam = bestEnd;
    return have;
  }

  bool tryStartLadderClimb() {
    glm::vec3 end{};
    if (!findLadderClimbTarget(end))
      return false;
    wallRunActive = false;
    wallRunTimer = 0.f;
    wallClimbActive = false;
    wallClimbTimer = 0.f;
    speedVaultActive = false;
    speedVaultTimer = 0.f;
    wallRunCameraTilt = 0.f;
    slideActive = false;
    slideAnimClip = -1;
    slideAnimElapsed = 0.f;
    slideAnimDurSec = 0.f;
    slideStartSpeed = 0.f;
    slideClearClipNextFrame = false;
    playerJumpAnimRemain = 0.f;
    playerJumpArchActive = false;
    playerJumpAirTimeTargetSec = 0.f;
    playerJumpPostLandRemain = 0.f;
    playerJumpPostLandDurationInit = 0.f;
    playerJumpPostLandSecondHalfScrub = false;
    playerJumpPostLandClipIndex = -1;
    playerJumpRunTailActive = false;
    playerPreFallAnimRemain = 0.f;
    playerPreFallFeetLockY = 0.f;
    playerPreFallUseRunClip = false;
    playerFallAnimTime = 0.f;
    playerJumpAwaitPreLandSecondHalf = false;
    playerJumpLedgeSecondHalfAir = false;
    playerVaultCrateJumpActive = false;
    ladderClimbApproachVel = horizVel;
    velY = 0.f;
    horizVel = glm::vec2(0.f);
    ladderClimbStartCam = camPos;
    ladderClimbEndCam = end;
    const glm::vec2 pullXZ{end.x - camPos.x, end.z - camPos.z};
    const float pullLen = glm::length(pullXZ);
    ladderClimbExitHoriz = pullLen > 1e-4f ? pullXZ * (1.f / pullLen) : glm::vec2(0.f);
    ladderClimbT = 0.f;
    ladderClimbActive = true;
    footstepDistAccum = 0.f;
    playerFallChainMaxFeetY = std::numeric_limits<float>::quiet_NaN();
    return true;
  }

  bool advanceLadderClimb(float dt, bool wantUp, bool wantDown) {
    if (!ladderClimbActive || avClipLedgeClimb < 0)
      return false;
    const double durD =
        staff_skin::clipDuration(staffRig, avClipLedgeClimb) / static_cast<double>(kAvatarAnimPlaybackScale);
    const float durSec = static_cast<float>(glm::max(durD, 0.25));
    float dir = 0.f;
    if (wantUp) dir += 1.f;
    if (wantDown) dir -= 1.f;
    ladderClimbT = std::clamp(ladderClimbT + dir * dt / durSec, 0.f, 1.f);
    const float t = ladderClimbT;
    const float s = t * t * (3.f - 2.f * t);
    camPos = glm::mix(ladderClimbStartCam, ladderClimbEndCam, s);
    velY = 0.f;
    horizVel = glm::vec2(0.f);
    slideActive = false;
    slideAnimClip = -1;
    slideAnimElapsed = 0.f;
    slideAnimDurSec = 0.f;
    slideStartSpeed = 0.f;
    slideClearClipNextFrame = false;
    jumpBuffer = 0.f;
    coyoteTime = 0.f;
    if (t <= 0.f && dir < 0.f) {
      ladderClimbActive = false;
      ladderClimbT = 0.f;
      camPos = ladderClimbStartCam;
      ladderClimbApproachVel = glm::vec2(0.f);
      ladderClimbExitHoriz = glm::vec2(0.f);
    }
    if (t >= 1.f) {
      ladderClimbActive = false;
      ladderClimbT = 0.f;
      camPos = ladderClimbEndCam;
      resolvePillarCollisions();
      const glm::vec2 appr = ladderClimbApproachVel;
      ladderClimbApproachVel = glm::vec2(0.f);
      if (glm::length(ladderClimbExitHoriz) > 1e-4f) {
        const glm::vec2 exitN = glm::normalize(ladderClimbExitHoriz);
        const float fwdIn = glm::dot(appr, exitN);
        const glm::vec2 side = appr - exitN * fwdIn;
        const float fwdSp = glm::max(
            kLedgeGrabExitSpeed * kLedgeMantleExitMinForward,
            kLedgeGrabExitSpeed + std::max(0.f, fwdIn) * kLedgeMantleExitFwdCarry);
        horizVel = exitN * fwdSp + side * kLedgeMantleExitSideCarry;
      } else
        horizVel = glm::vec2(std::cos(yaw), std::sin(yaw)) * kLedgeGrabExitSpeed;
      ladderClimbExitHoriz = glm::vec2(0.f);
      if (staffSkinnedActive && !staffRig.clips.empty()) {
        const int landC = (avClipJump >= 0) ? avClipJump : avClipJumpRun;
        if (landC >= 0 && static_cast<size_t>(landC) < staffRig.clips.size()) {
          const double durJ = staff_skin::clipDuration(staffRig, landC);
          const double tailDur = durJ * (1.0 - static_cast<double>(kJumpClipLedgeFirstHalfFrac));
          const float wall = std::min(static_cast<float>(tailDur), kPlayerLandClipMaxWallSec);
          playerJumpAnimRemain = 0.f;
          playerJumpPostLandClipIndex = landC;
          playerJumpPostLandDurationInit = wall;
          playerJumpPostLandRemain = wall;
          playerJumpPostLandSecondHalfScrub = true;
          playerJumpRunTailActive = (landC == avClipJumpRun);
          playerJumpArchActive = false;
        }
      }
    }
    return true;
  }

  void resolveStaffNpcAgainstWorld(ShelfEmployeeNpc& e) {
    if (!e.inited)
      return;
    const float chaseShelfStepRise =
        (e.nightPhase == 2 && e.meleeState == 0) ? (kShelfGapBetweenLevels + kShelfDeckThickness + 0.28f)
                                                  : -1.f;
    AABB s = staffNpcWorldHitbox(e.posXZ.x, e.posXZ.y, e.yaw, e.feetWorldY, e.bodyScale);
    const glm::vec2 c0(0.5f * (s.min.x + s.max.x), 0.5f * (s.min.z + s.max.z));

    const float bsMax = std::max({e.bodyScale.x, e.bodyScale.y, e.bodyScale.z});
    const float shelfReachSq = kShelfStaffXZReachSq * bsMax * bsMax;

    const int gcx = static_cast<int>(std::floor(e.posXZ.x / kPillarSpacing));
    const int gcz = static_cast<int>(std::floor(e.posXZ.y / kPillarSpacing));
    for (int dx = -kPillarGridRadius; dx <= kPillarGridRadius; ++dx) {
      for (int dz = -kPillarGridRadius; dz <= kPillarGridRadius; ++dz) {
        const float px = static_cast<float>(gcx + dx) * kPillarSpacing;
        const float pz = static_cast<float>(gcz + dz) * kPillarSpacing;
        resolveAABBMinPenetrationXZ(s, pillarCollisionAABB(px, pz));
      }
    }

    // Match resolveShelfRackCollisions player window so staff don’t miss racks the player hits.
    constexpr float kStaffShelfGridRangeM = 22.f;
    int waMin, waMax, wlMin, wlMax;
    shelfGridWindowForRange(e.posXZ.x, e.posXZ.y, kStaffShelfGridRangeM, waMin, waMax, wlMin, wlMax);
    for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
      const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
      const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
      const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
      for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
        const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
        for (int side = 0; side < 2; ++side) {
          if (!shelfSlotOccupied(worldAisle, worldAlong, side))
            continue;
          const float cx = side ? cxRight : cxLeft;
          const float yawDeg = side ? -90.0f : 90.0f;
          const float dxw = e.posXZ.x - cx;
          const float dzw = e.posXZ.y - cz;
          if (dxw * dxw + dzw * dzw > shelfReachSq)
            continue;
          const glm::vec3 shelfPos{cx, kGroundY, cz};
          const float shelfYawRad = glm::radians(yawDeg);
          const float hw = kShelfMeshHalfW;
          const float hd = kShelfMeshHalfD;
          const float ph = kShelfPostGauge;
          const float H = kShelfMeshHeight;
          const float shelfT = kShelfDeckThickness;
          const int numShelves = kShelfDeckCount;
          constexpr float yBase = 0.12f;
          const float yStep = kShelfGapBetweenLevels + shelfT;

          auto collidePost = [&](const glm::vec3& mn, const glm::vec3& mx) {
            const AABB box = shelfLocalBoxWorldAABB(shelfPos, shelfYawRad, mn, mx);
            resolveAABBMinPenetrationXZ(s, box);
          };
          collidePost({-hw, 0.f, -hd}, {-hw + 2.f * ph, H, -hd + 2.f * ph});
          collidePost({hw - 2.f * ph, 0.f, -hd}, {hw, H, -hd + 2.f * ph});
          collidePost({-hw, 0.f, hd - 2.f * ph}, {-hw + 2.f * ph, H, hd});
          collidePost({hw - 2.f * ph, 0.f, hd - 2.f * ph}, {hw, H, hd});
          for (int si = 0; si < numShelves; ++si) {
            const float y0 = yBase + static_cast<float>(si) * yStep;
            const float y1 = y0 + shelfT;
            const AABB deck =
                shelfLocalBoxWorldAABB(shelfPos, shelfYawRad,
                                       {-hw + kShelfDeckInset, y0, -hd + kShelfDeckInset},
                                       {hw - kShelfDeckInset, y1, hd - kShelfDeckInset});
            resolveStaffAgainstShelfWalkableSurface(s, e, deck, chaseShelfStepRise);
          }
          float lx, lz, yDeck, chx, chy, chz;
          const bool hasCrate =
              shelfCrateLocalLayout(worldAisle, worldAlong, side, lx, lz, yDeck, chx, chy, chz);
          if (hasCrate) {
            const AABB crate = shelfLocalBoxWorldAABB(
                shelfPos, shelfYawRad, {lx - chx, yDeck, lz - chz},
                {lx + chx, yDeck + 2.f * chy, lz + chz});
            resolveStaffAgainstShelfWalkableSurface(s, e, crate, chaseShelfStepRise);
            const float topY = yDeck + 2.f * chy;
            const AABB crateTop = shelfLocalBoxWorldAABB(
                shelfPos, shelfYawRad,
                {lx - chx + 0.04f, topY - 0.06f, lz - chz + 0.04f},
                {lx + chx - 0.04f, topY, lz + chz - 0.04f});
            resolveStaffAgainstShelfWalkableSurface(s, e, crateTop, chaseShelfStepRise);
          }
          forEachShelfPalletStackBoxLocal(worldAisle, worldAlong, side, hasCrate,
                                          [&](float mnx, float y0, float mnz, float mxx, float y1, float mxz) {
                                            const AABB pallet = shelfLocalBoxWorldAABB(
                                                shelfPos, shelfYawRad, {mnx, y0, mnz}, {mxx, y1, mxz});
                                            resolveStaffAgainstShelfWalkableSurface(s, e, pallet,
                                                                                    chaseShelfStepRise);
                                            const AABB palletTop = shelfLocalBoxWorldAABB(
                                                shelfPos, shelfYawRad,
                                                {mnx + 0.035f, y1 - 0.05f, mnz + 0.035f},
                                                {mxx - 0.035f, y1, mxz - 0.035f});
                                            resolveStaffAgainstShelfWalkableSurface(s, e, palletTop,
                                                                                    chaseShelfStepRise);
                                          });
          float llx, llz, leanR;
          glm::vec3 ladderScl;
          if (shelfLadderLocalLayout(worldAisle, worldAlong, side, llx, llz, leanR, ladderScl)) {
            const glm::mat4 ladderM = shelfLadderWorldModel(cx, cz, yawDeg, llx, llz, leanR, ladderScl);
            resolveAABBMinPenetrationXZ(s, shelfLadderCollisionAABB(ladderM));
          }
        }
      }
    }

    {
      const float px = (s.min.x + s.max.x) * 0.5f;
      const float pz = (s.min.z + s.max.z) * 0.5f;
      constexpr float kDeliCollScan = 8.f;
      const int daMin = static_cast<int>(std::floor((px - kDeliCollScan) / kShelfAisleModulePitch));
      const int daMax = static_cast<int>(std::ceil((px + kDeliCollScan) / kShelfAisleModulePitch));
      const int dlMin = static_cast<int>(std::floor((pz - kDeliCollScan) / kShelfAlongAislePitch));
      const int dlMax = static_cast<int>(std::ceil((pz + kDeliCollScan) / kShelfAlongAislePitch));
      for (int da = daMin; da <= daMax; ++da)
        for (int dl = dlMin; dl <= dlMax; ++dl) {
          if (!deliBarSlotOccupied(da, dl))
            continue;
          const float cx = (static_cast<float>(da) + 0.5f) * kShelfAisleModulePitch;
          const float cz = (static_cast<float>(dl) + 0.5f) * kShelfAlongAislePitch;
          const float hl = kDeliCounterHalfL + kDeliCounterTopOverhang;
          const float hd = kDeliCounterHalfD + kDeliCounterTopOverhang;
          const float th = kDeliCounterH + kDeliCounterTopH;
          const AABB bar = {{cx - hl, kGroundY, cz - hd}, {cx + hl, kGroundY + th, cz + hd}};
          resolveAABBMinPenetrationXZ(s, bar);

          const float ghl = kDeliCounterHalfL - 0.04f;
          const float gBase = kDeliCounterH + kDeliCounterTopH + 0.32f;
          const float gPeak = kDeliCounterH + kDeliGlassHeight;
          const float outerZ = kDeliCounterHalfD - 0.01f;
          const float innerZ = outerZ - 0.38f;
          const AABB glassBack = {{cx - ghl, kGroundY + gBase, cz - outerZ},
                                  {cx + ghl, kGroundY + gPeak, cz - innerZ}};
          const AABB glassFront = {{cx - ghl, kGroundY + gBase, cz + innerZ},
                                   {cx + ghl, kGroundY + gPeak, cz + outerZ}};
          const AABB roof = {{cx - ghl, kGroundY + gPeak - 0.04f, cz - innerZ},
                             {cx + ghl, kGroundY + gPeak, cz + innerZ}};
          resolveAABBMinPenetrationXZ(s, glassBack);
          resolveAABBMinPenetrationXZ(s, glassFront);
          resolveAABBMinPenetrationXZ(s, roof);
        }
    }

#if defined(VULKAN_GAME_SHREK_EGG_GLB)
    if (shrekEggAssetLoaded && shrekEggActive && shrekEggVertexCount > 0u && staffSkinnedActive) {
      constexpr float kShrekEggTargetHeightM = 1.78f;
      const float shrekScaleY = kShrekEggTargetHeightM / std::max(1e-4f, kEmployeeVisualHeight);
      const glm::vec3 shrekBodyScale(1.f, shrekScaleY, 1.f);
      const float shrekCullR = staffNpcFootprintRadiusXZ(shrekBodyScale) +
                               staffNpcFootprintRadiusXZ(e.bodyScale) + 2.5f;
      const glm::vec2 dSh(e.posXZ.x - shrekEggPos.x, e.posXZ.y - shrekEggPos.z);
      if (glm::dot(dSh, dSh) <= shrekCullR * shrekCullR) {
        const AABB shrekBox = staffNpcWorldHitbox(shrekEggPos.x, shrekEggPos.z, shrekEggYaw, shrekEggPos.y,
                                                  shrekBodyScale);
        resolveAABBMinPenetrationXZ(s, shrekBox);
      }
    }
#endif

    const glm::vec2 c1(0.5f * (s.min.x + s.max.x), 0.5f * (s.min.z + s.max.z));
    const glm::vec2 pushOut = c1 - c0;
    e.posXZ += pushOut;
    const float pushSq = glm::dot(pushOut, pushOut);
    if (pushSq > 1e-8f) {
      const glm::vec2 n = pushOut * (1.f / std::sqrt(pushSq));
      const float vn = glm::dot(e.velXZ, n);
      if (vn < 0.f)
        e.velXZ -= n * vn;
      const float vkb = glm::dot(e.staffShoveKnockbackVelXZ, n);
      if (vkb < 0.f)
        e.staffShoveKnockbackVelXZ -= n * vkb;
    }
  }

  void resolveShelfRackCollisions(AABB& player, float velY) {
    constexpr float kCullR2 = 21.f * 21.f;
    constexpr float kGridRangeM = 22.f;
    int waMin, waMax, wlMin, wlMax;
    shelfGridWindowForRange(camPos.x, camPos.z, kGridRangeM, waMin, waMax, wlMin, wlMax);
    for (int worldAisle = waMin; worldAisle <= waMax; ++worldAisle) {
      const float aisleCX = (static_cast<float>(worldAisle) + 0.5f) * kShelfAisleModulePitch;
      const float cxLeft = aisleCX - kStoreAisleWidth * 0.5f - kShelfMeshHalfD;
      const float cxRight = aisleCX + kStoreAisleWidth * 0.5f + kShelfMeshHalfD;
      for (int worldAlong = wlMin; worldAlong <= wlMax; ++worldAlong) {
        const float cz = (static_cast<float>(worldAlong) + 0.5f) * kShelfAlongAislePitch;
        for (int side = 0; side < 2; ++side) {
          if (!shelfSlotOccupied(worldAisle, worldAlong, side))
            continue;
          const float cx = side ? cxRight : cxLeft;
          const float yawDeg = side ? -90.0f : 90.0f;
          const float dx = camPos.x - cx;
          const float dz = camPos.z - cz;
          if (dx * dx + dz * dz > kCullR2)
            continue;
          const glm::vec3 shelfPos{cx, kGroundY, cz};
          const float shelfYawRad = glm::radians(yawDeg);
          const float hw = kShelfMeshHalfW;
          const float hd = kShelfMeshHalfD;
          const float ph = kShelfPostGauge;
          const float H = kShelfMeshHeight;
          const float shelfT = kShelfDeckThickness;
          const int numShelves = kShelfDeckCount;
          constexpr float yBase = 0.12f;
          const float yStep = kShelfGapBetweenLevels + shelfT;

          auto collidePost = [&](const glm::vec3& mn, const glm::vec3& mx) {
            const AABB box = shelfLocalBoxWorldAABB(shelfPos, shelfYawRad, mn, mx);
            if (aabbOverlap(player, box))
              resolveAABBMinPenetration(player, box);
          };
          collidePost({-hw, 0.f, -hd}, {-hw + 2.f * ph, H, -hd + 2.f * ph});
          collidePost({hw - 2.f * ph, 0.f, -hd}, {hw, H, -hd + 2.f * ph});
          collidePost({-hw, 0.f, hd - 2.f * ph}, {-hw + 2.f * ph, H, hd});
          collidePost({hw - 2.f * ph, 0.f, hd - 2.f * ph}, {hw, H, hd});
          for (int si = 0; si < numShelves; ++si) {
            const float y0 = yBase + static_cast<float>(si) * yStep;
            const float y1 = y0 + shelfT;
            const AABB deck =
                shelfLocalBoxWorldAABB(shelfPos, shelfYawRad,
                                       {-hw + kShelfDeckInset, y0, -hd + kShelfDeckInset},
                                       {hw - kShelfDeckInset, y1, hd - kShelfDeckInset});
            resolveShortLedgeStepUp(player, velY, deck);
          }
          float lx, lz, yDeck, chx, chy, chz;
          const bool hasCrate =
              shelfCrateLocalLayout(worldAisle, worldAlong, side, lx, lz, yDeck, chx, chy, chz);
          if (hasCrate) {
            const AABB crate = shelfLocalBoxWorldAABB(
                shelfPos, shelfYawRad, {lx - chx, yDeck, lz - chz},
                {lx + chx, yDeck + 2.f * chy, lz + chz});
            if (aabbOverlap(player, crate))
              resolveAABBMinPenetration(player, crate);
            const float topY = yDeck + 2.f * chy;
            const AABB crateTop = shelfLocalBoxWorldAABB(
                shelfPos, shelfYawRad,
                {lx - chx + 0.04f, topY - 0.06f, lz - chz + 0.04f},
                {lx + chx - 0.04f, topY, lz + chz - 0.04f});
            resolveShortLedgeStepUp(player, velY, crateTop);
          }
          forEachShelfPalletStackBoxLocal(worldAisle, worldAlong, side, hasCrate,
                                          [&](float mnx, float y0, float mnz, float mxx, float y1, float mxz) {
                                            const AABB pallet = shelfLocalBoxWorldAABB(
                                                shelfPos, shelfYawRad, {mnx, y0, mnz}, {mxx, y1, mxz});
                                            resolveShortLedgeStepUp(player, velY, pallet);
                                          });
          float llx, llz, leanR;
          glm::vec3 ladderScl;
          if (shelfLadderLocalLayout(worldAisle, worldAlong, side, llx, llz, leanR, ladderScl)) {
            const glm::mat4 ladderM = shelfLadderWorldModel(cx, cz, yawDeg, llx, llz, leanR, ladderScl);
            const AABB ladderBox = shelfLadderCollisionAABB(ladderM);
            if (aabbOverlap(player, ladderBox))
              resolveAABBMinPenetration(player, ladderBox);
          }
        }
      }
    }
  }

  void applyStaffPlayerHitAggro(ShelfEmployeeNpc& e) {
    const glm::vec2 pXZ(camPos.x, camPos.z);
    if (audioAreStoreFluorescentsOn()) {
      e.staffPushAggro = true;
      e.staffPushAggroCalmRemain = kStaffDayPushAggroCalmSec;
      e.staffNightShoveChase = false;
      e.staffNightShoveRevealRemain = 0.f;
    } else {
      e.staffPushAggro = false;
      e.staffPushAggroCalmRemain = 0.f;
      e.staffNightShoveChase = true;
      e.staffNightShoveRevealRemain = kStaffNightShoveRevealSec;
    }
    e.nightPhase = 2;
    e.nightLastKnownPlayerXZ = pXZ;
    e.nightSpotTimer = 0.f;
    e.nightInvestigateTimer = 0.f;
  }

  bool applyStaffDamageFromPlayerHit(uint64_t key, ShelfEmployeeNpc& e, float damage) {
    if (e.staffHpMax <= 1e-4f) {
      e.staffClassArchetype = staffClassArchetypeFromKey(key);
      e.staffHpMax = staffClassMaxHp(e.staffClassArchetype);
      e.staffHp = e.staffHpMax;
      e.staffDead = false;
      e.deadRagdollPitch = 0.f;
      e.deadRagdollRoll = 0.f;
      e.deadRagdollPitchVel = 0.f;
      e.deadRagdollRollVel = 0.f;
      e.deadRagdollYaw = e.yaw;
      e.deadRagdollYawVel = 0.f;
      shelfEmpResetDeadJointSim(e);
    }
    if (e.staffDead)
      return false;
    if (e.meleeState >= 2) {
      e.staffHp = std::max(0.f, e.staffHp - damage * 0.35f);
      if (e.staffHp <= 1e-4f) {
        e.staffDead = true;
        e.deadRagdollPitchVel += 1.4f;
        e.deadRagdollRollVel += 0.7f;
        e.deadRagdollYaw = e.yaw;
        e.deadRagdollYawVel += ((key & 1ull) != 0ull) ? 2.8f : -2.8f;
      }
      return false;
    }
    applyStaffPlayerHitAggro(e);
    e.staffHp = std::max(0.f, e.staffHp - damage);
    if (e.staffHp <= 1e-4f) {
      e.staffDead = true;
      e.deadRagdollPitchVel += 1.9f;
      e.deadRagdollRollVel += 1.0f;
      e.deadRagdollYaw = e.yaw;
      e.deadRagdollYawVel += ((key & 2ull) != 0ull) ? 3.4f : -3.4f;
      return true;
    }
    return false;
  }

  // toKnockDirXZ: horizontal direction the staff is knocked (away from player), normalized; same as LMB shove.
  void applyStaffShoveKnockdown(uint64_t key, ShelfEmployeeNpc& e, const glm::vec2& toKnockDirXZ) {
    if (!staffSkinnedActive || (staffClipMeleeFall < 0 && staffClipShoveHair < 0))
      return;
    if (e.meleeState == 2 || e.meleeState == 3 || e.meleeState == 4)
      return;
    glm::vec2 toEN = toKnockDirXZ;
    if (glm::dot(toEN, toEN) < 1e-8f)
      toEN = glm::vec2(std::sin(yaw), std::cos(yaw));
    else
      toEN *= 1.f / glm::length(toEN);
    const glm::vec2 pXZ(camPos.x, camPos.z);
    if (audioAreStoreFluorescentsOn()) {
      e.staffPushAggro = true;
      e.staffPushAggroCalmRemain = kStaffDayPushAggroCalmSec;
      e.staffNightShoveChase = false;
    } else {
      e.staffPushAggro = false;
      e.staffPushAggroCalmRemain = 0.f;
      e.staffNightShoveChase = true;
      e.staffNightShoveRevealRemain = kStaffNightShoveRevealSec;
    }
    e.nightPhase = 2;
    e.nightLastKnownPlayerXZ = pXZ;
    e.nightSpotTimer = 0.f;
    e.chaseLedgeClimbRem = -1.f;
    e.chaseLedgeClimbTotalDur = 0.f;
    e.staffMantelAnimPhaseSpanSec = 0.f;
    e.staffMantelRunnerChase = 0;
    e.staffAirLocoRemain = 0.f;
    e.staffAirFallClip = -1;
    e.staffAirLandRemain = 0.f;
    e.staffAirLandClip = -1;
    {
      int fc = 0;
      double fp = 0.0;
      bool fLoop = true;
      staffNpcLocomotionClip(e, key, fc, fp, fLoop);
      e.meleeAnimFromClip = fc;
      e.meleeAnimFromPhase = fp;
      e.meleeAnimFromLoop = fLoop ? 1u : 0u;
      e.meleeAnimBlend = 0.f;
    }
    {
      const glm::vec2 toP = pXZ - e.posXZ;
      if (glm::dot(toP, toP) > 1e-8f)
        e.yaw = std::atan2(toP.x, toP.y);
    }
    if (staffClipShoveHair >= 0 && !e.staffDead) {
      e.meleeKnockdownFeetAnchorY = e.feetWorldY;
      e.meleeState = 4;
      e.meleePhaseSec = 0.0;
    } else {
      e.meleeKnockdownFeetAnchorY = e.feetWorldY;
      e.meleeState = 2;
      e.meleePhaseSec = 0.0;
    }
    e.staffShoveKnockbackVelXZ = toEN * kStaffShoveKnockbackSpeed;
    e.velXZ = glm::vec2(0.f);
    horizVel -= toEN * kStaffShovePlayerRecoil;
    crosshairShoveAnimRemain = kCrosshairShoveAnimDur;
    if (avClipStepPush >= 0) {
      const double dPush = staff_skin::clipDuration(staffRig, avClipStepPush);
      e.shovePlayerPushDurSec =
          static_cast<float>(dPush / static_cast<double>(kPushAnimPlaybackScale));
      playerPushAnimRemain = static_cast<float>(dPush);
    } else {
      e.shovePlayerPushDurSec = 0.22f;
      playerPushAnimRemain = e.shovePlayerPushDurSec * kPushAnimPlaybackScale;
    }
    audioPlayStaffMeleeImpact();
  }

  // Environmental tall fall: melee fall clip only (no shove aggro, player recoil, or hair wind-up).
  void applyStaffTallFallRagdoll(uint64_t key, ShelfEmployeeNpc& e) {
    if (!staffSkinnedActive || staffClipMeleeFall < 0)
      return;
    if (e.meleeState >= 2)
      return;
    e.chaseLedgeClimbRem = -1.f;
    e.chaseLedgeClimbTotalDur = 0.f;
    e.staffMantelAnimPhaseSpanSec = 0.f;
    e.staffMantelRunnerChase = 0;
    e.staffChaseMantelCooldownRem = 0.f;
    e.staffVelY = 0.f;
    e.velXZ = glm::vec2(0.f);
    e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
    e.meleeKnockdownFeetAnchorY = e.feetWorldY;
    e.staffAirLocoRemain = 0.f;
    e.staffAirFallClip = -1;
    e.staffAirLandRemain = 0.f;
    e.staffAirLandClip = -1;
    {
      int fc = 0;
      double fp = 0.0;
      bool fLoop = true;
      staffNpcLocomotionClip(e, key, fc, fp, fLoop);
      e.meleeAnimFromClip = fc;
      e.meleeAnimFromPhase = fp;
      e.meleeAnimFromLoop = fLoop ? 1u : 0u;
      e.meleeAnimBlend = 0.f;
    }
    const glm::vec2 pXZ(camPos.x, camPos.z);
    const glm::vec2 toP = pXZ - e.posXZ;
    if (glm::dot(toP, toP) > 1e-8f)
      e.yaw = std::atan2(toP.x, toP.y);
    e.meleeState = 2;
    e.meleePhaseSec = 0.0;
    e.shovePlayerPushDurSec = 0.f;
  }

  void resolveStaffNpcCollisions(AABB& player, float playerVelY, bool allowBodySlamFromAir) {
    const glm::vec2 pc(camPos.x, camPos.z);
    for (uint32_t si : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& npc = shelfEmpPool[si];
      if (!npc.inited)
        continue;
      if (npc.meleeState >= 2)
        continue;
      const glm::vec2 d = npc.posXZ - pc;
      const float cullR =
          kPlayerHalfXZ + staffNpcFootprintRadiusXZ(npc.bodyScale) + kStaffPlayerCollisionPadM;
      if (glm::dot(d, d) > cullR * cullR)
        continue;
      const AABB box =
          staffNpcWorldHitbox(npc.posXZ.x, npc.posXZ.y, npc.yaw, npc.feetWorldY, npc.bodyScale);
      if (!aabbOverlap(player, box))
        continue;

      const float feet = player.min.y;
      const float top = box.max.y;
      const bool feetOnCapBand = feet >= top - kPlayerStaffBodySlamFeetBelowTopM &&
                                 feet <= top + kPlayerStaffBodySlamFeetAboveTopM;
      const bool slamVelOk = playerVelY <= kPlayerStaffBodySlamMaxVelY;
      if (allowBodySlamFromAir && feetOnCapBand && slamVelOk && playerStaffBodySlamCooldownRem <= 0.f &&
          staffSkinnedActive && (staffClipMeleeFall >= 0 || staffClipShoveHair >= 0)) {
        glm::vec2 toEN = npc.posXZ - pc;
        if (glm::dot(toEN, toEN) > 1e-8f)
          toEN *= 1.f / glm::length(toEN);
        else
          toEN = glm::vec2(std::sin(yaw), std::cos(yaw));
        const float rise = top - feet;
        if (rise > -0.18f && rise < 0.9f) {
          player.min.y = top;
          player.max.y += rise;
        }
        if (applyStaffDamageFromPlayerHit(npc.residentKey, npc, kStaffHitDamageBodySlam)) {
          applyStaffShoveKnockdown(npc.residentKey, npc, toEN);
        } else {
          npc.staffShoveKnockbackVelXZ = toEN * (kStaffShoveKnockbackSpeed * 0.5f);
          audioPlayStaffMeleeImpact();
        }
        playerStaffBodySlamCooldownRem = kPlayerStaffBodySlamCooldownSec;
        continue;
      }
      resolveAABBMinPenetration(player, box);
    }
  }

#if defined(VULKAN_GAME_SHREK_EGG_GLB)
  // Same footprint as staff NPCs; height matches loadSkinnedIdleGlb(..., 1.78f, ...) for the egg mesh.
  void resolveShrekEggPlayerCollision(AABB& player) {
    if (!shrekEggAssetLoaded || !shrekEggActive || shrekEggVertexCount == 0u || !staffSkinnedActive)
      return;
    constexpr float kShrekEggTargetHeightM = 1.78f;
    const float shrekScaleY = kShrekEggTargetHeightM / std::max(1e-4f, kEmployeeVisualHeight);
    const glm::vec3 shrekBodyScale(1.f, shrekScaleY, 1.f);
    const glm::vec2 pc(camPos.x, camPos.z);
    const glm::vec2 d(shrekEggPos.x - pc.x, shrekEggPos.z - pc.y);
    const float cullR =
        kPlayerHalfXZ + staffNpcFootprintRadiusXZ(shrekBodyScale) + kStaffPlayerCollisionPadM;
    if (glm::dot(d, d) > cullR * cullR)
      return;
    const AABB box = staffNpcWorldHitbox(shrekEggPos.x, shrekEggPos.z, shrekEggYaw, shrekEggPos.y,
                                         shrekBodyScale);
    if (aabbOverlap(player, box))
      resolveAABBMinPenetration(player, box);
  }
#endif

  // Ray vs all staff hitboxes; returns true if something was hit before maxT (tHit = closest along rd).
  bool rayStaffNpcFirstHit(const glm::vec3& ro, const glm::vec3& rd, float maxT, float& tHit) const {
    tHit = maxT;
    bool any = false;
    for (uint32_t si : shelfEmpActiveSlots) {
      const ShelfEmployeeNpc& npc = shelfEmpPool[si];
      if (!npc.inited)
        continue;
      const AABB box =
          staffNpcWorldHitbox(npc.posXZ.x, npc.posXZ.y, npc.yaw, npc.feetWorldY, npc.bodyScale);
      float t = 0.f;
      if (rayAABBFirstHit(ro, rd, box, t) && t >= 0.f && t <= maxT && t < tHit) {
        tHit = t;
        any = true;
      }
    }
    return any;
  }

  bool findStaffNpcCrosshairHit(uint64_t& outKey, float& outT, float maxDist) const {
    outKey = 0;
    outT = maxDist;
    glm::vec3 ro, fwd, right, up;
    getFirstPersonViewBasis(ro, fwd, right, up);
    if (glm::length(fwd) < 1e-5f)
      return false;
    const glm::vec3 rd = glm::normalize(fwd);
    bool any = false;
    for (uint32_t si : shelfEmpActiveSlots) {
      const ShelfEmployeeNpc& npc = shelfEmpPool[si];
      if (!npc.inited)
        continue;
      const AABB box =
          staffNpcWorldHitbox(npc.posXZ.x, npc.posXZ.y, npc.yaw, npc.feetWorldY, npc.bodyScale);
      float t = 0.f;
      if (!rayAABBFirstHit(ro, rd, box, t) || t < 0.f || t > maxDist || t > outT)
        continue;
      outT = t;
      outKey = npc.residentKey;
      any = true;
    }
    return any;
  }

  void processPendingStaffShove() {
    if (!pendingStaffShoveLmb)
      return;
    pendingStaffShoveLmb = false;
    if (!mouseGrab || showControlsOverlay || inTitleMenu || showPauseMenu)
      return;
    if (!staffSkinnedActive || (staffClipMeleeFall < 0 && staffClipShoveHair < 0))
      return;
    uint64_t key = 0;
    float tHit = kStaffShoveMaxDist;
    if (!findStaffNpcCrosshairHit(key, tHit, kStaffShoveMaxDist))
      return;
    ShelfEmployeeNpc* ep = shelfEmpFind(key);
    if (ep == nullptr || !ep->inited)
      return;
    ShelfEmployeeNpc& e = *ep;
    const glm::vec2 pXZ(camPos.x, camPos.z);
    const glm::vec2 toStaff = e.posXZ - pXZ;
    const float distP = glm::length(toStaff);
    if (distP > kStaffShoveMaxDist || distP < 1e-4f)
      return;
    glm::vec3 ro, fwd, right, up;
    getFirstPersonViewBasis(ro, fwd, right, up);
    glm::vec2 f(fwd.x, fwd.z);
    const float fl = glm::length(f);
    if (fl < 1e-4f)
      return;
    f *= 1.f / fl;
    const glm::vec2 toEN = toStaff * (1.f / distP);
    if (glm::dot(f, toEN) < kStaffShoveCosCone)
      return;
    if (e.meleeState == 2 || e.meleeState == 3 || e.meleeState == 4)
      return;
    // Shove should knock staff down, but HP loss stays at zero.
    applyStaffShoveKnockdown(key, e, toEN);
  }

  void processPendingPlayerKick() {
    if (!pendingPlayerKick)
      return;
    pendingPlayerKick = false;
    if (!mouseGrab || showControlsOverlay || inTitleMenu || showPauseMenu)
      return;
    if (playerKickAnimRemain > 0.f || playerPushAnimRemain > 0.f || dropKickActive)
      return;

    const bool airborne = !isGrounded();
    const float runSpd = glm::length(horizVel);
    const bool canDropKick = airborne && runSpd >= kDropKickMinRunSpeed &&
                             (avClipSlideRight >= 0 || avClipSlideLight >= 0);
    if (canDropKick) {
      dropKickActive = true;
      dropKickTimer = kDropKickMaxAirSec;
      dropKickHitApplied = false;
      glm::vec3 ro, fwd, right, up;
      getFirstPersonViewBasis(ro, fwd, right, up);
      glm::vec2 f(fwd.x, fwd.z);
      const float fl = glm::length(f);
      dropKickDir = fl > 1e-4f ? f * (1.f / fl) : glm::normalize(horizVel);
      horizVel = dropKickDir * kDropKickLungeSpeed;
      velY = std::max(velY, 0.3f);
      return;
    }

    if (staffClipMeleeKick < 0)
      return;
    const double dKick = staff_skin::clipDuration(staffRig, staffClipMeleeKick);
    playerKickAnimRemain = static_cast<float>(dKick);
    if (!staffSkinnedActive || (staffClipMeleeFall < 0 && staffClipShoveHair < 0))
      return;
    uint64_t key = 0;
    float tHit = kStaffShoveMaxDist;
    if (!findStaffNpcCrosshairHit(key, tHit, kStaffShoveMaxDist))
      return;
    ShelfEmployeeNpc* ep = shelfEmpFind(key);
    if (ep == nullptr || !ep->inited)
      return;
    ShelfEmployeeNpc& e = *ep;
    const glm::vec2 pXZ(camPos.x, camPos.z);
    const glm::vec2 toStaff = e.posXZ - pXZ;
    const float distP = glm::length(toStaff);
    if (distP > kStaffShoveMaxDist || distP < 1e-4f)
      return;
    glm::vec3 ro, fwd, right, up;
    getFirstPersonViewBasis(ro, fwd, right, up);
    glm::vec2 f(fwd.x, fwd.z);
    const float fl = glm::length(f);
    if (fl < 1e-4f)
      return;
    f *= 1.f / fl;
    const glm::vec2 toEN = toStaff * (1.f / distP);
    if (glm::dot(f, toEN) < kStaffShoveCosCone)
      return;
    if (e.meleeState == 2 || e.meleeState == 3 || e.meleeState == 4) {
      const bool killedOnGround = applyStaffDamageFromPlayerHit(key, e, kStaffHitDamageKick);
      // Dead staff stay ragdolled on the floor; downed alive staff can be re-staggered.
      e.meleeState = 2;
      if (killedOnGround || e.staffDead) {
        const double dFallDead = staffClipMeleeFall >= 0
                                     ? staff_skin::clipDuration(staffRig, staffClipMeleeFall)
                                     : 0.0;
        e.meleePhaseSec = std::max(0.0, dFallDead);
      } else {
        e.meleePhaseSec = 0.0;
      }
      e.meleeKnockdownFeetAnchorY = e.feetWorldY;
      e.staffVelY = 0.f;
      e.velXZ = glm::vec2(0.f);
      e.staffShoveKnockbackVelXZ = toEN * (kStaffShoveKnockbackSpeed * 0.55f);
      audioPlayStaffMeleeImpact();
      return;
    }
    // Normal kick should not drop standing staff unless this hit is lethal.
    if (applyStaffDamageFromPlayerHit(key, e, kStaffHitDamageKick)) {
      applyStaffShoveKnockdown(key, e, toEN);
    } else {
      e.staffShoveKnockbackVelXZ = toEN * (kStaffShoveKnockbackSpeed * 0.48f);
      audioPlayStaffMeleeImpact();
    }
  }

  // fullWorld: shelf + staff + many pillar passes (expensive). false = pillar-only sweeps for
  // horizontal micro-steps — avoids scanning every nearby rack up to moveSteps times per tick.
  void resolvePillarCollisions(bool fullWorld = true, bool allowStaffBodySlamFromAir = false) {
    AABB player = playerCollisionBox();
    const int gcx = static_cast<int>(std::floor(camPos.x / kPillarSpacing));
    const int gcz = static_cast<int>(std::floor(camPos.z / kPillarSpacing));
    auto resolvePillarsOnly = [&]() {
      for (int dx = -kPillarGridRadius; dx <= kPillarGridRadius; ++dx) {
        for (int dz = -kPillarGridRadius; dz <= kPillarGridRadius; ++dz) {
          const float px = static_cast<float>(gcx + dx) * kPillarSpacing;
          const float pz = static_cast<float>(gcz + dz) * kPillarSpacing;
          const AABB pillar = pillarCollisionAABB(px, pz);
          if (aabbOverlap(player, pillar))
            resolveAABBMinPenetration(player, pillar);
        }
      }
    };
    if (fullWorld) {
      for (int pass = 0; pass < 3; ++pass)
        resolvePillarsOnly();
      for (int sp = 0; sp < 1; ++sp) {
        resolveShelfRackCollisions(player, velY);
        resolveStaffNpcCollisions(player, velY, allowStaffBodySlamFromAir);
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
        resolveShrekEggPlayerCollision(player);
#endif
        {
          const float px = (player.min.x + player.max.x) * 0.5f;
          const float pz = (player.min.z + player.max.z) * 0.5f;
          constexpr float kDeliCollScan = 8.f;
          const int daMin = static_cast<int>(std::floor((px - kDeliCollScan) / kShelfAisleModulePitch));
          const int daMax = static_cast<int>(std::ceil((px + kDeliCollScan) / kShelfAisleModulePitch));
          const int dlMin = static_cast<int>(std::floor((pz - kDeliCollScan) / kShelfAlongAislePitch));
          const int dlMax = static_cast<int>(std::ceil((pz + kDeliCollScan) / kShelfAlongAislePitch));
          for (int da = daMin; da <= daMax; ++da)
            for (int dl = dlMin; dl <= dlMax; ++dl) {
              if (!deliBarSlotOccupied(da, dl)) continue;
              const float cx = (static_cast<float>(da) + 0.5f) * kShelfAisleModulePitch;
              const float cz = (static_cast<float>(dl) + 0.5f) * kShelfAlongAislePitch;
              const float hl = kDeliCounterHalfL + kDeliCounterTopOverhang;
              const float hd = kDeliCounterHalfD + kDeliCounterTopOverhang;
              const float th = kDeliCounterH + kDeliCounterTopH;
              const AABB bar = {{cx - hl, kGroundY, cz - hd},
                                {cx + hl, kGroundY + th, cz + hd}};
              if (aabbOverlap(player, bar))
                resolveAABBMinPenetration(player, bar);
              const float ghl   = kDeliCounterHalfL - 0.04f;
              const float gBase = kDeliCounterH + kDeliCounterTopH + 0.32f;
              const float gPeak = kDeliCounterH + kDeliGlassHeight;
              const float outerZ = kDeliCounterHalfD - 0.01f;
              const float innerZ = outerZ - 0.38f;
              const AABB glassBack = {{cx - ghl, kGroundY + gBase, cz - outerZ},
                                      {cx + ghl, kGroundY + gPeak, cz - innerZ}};
              const AABB glassFront = {{cx - ghl, kGroundY + gBase, cz + innerZ},
                                       {cx + ghl, kGroundY + gPeak, cz + outerZ}};
              const AABB roof = {{cx - ghl, kGroundY + gPeak - 0.04f, cz - innerZ},
                                 {cx + ghl, kGroundY + gPeak, cz + innerZ}};
              if (aabbOverlap(player, glassBack))
                resolveAABBMinPenetration(player, glassBack);
              if (aabbOverlap(player, glassFront))
                resolveAABBMinPenetration(player, glassFront);
              if (aabbOverlap(player, roof))
                resolveAABBMinPenetration(player, roof);
            }
        }
        resolvePillarsOnly();
      }
    } else {
      for (int pass = 0; pass < 2; ++pass)
        resolvePillarsOnly();
    }
    syncCamFromPlayerAABB(player, camPos, eyeHeight);
  }

  // Same as isGrounded() but reuses a terrainSupportY already computed for this (x,z,feet) sample.
  bool isGroundedUsingSupport(float terrainSupportHeight) const {
    const float feet = camPos.y - eyeHeight;
    if (feet <= terrainSupportHeight + kGroundedFeetAboveSupport && velY <= 0.01f)
      return true;
    const int gcx = static_cast<int>(std::floor(camPos.x / kPillarSpacing));
    const int gcz = static_cast<int>(std::floor(camPos.z / kPillarSpacing));
    const float top = kGroundY + kPillarHeight;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dz = -1; dz <= 1; ++dz) {
        const float px = static_cast<float>(gcx + dx) * kPillarSpacing;
        const float pz = static_cast<float>(gcz + dz) * kPillarSpacing;
        if (std::abs(feet - top) > 0.12f || velY > 0.1f)
          continue;
        if (std::abs(camPos.x - px) < kPillarHalfW + kPlayerHalfXZ - 0.05f &&
            std::abs(camPos.z - pz) < kPillarHalfD + kPlayerHalfXZ - 0.05f)
          return true;
      }
    }
    return false;
  }

  bool isGrounded() const {
    const float feet = camPos.y - eyeHeight;
    return isGroundedUsingSupport(playerTerrainSupportY(camPos.x, camPos.z, feet));
  }

  // True while descending in air (ledge / jump / fall): drive jump-clip phase to mid-frame, not walk-in-air.
  bool playerAvatarJumpFallMidPose() const {
    if (ledgeClimbT >= 0.f || ladderClimbActive || slideActive || wallRunActive || wallClimbActive ||
        speedVaultActive || dropKickActive)
      return false;
    if (playerAirWalkSmallGap)
      return false;
    if (isGrounded())
      return false;
    // Narrow hop + horizontal motion: prefer small-gap walk/run once probe says short drop (avoids jump “hang”
    // one frame before playerAirWalkSmallGap is set, e.g. after a space jump then striding over a shelf gap).
    if (playerWalkOffWalkableGapDropCached <= kPlayerWalkOffSmallGapMaxDropM && playerPreFallAnimRemain <= 1e-4f) {
      const float sp = glm::length(horizVel);
      if (sp > kPlayerWalkOffLedgeAnimMinHorizSp * 0.4f)
        return false;
    }
    // Big walk-off: after lip teeter, hold mid-jump until pre-land starts the second half (don’t require velY yet).
    if (playerJumpAwaitPreLandSecondHalf && playerJumpAnimRemain <= 1e-4f && playerPreFallAnimRemain <= 1e-4f &&
        playerWalkOffWalkableGapDropCached > kPlayerWalkOffSmallGapMaxDropM)
      return true;
    // Space / vault jump: clip was timed for base kJumpVel; charged jumps stay up longer — if the clip ends
    // while vy is still ≥ lip threshold, we must keep jump/air pose (not walk-in-air) through apex.
    if (playerJumpArchActive && velY >= kAvatarJumpFallVelYThr && ledgeClimbT < 0.f && !slideActive)
      return true;
    return velY < kAvatarJumpFallVelYThr;
  }

  void rebuildTerrainIfNeeded() {
    const float cw = static_cast<float>(kChunkCellCount) * kCellSize;
    const float tx = (inTitleMenu || inIntroSplash || inLoadingScreen) ? titleMenuSceneAnchor.x : camPos.x;
    const float tz = (inTitleMenu || inIntroSplash || inLoadingScreen) ? titleMenuSceneAnchor.z : camPos.z;
    const int pcx = static_cast<int>(std::floor(tx / cw));
    const int pcz = static_cast<int>(std::floor(tz / cw));
    const int m = kTerrainStreamMarginChunks;
    const int loX = lastTerrainChunkX - kChunkRadius + m;
    const int hiX = lastTerrainChunkX + kChunkRadius - m;
    const int loZ = lastTerrainChunkZ - kChunkRadius + m;
    const int hiZ = lastTerrainChunkZ + kChunkRadius - m;
    if (pcx >= loX && pcx <= hiX && pcz >= loZ && pcz <= hiZ)
      return;
    const int oldCx = lastTerrainChunkX;
    const int oldCz = lastTerrainChunkZ;
    const int dx = pcx - oldCx;
    const int dz = pcz - oldCz;
    lastTerrainChunkX = pcx;
    lastTerrainChunkZ = pcz;

    constexpr int kSide = kChunkRadius * 2 + 1;
    constexpr int kCv = 6;
    constexpr int kRowStride = kSide * kCv;
    const bool singleAxisStep = (std::abs(dx) + std::abs(dz) == 1);

    if (!singleAxisStep) {
      static std::vector<Vertex> sTerrainRebuildVerts;
      static std::vector<Vertex> sCeilingRebuildVerts;
      buildTerrainMesh(pcx, pcz, sTerrainRebuildVerts);
      groundVertexCount = static_cast<uint32_t>(sTerrainRebuildVerts.size());
      std::memcpy(groundMapped, sTerrainRebuildVerts.data(), sTerrainRebuildVerts.size() * sizeof(Vertex));
      buildCeilingMesh(pcx, pcz, sCeilingRebuildVerts);
      ceilingVertexCount = static_cast<uint32_t>(sCeilingRebuildVerts.size());
      std::memcpy(ceilingMapped, sCeilingRebuildVerts.data(), sCeilingRebuildVerts.size() * sizeof(Vertex));
      return;
    }

    auto* g = static_cast<Vertex*>(groundMapped);
    auto* c = static_cast<Vertex*>(ceilingMapped);
    const size_t rowBytes = sizeof(Vertex) * static_cast<size_t>(kRowStride);
    const size_t stripBytes = rowBytes - sizeof(Vertex) * static_cast<size_t>(kCv);

    if (dx == 1) {
      for (int cz = -kChunkRadius; cz <= kChunkRadius; ++cz) {
        Vertex* rowG = g + (cz + kChunkRadius) * kRowStride;
        Vertex* rowC = c + (cz + kChunkRadius) * kRowStride;
        std::memmove(rowG, rowG + kCv, stripBytes);
        std::memmove(rowC, rowC + kCv, stripBytes);
        const int gcx = pcx + kChunkRadius;
        const int gcz = pcz + cz;
        writeTerrainChunkVerts(gcx, gcz, rowG + kRowStride - kCv);
        writeCeilingChunkVerts(gcx, gcz, rowC + kRowStride - kCv);
      }
    } else if (dx == -1) {
      for (int cz = -kChunkRadius; cz <= kChunkRadius; ++cz) {
        Vertex* rowG = g + (cz + kChunkRadius) * kRowStride;
        Vertex* rowC = c + (cz + kChunkRadius) * kRowStride;
        std::memmove(rowG + kCv, rowG, stripBytes);
        std::memmove(rowC + kCv, rowC, stripBytes);
        const int gcx = pcx - kChunkRadius;
        const int gcz = pcz + cz;
        writeTerrainChunkVerts(gcx, gcz, rowG);
        writeCeilingChunkVerts(gcx, gcz, rowC);
      }
    } else if (dz == 1) {
      std::memmove(g, g + kRowStride, rowBytes * static_cast<size_t>(kSide - 1));
      std::memmove(c, c + kRowStride, rowBytes * static_cast<size_t>(kSide - 1));
      Vertex* lastG = g + kRowStride * (kSide - 1);
      Vertex* lastC = c + kRowStride * (kSide - 1);
      for (int cx = -kChunkRadius; cx <= kChunkRadius; ++cx) {
        const int gcx = pcx + cx;
        const int gcz = pcz + kChunkRadius;
        writeTerrainChunkVerts(gcx, gcz, lastG + (cx + kChunkRadius) * kCv);
        writeCeilingChunkVerts(gcx, gcz, lastC + (cx + kChunkRadius) * kCv);
      }
    } else if (dz == -1) {
      std::memmove(g + kRowStride, g, rowBytes * static_cast<size_t>(kSide - 1));
      std::memmove(c + kRowStride, c, rowBytes * static_cast<size_t>(kSide - 1));
      for (int cx = -kChunkRadius; cx <= kChunkRadius; ++cx) {
        const int gcx = pcx + cx;
        const int gcz = pcz - kChunkRadius;
        writeTerrainChunkVerts(gcx, gcz, g + (cx + kChunkRadius) * kCv);
        writeCeilingChunkVerts(gcx, gcz, c + (cx + kChunkRadius) * kCv);
      }
    }
  }

  bool avatarClipsAllowCrossfade(int fromC, int toC) const {
    if (fromC == toC)
      return false;
    auto isOneShot = [&](int c) {
      if (staffClipMeleeFall >= 0 && c == staffClipMeleeFall)
        return true;
      if (avClipStepPush >= 0 && c == avClipStepPush)
        return true;
      if (staffClipMeleeKick >= 0 && c == staffClipMeleeKick)
        return true;
      if (avClipJump >= 0 && c == avClipJump)
        return true;
      if (avClipJumpRun >= 0 && c == avClipJumpRun)
        return true;
      if (avClipLedgeClimb >= 0 && c == avClipLedgeClimb)
        return true;
      if (avClipSlideRight >= 0 && c == avClipSlideRight)
        return true;
      if (avClipSlideLight >= 0 && c == avClipSlideLight)
        return true;
      if (avClipLand >= 0 && c == avClipLand)
        return true;
      return false;
    };
    const bool fromJumpOrLand = (avClipJump >= 0 && fromC == avClipJump) ||
                               (avClipJumpRun >= 0 && fromC == avClipJumpRun) ||
                               (avClipLand >= 0 && fromC == avClipLand);
    const bool toLoco = (avClipWalk >= 0 && toC == avClipWalk) || (avClipSprint >= 0 && toC == avClipSprint) ||
                        (avClipIdle >= 0 && toC == avClipIdle);
    if (fromJumpOrLand && toLoco)
      return true;
    return !isOneShot(fromC) && !isOneShot(toC);
  }

  void resolvePlayerAvatarPhase(int clipIn, double& outPhase, bool& outLoop) const {
    int clip = clipIn;
    if (clip < 0 || static_cast<size_t>(clip) >= staffRig.clips.size())
      clip = 0;
    const double dur = staff_skin::clipDuration(staffRig, clip);

    if (playerDeathActive && playerDeathClipIndex >= 0 && clip == playerDeathClipIndex) {
      outLoop = false;
      const double endT = dur * static_cast<double>(playerDeathClipFracEnd);
      const double capEnd = std::max(1e-6, std::min(endT, dur - 1e-6));
      const double ph = playerDeathPlayingFallClip ? playerDeathAnimTime : endT;
      outPhase = std::clamp(ph, 0.0, capEnd);
      return;
    }
    if (playerDanceEmoteActive && staffClipShrekProximityDance >= 0 && clip == staffClipShrekProximityDance) {
      outLoop = true;
      double t = static_cast<double>(staffSimTime);
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
      if (shrekEggAssetLoaded && shrekEggActive)
        t = static_cast<double>(shrekEggAnimPhase);
#endif
      double ph = std::fmod(t, std::max(dur, 1e-6));
      if (ph < 0.0)
        ph += dur;
      outPhase = ph;
      return;
    }

    // Drop kick: slide clip — hold at legs-extended in air, play out during ground slide.
    if (dropKickActive && (clip == avClipSlideLight || clip == avClipSlideRight)) {
      outLoop = false;
      double u;
      if (dropKickTimer > kDropKickGroundSlideSec) {
        const float airElapsed = kDropKickMaxAirSec - dropKickTimer;
        u = std::min(0.45, static_cast<double>(airElapsed) / 0.25);
      } else {
        const float gProg = 1.f - dropKickTimer / kDropKickGroundSlideSec;
        u = 0.45 + 0.55 * static_cast<double>(gProg);
      }
      outPhase = std::clamp(u * dur, 0.0, std::max(1e-6, dur - 1e-6));
      return;
    }
    // One-shots: fixed timeline, no wrap (computePalette clamps when loopPhase is false).
    if (staffClipMeleeKick >= 0 && clip == staffClipMeleeKick && playerKickAnimRemain > 0.f) {
      outLoop = false;
      outPhase = std::clamp(static_cast<double>(dur) - static_cast<double>(playerKickAnimRemain), 0.0,
                            std::max(1e-6, dur - 1e-6));
      return;
    }
    if (avClipStepPush >= 0 && clip == avClipStepPush && playerPushAnimRemain > 0.f) {
      outLoop = false;
      outPhase = std::clamp(static_cast<double>(dur) - static_cast<double>(playerPushAnimRemain), 0.0,
                            std::max(1e-6, dur - 1e-6));
      return;
    }
    if (avClipLand >= 0 && clip == avClipLand && playerJumpPostLandClipIndex == avClipLand &&
        playerJumpPostLandRemain > 0.f && playerJumpPostLandDurationInit > 1e-6f) {
      outLoop = false;
      const double u = static_cast<double>(playerJumpPostLandDurationInit - playerJumpPostLandRemain) /
                       static_cast<double>(playerJumpPostLandDurationInit);
      outPhase = std::clamp(u * dur, 0.0, std::max(1e-6, dur - 1e-6));
      return;
    }
    const bool jumpClipPhase =
        (avClipJump >= 0 && clip == avClipJump) || (avClipJumpRun >= 0 && clip == avClipJumpRun);
    if (jumpClipPhase && speedVaultActive) {
      outLoop = false;
      const double u =
          std::clamp(1.0 - static_cast<double>(speedVaultTimer) / static_cast<double>(kSpeedVaultDuration), 0.0,
                     1.0);
      outPhase = std::clamp(u * dur, 0.0, std::max(1e-6, dur - 1e-6));
      return;
    }
    const bool useJumpFallMidPose =
        jumpClipPhase && playerJumpPostLandRemain <= 0.f && playerAvatarJumpFallMidPose();
    if (jumpClipPhase &&
        (playerJumpAnimRemain > 0.f || playerJumpPostLandRemain > 0.f || playerPreFallAnimRemain > 0.f ||
         useJumpFallMidPose)) {
      outLoop = false;
      if (playerJumpPostLandRemain > 0.f) {
        if (playerJumpPostLandSecondHalfScrub && playerJumpPostLandDurationInit > 1e-6f) {
          const double halfDur = dur * static_cast<double>(kJumpClipLedgeFirstHalfFrac);
          const double tail = std::max(dur - halfDur, 1e-6);
          const double u = static_cast<double>(playerJumpPostLandDurationInit - playerJumpPostLandRemain) /
                           static_cast<double>(playerJumpPostLandDurationInit);
          outPhase = std::clamp(halfDur + u * tail, halfDur, std::max(halfDur + 1e-6, dur - 1e-6));
        } else
          outPhase = std::clamp(dur - 1e-4, 0.0, std::max(1e-6, dur - 1e-6));
      } else if (playerPreFallAnimRemain > 0.f) {
        // First half of jump clip on the ledge; second half plays just before touchdown (see playerJumpAwaitPreLandSecondHalf).
        const float u =
            glm::clamp(1.f - playerPreFallAnimRemain / kPlayerPreFallBeforeFallSec, 0.f, 1.f);
        const double halfDur = dur * static_cast<double>(kJumpClipLedgeFirstHalfFrac);
        outPhase = std::clamp(static_cast<double>(u) * halfDur, 0.0, std::max(1e-6, halfDur - 1e-6));
      } else if (useJumpFallMidPose && velY < kAvatarJumpFallVelYThr) {
        // Once descending, force the jump clip into the falling section even if jumpAnimRemain is still active.
        const bool isRunClip = avClipJumpRun >= 0 && clip == avClipJumpRun;
        const double uMid = isRunClip ? static_cast<double>(kJumpClipRunMidPhaseFrac)
                                      : static_cast<double>(kJumpClipMidPhaseFrac);
        const double uMax = isRunClip ? static_cast<double>(kFallAnimMaxFracRun)
                                      : static_cast<double>(kFallAnimMaxFrac);
        const double uFall = std::min(uMid + static_cast<double>(playerFallAnimTime * kFallAnimScrubRate), uMax);
        outPhase = std::clamp(dur * uFall, 0.0, std::max(1e-6, dur - 1e-6));
      } else if (playerJumpAnimRemain > 0.f) {
        // Second half of clip (ledge pre-land) or full clip (ground jump): phase = dur − remain.
        outPhase = std::clamp(static_cast<double>(dur) - static_cast<double>(playerJumpAnimRemain), 0.0,
                              std::max(1e-6, dur - 1e-6));
      } else if (useJumpFallMidPose) {
        const bool isRunClip = avClipJumpRun >= 0 && clip == avClipJumpRun;
        const double uMid = isRunClip ? static_cast<double>(kJumpClipRunMidPhaseFrac)
                                      : static_cast<double>(kJumpClipMidPhaseFrac);
        const double uMax = isRunClip ? static_cast<double>(kFallAnimMaxFracRun)
                                      : static_cast<double>(kFallAnimMaxFrac);
        const double uFall = std::min(uMid + static_cast<double>(playerFallAnimTime * kFallAnimScrubRate), uMax);
        outPhase = std::clamp(dur * uFall, 0.0, std::max(1e-6, dur - 1e-6));
      }
      return;
    }
    if (ledgeHangActive && avClipLedgeGrab >= 0 && clip == avClipLedgeGrab) {
      outLoop = false;
      if (ledgeHangGrabTimer > 0.f) {
        const double grabDur = staff_skin::clipDuration(staffRig, avClipLedgeGrab);
        const double u = std::clamp(
            1.0 - static_cast<double>(ledgeHangGrabTimer) / static_cast<double>(kLedgeHangGrabDuration),
            0.0, 1.0);
        outPhase = std::clamp(u * grabDur, 0.0, std::max(1e-6, grabDur - 1e-6));
      } else {
        outPhase = std::max(0.0, dur - 1e-4);
      }
      return;
    }
    if (ledgeHangActive && avClipShimmyLeft >= 0 && clip == avClipShimmyLeft) {
      // Loop phase while shimming. Blocked strafe (A/D into deck limit) clears ledgeHangShimmyDir so idle
      // hang applies instead of treadmill phase advance at the clamp edge.
      outLoop = true;
      outPhase = std::fmod(static_cast<double>(ledgeHangShimmyPhase), std::max(dur, 1e-6));
      if (outPhase < 0.0)
        outPhase += std::max(dur, 1e-6);
      return;
    }
    if (ledgeHangActive && avClipShimmyRight >= 0 && clip == avClipShimmyRight) {
      outLoop = true;
      outPhase = std::fmod(static_cast<double>(ledgeHangShimmyPhase), std::max(dur, 1e-6));
      if (outPhase < 0.0)
        outPhase += std::max(dur, 1e-6);
      return;
    }
    if (ladderClimbActive && avClipLedgeClimb >= 0 && clip == avClipLedgeClimb) {
      outLoop = false;
      const double t = std::clamp(static_cast<double>(ladderClimbT), 0.0, 1.0);
      outPhase = std::clamp(t * dur, 0.0, std::max(1e-6, dur - 1e-6));
      return;
    }
    if (ledgeHangActive && avClipLedgeClimb >= 0 && clip == avClipLedgeClimb
        && ledgeClimbT < 0.f && !wallClimbActive && !ladderClimbActive) {
      outLoop = true;
      outPhase = std::fmod(static_cast<double>(ledgeHangShimmyPhase), std::max(dur, 1e-6));
      if (outPhase < 0.0)
        outPhase += std::max(dur, 1e-6);
      return;
    }
    if (avClipLedgeClimb >= 0 && clip == avClipLedgeClimb && ledgeClimbT >= 0.f) {
      outLoop = false;
      const double t = std::clamp(static_cast<double>(ledgeClimbT), 0.0, 1.0);
      const double halfDur = dur * static_cast<double>(kLedgeClimbAnimClipFrac);
      outPhase = std::clamp(t * halfDur, 0.0, std::max(1e-6, halfDur - 1e-6));
      return;
    }
    if (avClipLedgeClimb >= 0 && clip == avClipLedgeClimb && wallClimbActive) {
      outLoop = false;
      const double u =
          std::clamp(1.0 - static_cast<double>(wallClimbTimer) / static_cast<double>(kWallClimbMaxDuration), 0.0,
                     1.0);
      const double halfDur = dur * static_cast<double>(kLedgeClimbAnimClipFrac);
      outPhase = std::clamp(u * halfDur, 0.0, std::max(1e-6, halfDur - 1e-6));
      return;
    }
    if ((slideActive || slideClearClipNextFrame) && slideAnimClip >= 0 && clip == slideAnimClip) {
      outLoop = false;
      const double elapsed =
          static_cast<double>(std::min(slideAnimElapsed, slideAnimDurSec));
      outPhase = std::clamp(elapsed, 0.0, std::max(1e-6, dur - 1e-6));
      return;
    }

    // Looping clips: idle / walk / sprint / crouch use distance-synced avatarLocoPhaseSec (updated in
    // update()). Fallback slide-without-mesh uses speed-scaled sim time.
    const bool motionPhaseClip =
        (avClipIdle >= 0 && clip == avClipIdle) || (avClipWalk >= 0 && clip == avClipWalk) ||
        (avClipSprint >= 0 && clip == avClipSprint) ||
        (avClipCrouchFwd >= 0 && clip == avClipCrouchFwd) ||
        (avClipCrouchBack >= 0 && clip == avClipCrouchBack) ||
        (avClipCrouchLeft >= 0 && clip == avClipCrouchLeft) ||
        (avClipCrouchRight >= 0 && clip == avClipCrouchRight) ||
        (avClipCrouchIdleBow >= 0 && clip == avClipCrouchIdleBow);

    double phaseClock = avatarLocoPhaseSec;
    if (!motionPhaseClip) {
      const float spXZ = avatarHorizSpeedSmoothed;
      double phaseMul = 1.0;
      if (clip == avClipIdle)
        phaseMul = 0.80;
      else if (avClipSprint >= 0 && clip == avClipSprint)
        phaseMul = 0.98;
      if (spXZ > 0.04f) {
        if (slideAnimClip < 0 &&
            ((avClipSlideRight >= 0 && clip == avClipSlideRight) ||
             (avClipSlideLight >= 0 && clip == avClipSlideLight)))
          phaseMul *= static_cast<double>(glm::clamp(spXZ / kSlideStartSpeed, 0.30f, 1.08f));
      }
      phaseMul *= static_cast<double>(kAvatarAnimPlaybackScale);
      phaseClock = static_cast<double>(staffSimTime) * phaseMul;
    }

    outLoop = true;
    const bool isWalk = avClipWalk >= 0 && clip == avClipWalk;
    if (isWalk && playerWalkAnimReverse) {
      double tPh = std::fmod(phaseClock, dur);
      if (tPh < 0.0)
        tPh += dur;
      outPhase = std::clamp(dur - tPh, 1e-6, std::max(1e-6, dur - 1e-6));
      return;
    }
    outPhase = phaseClock;
  }

  template <typename DownFn>
  void syncPlayerAvatarClip(bool groundedEnd, DownFn down) {
    if (!staffSkinnedActive || staffRig.clips.empty()) {
      playerAvatarClip = 0;
      return;
    }
    if (playerDeathActive) {
      if (playerDeathClipIndex >= 0 && static_cast<size_t>(playerDeathClipIndex) < staffRig.clips.size())
        playerAvatarClip = playerDeathClipIndex;
      else if (avClipIdle >= 0)
        playerAvatarClip = avClipIdle;
      else
        playerAvatarClip = 0;
      return;
    }
    if (playerDanceEmoteActive && staffClipShrekProximityDance >= 0 &&
        static_cast<size_t>(staffClipShrekProximityDance) < staffRig.clips.size()) {
      playerAvatarClip = staffClipShrekProximityDance;
      return;
    }
    const float sp = glm::length(horizVel);
    // Schmitt: avoid idle/walk/sprint flicker when |v| hovers near the old 0.12 threshold.
    const bool wasLoco = (avClipWalk >= 0 && playerAvatarClip == avClipWalk) ||
                         (avClipSprint >= 0 && playerAvatarClip == avClipSprint);
    constexpr float kLocoStaySpeed = 0.072f;
    constexpr float kLocoEnterSpeed = 0.118f;
    const bool moving = wasLoco ? (sp > kLocoStaySpeed) : (sp > kLocoEnterSpeed);
    if (sp < 0.035f)
      playerWalkReverseHold = false;
    // Match gameplay: C held = crouch immediately (don’t wait for eyeHeight lerp); also low ceiling / lerped crouch.
    const bool crouchHeldRaw = down(SDL_SCANCODE_C);
    const bool crouchAnim = crouchHeldRaw || eyeHeight < (kEyeHeight - 0.12f);
    const bool runGroundedAnim =
        groundedEnd && inputSprintHeld(down) && !crouchAnim && !slideActive;

    const glm::vec2 f(std::cos(yaw), std::sin(yaw));
    const glm::vec2 r(std::cos(yaw + glm::half_pi<float>()), std::sin(yaw + glm::half_pi<float>()));

    if (ledgeHangActive) {
      playerWalkReverseHold = false;
      if (ledgeHangGrabTimer > 0.f && avClipLedgeGrab >= 0) {
        playerAvatarClip = avClipLedgeGrab;
      } else if (ledgeHangShimmyDir > 0) {
        playerAvatarClip = (avClipShimmyLeft >= 0) ? avClipShimmyLeft
                         : (avClipLedgeClimb >= 0) ? avClipLedgeClimb
                         : (avClipLedgeGrab >= 0)  ? avClipLedgeGrab
                                                   : avClipIdle;
      } else if (ledgeHangShimmyDir < 0) {
        playerAvatarClip = (avClipShimmyRight >= 0) ? avClipShimmyRight
                         : (avClipLedgeClimb >= 0)  ? avClipLedgeClimb
                         : (avClipLedgeGrab >= 0)   ? avClipLedgeGrab
                                                    : avClipIdle;
      } else if (avClipLedgeGrab >= 0) {
        playerAvatarClip = avClipLedgeGrab;
      } else {
        playerAvatarClip = avClipIdle;
      }
      static int shimmyDbgCount = 0;
      if (ledgeHangShimmyDir != 0 && shimmyDbgCount < 30) {
        std::fprintf(stderr, "[shimmy frame] dir=%d clip=%d phase=%.3f shimmyL=%d shimmyR=%d "
                     "blend=%.2f 3P=%d\n",
                     ledgeHangShimmyDir, playerAvatarClip, ledgeHangShimmyPhase,
                     avClipShimmyLeft, avClipShimmyRight, playerAvatarClipBlend,
                     thirdPersonTestMode ? 1 : 0);
        ++shimmyDbgCount;
      }
      return;
    }
    if (ladderClimbActive && avClipLedgeClimb >= 0) {
      playerWalkReverseHold = false;
      playerAvatarClip = avClipLedgeClimb;
      return;
    }
    if (ledgeClimbT >= 0.f && avClipLedgeClimb >= 0) {
      playerWalkReverseHold = false;
      playerAvatarClip = avClipLedgeClimb;
      return;
    }
    if (wallClimbActive && avClipLedgeClimb >= 0) {
      playerWalkReverseHold = false;
      playerAvatarClip = avClipLedgeClimb;
      return;
    }
    if (wallRunActive) {
      playerWalkReverseHold = false;
      if (avClipSprint >= 0)
        playerAvatarClip = avClipSprint;
      else if (avClipWalk >= 0)
        playerAvatarClip = avClipWalk;
      else
        playerAvatarClip = avClipIdle;
      return;
    }
    if (speedVaultActive && (avClipJump >= 0 || avClipJumpRun >= 0)) {
      playerWalkReverseHold = false;
      playerAvatarClip = (avClipJump >= 0) ? avClipJump : avClipJumpRun;
      return;
    }

    if (slideActive || slideClearClipNextFrame) {
      playerWalkReverseHold = false;
      if (!thirdPersonTestMode && avClipSlideLight >= 0)
        playerAvatarClip = avClipSlideLight;
      else if (avClipSlideRight >= 0 && glm::dot(slideDir, r) > 0.55f)
        playerAvatarClip = avClipSlideRight;
      else if (avClipSlideLight >= 0)
        playerAvatarClip = avClipSlideLight;
      else if (avClipSprint >= 0)
        playerAvatarClip = avClipSprint;
      else if (avClipWalk >= 0)
        playerAvatarClip = avClipWalk;
      else
        playerAvatarClip = avClipIdle;
      return;
    }

    if (dropKickActive) {
      playerWalkReverseHold = false;
      if (avClipSlideLight >= 0)
        playerAvatarClip = avClipSlideLight;
      else if (avClipSlideRight >= 0)
        playerAvatarClip = avClipSlideRight;
      return;
    }
    if (playerKickAnimRemain > 0.f && staffClipMeleeKick >= 0) {
      playerWalkReverseHold = false;
      playerAvatarClip = staffClipMeleeKick;
      return;
    }
    if (playerPushAnimRemain > 0.f && avClipStepPush >= 0) {
      playerWalkReverseHold = false;
      playerAvatarClip = avClipStepPush;
      return;
    }

    if (playerJumpPostLandRemain > 0.f && playerJumpPostLandClipIndex >= 0) {
      playerWalkReverseHold = false;
      playerAvatarClip = playerJumpPostLandClipIndex;
      return;
    }
    if (playerPreFallAnimRemain > 0.f) {
      playerWalkReverseHold = false;
      if (avClipJump >= 0 || avClipJumpRun >= 0) {
        // Match resolvePlayerAvatarPhase + pre-land clip: sprint off ledge uses run-jump when loaded.
        playerAvatarClip = (playerPreFallUseRunClip && avClipJumpRun >= 0) ? avClipJumpRun
                            : (avClipJump >= 0 ? avClipJump : avClipJumpRun);
        return;
      }
      if (avClipIdle >= 0) {
        playerAvatarClip = avClipIdle;
        return;
      }
      if (avClipWalk >= 0) {
        playerAvatarClip = avClipWalk;
        return;
      }
      playerAvatarClip = 0;
      return;
    }
    if (playerJumpAnimRemain > 0.f && (avClipJump >= 0 || avClipJumpRun >= 0)) {
      playerWalkReverseHold = false;
      playerAvatarClip =
          (playerJumpRunTailActive && avClipJumpRun >= 0) ? avClipJumpRun
          : (avClipJump >= 0 ? avClipJump : avClipJumpRun);
      return;
    }
    if (playerAvatarJumpFallMidPose() && (avClipJump >= 0 || avClipJumpRun >= 0)) {
      playerWalkReverseHold = false;
      playerAvatarClip =
          (playerJumpRunTailActive && avClipJumpRun >= 0) ? avClipJumpRun
          : (avClipJump >= 0 ? avClipJump : avClipJumpRun);
      return;
    }

    // Narrow shelf gaps: walk/sprint in air only after jump / pre-fall / fall-mid clips (above) decline.
    if (playerAirWalkSmallGap) {
      playerWalkReverseHold = false;
      const bool movingGap =
          wasLoco ? (sp > kLocoStaySpeed) : (sp > std::min(kLocoEnterSpeed, 0.064f));
      const bool runAirSmall = inputSprintHeld(down) && !crouchAnim && !slideActive;
      if (runAirSmall && movingGap && avClipSprint >= 0) {
        playerWalkAnimReverse = false;
        playerWalkReverseHold = false;
        playerAvatarClip = avClipSprint;
        return;
      }
      if (movingGap && avClipWalk >= 0) {
        constexpr float kWalkBackEnter = -0.09f;
        constexpr float kWalkBackExit = 0.04f;
        const float fd = glm::dot(horizVel, f);
        if (playerWalkReverseHold) {
          if (fd > kWalkBackExit)
            playerWalkReverseHold = false;
        } else {
          if (fd < kWalkBackEnter)
            playerWalkReverseHold = true;
        }
        playerWalkAnimReverse = playerWalkReverseHold;
        playerAvatarClip = avClipWalk;
        return;
      }
      if (avClipWalk >= 0) {
        playerWalkAnimReverse = false;
        playerWalkReverseHold = false;
        playerAvatarClip = avClipWalk;
        return;
      }
    }

    // Crouch clips: use as soon as C is held / crouched height (incl. stationary crouch “idle” = forward cautious pose).
    if (crouchAnim && groundedEnd) {
      playerWalkReverseHold = false;
      constexpr float kCrouchVelEps = 0.055f;
      const bool movingCrouch = sp > kCrouchVelEps;
      if (movingCrouch) {
        const float fd = glm::dot(horizVel, f);
        const float rd = glm::dot(horizVel, r);
        if (std::abs(fd) >= std::abs(rd)) {
          if (fd > kCrouchVelEps && avClipCrouchFwd >= 0) {
            playerAvatarClip = avClipCrouchFwd;
            return;
          }
          if (fd < -kCrouchVelEps && avClipCrouchBack >= 0) {
            playerAvatarClip = avClipCrouchBack;
            return;
          }
        } else {
          if (rd > kCrouchVelEps && avClipCrouchRight >= 0) {
            playerAvatarClip = avClipCrouchRight;
            return;
          }
          if (rd < -kCrouchVelEps && avClipCrouchLeft >= 0) {
            playerAvatarClip = avClipCrouchLeft;
            return;
          }
        }
        if (avClipCrouchFwd >= 0) {
          playerAvatarClip = avClipCrouchFwd;
          return;
        }
      } else {
        if (avClipCrouchIdleBow >= 0) {
          playerAvatarClip = avClipCrouchIdleBow;
          return;
        }
        if (avClipCrouchFwd >= 0) {
          playerAvatarClip = avClipCrouchFwd;
          return;
        }
      }
    }

    if (runGroundedAnim && moving && avClipSprint >= 0) {
      playerWalkAnimReverse = false;
      playerWalkReverseHold = false;
      playerAvatarClip = avClipSprint;
      return;
    }
    if (moving && avClipWalk >= 0) {
      // Camera-forward (f): reverse walk cycle when moving backward; hysteresis stops strafe flicker.
      constexpr float kWalkBackEnter = -0.09f;
      constexpr float kWalkBackExit = 0.04f;
      const float fd = glm::dot(horizVel, f);
      if (playerWalkReverseHold) {
        if (fd > kWalkBackExit)
          playerWalkReverseHold = false;
      } else {
        if (fd < kWalkBackEnter)
          playerWalkReverseHold = true;
      }
      playerWalkAnimReverse = playerWalkReverseHold;
      playerAvatarClip = avClipWalk;
      return;
    }
    if (moving && avClipSprint >= 0) {
      playerWalkAnimReverse = false;
      playerWalkReverseHold = false;
      playerAvatarClip = avClipSprint;
      return;
    }
    playerWalkAnimReverse = false;
    playerWalkReverseHold = false;
    playerAvatarClip = avClipIdle;
  }

  void refreshWindowTitleWithHealth() {
    if (!window)
      return;
    char tb[220];
    std::snprintf(tb, sizeof(tb),
                  "retro ikea v" VULKAN_GAME_VERSION_STRING
                  " — hall — HP %.0f/%.0f — WASD | sprint | slide | Space jump + ledge hang | Shift+wall run | "
                  "wall+Space jump/climb | speed vault",
                  playerHealth, kPlayerHealthMax);
    SDL_SetWindowTitle(window, tb);
  }

  void resetShelfStaffAfterPlayerRespawn() {
    for (uint32_t si : shelfEmpActiveSlots) {
      ShelfEmployeeNpc& e = shelfEmpPool[si];
      if (!e.inited)
        continue;
      e.nightPhase = 0;
      e.nightSpotTimer = 0.f;
      e.nightInvestigateTimer = 0.f;
      e.nightLastKnownPlayerXZ = glm::vec2(0.f);
      e.stuckTimer = 0.f;
      e.chaseUnstuckTimer = 0.f;
      e.meleeState = 0;
      e.meleePhaseSec = 0.0;
      e.meleeAnimBlend = 1.f;
      e.staffPushAggro = false;
      e.staffPushAggroCalmRemain = 0.f;
      e.staffNightShoveChase = false;
      e.staffNightShoveRevealRemain = 0.f;
      e.chaseLedgeClimbRem = -1.f;
      e.chaseLedgeClimbTotalDur = 0.f;
      e.chaseLedgeClimbY0 = 0.f;
      e.chaseLedgeClimbY1 = 0.f;
      e.staffChaseMantelCooldownRem = 0.f;
      e.staffLastMantelTargetY = kGroundY;
      e.staffMantelAnimPhaseSpanSec = 0.f;
      e.staffMantelRunnerChase = 0;
      e.staffShoveKnockbackVelXZ = glm::vec2(0.f);
      e.staffVelY = 0.f;
      e.velXZ = glm::vec2(0.f);
      e.lastHorizSpeed = 0.f;
      e.staffAirLocoRemain = 0.f;
      e.staffAirFallClip = -1;
      e.staffAirLandRemain = 0.f;
      e.staffAirLandClip = -1;
      e.staffGroundedPrev = true;
      e.staffFallPeakFeetY = kGroundY;
      e.staffTallFallKnockdownPending = 0;
      e.meleeKnockdownFeetAnchorY = kGroundY;
      e.shovePlayerPushDurSec = 0.f;
      e.staffFootstepHavePrev = false;
      e.staffFootstepAccum = 0.f;
      const float fy = playerTerrainSupportY(e.posXZ.x, e.posXZ.y, e.feetWorldY);
      if (std::isfinite(fy))
        e.feetWorldY = fy;
    }
  }

  void beginPlayerDeath() {
    if (playerDeathActive)
      return;
    showPauseMenu = false;
    showInventoryMenu = false;
    audioSetStoreDayNightCyclePaused(true);
    playerDeathActive = true;
    playerDeathShowMenu = false;
    playerHealth = 0.f;
    playerDeathClipIndex = -1;
    playerDeathClipFracEnd = kPlayerDeathFallClipPortion;
    if (staffClipMeleeFall >= 0 && static_cast<size_t>(staffClipMeleeFall) < staffRig.clips.size()) {
      playerDeathClipIndex = staffClipMeleeFall;
    } else if (avClipLand >= 0 && static_cast<size_t>(avClipLand) < staffRig.clips.size()) {
      playerDeathClipIndex = avClipLand;
      playerDeathClipFracEnd = kPlayerDeathLandClipPortion;
    }
    if (playerDeathClipIndex >= 0) {
      playerDeathPlayingFallClip = true;
      playerDeathAnimTime = 0.0;
      playerDeathHoldRemain = 0.f;
    } else {
      playerDeathPlayingFallClip = false;
      playerDeathAnimTime = 0.0;
      playerDeathHoldRemain = kPlayerDeathNoClipHoldSec;
    }
    wallRunActive = false;
    wallRunTimer = 0.f;
    wallRunCooldownTimer = 0.f;
    wallRunCameraTilt = 0.f;
    wallClimbActive = false;
    wallClimbTimer = 0.f;
    wallClimbCooldownTimer = 0.f;
    wallJumpCooldownTimer = 0.f;
    speedVaultActive = false;
    speedVaultTimer = 0.f;
    ledgeHangActive = false;
    ledgeHangGrabTimer = 0.f;
    ledgeHangShimmyDir = 0;
    ledgeHangShimmyPhase = 0.f;
    ledgeHangFeetYCaptured = false;
    slideActive = false;
    slideAnimClip = -1;
    slideAnimElapsed = 0.f;
    slideAnimDurSec = 0.f;
    slideStartSpeed = 0.f;
    slideClearClipNextFrame = false;
    if (ledgeClimbT >= 0.f) {
      ledgeClimbT = -1.f;
      ledgeClimbVisPhase = 0.f;
      ledgeClimbApproachVel = glm::vec2(0.f);
    }
    ladderClimbActive = false;
    ladderClimbT = 0.f;
    ladderClimbApproachVel = glm::vec2(0.f);
    playerJumpAnimRemain = 0.f;
    playerPreFallAnimRemain = 0.f;
    playerJumpPostLandRemain = 0.f;
    playerJumpPostLandClipIndex = -1;
    playerJumpArchActive = false;
    playerJumpAirTimeTargetSec = 0.f;
    {
      const float feetProbe = camPos.y - eyeHeight;
      float fy = playerTerrainSupportY(camPos.x, camPos.z, feetProbe);
      if (!std::isfinite(fy))
        fy = playerTerrainSupportY(camPos.x, camPos.z, kGroundY + 2.8f);
      if (!std::isfinite(fy))
        fy = kGroundY;
      camPos.y = fy + eyeHeight;
      const float fyGlue = playerTerrainSupportY(camPos.x, camPos.z, camPos.y - eyeHeight);
      if (std::isfinite(fyGlue))
        camPos.y = fyGlue + eyeHeight;
    }
    horizVel = glm::vec2(0.f);
    velY = 0.f;
    playerAvatarClipBlend = 1.f;
    playerDanceEmoteActive = false;
    playerDanceEmoteStopGraceRemain = 0.f;
    // Persist a "dead" snapshot immediately so closing the game on death does not rewind to an old alive save.
    gameSaveWrite();
    refreshWindowTitleWithHealth();
  }

  void respawnPlayerAfterDeath() {
    playerDeathActive = false;
    playerDeathShowMenu = false;
    playerDeathPlayingFallClip = false;
    playerDeathAnimTime = 0.0;
    playerDeathHoldRemain = 0.f;
    playerDeathClipIndex = -1;
    playerHealth = kPlayerHealthMax;
    playerHunger = kPlayerHungerMax;
    playerStaffMeleeInvulnRem = 0.f;
    dropKickActive = false;
    dropKickTimer = 0.f;
    resetShelfStaffAfterPlayerRespawn();

    constexpr float kRespawnRandHalfSpanM = 118.f;
    std::random_device respawnRd;
    std::mt19937 respawnGen(respawnRd());
    std::uniform_real_distribution<float> xzDist(-kRespawnRandHalfSpanM, kRespawnRandHalfSpanM);
    bool respawnPlaced = false;
    for (int attempt = 0; attempt < 80; ++attempt) {
      const float rx = xzDist(respawnGen);
      const float rz = xzDist(respawnGen);
      if (!playerRespawnFloorXZClear(rx, rz))
        continue;
      camPos.x = rx;
      camPos.z = rz;
      const float probeFeet = kGroundY + 0.75f;
      float fy = playerTerrainSupportY(camPos.x, camPos.z, probeFeet);
      if (!std::isfinite(fy))
        continue;
      camPos.y = fy + eyeHeight;
      float fyGlue = playerTerrainSupportY(camPos.x, camPos.z, camPos.y - eyeHeight);
      if (std::isfinite(fyGlue))
        camPos.y = fyGlue + eyeHeight;
      resolvePillarCollisions(true);
      respawnPlaced = true;
      break;
    }
    if (!respawnPlaced) {
      const float feetProbe = camPos.y - eyeHeight;
      float fy = playerTerrainSupportY(camPos.x, camPos.z, feetProbe);
      if (!std::isfinite(fy))
        fy = playerTerrainSupportY(camPos.x, camPos.z, kGroundY + 2.8f);
      if (!std::isfinite(fy))
        fy = kGroundY;
      camPos.y = fy + eyeHeight;
      const float fyGlue = playerTerrainSupportY(camPos.x, camPos.z, camPos.y - eyeHeight);
      if (std::isfinite(fyGlue))
        camPos.y = fyGlue + eyeHeight;
    } else {
      std::uniform_real_distribution<float> yawDist(-glm::pi<float>(), glm::pi<float>());
      yaw = wrapAnglePi(yawDist(respawnGen));
    }

    velY = 0.f;
    horizVel = glm::vec2(0.f);
    {
      const float feetNow = camPos.y - eyeHeight;
      float supY = playerTerrainSupportY(camPos.x, camPos.z, feetNow);
      if (!std::isfinite(supY))
        supY = kGroundY;
      const float supGlue = playerTerrainSupportY(camPos.x, camPos.z, camPos.y - eyeHeight);
      playerFallLastGroundedSupportY = std::isfinite(supGlue) ? supGlue : supY;
    }
    playerFallChainMaxFeetY = std::numeric_limits<float>::quiet_NaN();
    audioSetStoreDayNightCyclePaused(false);
    mouseGrab = true;
    syncInputGrab();
    refreshWindowTitleWithHealth();
  }

  std::string gameSavePrefDirectory() const {
    char* pref = SDL_GetPrefPath("NightShift", "VulkanGame");
    if (!pref)
      return {};
    std::string path(pref);
    SDL_free(pref);
    return path;
  }

  std::string gameLegacySavePath() const {
    return gameSavePrefDirectory() + "save.bin";
  }

  std::string gameSaveSlotPath(int slot) const {
    if (slot < 0 || slot >= kGameSaveSlotCount)
      return {};
    return gameSavePrefDirectory() + "save_slot_" + std::to_string(slot) + ".bin";
  }

  std::string gameSaveInventorySlotPath(int slot) const {
    if (slot < 0 || slot >= kGameSaveSlotCount)
      return {};
    return gameSavePrefDirectory() + "save_slot_" + std::to_string(slot) + ".inv";
  }

  std::string gameLastSaveSlotMetaPath() const {
    return gameSavePrefDirectory() + "last_save_slot.txt";
  }

  bool loadInventoryStateFromSlot(int slot) {
    inventoryItems.clear();
    deliPizzaSlicesBySlot.clear();
    deliPizzaReplenishTimerBySlot.clear();
    deliMeatballsBySlot.clear();
    deliMeatballReplenishTimerBySlot.clear();
    playerHunger = kPlayerHungerMax;
    ++inventoryRevision;
    inventoryScrollRow = 0;
    inventoryMenuCacheScroll = -1;
    const std::string path = gameSaveInventorySlotPath(slot);
    if (path.empty())
      return false;
    std::ifstream f(path, std::ios::binary);
    if (!f)
      return false;
    struct HeaderV1 {
      uint32_t magic;
      uint32_t version;
      float playerHunger;
      uint32_t itemCount;
      uint32_t sliceCount;
      uint32_t timerCount;
    };
    struct HeaderV2 {
      uint32_t magic;
      uint32_t version;
      float playerHunger;
      uint32_t itemCount;
      uint32_t pizzaSliceCount;
      uint32_t pizzaTimerCount;
      uint32_t meatballCount;
      uint32_t meatballTimerCount;
    };
    uint32_t magic = 0;
    uint32_t version = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!f || magic != kInventoryStateSaveMagic)
      return false;
    uint32_t itemCount = 0;
    uint32_t pizzaSliceCount = 0;
    uint32_t pizzaTimerCount = 0;
    uint32_t meatballCount = 0;
    uint32_t meatballTimerCount = 0;
    float savedHunger = kPlayerHungerMax;
    f.clear();
    f.seekg(0, std::ios::beg);
    if (version == 1u) {
      f.clear();
      HeaderV1 h1{};
      f.read(reinterpret_cast<char*>(&h1), sizeof(h1));
      if (!f || h1.magic != kInventoryStateSaveMagic || h1.version != 1u)
        return false;
      itemCount = h1.itemCount;
      pizzaSliceCount = h1.sliceCount;
      pizzaTimerCount = h1.timerCount;
      savedHunger = h1.playerHunger;
    } else if (version == kInventoryStateSaveVersion) {
      HeaderV2 h2{};
      f.read(reinterpret_cast<char*>(&h2), sizeof(h2));
      if (!f || h2.magic != kInventoryStateSaveMagic || h2.version != kInventoryStateSaveVersion)
        return false;
      itemCount = h2.itemCount;
      pizzaSliceCount = h2.pizzaSliceCount;
      pizzaTimerCount = h2.pizzaTimerCount;
      meatballCount = h2.meatballCount;
      meatballTimerCount = h2.meatballTimerCount;
    } else {
      return false;
    }
    constexpr uint32_t kMaxItems = 20000u;
    constexpr uint32_t kMaxSlices = 20000u;
    constexpr uint32_t kMaxTimers = 20000u;
    if (itemCount > kMaxItems || pizzaSliceCount > kMaxSlices || pizzaTimerCount > kMaxTimers ||
        meatballCount > kMaxSlices || meatballTimerCount > kMaxTimers)
      return false;
    if (std::isfinite(savedHunger))
      playerHunger = std::clamp(savedHunger, 0.f, kPlayerHungerMax);
    inventoryItems.reserve(itemCount);
    for (uint32_t i = 0; i < itemCount; ++i) {
      uint16_t len = 0;
      f.read(reinterpret_cast<char*>(&len), sizeof(len));
      if (!f || len > 256u)
        return false;
      std::string item(static_cast<size_t>(len), '\0');
      if (len > 0)
        f.read(item.data(), static_cast<std::streamsize>(len));
      if (!f)
        return false;
      inventoryItems.emplace_back(std::move(item));
    }
    for (uint32_t i = 0; i < pizzaSliceCount; ++i) {
      uint64_t key = 0;
      uint8_t count = 0;
      f.read(reinterpret_cast<char*>(&key), sizeof(key));
      f.read(reinterpret_cast<char*>(&count), sizeof(count));
      if (!f)
        return false;
      deliPizzaSlicesBySlot[key] =
          static_cast<uint8_t>(std::min<int>(static_cast<int>(count), kDeliPizzaSlicesPerCounter));
    }
    for (uint32_t i = 0; i < pizzaTimerCount; ++i) {
      uint64_t key = 0;
      float remain = 0.f;
      f.read(reinterpret_cast<char*>(&key), sizeof(key));
      f.read(reinterpret_cast<char*>(&remain), sizeof(remain));
      if (!f)
        return false;
      if (std::isfinite(remain) && remain > 0.f)
        deliPizzaReplenishTimerBySlot[key] = std::clamp(remain, 0.f, kDeliPizzaReplenishSec);
    }
    for (uint32_t i = 0; i < meatballCount; ++i) {
      uint64_t key = 0;
      uint8_t count = 0;
      f.read(reinterpret_cast<char*>(&key), sizeof(key));
      f.read(reinterpret_cast<char*>(&count), sizeof(count));
      if (!f)
        return false;
      deliMeatballsBySlot[key] =
          static_cast<uint8_t>(std::min<int>(static_cast<int>(count), kDeliMeatballsPerCounter));
    }
    for (uint32_t i = 0; i < meatballTimerCount; ++i) {
      uint64_t key = 0;
      float remain = 0.f;
      f.read(reinterpret_cast<char*>(&key), sizeof(key));
      f.read(reinterpret_cast<char*>(&remain), sizeof(remain));
      if (!f)
        return false;
      if (std::isfinite(remain) && remain > 0.f)
        deliMeatballReplenishTimerBySlot[key] = std::clamp(remain, 0.f, kDeliPizzaReplenishSec);
    }
    const int maxScroll = std::max(0, inventoryStackRowCount() - 8);
    inventoryScrollRow = std::clamp(inventoryScrollRow, 0, maxScroll);
    inventoryMenuCacheScroll = -1;
    return true;
  }

  void saveInventoryStateToSlot(int slot) {
    const std::string path = gameSaveInventorySlotPath(slot);
    if (path.empty())
      return;
    std::ofstream f(path, std::ios::binary);
    if (!f)
      return;
    struct Header {
      uint32_t magic;
      uint32_t version;
      float playerHunger;
      uint32_t itemCount;
      uint32_t pizzaSliceCount;
      uint32_t pizzaTimerCount;
      uint32_t meatballCount;
      uint32_t meatballTimerCount;
    };
    Header h{};
    h.magic = kInventoryStateSaveMagic;
    h.version = kInventoryStateSaveVersion;
    h.playerHunger = playerHunger;
    h.itemCount = static_cast<uint32_t>(inventoryItems.size());
    h.pizzaSliceCount = static_cast<uint32_t>(deliPizzaSlicesBySlot.size());
    h.pizzaTimerCount = static_cast<uint32_t>(deliPizzaReplenishTimerBySlot.size());
    h.meatballCount = static_cast<uint32_t>(deliMeatballsBySlot.size());
    h.meatballTimerCount = static_cast<uint32_t>(deliMeatballReplenishTimerBySlot.size());
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    if (!f)
      return;
    for (const std::string& item : inventoryItems) {
      const uint16_t len = static_cast<uint16_t>(std::min<size_t>(item.size(), 256u));
      f.write(reinterpret_cast<const char*>(&len), sizeof(len));
      if (len > 0)
        f.write(item.data(), static_cast<std::streamsize>(len));
      if (!f)
        return;
    }
    for (const auto& kv : deliPizzaSlicesBySlot) {
      f.write(reinterpret_cast<const char*>(&kv.first), sizeof(kv.first));
      f.write(reinterpret_cast<const char*>(&kv.second), sizeof(kv.second));
      if (!f)
        return;
    }
    for (const auto& kv : deliPizzaReplenishTimerBySlot) {
      const float remain = std::clamp(kv.second, 0.f, kDeliPizzaReplenishSec);
      f.write(reinterpret_cast<const char*>(&kv.first), sizeof(kv.first));
      f.write(reinterpret_cast<const char*>(&remain), sizeof(remain));
      if (!f)
        return;
    }
    for (const auto& kv : deliMeatballsBySlot) {
      f.write(reinterpret_cast<const char*>(&kv.first), sizeof(kv.first));
      f.write(reinterpret_cast<const char*>(&kv.second), sizeof(kv.second));
      if (!f)
        return;
    }
    for (const auto& kv : deliMeatballReplenishTimerBySlot) {
      const float remain = std::clamp(kv.second, 0.f, kDeliPizzaReplenishSec);
      f.write(reinterpret_cast<const char*>(&kv.first), sizeof(kv.first));
      f.write(reinterpret_cast<const char*>(&remain), sizeof(remain));
      if (!f)
        return;
    }
  }

  bool saveSlotFileLooksValid(const std::string& path) const {
    if (path.empty())
      return false;
    std::ifstream file(path, std::ios::binary);
    if (!file)
      return false;
    uint32_t magic = 0;
    uint32_t version = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!file)
      return false;
    file.clear();
    file.seekg(0, std::ios::beg);
    if (magic != kGameSaveMagic || (version != 1u && version != 2u && version != 3u))
      return false;
    const size_t want = version == 1u ? sizeof(GameSaveFileV1)
                         : version == 2u ? sizeof(GameSaveFileV2Read)
                                          : sizeof(GameSaveFileV3);
    std::vector<char> buf(want);
    file.read(buf.data(), static_cast<std::streamsize>(want));
    if (!file || static_cast<size_t>(file.gcount()) != want)
      return false;
    float hdr[7];
    std::memcpy(hdr, buf.data() + 8, sizeof(hdr));
    const float camX = hdr[0], camY = hdr[1], camZ = hdr[2], yaw = hdr[3], pitch = hdr[4], ph = hdr[5],
                eh = hdr[6];
    return std::isfinite(camX) && std::isfinite(camY) && std::isfinite(camZ) &&
           std::isfinite(yaw) && std::isfinite(pitch) && std::isfinite(ph) && std::isfinite(eh);
  }

  void migrateLegacySaveIfNeeded() {
    const std::string leg = gameLegacySavePath();
    std::ifstream testLeg(leg, std::ios::binary);
    if (!testLeg.good())
      return;
    for (int i = 0; i < kGameSaveSlotCount; ++i) {
      if (saveSlotFileLooksValid(gameSaveSlotPath(i)))
        return;
    }
    testLeg.close();
    std::ifstream in(leg, std::ios::binary);
    std::ofstream out(gameSaveSlotPath(0), std::ios::binary);
    if (!in || !out)
      return;
    out << in.rdbuf();
    in.close();
    out.close();
    std::remove(leg.c_str());
    std::ofstream meta(gameLastSaveSlotMetaPath());
    if (meta)
      meta << "0\n";
  }

  void refreshTitleMenuContinueState() {
    titleMenuHasContinue = false;
    titleMenuLastSlot = 0;
    const std::string metaPath = gameLastSaveSlotMetaPath();
    if (!metaPath.empty()) {
      std::ifstream mf(metaPath);
      int slot = -1;
      if (mf >> slot) {
        if (slot >= 0 && slot < kGameSaveSlotCount && saveSlotFileLooksValid(gameSaveSlotPath(slot))) {
          titleMenuHasContinue = true;
          titleMenuLastSlot = slot;
        }
      }
    }
    if (!titleMenuHasContinue) {
      for (int i = kGameSaveSlotCount - 1; i >= 0; --i) {
        if (saveSlotFileLooksValid(gameSaveSlotPath(i))) {
          titleMenuHasContinue = true;
          titleMenuLastSlot = i;
          break;
        }
      }
    }
  }

  void deleteSaveSlot(int slot) {
    if (slot < 0 || slot >= kGameSaveSlotCount)
      return;
    const std::string path = gameSaveSlotPath(slot);
    const std::string invPath = gameSaveInventorySlotPath(slot);
    if (path.empty())
      return;
    std::remove(path.c_str());
    if (!invPath.empty())
      std::remove(invPath.c_str());
    refreshTitleMenuContinueState();
    const std::string metaPath = gameLastSaveSlotMetaPath();
    if (metaPath.empty())
      return;
    if (titleMenuHasContinue) {
      std::ofstream mf(metaPath);
      if (mf)
        mf << titleMenuLastSlot << '\n';
    } else {
      std::remove(metaPath.c_str());
    }
  }

  void writeLastSaveSlotMeta() {
    const std::string p = gameLastSaveSlotMetaPath();
    if (p.empty())
      return;
    std::ofstream f(p);
    if (f)
      f << activeSaveSlot << '\n';
  }

  void uploadUiMeshToGpu(const std::vector<Vertex>& verts, VkBuffer& outBuf, VkDeviceMemory& outMem,
                         uint32_t& outCount) {
    outCount = static_cast<uint32_t>(verts.size());
    if (outBuf != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, outBuf, nullptr);
      vkFreeMemory(device, outMem, nullptr);
      outBuf = VK_NULL_HANDLE;
      outMem = VK_NULL_HANDLE;
    }
    if (verts.empty())
      return;
    const VkDeviceSize sz = sizeof(Vertex) * verts.size();
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    createBuffer(physicalDevice, device, sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging,
                 stagingMem);
    void* data = nullptr;
    vkMapMemory(device, stagingMem, 0, sz, 0, &data);
    std::memcpy(data, verts.data(), static_cast<size_t>(sz));
    vkUnmapMemory(device, stagingMem);
    createBuffer(physicalDevice, device, sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuf, outMem);
    copyBuffer(device, commandPool, graphicsQueue, staging, outBuf, sz);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);
  }

  void recreateTitleMenuMainGpuMesh() {
    refreshTitleMenuContinueState();
    uploadUiMeshToGpu(buildTitleMenuMainOverlayVertices(titleMenuHasContinue), titleMenuMainVertexBuffer,
                      titleMenuMainVertexBufferMemory, titleMenuMainVertexCount);
  }

  void recreateTitleMenuSlotGpuMesh() {
    std::array<bool, 4> used{};
    for (int i = 0; i < kGameSaveSlotCount; ++i)
      used[static_cast<size_t>(i)] = saveSlotFileLooksValid(gameSaveSlotPath(i));
    uploadUiMeshToGpu(buildTitleMenuSlotPickerVertices(used), titleMenuSlotVertexBuffer,
                      titleMenuSlotVertexBufferMemory, titleMenuSlotVertexCount);
  }

  void destroyTitleMenuGpuMeshes() {
    if (titleMenuMainVertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, titleMenuMainVertexBuffer, nullptr);
      vkFreeMemory(device, titleMenuMainVertexBufferMemory, nullptr);
      titleMenuMainVertexBuffer = VK_NULL_HANDLE;
      titleMenuMainVertexBufferMemory = VK_NULL_HANDLE;
      titleMenuMainVertexCount = 0;
    }
    if (titleMenuSlotVertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, titleMenuSlotVertexBuffer, nullptr);
      vkFreeMemory(device, titleMenuSlotVertexBufferMemory, nullptr);
      titleMenuSlotVertexBuffer = VK_NULL_HANDLE;
      titleMenuSlotVertexBufferMemory = VK_NULL_HANDLE;
      titleMenuSlotVertexCount = 0;
    }
  }

  void ensureYellowMenuCursor() {
    if (yellowMenuCursor)
      return;
    // Real arrow art in assets (yellow fill, black 1px rim, 64² nearest-upscaled for chunky Win9x look).
    // Note: Microsoft Learn’s CreateCursor sample AND/XOR arrays are a *yin-yang* demo — not IDC_ARROW.
    SDL_Surface* raw = nullptr;
#if defined(VULKAN_GAME_ASSETS_DIR)
    raw = IMG_Load(VULKAN_GAME_ASSETS_DIR "/ui/yellow_win_arrow.png");
#endif
    if (!raw)
      raw = IMG_Load("assets/ui/yellow_win_arrow.png");
    if (!raw) {
      std::cerr << "[ui] yellow_win_arrow.png missing — menu uses default cursor (" << IMG_GetError() << ")\n";
      return;
    }
    SDL_Surface* conv = SDL_ConvertSurfaceFormat(raw, SDL_PIXELFORMAT_RGBA8888, 0);
    SDL_FreeSurface(raw);
    if (!conv)
      return;
    const int scaledW =
        std::max(1, static_cast<int>(std::lround(static_cast<double>(conv->w) * kMenuCursorScale)));
    const int scaledH =
        std::max(1, static_cast<int>(std::lround(static_cast<double>(conv->h) * kMenuCursorScale)));
    SDL_Surface* scaled = SDL_CreateRGBSurfaceWithFormat(0, scaledW, scaledH, 32, SDL_PIXELFORMAT_RGBA8888);
    if (scaled) {
      SDL_Rect dst{0, 0, scaledW, scaledH};
      SDL_BlitScaled(conv, nullptr, scaled, &dst);
      yellowMenuCursor = SDL_CreateColorCursor(scaled, 0, 0);
      SDL_FreeSurface(scaled);
    } else {
      yellowMenuCursor = SDL_CreateColorCursor(conv, 0, 0);
    }
    SDL_FreeSurface(conv);
  }

  void applyYellowMenuCursorIfNeeded() {
    ensureYellowMenuCursor();
    if (yellowMenuCursor)
      SDL_SetCursor(yellowMenuCursor);
  }

  void returnToTitleMenuFromGame() {
    showPauseMenu = false;
    showInventoryMenu = false;
    inTitleMenu = true;
    titleMenuSceneTime = 0.f;
    titleMenuSlideTime = 0.f;
    titleMenuSlideWasSlot = false;
    titleMenuPickSlot = false;
    playerDeathActive = false;
    playerDeathShowMenu = false;
    playerHealth = kPlayerHealthMax;
    playerHunger = kPlayerHungerMax;
    velY = 0.f;
    horizVel = glm::vec2(0.f);
    ledgeClimbT = -1.f;
    ledgeClimbVisPhase = 0.f;
    ledgeClimbApproachVel = glm::vec2(0.f);
    ladderClimbActive = false;
    ladderClimbT = 0.f;
    ladderClimbApproachVel = glm::vec2(0.f);
    audioSetStoreDayNightCyclePaused(true);
    audioSetTitleMenuMusicActive(true);
    recreateTitleMenuMainGpuMesh();
    recreateTitleMenuSlotGpuMesh();
    mouseGrab = false;
    syncInputGrab();
    refreshWindowTitleWithHealth();
  }

  void applyDefaultNewGameState() {
    camPos = glm::vec3(0.f, kGroundY + kEyeHeight, 6.f);
    yaw = -glm::pi<float>() * 0.5f;
    pitch = 0.f;
    eyeHeight = kEyeHeight;
    playerHealth = kPlayerHealthMax;
    playerHunger = kPlayerHungerMax;
    inventoryItems.clear();
    deliPizzaSlicesBySlot.clear();
    deliPizzaReplenishTimerBySlot.clear();
    deliMeatballsBySlot.clear();
    deliMeatballReplenishTimerBySlot.clear();
    ++inventoryRevision;
    inventoryScrollRow = 0;
    inventoryMenuCacheScroll = -1;
    velY = 0.f;
    horizVel = glm::vec2(0.f);
    playerFallChainMaxFeetY = std::numeric_limits<float>::quiet_NaN();
    ladderClimbActive = false;
    ladderClimbT = 0.f;
    ladderClimbApproachVel = glm::vec2(0.f);
  }

  bool tryLoadGameSaveFromSlot(int slot) {
    const std::string path = gameSaveSlotPath(slot);
    if (path.empty())
      return false;
    std::ifstream file(path, std::ios::binary);
    if (!file)
      return false;
    uint32_t magic = 0;
    uint32_t version = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!file)
      return false;
    file.clear();
    file.seekg(0, std::ios::beg);
    if (magic != kGameSaveMagic || (version != 1u && version != 2u && version != 3u))
      return false;
    const size_t want = version == 1u ? sizeof(GameSaveFileV1)
                         : version == 2u ? sizeof(GameSaveFileV2Read)
                                          : sizeof(GameSaveFileV3);
    std::vector<char> buf(want);
    file.read(buf.data(), static_cast<std::streamsize>(want));
    if (!file || static_cast<size_t>(file.gcount()) != want)
      return false;
    float hdr[7];
    std::memcpy(hdr, buf.data() + 8, sizeof(hdr));
    const float cx = hdr[0], cy = hdr[1], cz = hdr[2], sy = hdr[3], sp = hdr[4], ph = hdr[5], eh = hdr[6];
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(cz))
      return false;
    if (!std::isfinite(sy) || !std::isfinite(sp))
      return false;
    if (!std::isfinite(ph) || !std::isfinite(eh))
      return false;
    camPos = glm::vec3(cx, cy, cz);
    yaw = wrapAnglePi(sy);
    pitch = std::clamp(sp, kPitchMaxLookDown, kPitchMaxLookUp);
    playerHealth = std::clamp(ph, 0.f, kPlayerHealthMax);
    playerHunger = kPlayerHungerMax;
    eyeHeight = std::clamp(eh, kCrouchEyeHeight, kEyeHeight);
    pendingLoadedSaveWasDead = playerHealth <= 0.f;
    velY = 0.f;
    horizVel = glm::vec2(0.f);
    pendingLoadedAudioStateValid = false;
    if (version == 3u) {
      GameSaveFileV3 s{};
      std::memcpy(&s, buf.data(), sizeof(s));
      pendingLoadedAudioState = s.audioState;
      pendingLoadedAudioStateValid = true;
    } else if (version == 2u) {
      GameSaveFileV2Read s{};
      std::memcpy(&s, buf.data(), sizeof(s));
      AudioStoreCycleSaveState a{};
      a.version = 2;
      a.storePhase = s.audio.storePhase;
      a.flags = s.audio.flags;
      a.storeCursorFrames = s.audio.storeCursorFrames;
      a.horrorCursorFrames = s.audio.horrorCursorFrames;
      a.chaseCursorFrames = s.audio.chaseCursorFrames;
      a.shrekCursorFrames = s.audio.shrekCursorFrames;
      a.blackoutRemainingMs = s.audio.blackoutRemainingMs;
      a.dayRestoreRemainingMs = s.audio.dayRestoreRemainingMs;
      a.storeDayMusicTrackIdx = 0;
      pendingLoadedAudioState = a;
      pendingLoadedAudioStateValid = true;
    }
    loadInventoryStateFromSlot(slot);
    refreshWindowTitleWithHealth();
    return true;
  }

  void beginGameFromSaveSlot(int slot) {
    if (slot < 0 || slot >= kGameSaveSlotCount)
      return;
    inLoadingScreen = true;
    loadingScreenTime = 0.f;
    loadingScreenSlot = slot;
    loadingScreenIsNewSave = !saveSlotFileLooksValid(gameSaveSlotPath(slot));
    inTitleMenu = false;
    titleMenuPickSlot = false;
    audioSetTitleMenuMusicActive(false);
    audioSetLoadingScreenActive(true);
    syncInputGrab();
  }

  void finishLoadFromSlot(int slot) {
    activeSaveSlot = slot;
    playerDeathActive = false;
    playerDeathShowMenu = false;
    pendingLoadedAudioStateValid = false;
    pendingLoadedSaveWasDead = false;
    const bool hadSave = tryLoadGameSaveFromSlot(slot);
    if (!hadSave)
      applyDefaultNewGameState();
    audioSetTitleMenuMusicActive(false);
    if (!hadSave)
      audioResetToNewGame();
    // Restore store phase / cursors while still paused so unpause logic does not cold-start day music
    // (wrong phase) before apply — that used to strand Continue loads with no repeating day↔night cycle.
    if (pendingLoadedAudioStateValid)
      audioRestoreStoreCycleSaveState(pendingLoadedAudioState);
    audioSetStoreDayNightCyclePaused(false);
    if (pendingLoadedSaveWasDead) {
      respawnPlayerAfterDeath();
      pendingLoadedSaveWasDead = false;
    }
    const bool wasNewSave = loadingScreenIsNewSave;
    audioSetLoadingScreenActive(false);
    inLoadingScreen = false;
    mouseGrab = true;
    syncInputGrab();
    gameSaveWrite();
    refreshWindowTitleWithHealth();
    if (wasNewSave)
      newGameControlsPopupDelay = 1.0f;
  }

  void continueFromLastSave() {
    refreshTitleMenuContinueState();
    if (!titleMenuHasContinue)
      return;
    beginGameFromSaveSlot(titleMenuLastSlot);
  }

  void gameSaveWrite() {
    if (inTitleMenu)
      return;
    const std::string path = gameSaveSlotPath(activeSaveSlot);
    if (path.empty())
      return;
    GameSaveFileV3 s{};
    s.magic = kGameSaveMagic;
    s.version = kGameSaveVersion;
    s.camX = camPos.x;
    s.camY = camPos.y;
    s.camZ = camPos.z;
    s.yaw = yaw;
    s.pitch = pitch;
    s.playerHealth = playerHealth;
    s.eyeHeight = eyeHeight;
    if (!audioCaptureStoreCycleSaveState(&s.audioState))
      s.audioState = AudioStoreCycleSaveState{};
    std::ofstream f(path, std::ios::binary);
    if (!f)
      return;
    f.write(reinterpret_cast<const char*>(&s), sizeof(s));
    if (!f)
      return;
    saveInventoryStateToSlot(activeSaveSlot);
    writeLastSaveSlotMeta();
  }

  void tickPlayerDeathScene(float dt) {
    if (!playerDeathActive || playerDeathShowMenu || !std::isfinite(dt) || dt < 1e-8f)
      return;
    if (playerDeathPlayingFallClip) {
      if (playerDeathClipIndex < 0 || static_cast<size_t>(playerDeathClipIndex) >= staffRig.clips.size()) {
        playerDeathPlayingFallClip = false;
        playerDeathHoldRemain = kPlayerDeathHoldBeforeRespawnSec;
        return;
      }
      const double dur = staff_skin::clipDuration(staffRig, playerDeathClipIndex);
      const double endT = dur * static_cast<double>(playerDeathClipFracEnd);
      playerDeathAnimTime += static_cast<double>(dt);
      if (playerDeathAnimTime >= endT) {
        playerDeathAnimTime = endT;
        playerDeathPlayingFallClip = false;
        playerDeathHoldRemain = kPlayerDeathHoldBeforeRespawnSec;
      }
    } else {
      playerDeathHoldRemain -= dt;
      if (playerDeathHoldRemain <= 0.f) {
        playerDeathShowMenu = true;
        syncInputGrab();
      }
    }
  }

  void update(float dt) {
    PerfScopeTimer _updT(&gPerfTrace.accumUpdate, gPerfTrace.enabled);
    audioUpdateStore(dt);
    if (!std::isfinite(dt) || dt < 1e-5f)
      dt = 1.f / static_cast<float>(kTargetFps);
    const bool uiMenuFreeze =
        (showPauseMenu || showInventoryMenu || showControlsOverlay || inTitleMenu) && !playerDeathActive;
    if (!uiMenuFreeze) {
      tickDeliPizzaReplenish(dt);
      playerHunger = std::max(0.f, playerHunger - dt * kPlayerHungerDrainPerSec);
      if (crosshairShoveAnimRemain > 0.f)
        crosshairShoveAnimRemain = std::max(0.f, crosshairShoveAnimRemain - dt);
      if (playerPushAnimRemain > 0.f)
        playerPushAnimRemain = std::max(0.f, playerPushAnimRemain - dt * kPushAnimPlaybackScale);
      if (playerKickAnimRemain > 0.f)
        playerKickAnimRemain = std::max(0.f, playerKickAnimRemain - dt * kKickAnimPlaybackScale);
      if (dropKickActive) {
        dropKickTimer = std::max(0.f, dropKickTimer - dt);
        if (!dropKickHitApplied && staffSkinnedActive &&
            (staffClipMeleeFall >= 0 || staffClipShoveHair >= 0)) {
          const glm::vec2 pXZ(camPos.x, camPos.z);
          bool hitAny = false;
          for (uint32_t si : shelfEmpActiveSlots) {
            ShelfEmployeeNpc& npc = shelfEmpPool[si];
            if (!npc.inited)
              continue;
            if (npc.meleeState == 2 || npc.meleeState == 3 || npc.meleeState == 4)
              continue;
            const glm::vec2 toNpc = npc.posXZ - pXZ;
            const float dist = glm::length(toNpc);
            if (dist > kDropKickMaxDist || dist < 1e-4f)
              continue;
            const glm::vec2 toN = toNpc * (1.f / dist);
            if (glm::dot(dropKickDir, toN) < kDropKickCosCone)
              continue;
            if (applyStaffDamageFromPlayerHit(npc.residentKey, npc, kStaffHitDamageDropKick)) {
              applyStaffShoveKnockdown(npc.residentKey, npc, toN);
              npc.staffShoveKnockbackVelXZ = dropKickDir * kDropKickKnockbackSpeed;
              npc.meleeState = 2;
              npc.meleePhaseSec = 0.0;
              npc.staffVelY = kDropKickNpcVerticalPop;
            } else {
              npc.staffShoveKnockbackVelXZ = dropKickDir * (kDropKickKnockbackSpeed * 0.62f);
              audioPlayStaffMeleeImpact();
            }
            hitAny = true;
          }
          if (hitAny) {
            dropKickHitApplied = true;
          }
        }
        if (isGrounded() && dropKickTimer > kDropKickGroundSlideSec)
          dropKickTimer = kDropKickGroundSlideSec;
        if (dropKickTimer <= 0.f) {
          dropKickActive = false;
          dropKickTimer = 0.f;
        }
      }
      if (playerJumpPostLandRemain > 0.f)
        playerJumpPostLandRemain = std::max(0.f, playerJumpPostLandRemain - dt);
      if (playerJumpPostLandRemain <= 0.f) {
        playerJumpPostLandClipIndex = -1;
        playerJumpPostLandDurationInit = 0.f;
        playerJumpPostLandSecondHalfScrub = false;
      }
      if (playerFallDamageChainImmuneRemain > 0.f)
        playerFallDamageChainImmuneRemain = std::max(0.f, playerFallDamageChainImmuneRemain - dt);
      if (playerStaffBodySlamCooldownRem > 0.f)
        playerStaffBodySlamCooldownRem = std::max(0.f, playerStaffBodySlamCooldownRem - dt);
      if (playerScreenDamagePulse > 0.f)
        playerScreenDamagePulse =
            std::max(0.f, playerScreenDamagePulse - dt * kPlayerScreenDamagePulseDecayPerSec);
      const int jumpRateClip =
          (playerJumpRunTailActive && avClipJumpRun >= 0) ? avClipJumpRun : avClipJump;
      if (playerJumpAnimRemain > 0.f && jumpRateClip >= 0) {
        const double durJ = staff_skin::clipDuration(staffRig, jumpRateClip);
        float rate;
        if (playerJumpLedgeSecondHalfAir) {
          const float secondHalfDur =
              static_cast<float>(durJ * (1.0 - static_cast<double>(kJumpClipLedgeFirstHalfFrac)));
          rate = secondHalfDur / std::max(kJumpLedgePreLandLeadSec, 0.06f);
          rate = glm::clamp(rate, 0.4f, 3.2f) * kJumpAnimPlaybackScale;
        } else {
          const float tPred = std::max(
              playerJumpAirTimeTargetSec > 0.05f ? playerJumpAirTimeTargetSec : kJumpPredictedAirTimeSec,
              0.12f);
          rate = glm::clamp(static_cast<float>(durJ) / tPred, 0.52f, 2.28f) * kJumpAnimPlaybackScale;
          if (playerJumpRunTailActive)
            rate *= kJumpRunTailRateScale;
          if (playerVaultCrateJumpActive)
            rate *= kVaultJumpAnimRateScale;
        }
        playerJumpAnimRemain = std::max(0.f, playerJumpAnimRemain - dt * rate);
        if (playerJumpAnimRemain <= 1e-4f)
          playerJumpLedgeSecondHalfAir = false;
      } else if (playerJumpAnimRemain > 0.f)
        playerJumpAnimRemain = std::max(0.f, playerJumpAnimRemain - dt * kJumpAnimPlaybackScale);
      if (playerAvatarJumpFallMidPose() && playerPreFallAnimRemain <= 1e-4f &&
          velY < kAvatarJumpFallVelYThr)
        playerFallAnimTime += dt;
      else if (isGrounded())
        playerFallAnimTime = 0.f;
    }
    if (!playerDeathActive && !showPauseMenu && !showInventoryMenu && !inTitleMenu && !inLoadingScreen) {
      staffSimTime += dt;
      processPendingStaffShove();
      processPendingPlayerKick();
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
      const char* shrekAuto = std::getenv("VULKAN_GAME_SHREK_AUTOSPAWN");
      if (!shrekEggDidAutoSpawnOnce && shrekEggAssetLoaded && staffSkinnedActive && shrekAuto != nullptr &&
          shrekAuto[0] == '1' && shrekAuto[1] == '\0') {
        spawnShrekEggNearPlayer();
        shrekEggDidAutoSpawnOnce = true;
      }
      updateShrekEggEaster(dt);
#endif
      updateShelfEmployees(dt);
    }
    if (inIntroSplash) {
      introSplashTime += dt;
      constexpr float kIntroDuration = 17.0f;
      if (introSplashTime >= kIntroDuration) {
        inIntroSplash = false;
        inTitleMenu = true;
        titleMenuSceneTime = 0.f;
        titleMenuSlideTime = 0.f;
        titleMenuSlideWasSlot = false;
      }
      rebuildTerrainIfNeeded();
      return;
    }
    if (inLoadingScreen) {
      loadingScreenTime += dt;
      constexpr float kLoadDurationExisting = 8.0f;
      constexpr float kLoadDurationNewSave  = 14.0f;
      const float kLoadDuration = loadingScreenIsNewSave ? kLoadDurationNewSave : kLoadDurationExisting;
      if (loadingScreenTime >= kLoadDuration) {
        finishLoadFromSlot(loadingScreenSlot);
      }
      rebuildTerrainIfNeeded();
      return;
    }
    if ((showPauseMenu || showInventoryMenu || inTitleMenu) && !playerDeathActive) {
      audioSetLowHealthHeartbeat(playerHealth > 0.f && playerHealth < kPlayerHealthMercyCap);
      tickPlayerDeathScene(dt);
      if (inTitleMenu) {
        titleMenuSceneTime += dt;
        if (titleMenuSlideWasSlot != titleMenuPickSlot) {
          titleMenuSlideTime = 0.f;
          titleMenuSlideWasSlot = titleMenuPickSlot;
        }
        titleMenuSlideTime += dt;
        staffSimTime += dt;
        syncTitleMenuSceneAnchor();
      }
      rebuildTerrainIfNeeded();
      return;
    }
    if (!playerDeathActive && !showPauseMenu && !showInventoryMenu && !inTitleMenu) {
      autoSaveAccumSec += dt;
      constexpr float kAutoSaveIntervalSec = 45.f;
      if (autoSaveAccumSec >= kAutoSaveIntervalSec) {
        autoSaveAccumSec = 0.f;
        gameSaveWrite();
      }
    }
    if (newGameControlsPopupDelay >= 0.f && !showControlsOverlay && !showPauseMenu &&
        !showInventoryMenu && !playerDeathActive && !inTitleMenu) {
      newGameControlsPopupDelay -= dt;
      if (newGameControlsPopupDelay <= 0.f) {
        newGameControlsPopupDelay = -1.f;
        showControlsOverlay = true;
        audioSetStoreDayNightCyclePaused(true);
        mouseGrab = false;
        syncInputGrab();
      }
    }
    SDL_PumpEvents();
    const Uint8* keys = SDL_GetKeyboardState(nullptr);
    const auto down = [keys, this](SDL_Scancode sc) -> bool {
      const int i = static_cast<int>(sc);
      if (i < 0 || i >= SDL_NUM_SCANCODES)
        return keys[sc] != 0;
      return keys[sc] != 0 || scancodeDown[static_cast<size_t>(sc)];
    };
    int mx = 0, my = 0;
    if (mouseGrab && !showControlsOverlay && !showPauseMenu && !showInventoryMenu && !inTitleMenu) {
      SDL_GetRelativeMouseState(&mx, &my);
      if (playerDeathActive)
        mx = my = 0;
    }
    const float sens = 0.0022f;
    if (!showControlsOverlay && !playerDeathActive && !showPauseMenu && !showInventoryMenu &&
        !inTitleMenu) {
      yaw += static_cast<float>(mx) * sens;
      pitch -= static_cast<float>(my) * sens;
      // Keyboard look if the mouse is ignored (Wayland / bad relative-mode drivers).
      const float lk = 2.4f * dt;
      if (down(SDL_SCANCODE_J))
        yaw -= lk;
      if (down(SDL_SCANCODE_L))
        yaw += lk;
      if (down(SDL_SCANCODE_I))
        pitch += lk;
      if (down(SDL_SCANCODE_K))
        pitch -= lk;
      pitch = std::clamp(pitch, kPitchMaxLookDown, kPitchMaxLookUp);
      if (!thirdPersonTestMode &&
          (ledgeHangActive || ledgeClimbT >= 0.f || ladderClimbActive || wallClimbActive)) {
        float climbAnchorYaw = yaw;
        bool haveClimbAnchor = false;
        if (ledgeHangActive) {
          climbAnchorYaw = ledgeHangTargetYaw;
          haveClimbAnchor = true;
        } else if (ledgeClimbT >= 0.f && glm::length(ledgeClimbExitHoriz) > 1e-4f) {
          // Face into the ledge while climbing up (exit vector points out from ledge face).
          climbAnchorYaw = std::atan2(-ledgeClimbExitHoriz.y, -ledgeClimbExitHoriz.x);
          haveClimbAnchor = true;
        } else if (ladderClimbActive && glm::length(ladderClimbExitHoriz) > 1e-4f) {
          // Ladder climb uses the same "exit points outward" convention.
          climbAnchorYaw = std::atan2(-ladderClimbExitHoriz.y, -ladderClimbExitHoriz.x);
          haveClimbAnchor = true;
        } else if (wallClimbActive) {
          const glm::vec2 nXZ(wallClimbNormal.x, wallClimbNormal.z);
          if (glm::length(nXZ) > 1e-4f) {
            // Wall normal points away from wall; face into wall during climb.
            climbAnchorYaw = std::atan2(-nXZ.y, -nXZ.x);
            haveClimbAnchor = true;
          }
        }
        if (haveClimbAnchor) {
          const float yawDelta = wrapAnglePi(yaw - climbAnchorYaw);
          yaw = wrapAnglePi(climbAnchorYaw +
                            std::clamp(yawDelta, -kClimbLookYawClamp, kClimbLookYawClamp));
        }
        pitch = std::clamp(pitch, std::max(kPitchMaxLookDown, kClimbLookPitchDown),
                           std::min(kPitchMaxLookUp, kClimbLookPitchUp));
      }
    }

    if (thirdPersonTestMode && !playerDeathActive && !showPauseMenu && !showInventoryMenu &&
        !inTitleMenu) {
      constexpr float kTpZoomMin = 1.15f;
      constexpr float kTpZoomMax = 14.f;
      if (down(SDL_SCANCODE_LEFTBRACKET))
        thirdPersonCamDist = glm::clamp(thirdPersonCamDist - 3.2f * dt, kTpZoomMin, kTpZoomMax);
      if (down(SDL_SCANCODE_RIGHTBRACKET))
        thirdPersonCamDist = glm::clamp(thirdPersonCamDist + 3.2f * dt, kTpZoomMin, kTpZoomMax);
    }

    const glm::vec3 forward{std::cos(yaw), 0.f, std::sin(yaw)};
    const float feetBeforeCrouch = camPos.y - eyeHeight;
    const float supStart = playerTerrainSupportY(camPos.x, camPos.z, feetBeforeCrouch);
    const bool groundedStart = isGroundedUsingSupport(supStart);
    const bool crouchHeldRaw = down(SDL_SCANCODE_C);
    const bool slideEdge = pendingSlideCrouchEdge;
    pendingSlideCrouchEdge = false;
    const bool slideInput =
        slideEdge && crouchHeldRaw && inputForward(down) && inputSprintHeld(down);
    if (slideCooldownTimer > 0.f)
      slideCooldownTimer -= dt;
    if (wallRunCooldownTimer > 0.f)
      wallRunCooldownTimer = std::max(0.f, wallRunCooldownTimer - dt);
    if (wallClimbCooldownTimer > 0.f)
      wallClimbCooldownTimer = std::max(0.f, wallClimbCooldownTimer - dt);
    if (wallJumpCooldownTimer > 0.f)
      wallJumpCooldownTimer = std::max(0.f, wallJumpCooldownTimer - dt);
    if (wallRunActive)
      wallRunTimer = std::max(0.f, wallRunTimer - dt);
    if (wallClimbActive)
      wallClimbTimer = std::max(0.f, wallClimbTimer - dt);
    if (speedVaultActive)
      speedVaultTimer = std::max(0.f, speedVaultTimer - dt);
    if (slideClearClipNextFrame) {
      slideAnimClip = -1;
      slideAnimElapsed = 0.f;
      slideAnimDurSec = 0.f;
      slideStartSpeed = 0.f;
      slideClearClipNextFrame = false;
    }
    if (slideActive && slideAnimClip < 0) {
      slideTimer -= dt;
      if (slideTimer <= 0.f)
        slideActive = false;
    }
    const bool crouchHeld = crouchHeldRaw || slideActive;
    float targetEyeHeight = crouchHeld ? kCrouchEyeHeight : kEyeHeight;
    if (!crouchHeld) {
      const float standHeadY = feetBeforeCrouch + kEyeHeight + 0.12f;
      if (standHeadY >= kCeilingY - 0.08f)
        targetEyeHeight = kCrouchEyeHeight;
    }
    eyeHeight = glm::mix(eyeHeight, targetEyeHeight, 1.f - std::exp(-dt * kCrouchTransitionRate));
    camPos.y = feetBeforeCrouch + eyeHeight;
    // Pin feet to queried support while grounded (crouch/slide eye lerp + collision can leave a few mm drift).
    if (groundedStart && velY <= 0.18f) {
      const float feetPin = camPos.y - eyeHeight;
      const float supPin = playerTerrainSupportY(camPos.x, camPos.z, feetPin);
      if (isGroundedUsingSupport(supPin) && feetPin > supPin + 2e-4f)
        camPos.y = supPin + eyeHeight;
    }
    if (groundedStart) {
      const bool onShelfOrCrate =
          playerFallSupportIsShelfDeckOrCrateTop(camPos.x, camPos.z, supStart);
      coyoteTime = onShelfOrCrate ? kCoyoteTimeLedge : kCoyoteTime;
      if (playerDepthJumpWindowRemain > 0.f)
        playerDepthJumpWindowRemain = std::max(0.f, playerDepthJumpWindowRemain - dt);
    } else
      coyoteTime -= dt;

    if (!playerDeathActive && down(SDL_SCANCODE_SPACE) && !spaceWasDown)
      jumpBuffer = kJumpBufferTime;
    else
      jumpBuffer -= dt;
    jumpBuffer = std::max(0.f, jumpBuffer);
    if (playerJumpMinIntervalRem > 0.f)
      playerJumpMinIntervalRem = std::max(0.f, playerJumpMinIntervalRem - dt);
    if (playerJumpRepeatWindowRem > 0.f)
      playerJumpRepeatWindowRem = std::max(0.f, playerJumpRepeatWindowRem - dt);

    if (playerDanceEmoteStopGraceRemain > 0.f)
      playerDanceEmoteStopGraceRemain = std::max(0.f, playerDanceEmoteStopGraceRemain - dt);

    if (kPlayerLedgeMantleEnabled && ledgeClimbT >= 0.f) {
      if (down(SDL_SCANCODE_X)) {
        ledgeClimbT = -1.f;
        ledgeClimbVisPhase = 0.f;
      } else
        advanceLedgeClimb(dt);
    } else if (!kPlayerLedgeMantleEnabled && ledgeClimbT >= 0.f) {
      ledgeClimbT = -1.f;
      ledgeClimbVisPhase = 0.f;
      ledgeClimbApproachVel = glm::vec2(0.f);
    }

    if (ledgeHangActive) {
      velY = 0.f;
      horizVel = glm::vec2(0.f);
      if (ledgeHangGrabTimer > 0.f) {
        ledgeHangGrabTimer = std::max(0.f, ledgeHangGrabTimer - dt);
        const float grabT = 1.f - ledgeHangGrabTimer / kLedgeHangGrabDuration;
        const float ease = grabT * grabT * (3.f - 2.f * grabT);
        camPos = glm::mix(ledgeHangStartCam, ledgeHangCamPos, ease);
      } else {
        camPos = ledgeHangCamPos;
      }
      // Safety: if feet are already supported (crate/deck/floor), leave hang so clip/state doesn't stick.
      if (ledgeHangGrabTimer <= 0.f) {
        const float feetHang = camPos.y - eyeHeight;
        const float supHang = playerTerrainSupportY(camPos.x, camPos.z, feetHang);
        if (isGroundedUsingSupport(supHang)) {
          ledgeHangActive = false;
          ledgeHangFeetYCaptured = false;
          ledgeHangShimmyDir = 0;
          ledgeHangApproachVel = glm::vec2(0.f);
        }
      }
      if (ledgeHangActive && down(SDL_SCANCODE_SPACE) && inputBack(down) && ledgeHangGrabTimer <= 0.f) {
        ledgeHangActive = false;
        ledgeHangFeetYCaptured = false;
        velY = kLedgeHangJumpAwayVelY;
        if (glm::length(ledgeHangExitHoriz) > 1e-4f)
          horizVel = glm::normalize(ledgeHangExitHoriz) * kLedgeHangJumpAwaySpeed;
        jumpBuffer = 0.f;
        coyoteTime = 0.f;
      } else if (ledgeHangActive && down(SDL_SCANCODE_SPACE) && ledgeHangGrabTimer <= 0.f) {
        ledgeHangActive = false;
        ledgeHangFeetYCaptured = false;
        ledgeClimbStartCam = ledgeHangCamPos;
        ledgeClimbEndCam = ledgeHangEndCam;
        ledgeClimbExitHoriz = ledgeHangExitHoriz;
        ledgeClimbApproachVel = ledgeHangApproachVel;
        ledgeClimbT = 0.f;
        ledgeClimbVisPhase = 0.f;
        jumpBuffer = 0.f;
        coyoteTime = 0.f;
      } else if (ledgeHangActive && (down(SDL_SCANCODE_X) || down(SDL_SCANCODE_C)) && ledgeHangGrabTimer <= 0.f) {
        ledgeHangActive = false;
        ledgeHangFeetYCaptured = false;
        velY = kLedgeHangDropVelY;
        if (glm::length(ledgeHangExitHoriz) > 1e-4f)
          horizVel = -glm::normalize(ledgeHangExitHoriz) * 1.6f;
      } else if (ledgeHangActive && ledgeHangGrabTimer <= 0.f) {
        const bool strafeL = inputStrafeLeft(down);
        const bool strafeR = inputStrafeRight(down);
        // A / strafe-left → +dir → Meshy left climb (ledge_shimmy_left); D → right climb.
        // Only treat as shimmy when the deck clamp actually moves the hang point — at the strafe AABB edge,
        // hold strafe without advancing shimmy clip/phase (no treadmill against the limit).
        int wishShimmyDir = 0;
        if (strafeL && !strafeR)
          wishShimmyDir = 1;
        if (strafeR && !strafeL)
          wishShimmyDir = -1;
        ledgeHangShimmyDir = 0;
        if (wishShimmyDir != 0) {
          const glm::vec2 shimmyPerp(-ledgeHangExitHoriz.y, ledgeHangExitHoriz.x);
          const glm::vec2 move =
              shimmyPerp * (static_cast<float>(wishShimmyDir) * kLedgeHangShimmySpeed * dt);
          const float ox = ledgeHangEndCam.x;
          const float oz = ledgeHangEndCam.z;
          float nx = ox + move.x;
          float nz = oz + move.y;
          const float mx = kPlayerHalfXZ + 0.1f;
          nx = std::clamp(nx, ledgeHangDeckAABB.min.x + mx, ledgeHangDeckAABB.max.x - mx);
          nz = std::clamp(nz, ledgeHangDeckAABB.min.z + mx, ledgeHangDeckAABB.max.z - mx);
          constexpr float kLedgeHangShimmyMovedEps = 2e-4f;
          if (std::abs(nx - ox) > kLedgeHangShimmyMovedEps || std::abs(nz - oz) > kLedgeHangShimmyMovedEps) {
            ledgeHangShimmyDir = wishShimmyDir;
            ledgeHangEndCam.x = nx;
            ledgeHangEndCam.z = nz;
          }
        }
        auto snapHangToActiveDeckFace = [&](float pullBack) {
          const float mx = kPlayerHalfXZ + 0.1f;
          if (ledgeHangExitHoriz.x > 0.5f) {
            ledgeHangEndCam.x = ledgeHangDeckAABB.max.x - mx;
            ledgeHangEndCam.z =
                std::clamp(ledgeHangEndCam.z, ledgeHangDeckAABB.min.z + mx, ledgeHangDeckAABB.max.z - mx);
          } else if (ledgeHangExitHoriz.x < -0.5f) {
            ledgeHangEndCam.x = ledgeHangDeckAABB.min.x + mx;
            ledgeHangEndCam.z =
                std::clamp(ledgeHangEndCam.z, ledgeHangDeckAABB.min.z + mx, ledgeHangDeckAABB.max.z - mx);
          } else if (ledgeHangExitHoriz.y > 0.5f) {
            ledgeHangEndCam.z = ledgeHangDeckAABB.max.z - mx;
            ledgeHangEndCam.x =
                std::clamp(ledgeHangEndCam.x, ledgeHangDeckAABB.min.x + mx, ledgeHangDeckAABB.max.x - mx);
          } else if (ledgeHangExitHoriz.y < -0.5f) {
            ledgeHangEndCam.z = ledgeHangDeckAABB.min.z + mx;
            ledgeHangEndCam.x =
                std::clamp(ledgeHangEndCam.x, ledgeHangDeckAABB.min.x + mx, ledgeHangDeckAABB.max.x - mx);
          }
          ledgeHangCamPos.x = ledgeHangEndCam.x + ledgeHangExitHoriz.x * pullBack;
          ledgeHangCamPos.z = ledgeHangEndCam.z + ledgeHangExitHoriz.y * pullBack;
          if (ledgeHangExitHoriz.x > 0.5f)
            ledgeHangCamPos.x = std::max(ledgeHangCamPos.x, ledgeHangDeckAABB.max.x + kLedgeHangOutsideClearance);
          else if (ledgeHangExitHoriz.x < -0.5f)
            ledgeHangCamPos.x = std::min(ledgeHangCamPos.x, ledgeHangDeckAABB.min.x - kLedgeHangOutsideClearance);
          if (ledgeHangExitHoriz.y > 0.5f)
            ledgeHangCamPos.z = std::max(ledgeHangCamPos.z, ledgeHangDeckAABB.max.z + kLedgeHangOutsideClearance);
          else if (ledgeHangExitHoriz.y < -0.5f)
            ledgeHangCamPos.z = std::min(ledgeHangCamPos.z, ledgeHangDeckAABB.min.z - kLedgeHangOutsideClearance);
        };
        const float pullBack = (ledgeHangShimmyDir != 0)
            ? (kLedgeHangPullBack - kLedgeHangShimmyPullBackTowardLedge)
            : kLedgeHangPullBack;
        snapHangToActiveDeckFace(pullBack);
        camPos = ledgeHangCamPos;
        resolvePillarCollisions();
        if (camPos.x != ledgeHangCamPos.x || camPos.z != ledgeHangCamPos.z) {
          const float dx = camPos.x - ledgeHangCamPos.x;
          const float dz = camPos.z - ledgeHangCamPos.z;
          ledgeHangEndCam.x += dx;
          ledgeHangEndCam.z += dz;
          snapHangToActiveDeckFace(pullBack);
          camPos = ledgeHangCamPos;
        }
        if (ledgeHangShimmyDir != 0)
          ledgeHangShimmyPhase += dt * 2.5f;
        else
          ledgeHangShimmyPhase += dt * kLedgeHangIdleShimmyPhaseRate;
      }
    }
    if (ladderClimbActive && avClipLedgeClimb >= 0) {
      if (down(SDL_SCANCODE_X) || down(SDL_SCANCODE_SPACE)) {
        ladderClimbActive = false;
        ladderClimbT = 0.f;
        ladderClimbApproachVel = glm::vec2(0.f);
      } else
        advanceLadderClimb(dt, inputForward(down), inputBack(down));
    }

    if (ledgeClimbT < 0.f && !ledgeHangActive && !ladderClimbActive) {
    float velYPreGround = 0.f;
    bool jumped = false;
    float horizDist = 0.f;
    float feetFinalVert = 0.f;
    float terrainY = 0.f;
    bool groundedEnd = false;
    glm::vec2 wish(0.f, 0.f);
    bool hasWishInput = false;
    bool runGrounded = false;
    float sp = 0.f;
    if (!playerDeathActive) {
    glm::vec3 rightDir{std::cos(yaw + glm::half_pi<float>()), 0,
                         std::sin(yaw + glm::half_pi<float>())};
    if (inputForward(down))
      wish += glm::vec2(forward.x, forward.z);
    if (inputBack(down))
      wish -= glm::vec2(forward.x, forward.z);
    if (inputStrafeLeft(down))
      wish -= glm::vec2(rightDir.x, rightDir.z);
    if (inputStrafeRight(down))
      wish += glm::vec2(rightDir.x, rightDir.z);
    hasWishInput = glm::length(wish) > 1e-4f;
    if (playerDanceEmoteActive && !playerDeathActive &&
        (hasWishInput || slideActive ||
         (playerDanceEmoteStopGraceRemain <= 0.f &&
          (down(SDL_SCANCODE_SPACE) || crouchHeldRaw)))) {
      playerDanceEmoteActive = false;
      playerDanceEmoteStopGraceRemain = 0.f;
    }
    if (hasWishInput)
      wish = glm::normalize(wish);
    if (!playerDeathActive && !slideActive && groundedStart && slideCooldownTimer <= 0.f && slideInput) {
      glm::vec2 startDir = horizVel;
      if (glm::length(startDir) < 0.2f)
        startDir = glm::vec2(forward.x, forward.z);
      if (glm::length(startDir) > 1e-4f) {
        slideDir = glm::normalize(startDir);
        const float curSpeed = glm::length(horizVel);
        // Blend current speed into slide; cap at kSlideStartSpeed (sprint carries farther).
        const float slideSp0 = std::clamp(curSpeed * 0.68f + 8.0f * kMovementSpeedScale,
                                          11.0f * kMovementSpeedScale, kSlideStartSpeed);
        horizVel = slideDir * slideSp0;
        slideStartSpeed = slideSp0;
        slideAnimElapsed = 0.f;
        slideClearClipNextFrame = false;
        const glm::vec2 r(std::cos(yaw + glm::half_pi<float>()), std::sin(yaw + glm::half_pi<float>()));
        int clipPick = -1;
        if (avClipSlideRight >= 0 && glm::dot(slideDir, r) > 0.55f)
          clipPick = avClipSlideRight;
        else if (avClipSlideLight >= 0)
          clipPick = avClipSlideLight;
        slideAnimClip = clipPick;
        if (clipPick >= 0) {
          slideAnimDurSec = static_cast<float>(staff_skin::clipDuration(staffRig, clipPick));
          slideCooldownTimer = slideAnimDurSec + kSlideCooldown;
        } else {
          slideAnimDurSec = 0.f;
          slideTimer = kSlideDuration;
          slideCooldownTimer = kSlideDuration + kSlideCooldown;
        }
        slideActive = true;
        playerPreFallAnimRemain = 0.f;
        playerJumpAwaitPreLandSecondHalf = false;
        playerJumpLedgeSecondHalfAir = false;
      }
    }

    if (!slideActive && playerPreFallAnimRemain > 1e-6f)
      horizVel *= std::exp(-kPreFallHorizBrakePerSec * dt);

    const bool crouchedMove = eyeHeight < (kEyeHeight - 0.12f);
    runGrounded = groundedStart && inputSprintHeld(down) && !crouchedMove && !slideActive;
    const bool airSmallGapSlow = !groundedStart && playerAirWalkSmallGap;
    const float accel =
        groundedStart ? kWalkAccel * (runGrounded ? kSprintAccelMult : 1.f)
                      : (airSmallGapSlow ? kAirAccel * kAirWalkSmallGapAccelMul : kAirAccel);
    // Walk/sprint caps are not tied to fluorescents — blackout uses the same speeds as day.
    const float capWalk = kMaxSpeed;
    const float capSprint = kSprintSpeed;
    const float maxSp =
        groundedStart ? ((runGrounded ? capSprint : capWalk) * (crouchedMove ? kCrouchSpeedMult : 1.f))
                      : (airSmallGapSlow ? kAirSpeedCap * kAirWalkSmallGapMaxSpMul : kAirSpeedCap);
    glm::vec2 accel2(0.f);
    if (hasWishInput) {
      const float alongWish = glm::dot(horizVel, wish);
      const float headroom = std::max(0.f, maxSp - std::max(0.f, alongWish));
      const float headDen =
          maxSp * (groundedStart ? kAccelHeadroomFracGround : kAccelHeadroomFracAir) + 0.01f;
      const float minScale = groundedStart ? kAccelMinScaleGround : kAccelMinScaleAir;
      const float accelScale = std::clamp(headroom / headDen, minScale, 1.f);
      accel2 = wish * accel * accelScale;
    }
    // Slide: with Meshy clips, hold speed for exactly one clip length (matches anim once). Fallback:
    // decelerate until stop or timer.
    if (slideActive) {
      if (slideAnimClip >= 0) {
        if (!thirdPersonTestMode && slideAnimDurSec > 1e-4f) {
          const float u =
              glm::clamp(slideAnimElapsed / std::max(slideAnimDurSec, 1e-4f), 0.f, 1.f);
          const float spMul = glm::mix(1.0f, 0.72f, u);
          horizVel = slideDir * (slideStartSpeed * spMul);
        } else {
          horizVel = slideDir * slideStartSpeed;
        }
      } else {
        const float along = glm::dot(horizVel, slideDir);
        float slideSp = std::max(0.f, along - kSlideDecel * dt);
        if (slideSp < kSlideStopSpeed)
          slideActive = false;
        slideSp = std::max(slideSp, 0.f);
        horizVel = slideDir * slideSp;
      }
    } else if (groundedStart) {
      horizVel += accel2 * dt;
      const float fr = glm::length(horizVel);
      if (fr > 0.01f) {
        // Keep momentum while input is held; apply full braking only when input is released.
        const float frictionScale =
            hasWishInput ? (runGrounded ? kGroundFrictionScaleSprint : kGroundFrictionScaleWalk) : 1.0f;
        // Linear model: v *= (1 - min(1, k*dt)). When k*dt >= 1 (long frames / hitches), that zeroed
        // velocity after accel — walk felt dead while slide/jump still worked. Cap per-tick loss.
        const float reduce = std::min(kFriction * frictionScale * dt, 0.95f);
        horizVel *= (1.f - reduce);
      }
    } else {
      horizVel += accel2 * dt;
      horizVel *= std::exp(-kAirDrag * dt);
    }
    sp = glm::length(horizVel);
    if (!slideActive && sp > maxSp)
      horizVel *= maxSp / sp;

    const glm::vec2 horizDelta = horizVel * dt;
    horizDist = glm::length(horizDelta);
    const int moveSteps =
        std::max(1, std::min(10, static_cast<int>(std::ceil(horizDist / kMaxHorizMoveStep))));
    const glm::vec2 stepDelta = horizDelta / static_cast<float>(moveSteps);
    for (int i = 0; i < moveSteps; ++i) {
      camPos.x += stepDelta.x;
      camPos.z += stepDelta.y;
      resolvePillarCollisions(i + 1 == moveSteps);
    }

    const float feetAfterHoriz = camPos.y - eyeHeight;
    const float supMove = playerTerrainSupportY(camPos.x, camPos.z, feetAfterHoriz);
    const bool groundedMove = isGroundedUsingSupport(supMove);
    const glm::vec2 parkourMoveDir =
        hasWishInput ? wish
                     : (glm::length(horizVel) > 0.055f ? glm::normalize(horizVel)
                                                       : glm::vec2(forward.x, forward.z));
    ParkourWallHit wallHit{};
    const bool hasWallHit = !groundedMove && findNearbyWallSurface(parkourMoveDir, kWallProbeMaxDist, wallHit);
    // Do not cancel slide when groundedMove flickers false on shelf decks (thin probes / sub-steps);
    // slide ends via clip/timer, jump, or slide-stop logic below.

    if (spaceWasDown && !down(SDL_SCANCODE_SPACE) && velY > 0.2f)
      velY *= kJumpCutMult;
    spaceWasDown = down(SDL_SCANCODE_SPACE);

    velY -= kGravity * dt;
    if (dropKickActive) {
      if (dropKickTimer > kDropKickGroundSlideSec) {
        velY += kGravity * dt * (1.f - kDropKickGravityScale);
        horizVel = dropKickDir * kDropKickLungeSpeed;
      } else {
        horizVel *= std::exp(-kDropKickGroundSlideDecay * dt);
      }
    }
    jumped = false;
    bool vaultAssistThisFrame = false;
    glm::vec3 mantleProbeEnd{};
    const bool canSquatHere = groundedMove && velY <= 0.2f && !slideActive;
    const bool mantleProbe = canSquatHere && (down(SDL_SCANCODE_SPACE) || playerJumpSquatCharging ||
                                                jumpBuffer > 1e-5f);
    const bool squatFarTarget = mantleProbe && playerJumpSquatTargetIsFar(mantleProbeEnd);

    auto applyJumpCommon_ = [&]() {
      wallRunActive = false;
      wallRunTimer = 0.f;
      wallClimbActive = false;
      wallClimbTimer = 0.f;
      speedVaultActive = false;
      speedVaultTimer = 0.f;
      wallRunCameraTilt = 0.f;
      slideActive = false;
      slideAnimClip = -1;
      slideAnimElapsed = 0.f;
      slideAnimDurSec = 0.f;
      slideStartSpeed = 0.f;
      slideClearClipNextFrame = false;
      jumpBuffer = 0.f;
      coyoteTime = 0.f;
      bobOffsetY = 0.f;
      bobSideOffset = 0.f;
      walkPitchOsc = 0.f;
      bobPhase = std::fmod(bobPhase, glm::two_pi<float>());
      playerPreFallAnimRemain = 0.f;
      playerJumpAwaitPreLandSecondHalf = false;
      playerJumpLedgeSecondHalfAir = false;
      // Squat dip stacked into landingPitchOfs; without clearing, the view stays pitched down in the air
      // after charged (big) jumps.
      landingPitchOfs = 0.f;
    };

    const glm::vec2 wallNormalXZ =
        hasWallHit ? glm::normalize(glm::vec2(wallHit.normal.x, wallHit.normal.z)) : glm::vec2(0.f);
    glm::vec2 wallTangent(-wallNormalXZ.y, wallNormalXZ.x);
    if (hasWallHit && glm::dot(wallTangent, parkourMoveDir) < 0.f)
      wallTangent = -wallTangent;
    const float wallApproachDot = hasWallHit ? glm::dot(parkourMoveDir, -wallNormalXZ) : -1.f;
    const float wallParallelDot = hasWallHit ? std::abs(glm::dot(parkourMoveDir, wallTangent)) : 0.f;

    if (groundedMove) {
      wallRunActive = false;
      wallRunTimer = 0.f;
      wallClimbActive = false;
      wallClimbTimer = 0.f;
      wallRunCameraTilt = 0.f;
    }
    if (wallRunActive &&
        (groundedMove || wallRunTimer <= 1e-4f || !inputSprintHeld(down) || !hasWallHit ||
         wallHit.t > kWallRunWallDist || sp < kWallRunDetachSpeed)) {
      wallRunActive = false;
      wallRunTimer = 0.f;
      wallRunCooldownTimer = std::max(wallRunCooldownTimer, kWallRunCooldown);
    }
    if (wallClimbActive && (groundedMove || wallClimbTimer <= 1e-4f || !hasWallHit || wallApproachDot < 0.05f)) {
      wallClimbActive = false;
      wallClimbTimer = 0.f;
      wallClimbCooldownTimer = std::max(wallClimbCooldownTimer, kWallClimbCooldown);
    }
    if (speedVaultActive && speedVaultTimer <= 1e-4f)
      speedVaultActive = false;

    if (!speedVaultActive && groundedMove && runGrounded && !slideActive && !crouchedMove &&
        sp >= kSpeedVaultMinSpeed) {
      SpeedVaultTarget vaultTarget{};
      if (findSpeedVaultTarget(parkourMoveDir, vaultTarget)) {
        speedVaultActive = true;
        speedVaultTimer = kSpeedVaultDuration;
        speedVaultDir = vaultTarget.dir;
        speedVaultStartY = camPos.y;
        speedVaultTargetY = vaultTarget.topY + eyeHeight + 0.04f;
        horizVel = speedVaultDir * std::max(sp * kSpeedVaultSpeedRetain,
                                            kSpeedVaultMinSpeed * kSpeedVaultSpeedRetain);
        playerJumpRunTailActive = false;
        playerJumpPostLandClipIndex = -1;
        playerJumpPostLandDurationInit = 0.f;
        playerJumpPostLandSecondHalfScrub = false;
        playerJumpPostLandRemain = 0.f;
        playerJumpAwaitPreLandSecondHalf = false;
        playerJumpLedgeSecondHalfAir = false;
        const int speedVaultClip = (avClipJump >= 0) ? avClipJump : avClipJumpRun;
        playerJumpAnimRemain =
            speedVaultClip >= 0 ? static_cast<float>(staff_skin::clipDuration(staffRig, speedVaultClip)) : 0.f;
        playerJumpArchActive = true;
        playerJumpAirTimeTargetSec = glm::clamp(kSpeedVaultDuration, kJumpAirTimeTargetMinSec,
                                                kJumpAirTimeTargetMaxSec);
        playerVaultCrateJumpActive = true;
        playerJumpSquatCharging = false;
        playerJumpSquatCharge = 0.f;
        jumpBuffer = 0.f;
        coyoteTime = 0.f;
        landingPitchOfs = std::min(landingPitchOfs, -0.018f);
      }
    }

    if (!speedVaultActive && !groundedMove && !wallClimbActive && hasWallHit &&
        !wallHit.shelf &&
        jumpBuffer > 0.f &&
        inputSprintHeld(down) && wallClimbCooldownTimer <= 0.f && sp >= kWallClimbMinApproachSpeed &&
        wallApproachDot >= kWallClimbMinDotForward) {
      wallClimbActive = true;
      wallClimbTimer = kWallClimbMaxDuration;
      wallClimbNormal = wallHit.normal;
      wallClimbStartY = camPos.y;
      wallRunActive = false;
      wallRunTimer = 0.f;
      wallRunCameraTilt = 0.f;
      playerJumpSquatCharging = false;
      playerJumpSquatCharge = 0.f;
      jumpBuffer = 0.f;
      coyoteTime = 0.f;
    }

    if (!speedVaultActive && !wallClimbActive && !wallRunActive && !groundedMove && hasWallHit &&
        !wallHit.shelf &&
        inputSprintHeld(down) && wallRunCooldownTimer <= 0.f && sp >= kWallRunMinSpeed &&
        wallHit.t <= kWallRunWallDist && wallParallelDot >= 0.45f &&
        wallHit.box.max.y >= feetAfterHoriz + kWallRunMinHeight) {
      wallRunActive = true;
      wallRunTimer = kWallRunMaxDuration;
      wallRunNormal = wallHit.normal;
      wallRunDir = glm::vec3(wallTangent.x, 0.f, wallTangent.y);
      wallRunSide = wallHit.side;
      wallRunCameraTilt = static_cast<float>(wallRunSide) * kWallRunCameraTiltRad;
      wallClimbActive = false;
      wallClimbTimer = 0.f;
      velY = std::max(velY, kWallRunUpForce * 0.3f);
    }

    if (!speedVaultActive && !groundedMove && hasWallHit && jumpBuffer > 0.f && wallJumpCooldownTimer <= 0.f &&
        !wallClimbActive && inputSprintHeld(down)) {
      applyJumpCommon_();
      wallJumpCooldownTimer = kWallJumpCooldown;
      wallRunCooldownTimer = std::max(wallRunCooldownTimer, kWallRunCooldown * 0.7f);
      wallClimbCooldownTimer = std::max(wallClimbCooldownTimer, kWallClimbCooldown);
      velY = std::max(kWallJumpVelY, velY * kWallJumpUpMul);
      horizVel = wallTangent * std::max(sp * 0.35f, 1.1f) + (-wallNormalXZ) * kWallJumpAwaySpeed;
      jumped = true;
      vaultAssistThisFrame = false;
    }

    // Commit jump-squat on Space release even if mantle probe flickers this frame.
    if (playerJumpSquatCharging && canSquatHere && !down(SDL_SCANCODE_SPACE)) {
      const float uEase = 1.f - std::pow(1.f - playerJumpSquatCharge, 2.05f);
      float vJump = kJumpVel * glm::mix(kJumpSquatVelMinMul, kJumpSquatVelMaxMul, uEase);
      if (playerDepthJumpWindowRemain > 0.f) {
        vJump += kDepthJumpBonusVelY;
        playerDepthJumpWindowRemain = 0.f;
      }
      velY = vJump;
      horizVel *= kJumpHorizCarry;
      if (groundedMove && playerFallSupportIsShelfDeckOrCrateTop(camPos.x, camPos.z, supMove))
        horizVel *= kJumpFromLedgeHorizMul;
      applyJumpCommon_();
      jumped = true;
      vaultAssistThisFrame = false;
      playerJumpSquatCharging = false;
      playerJumpSquatCharge = 0.f;
    } else {
      if (!canSquatHere) {
        playerJumpSquatCharging = false;
        playerJumpSquatCharge = 0.f;
      } else if (squatFarTarget && down(SDL_SCANCODE_SPACE)) {
        if (!playerJumpSquatCharging)
          playerJumpSquatCharge = 0.f;
        playerJumpSquatCharging = true;
        playerJumpSquatCharge =
            std::min(1.f, playerJumpSquatCharge + dt / std::max(1e-4f, kJumpSquatFullChargeSec));
      } else if (!squatFarTarget) {
        playerJumpSquatCharging = false;
        playerJumpSquatCharge = 0.f;
      }
    }
    const bool blockInstantJumpForSquat = squatFarTarget && canSquatHere && down(SDL_SCANCODE_SPACE);
    // Walking off a deck: horizontal move can leave groundedMove false while coyote is still hot — that
    // fired a full jump and blocked small-gap walk. Skip coyote jump this tick when the next deck is a
    // short hop ahead (same fix as mantle skip).
    const bool lostGroundFromHorizStep = groundedStart && !groundedMove;
    const glm::vec2 coyoteFwd(std::cos(yaw), std::sin(yaw));
    const glm::vec2 coyoteVelXZ =
        hasWishInput ? wish
                     : (glm::length(horizVel) > 0.055f ? glm::normalize(horizVel) : coyoteFwd);
    const float coyoteHopDrop =
        lostGroundFromHorizStep
            ? playerWalkOffEffectiveWalkableGapDrop(supStart, camPos.x, camPos.z, coyoteVelXZ, coyoteFwd)
            : 1e30f;
    const bool blockCoyoteShelfStride =
        lostGroundFromHorizStep && coyoteHopDrop <= kPlayerWalkOffSmallGapMaxDropM * 1.22f;
    if (!jumped && !blockInstantJumpForSquat && jumpBuffer > 0.f && playerJumpMinIntervalRem <= 0.f &&
        groundedMove && velY <= 0.2f && ledgeClimbT < 0.f && !ledgeHangActive && !ladderClimbActive &&
        !slideActive && avClipLedgeClimb >= 0) {
      if (tryStartLadderClimb()) {
        jumpBuffer = 0.f;
        coyoteTime = 0.f;
        playerJumpSquatCharging = false;
        playerJumpSquatCharge = 0.f;
      }
    }
    if (!jumped && !blockInstantJumpForSquat && jumpBuffer > 0.f && playerJumpMinIntervalRem <= 0.f &&
        (groundedMove || (coyoteTime > 0.f && !blockCoyoteShelfStride)) && velY <= 0.2f) {
      float vaultFwdBoost = 0.f;
      float vaultVyBoost = 0.f;
      const bool vaultAssist =
          groundedMove && computeVaultCrateJumpAssist(wish, hasWishInput, vaultFwdBoost, vaultVyBoost);
      float vJump = kJumpVel;
      if (playerDepthJumpWindowRemain > 0.f) {
        vJump += kDepthJumpBonusVelY;
        playerDepthJumpWindowRemain = 0.f;
      }
      if (vaultAssist)
        vJump += vaultVyBoost;
      velY = vJump;
      horizVel *= kJumpHorizCarry;
      const bool onShelfOrCrateTop =
          groundedMove && playerFallSupportIsShelfDeckOrCrateTop(camPos.x, camPos.z, supMove);
      if (onShelfOrCrateTop && !vaultAssist)
        horizVel *= kJumpFromLedgeHorizMul;
      if (vaultAssist) {
        const glm::vec2 fj{std::cos(yaw), std::sin(yaw)};
        horizVel += fj * vaultFwdBoost;
      }
      if (playerJumpRepeatWindowRem > 1e-4f && !vaultAssist) {
        horizVel *= kJumpRepeatCarryMul;
        if (inputForward(down))
          horizVel *= kJumpRepeatForwardMul;
      }
      applyJumpCommon_();
      playerJumpMinIntervalRem = kJumpMinIntervalSec;
      playerJumpRepeatWindowRem = kJumpRepeatNerfWindowSec;
      jumped = true;
      vaultAssistThisFrame = vaultAssist;
    }
    if (jumped) {
      // Match syncPlayerAvatarClip sprint + moving threshold (kLocoEnterSpeed).
      constexpr float kRunJumpMinHorizSp = 0.118f;
      const bool runJumpAnim =
          runGrounded && sp > kRunJumpMinHorizSp && avClipJumpRun >= 0;
      playerJumpRunTailActive = runJumpAnim;
      playerJumpPostLandClipIndex = -1;
      playerJumpPostLandDurationInit = 0.f;
      playerJumpPostLandSecondHalfScrub = false;
      if (runJumpAnim)
        playerJumpAnimRemain = static_cast<float>(staff_skin::clipDuration(staffRig, avClipJumpRun));
      else if (avClipJump >= 0)
        playerJumpAnimRemain = static_cast<float>(staff_skin::clipDuration(staffRig, avClipJump));
      else
        playerJumpAnimRemain = 0.f;
      playerJumpArchActive = true;
      playerJumpAirTimeTargetSec =
          glm::clamp(2.f * velY / kGravity, kJumpAirTimeTargetMinSec, kJumpAirTimeTargetMaxSec);
      playerJumpPostLandRemain = 0.f;
      playerAirWalkSmallGap = false;
      playerJumpAwaitPreLandSecondHalf = false;
      playerJumpLedgeSecondHalfAir = false;
      playerVaultCrateJumpActive = vaultAssistThisFrame;
    }

    if (wallRunActive) {
      if (hasWallHit) {
        wallRunNormal = wallHit.normal;
        wallRunSide = wallHit.side;
        wallRunDir = glm::vec3(wallTangent.x, 0.f, wallTangent.y);
      }
      const glm::vec2 runDir(wallRunDir.x, wallRunDir.z);
      const float runSpeed = std::max(glm::length(horizVel), kWallRunMinSpeed);
      horizVel = runDir * runSpeed;
      velY += kGravity * dt * (1.f - kWallRunGravityScale);
      velY += kWallRunUpForce * dt;
      velY = std::max(velY, -1.2f);
    }
    if (wallClimbActive) {
      wallRunActive = false;
      wallRunTimer = 0.f;
      wallRunCameraTilt = 0.f;
      horizVel *= std::exp(-dt * 8.5f);
      velY = kWallClimbSpeed;
      landingPitchOfs = std::max(landingPitchOfs, 0.006f);
    }
    if (speedVaultActive) {
      const float u = 1.f - std::clamp(speedVaultTimer / std::max(kSpeedVaultDuration, 1e-4f), 0.f, 1.f);
      const float yBase = glm::mix(speedVaultStartY, speedVaultTargetY, u);
      camPos.y = std::max(camPos.y, yBase + std::sin(u * glm::pi<float>()) * kSpeedVaultArcHeight);
      horizVel = speedVaultDir * std::max(glm::length(horizVel), kSpeedVaultMinSpeed * kSpeedVaultSpeedRetain);
      velY = std::max(0.f, kSpeedVaultUpBoost * (1.f - u) * 0.15f);
    }

    camPos.y += velY * dt;

    const AABB playerBeforeBump = playerCollisionBox();
    const bool allowStaffSlam =
        playerStaffBodySlamCooldownRem <= 0.f &&
        (playerTrackingAirFall || velY < -0.38f) && velY <= kPlayerStaffBodySlamMaxVelY;
    resolvePillarCollisions(true, allowStaffSlam);
    if (velY > 0.f) {
      constexpr float kHeadBumpEps = 0.004f;
      const AABB playerAfterBump = playerCollisionBox();
      if (playerAfterBump.max.y < playerBeforeBump.max.y - kHeadBumpEps)
        velY = 0.f;
    }

    velYPreGround = velY;
    const float feetY = camPos.y - eyeHeight;
    terrainY = playerTerrainSupportY(camPos.x, camPos.z, feetY);
    // Do not snap to floor while moving up: with small dt, delta-y can stay within this band for
    // several frames and would cancel jump velocity without changing velY.
    if (feetY < terrainY) {
      camPos.y = terrainY + eyeHeight;
      if (velY < 0.0f)
        velY = 0.0f;
    } else if (feetY <= terrainY + kFeetSnapDownSlop && velY <= 0.04f) {
      // Allow tiny upward vel (FP / collision resolution); strict <= 0 left a 1–3 cm hover mid-crouch/slide.
      camPos.y = terrainY + eyeHeight;
      if (velY < 0.0f)
        velY = 0.0f;
    }

    const float headY = camPos.y + 0.12f;
    if (headY >= kCeilingY - 0.08f && velY > 0.0f) {
      camPos.y = kCeilingY - 0.12f - 0.08f;
      velY = 0.0f;
    }

    const float feetAfterVert = camPos.y - eyeHeight;
    if (std::abs(feetAfterVert - feetY) > 1e-5f)
      terrainY = playerTerrainSupportY(camPos.x, camPos.z, feetAfterVert);
    // If still within grounded band above support, glue feet down (covers cases snap missed when velY was > 0).
    if (feetAfterVert > terrainY &&
        feetAfterVert <= terrainY + kGroundedFeetAboveSupport + 0.045f && velY <= 0.12f) {
      camPos.y = terrainY + eyeHeight;
      velY = 0.f;
    }
    feetFinalVert = camPos.y - eyeHeight;
    terrainY = playerTerrainSupportY(camPos.x, camPos.z, feetFinalVert);
    groundedEnd = isGroundedUsingSupport(terrainY);
    if (groundedEnd) {
      wallRunActive = false;
      wallRunTimer = 0.f;
      wallClimbActive = false;
      wallClimbTimer = 0.f;
      speedVaultActive = false;
      speedVaultTimer = 0.f;
      wallRunCameraTilt = 0.f;
    }
    } else {
      horizVel = glm::vec2(0.f);
      velY = 0.f;
      velYPreGround = 0.f;
      jumped = false;
      horizDist = 0.f;
      {
        const float feetYDead = camPos.y - eyeHeight;
        terrainY = playerTerrainSupportY(camPos.x, camPos.z, feetYDead);
        if (std::isfinite(terrainY)) {
          if (feetYDead < terrainY)
            camPos.y = terrainY + eyeHeight;
          else if (feetYDead <= terrainY + kFeetSnapDownSlop)
            camPos.y = terrainY + eyeHeight;
        }
      }
      {
        const float headYDead = camPos.y + 0.12f;
        if (headYDead >= kCeilingY - 0.08f)
          camPos.y = kCeilingY - 0.12f - 0.08f;
      }
      feetFinalVert = camPos.y - eyeHeight;
      terrainY = playerTerrainSupportY(camPos.x, camPos.z, feetFinalVert);
      if (std::isfinite(terrainY) && feetFinalVert > terrainY &&
          feetFinalVert <= terrainY + kGroundedFeetAboveSupport + 0.045f)
        camPos.y = terrainY + eyeHeight;
      feetFinalVert = camPos.y - eyeHeight;
      terrainY = playerTerrainSupportY(camPos.x, camPos.z, feetFinalVert);
      groundedEnd = isGroundedUsingSupport(terrainY);
      wish = glm::vec2(0.f);
      hasWishInput = false;
      runGrounded = false;
      sp = 0.f;
    }
    if (groundedEnd && wasGrounded) {
      playerFallLastGroundedSupportY = terrainY;
      playerFallTakeoffDamageTier = -1;
    }

    {
      // Use movement *intent* (wish) for neighbor/ray direction — horizVel is often skewed by collisions
      // at the lip and misclassified big vs small gap. Forward yaw matches coyote/mantle probes.
      const glm::vec2 yawFwd(std::cos(yaw), std::sin(yaw));
      const glm::vec2 gapVelXZ =
          hasWishInput ? wish
                       : (glm::length(horizVel) > 0.055f ? glm::normalize(horizVel) : yawFwd);
      const float gapRefSupportY =
          (wasGrounded && !groundedEnd) ? std::max(playerFallLastGroundedSupportY, supStart)
                                        : playerFallLastGroundedSupportY;
      playerWalkOffWalkableGapDropCached =
          playerWalkOffEffectiveWalkableGapDrop(gapRefSupportY, camPos.x, camPos.z, gapVelXZ, yawFwd);
    }

    // Ledge gap pipeline (walk-off, no Space):
    // • Small drop + moving off → air walk/run (flat stride). Small drop + stopped at lip → no air bridge, you fall.
    // • Big drop + moving off → kPlayerPreFallBeforeFallSec on the lip (scrub first half of jump clip), then fall
    //   at mid-jump pose until impact is near, then play second half into touchdown + post-land scrub (same idea as
    //   split jump: first half / apex / second half).

    // Walk-off ledge: after first half on the deck, start second half when impact is soon (feet→support).
    if (playerJumpAwaitPreLandSecondHalf && !groundedEnd && ledgeClimbT < 0.f && !slideActive &&
        playerWalkOffWalkableGapDropCached > kPlayerWalkOffSmallGapMaxDropM &&
        playerJumpAnimRemain <= 1e-4f && playerPreFallAnimRemain <= 1e-4f && velY < -0.05f) {
      const float gapAbove = feetFinalVert - terrainY;
      if (gapAbove > 0.001f && std::isfinite(gapAbove)) {
        const float velDown = -velY;
        const float tHit = gapAbove / std::max(velDown, 0.02f);
        if (tHit <= kJumpLedgePreLandLeadSec && tHit >= 0.f) {
          const int clipSecond =
              (playerPreFallUseRunClip && avClipJumpRun >= 0) ? avClipJumpRun
              : (avClipJump >= 0 ? avClipJump : avClipJumpRun);
          if (clipSecond >= 0 && static_cast<size_t>(clipSecond) < staffRig.clips.size()) {
            const double durJ = staff_skin::clipDuration(staffRig, clipSecond);
            // Play the *second* half of the jump clip into the landing window (phase = dur − remain).
            playerJumpAnimRemain = static_cast<float>(
                durJ * (1.0 - static_cast<double>(kJumpClipLedgeFirstHalfFrac)));
            playerJumpRunTailActive = (clipSecond == avClipJumpRun);
            playerJumpAwaitPreLandSecondHalf = false;
            playerJumpLedgeSecondHalfAir = true;
          }
        }
      }
    }

    if (ledgeClimbT < 0.f) {
      if (!groundedEnd) {
        if (wasGrounded) {
          playerTrackingAirFall = true;
          playerFallTakeoffDamageTier =
              playerFallDamageTierFromSupportWorldY(playerFallLastGroundedSupportY);
          if (!std::isfinite(playerFallChainMaxFeetY))
            playerFallChainMaxFeetY =
                std::max(feetFinalVert, playerFallLastGroundedSupportY);
          else
            playerFallChainMaxFeetY =
                std::max(playerFallChainMaxFeetY,
                         std::max(feetFinalVert, playerFallLastGroundedSupportY));
          // First air frame: feet can sit below the deck one integration step — anchor peak to last support.
          playerAirFeetPeakY =
              std::max(playerAirFeetPeakY, std::max(feetFinalVert, playerFallLastGroundedSupportY));
          // Step off a ledge while moving: optional pre-fall jump clip for *large* drops only. Small gaps:
          // walk/sprint in air (no jump arch) — must not require jump clips to be loaded.
          const glm::vec2 fwdXZ(std::cos(yaw), std::sin(yaw));
          const float walkableGapDrop = playerWalkOffWalkableGapDropCached;
          // Large drop from shelf/crate (not narrow bay stride): bleed sprint momentum so the fall is shorter.
          if (playerLastGroundedOnShelfDeck && !jumped && !slideActive &&
              walkableGapDrop > kPlayerWalkOffSmallGapMaxDropM)
            horizVel *= kWalkOffLedgeHorizMul;
          const float spAir = glm::length(horizVel);
          constexpr float kRunJumpMinHorizSpPreFall = 0.118f;
          // Include slide: sprint-slide off a small bay reads as “moving off” so air-walk + slide clip can continue.
          // Do not require jump/landing clip timers to be zero: after a space jump you can still have
          // playerJumpAnimRemain or playerJumpPostLandRemain while grounded — that blocked all walk-off ledges.
          const bool movingOffLedge =
              !jumped && playerPreFallAnimRemain <= 1e-4f && velY <= kPlayerWalkOffLedgeMaxVelYForAnim &&
              (hasWishInput || spAir > kPlayerWalkOffLedgeAnimMinHorizSp || slideActive);
          if (movingOffLedge && walkableGapDrop <= kPlayerWalkOffSmallGapMaxDropM) {
            playerAirWalkSmallGap = true;
            // Short hop: never mix with big-ledge pre-land / timed second-half state.
            playerJumpAwaitPreLandSecondHalf = false;
            playerJumpLedgeSecondHalfAir = false;
            playerJumpPostLandClipIndex = -1;
            playerJumpPostLandDurationInit = 0.f;
            playerJumpPostLandSecondHalfScrub = false;
            playerJumpPostLandRemain = 0.f;
            playerJumpArchActive = false;
          } else if (!movingOffLedge && walkableGapDrop <= kPlayerWalkOffSmallGapMaxDropM && !jumped &&
                     velY <= kPlayerWalkOffLedgeMaxVelYForAnim) {
            // Stopped at a narrow gap: fall through like a real edge (no “hover walk” bridge).
            playerAirWalkSmallGap = false;
            playerJumpAwaitPreLandSecondHalf = false;
            playerJumpLedgeSecondHalfAir = false;
            if (avClipJump >= 0 || avClipJumpRun >= 0) {
              playerJumpArchActive = true;
            }
          } else if (movingOffLedge && walkableGapDrop > kPlayerWalkOffSmallGapMaxDropM)
            playerAirWalkSmallGap = false;

          const bool wantWalkOffPreFall =
              movingOffLedge && walkableGapDrop > kPlayerWalkOffSmallGapMaxDropM;
          if (wantWalkOffPreFall && slideActive) {
            slideActive = false;
            slideAnimClip = -1;
            slideAnimElapsed = 0.f;
            slideAnimDurSec = 0.f;
            slideStartSpeed = 0.f;
            slideClearClipNextFrame = false;
          }
          if (wantWalkOffPreFall) {
            // Cancel any lingering jump/land one-shots so pre-fall + first-half clip own the queue.
            playerJumpPostLandClipIndex = -1;
            playerJumpPostLandDurationInit = 0.f;
            playerJumpPostLandSecondHalfScrub = false;
            playerJumpPostLandRemain = 0.f;
            playerJumpAnimRemain = 0.f;
            // Lock to last grounded deck height (feetFinalVert can dip past the lip one tick and confuse snap).
            playerPreFallFeetLockY = playerFallLastGroundedSupportY;
            playerPreFallAnimRemain = kPlayerPreFallBeforeFallSec;
            horizVel *= kPlayerPreFallStartHorizMul;
            playerPreFallUseRunClip =
                runGrounded && spAir > kRunJumpMinHorizSpPreFall && avClipJumpRun >= 0;
            playerJumpRunTailActive = false;
            playerJumpArchActive = true;
            swayPitch += kJumpTakeoffPitch * 0.22f;
            playerJumpAwaitPreLandSecondHalf = true;
          }
        } else if (playerTrackingAirFall) {
          playerAirFeetPeakY = std::max(playerAirFeetPeakY, feetFinalVert);
          if (!std::isfinite(playerFallChainMaxFeetY))
            playerFallChainMaxFeetY =
                std::max(feetFinalVert, playerFallLastGroundedSupportY);
          else
            playerFallChainMaxFeetY = std::max(playerFallChainMaxFeetY, feetFinalVert);
          // Re-evaluate each air frame; don’t gate on playerJumpArchActive — after the jump clip elapses
          // (remain=0) we still need small-gap walk. Skip while jump/pre-fall clips own the timeline.
          if (!groundedEnd && ledgeClimbT < 0.f && playerJumpAnimRemain <= 1e-4f &&
              playerPreFallAnimRemain <= 1e-4f) {
            const float spTr = glm::length(horizVel);
            const float airWalkableDrop = playerWalkOffWalkableGapDropCached;
            if (airWalkableDrop <= kPlayerWalkOffSmallGapMaxDropM) {
              if (hasWishInput || spTr > kPlayerWalkOffLedgeAnimMinHorizSp || slideActive)
                playerAirWalkSmallGap = true;
              else
                playerAirWalkSmallGap = false;
            } else if (airWalkableDrop > kPlayerWalkOffSmallGapMaxDropM * 1.52f)
              playerAirWalkSmallGap = false;
          }
        }
      } else if (wasGrounded) {
        playerTrackingAirFall = false;
        playerAirFeetPeakY = feetFinalVert;
      }
    } else {
      playerTrackingAirFall = false;
      playerFallTakeoffDamageTier = -1;
      playerFallChainMaxFeetY = std::numeric_limits<float>::quiet_NaN();
    }

    if (groundedEnd) {
      // Do not clear playerPreFallAnimRemain here: after pre-fall snap, feet read as grounded on the same
      // deck for several frames — clearing would cancel the teeter immediately while the fall clip still ran.
      playerAirWalkSmallGap = false;
      playerLastGroundedOnShelfDeck =
          playerFallSupportIsShelfDeckOrCrateTop(camPos.x, camPos.z, terrainY);
    }
    if (ledgeClimbT < 0.f && playerPreFallAnimRemain > 1e-6f && !slideActive) {
      // Glue feet to the live deck/crate surface under (x,z); void probes read the floor far below — keep lock.
      const float feetProbe = camPos.y - eyeHeight;
      float surfY = playerTerrainSupportY(camPos.x, camPos.z, feetProbe);
      constexpr float kPreFallMaxDeckDrop = 0.48f;
      constexpr float kPreFallMaxDeckRise = 0.1f;
      const bool onDeckLike =
          playerFallSupportIsShelfDeckOrCrateTop(camPos.x, camPos.z, surfY) &&
          surfY >= playerPreFallFeetLockY - kPreFallMaxDeckDrop &&
          surfY <= playerPreFallFeetLockY + kPreFallMaxDeckRise;
      if (!onDeckLike)
        surfY = playerPreFallFeetLockY;
      camPos.y = surfY + eyeHeight;
      velY = 0.f;
      const float prevPreFall = playerPreFallAnimRemain;
      playerPreFallAnimRemain = std::max(0.f, playerPreFallAnimRemain - dt);
      if (prevPreFall > 0.f && playerPreFallAnimRemain <= 0.f) {
        glm::vec2 commitDir(std::cos(yaw), std::sin(yaw));
        const float hl = glm::length(horizVel);
        if (hl > 0.07f)
          commitDir = horizVel * (1.f / hl);
        camPos.x += commitDir.x * kWalkOffLedgeCommitPushM;
        camPos.z += commitDir.y * kWalkOffLedgeCommitPushM;
        resolvePillarCollisions(true);
        // If still over the same shelf deck, step again (thin lip / inset footprint).
        for (int step = 0; step < 2; ++step) {
          const float fp = camPos.y - eyeHeight;
          const float sy = playerTerrainSupportY(camPos.x, camPos.z, fp);
          if (!playerFallSupportIsShelfDeckOrCrateTop(camPos.x, camPos.z, sy) ||
              sy < playerPreFallFeetLockY - 0.55f)
            break;
          camPos.x += commitDir.x * kWalkOffLedgeCommitPushM;
          camPos.z += commitDir.y * kWalkOffLedgeCommitPushM;
          resolvePillarCollisions(true);
        }
        if (playerJumpAnimRemain <= 1e-4f) {
          const int clipGo =
              (playerPreFallUseRunClip && avClipJumpRun >= 0) ? avClipJumpRun
              : (avClipJump >= 0 ? avClipJump : avClipJumpRun);
          if (clipGo >= 0 && static_cast<size_t>(clipGo) < staffRig.clips.size()) {
            // First half played on the lip (0..kPlayerPreFallBeforeFallSec). Stay at mid-jump pose while falling;
            // playerJumpAwaitPreLandSecondHalf + kJumpLedgePreLandLeadSec starts the second half, landing uses scrub.
            playerJumpAnimRemain = 0.f;
            playerJumpRunTailActive = (clipGo == avClipJumpRun);
            playerJumpAwaitPreLandSecondHalf = true;
            playerJumpLedgeSecondHalfAir = false;
          }
        }
      }
    }

    lastShadowGroundUnderY = playerTerrainSupportY(camPos.x, camPos.z, camPos.y - eyeHeight);
    const float pitchDamp = groundedEnd ? kSwayPitchDamp : kSwayPitchDampAir;
    swayPitch *= std::exp(-dt * pitchDamp);
    if (jumped)
      swayPitch += kJumpTakeoffPitch;

    {
      float lpDecay = kLandingPitchDecay;
      if (groundedEnd)
        lpDecay += kLandingPitchDecayGroundBoost;
      landingPitchOfs *= std::exp(-dt * lpDecay);
    }
    if (playerPreFallAnimRemain > 1e-6f && ledgeClimbT < 0.f && !slideActive) {
      const float tCh =
          glm::clamp(1.f - playerPreFallAnimRemain / kPlayerPreFallBeforeFallSec, 0.f, 1.f);
      const float dip = std::sin(tCh * glm::pi<float>());
      landingPitchOfs = -kPreFallChargeViewPitchAmp * std::pow(dip, 1.22f);
    } else if (playerJumpSquatCharging && ledgeClimbT < 0.f && !slideActive) {
      // Absolute pitch from charge phase (matches pre-fall). Cumulative -= + decay stacked into a deep
      // negative offset and felt “camera locked” looking at the floor during/after big jumps.
      const float dip = std::sin(playerJumpSquatCharge * glm::pi<float>());
      landingPitchOfs = -kJumpSquatViewPitchAmp * std::pow(dip, 1.15f);
    } else if (wallClimbActive) {
      landingPitchOfs = glm::mix(landingPitchOfs, 0.014f, 1.f - std::exp(-dt * 8.f));
    } else if (speedVaultActive) {
      landingPitchOfs = glm::mix(landingPitchOfs, -0.026f, 1.f - std::exp(-dt * 10.f));
    } else if (ledgeHangActive && ledgeHangGrabTimer > 0.f) {
      landingPitchOfs = glm::mix(landingPitchOfs, -0.02f, 1.f - std::exp(-dt * 9.f));
    } else if (ledgeHangActive) {
      landingPitchOfs = glm::mix(landingPitchOfs, 0.f, 1.f - std::exp(-dt * 5.f));
    }
    if (!wasGrounded && groundedEnd) {
        playerJumpAwaitPreLandSecondHalf = false;
        playerJumpLedgeSecondHalfAir = false;
        playerVaultCrateJumpActive = false;
        const float peakForFall =
            std::max(playerAirFeetPeakY, playerFallLastGroundedSupportY);
        const float dropGeom = std::max(0.f, peakForFall - terrainY);
        if (dropGeom >= kDepthJumpMinDropM)
          playerDepthJumpWindowRemain = kDepthJumpWindowSec;
        const float vDown = std::max(0.f, -velYPreGround);
        const float vEarthDown = vDown * kFallDamageVelToEarthScale;
        const int fdTier = glm::clamp(playerFallDamageTierAtSupport(camPos.x, camPos.z, terrainY), 0,
                                      kPlayerFallDamageTierCount - 1);
        const PlayerFallDamageTierParams& fdP = kPlayerFallDamageTierParams[fdTier];
        const float fdSafe = fdP.safeImpactSpeed;
        const float fdScale = fdP.kineticScale;
        const float fdCap = fdP.singleHitCap;
        const float fdMinExcess = fdP.minExcessImpact;
        const float fdSafeEquivDropM = (fdSafe * fdSafe) / (2.f * kFallDamageEarthG);
        const float jumpArchDropLim = fdP.jumpArchMinDropM;
        const bool jumpBlocksSmallFall = playerJumpArchActive && dropGeom < jumpArchDropLim;
        const bool fallLikeAir =
            playerTrackingAirFall ||
            (!playerJumpArchActive && ledgeClimbT < 0.f && vEarthDown >= fdSafe - 0.3f);
        const bool skipChainFallDamage =
            playerFallDamageChainImmuneRemain > 0.f && dropGeom <= kPlayerFallDamageChainImmuneMaxDropM &&
            vEarthDown < fdSafe + 6.5f;
        const bool landedOnShelfOrCrate =
            playerFallSupportIsShelfDeckOrCrateTop(camPos.x, camPos.z, terrainY);
        const bool skipShortFallOntoObject =
            landedOnShelfOrCrate && dropGeom <= kPlayerFallNoDamageShelfOrCrateMaxDropM;
        // Second shelf (deck index 1): no fall injury — use height-only tier + chain max so lip XZ / skim
        // descents still count.
        const int chainTier = std::isfinite(playerFallChainMaxFeetY)
                                  ? playerFallDamageTierFromSupportWorldY(playerFallChainMaxFeetY)
                                  : 0;
        const bool skipFallDamageFromSecondShelfTakeoff =
            chainTier == 1 || playerFallTakeoffDamageTier == 1;
        const bool fdDebug = std::getenv("VULKAN_GAME_DEBUG_FALL_DAMAGE") != nullptr;
        if (fdDebug) {
          static const char* kFdTierNames[] = {"low", "midLow", "mid", "high"};
          const char* tierName =
              (fdTier >= 0 && fdTier < kPlayerFallDamageTierCount) ? kFdTierNames[fdTier] : "?";
          std::fprintf(stderr,
                       "[fall] land peakY=%.3f terrainY=%.3f dropGeom=%.3f | vyImpact=%.3f vEarth=%.3f | "
                       "tier=%d(%s) | trackAir=%d jumpArch=%d ledgeClimb=%d chainImm=%.2fs\n",
                       peakForFall, terrainY, dropGeom, velYPreGround, vEarthDown, fdTier, tierName,
                       playerTrackingAirFall ? 1 : 0, playerJumpArchActive ? 1 : 0, ledgeClimbT >= 0.f ? 1 : 0,
                       static_cast<double>(playerFallDamageChainImmuneRemain));
          std::fprintf(stderr,
                       "[fall] gates safeDropEquiv=%.2fm safeV=%.2f scale=%.3f cap=%.1f minExcess=%.3f "
                       "jumpArchSkip<%.2f\n",
                       fdSafeEquivDropM, fdSafe, fdScale, fdCap, fdMinExcess, jumpArchDropLim);
          std::fprintf(stderr,
                       "[fall] fallLikeAir=%d jumpBlockSmall=%d skipChain=%d skipObjLedge=%d velYOk=%d "
                       "skip2ndShelfTakeoff=%d takeoffTier=%d chainTier=%d chainY=%.3f\n",
                       fallLikeAir ? 1 : 0, jumpBlocksSmallFall ? 1 : 0, skipChainFallDamage ? 1 : 0,
                       skipShortFallOntoObject ? 1 : 0, velYPreGround < kPlayerFallDamageLandVelYMaxEps ? 1 : 0,
                       skipFallDamageFromSecondShelfTakeoff ? 1 : 0, playerFallTakeoffDamageTier, chainTier,
                       static_cast<double>(playerFallChainMaxFeetY));
        }
        if (fallLikeAir && !jumpBlocksSmallFall && ledgeClimbT < 0.f &&
            velYPreGround < kPlayerFallDamageLandVelYMaxEps && !skipChainFallDamage &&
            !skipShortFallOntoObject && !skipFallDamageFromSecondShelfTakeoff) {
          const float vFromDrop = std::sqrt(std::max(0.f, 2.f * kFallDamageEarthG * dropGeom));
          const float vImpact = std::max(vEarthDown, vFromDrop);
          const float dv = std::max(0.f, vImpact - fdSafe);
          if (fdDebug) {
            std::fprintf(stderr,
                         "[fall] calc dropGeom=%.3f vFromDrop=%.3f vMeas=%.3f vImpact=%.3f dv=%.3f "
                         "(need dv>=%.3f safeEquivDrop=%.2fm)\n",
                         dropGeom, vFromDrop, vEarthDown, vImpact, dv, fdMinExcess, fdSafeEquivDropM);
          }
          if (dv >= fdMinExcess && !playerDeathActive) {
            float dmg = fdScale * dv * dv;
            float landSurfMul = 1.f;
            float landEdgeMul = 1.f;
            playerFallLandingDamageMultipliers(camPos.x, camPos.z, terrainY, dv, fdMinExcess, landSurfMul,
                                               landEdgeMul);
            const float dmgPreCap = dmg * landSurfMul * landEdgeMul;
            dmg *= landSurfMul * landEdgeMul;
            dmg = std::min(dmg, fdCap);
            if (fdDebug) {
              const float sevDbg =
                  glm::clamp((dv - fdMinExcess) / std::max(1e-4f, kPlayerFallLandSeveritySpanDv), 0.f, 1.f);
              std::fprintf(stderr,
                           "[fall] dmg base=%.2f surf=%.3f edge=%.3f preCap=%.2f final=%.2f (cap=%.1f) "
                           "sev=%.3f HP %.1f -> %.1f\n",
                           fdScale * dv * dv, landSurfMul, landEdgeMul, dmgPreCap, dmg, fdCap,
                           static_cast<double>(sevDbg), static_cast<double>(playerHealth),
                           static_cast<double>(std::max(0.f, playerHealth - dmg)));
            }
            playerHealth = std::max(0.f, playerHealth - dmg);
            if (dmg > 0.f) {
              landingPitchOfs -=
                  kFallDamageViewPitchPerSqrtDmg * std::sqrt(std::min(dmg, 36.f));
              playerFallDamageChainImmuneRemain = kPlayerFallDamageChainImmuneSec;
              playerAirFeetPeakY = feetFinalVert;
              playerFallLastGroundedSupportY = terrainY;
              playerScreenDamagePulse = std::max(
                  playerScreenDamagePulse,
                  glm::clamp(dmg / std::max(1e-4f, kPlayerScreenDamagePulseRefDmg), 0.16f, 1.f));
              {
                const float sevDv =
                    glm::clamp((dv - fdMinExcess) / std::max(1e-4f, kPlayerFallLandSeveritySpanDv), 0.f, 1.f);
                const float hExcess = std::max(0.f, dropGeom - fdSafeEquivDropM);
                constexpr float kFallSfxDropSpanM = 9.f;
                const float sevDrop = glm::clamp(hExcess / kFallSfxDropSpanM, 0.f, 1.f);
                const float dmgNorm = glm::clamp(dmg / std::max(1e-4f, fdCap), 0.f, 1.f);
                const float impact01 = glm::clamp(
                    0.38f * sevDv + 0.48f * sevDrop + 0.22f * dmgNorm, 0.08f, 1.f);
                audioPlayBigFallImpact(impact01);
              }
            }
            refreshWindowTitleWithHealth();
            if (playerHealth <= 0.f && !playerDeathActive)
              beginPlayerDeath();
          } else if (fdDebug) {
            std::fprintf(stderr, "[fall] SKIP: excess speed dv=%.4f < minExcess=%.3f (safe landing band)\n",
                         static_cast<double>(dv), static_cast<double>(fdMinExcess));
          }
        } else if (fdDebug) {
          std::fprintf(stderr,
                       "[fall] SKIP: not counted as fall damage (fallLikeAir=%d jumpBlock=%d ledge=%d "
                       "chainSkip=%d objLedge=%d vyGate=%d)\n",
                       fallLikeAir ? 1 : 0, jumpBlocksSmallFall ? 1 : 0, ledgeClimbT >= 0.f ? 1 : 0,
                       skipChainFallDamage ? 1 : 0, skipShortFallOntoObject ? 1 : 0,
                       velYPreGround < kPlayerFallDamageLandVelYMaxEps ? 1 : 0);
        }
      if (velYPreGround <= 0.f && !slideActive) {
        const float impact = std::clamp(-velYPreGround, 0.f, kLandingPitchImpactRef * 1.35f);
        const float t = std::clamp(impact / kLandingPitchImpactRef, 0.f, 1.f);
        const float smooth = t * t * (3.f - 2.f * t);
        landingPitchOfs -= kLandingPitchMin + (kLandingPitchMax - kLandingPitchMin) * smooth;
        groundEase = 0.f;
        audioPlayFootstep(true);
        footstepDistAccum = 0.f;
        const auto tryStartPlayerLandClip = [&]() -> bool {
          if (avClipLand < 0 || static_cast<size_t>(avClipLand) >= staffRig.clips.size())
            return false;
          if (-velYPreGround < kPlayerLandClipMinDownVel)
            return false;
          const double dL = staff_skin::clipDuration(staffRig, avClipLand);
          if (dL <= 1e-6)
            return false;
          const float wall = std::min(static_cast<float>(dL), kPlayerLandClipMaxWallSec);
          playerJumpPostLandClipIndex = avClipLand;
          playerJumpPostLandDurationInit = wall;
          playerJumpPostLandRemain = wall;
          playerJumpPostLandSecondHalfScrub = false;
          playerJumpAnimRemain = 0.f;
          playerJumpArchActive = false;
          playerJumpRunTailActive = false;
          return true;
        };
        const auto startJumpSecondHalfLand = [&](int landClipIdx) {
          if (landClipIdx < 0 || static_cast<size_t>(landClipIdx) >= staffRig.clips.size())
            return false;
          const double durJ = staff_skin::clipDuration(staffRig, landClipIdx);
          if (durJ <= 1e-6)
            return false;
          const double tailDur = durJ * (1.0 - static_cast<double>(kJumpClipLedgeFirstHalfFrac));
          const float wall = std::min(static_cast<float>(tailDur), kPlayerLandClipMaxWallSec);
          playerJumpPostLandClipIndex = landClipIdx;
          playerJumpPostLandDurationInit = wall;
          playerJumpPostLandRemain = wall;
          playerJumpPostLandSecondHalfScrub = true;
          playerJumpAnimRemain = 0.f;
          return true;
        };
        const auto tryStartJumpSecondHalfAny = [&](int first, int second) -> bool {
          if (first >= 0 && startJumpSecondHalfLand(first))
            return true;
          if (second >= 0 && second != first && startJumpSecondHalfLand(second))
            return true;
          return false;
        };
        if (playerJumpArchActive) {
          // Touchdown: use standing jump tail when the asset exists (run-jump tail reads as running in place).
          const int landClipJ = (avClipJump >= 0) ? avClipJump : avClipJumpRun;
          const int altJump =
              (landClipJ == avClipJump && avClipJumpRun >= 0) ? avClipJumpRun : -1;
          if (tryStartJumpSecondHalfAny(landClipJ, altJump)) {
            playerJumpArchActive = false;
            playerJumpRunTailActive = false;
          } else if (avClipJump < 0 && avClipJumpRun < 0 && tryStartPlayerLandClip()) {
          } else {
            playerJumpArchActive = false;
            playerJumpRunTailActive = false;
          }
        } else if (!playerJumpArchActive && !playerAirWalkSmallGap && playerTrackingAirFall &&
                   ledgeClimbT < 0.f &&
                   (avClipJump >= 0 || avClipJumpRun >= 0 || avClipLand >= 0)) {
          const float dropWalk = std::max(0.f, peakForFall - terrainY);
          // No max drop: tall shelves were capped out; no min impact vel: fast falls were skipping this path
          // while jumpArch was false (e.g. edge cases without pre-fall arch).
          if (dropWalk > kPlayerWalkOffSmallGapMaxDropM && dropWalk >= kPlayerWalkOffLandMinDropM) {
            const int landC = (avClipJump >= 0) ? avClipJump : avClipJumpRun;
            const int altJump =
                (landC == avClipJump && avClipJumpRun >= 0) ? avClipJumpRun : -1;
            if (tryStartJumpSecondHalfAny(landC, altJump)) {
              playerJumpRunTailActive = (playerJumpPostLandClipIndex == avClipJumpRun);
            } else if (avClipJump < 0 && avClipJumpRun < 0 && tryStartPlayerLandClip()) {
            } else if (landC >= 0 && static_cast<size_t>(landC) < staffRig.clips.size()) {
              // Rare: second-half setup failed; still hold last frame of jump second half (no dedicated land clip).
              const double durJ = staff_skin::clipDuration(staffRig, landC);
              const double tailDur = durJ * (1.0 - static_cast<double>(kJumpClipLedgeFirstHalfFrac));
              const float wall =
                  std::max(std::min(static_cast<float>(tailDur), kPlayerLandClipMaxWallSec), 0.08f);
              playerJumpPostLandClipIndex = landC;
              playerJumpPostLandDurationInit = wall;
              playerJumpPostLandRemain = wall;
              playerJumpPostLandSecondHalfScrub = true;
              playerJumpAnimRemain = 0.f;
              playerJumpRunTailActive = (landC == avClipJumpRun);
            }
          }
        }
      }
      if (terrainY <= kGroundY + kPlayerFallLandFloorBandM)
        playerFallChainMaxFeetY = std::numeric_limits<float>::quiet_NaN();
      playerTrackingAirFall = false;
    }
    if (!wasGrounded && groundedEnd)
      swayPitch *= 0.48f;
    landingPitchOfs = glm::clamp(landingPitchOfs, -kLandingPitchOfsClampRad, 0.028f);
    wasGrounded = groundedEnd;

    if (groundedEnd)
      groundEase = std::min(1.f, groundEase + dt * kGroundEaseRate);
    else
      groundEase = 1.f;

    sp = glm::length(horizVel);
    // Footsteps + view-bob + avatar loco phase share stride (blended walk→sprint).
    const float strideM = glm::mix(kAvatarStrideWalkM, kAvatarStrideRunM, runAnimBlend);
    const float strideInv = 1.f / std::max(strideM, 0.08f);
    if (groundedEnd && !slideActive && sp > 0.24f)
      footstepDistAccum += horizDist * strideInv;
    else
      footstepDistAccum = 0.f;
    while (footstepDistAccum >= 1.f) {
      footstepDistAccum -= 1.f;
      audioPlayFootstep();
    }

    audioSetSlide(slideActive);

    const float runTarget =
        (groundedEnd && inputSprintHeld(down) && sp > 0.2f) ? std::clamp(sp / kSprintSpeed, 0.f, 1.f) : 0.f;
    const float runBlendRate = runTarget > runAnimBlend ? kRunAnimBlendInRate : kRunAnimBlendOutRate;
    runAnimBlend = glm::mix(runAnimBlend, runTarget, 1.f - std::exp(-dt * runBlendRate));

    const float idleTarget =
        groundedEnd ? (1.f - std::clamp((sp - 0.1f) / 0.8f, 0.f, 1.f)) * (1.f - runAnimBlend * 0.6f) : 0.f;
    idleAnimBlend = glm::mix(idleAnimBlend, idleTarget, 1.f - std::exp(-dt * kIdleAnimBlendRate));

    float strafe = 0.f;
    if (inputStrafeLeft(down))
      strafe -= 1.f;
    if (inputStrafeRight(down))
      strafe += 1.f;
    const float moveBlend = std::clamp(sp / std::max(kMaxSpeed, 0.01f), 0.f, 1.f);
    const float runRollScale = glm::mix(1.f, kRunRollScale, runAnimBlend);
    const float targetRoll =
        -strafe * moveBlend * kSwayRollStr * runRollScale * (groundedEnd ? 1.f : 0.55f);
    swayRoll = glm::mix(swayRoll, targetRoll, 1.f - std::exp(-dt * 8.2f));
    const float parkourRollTarget = wallRunActive ? wallRunCameraTilt : 0.f;
    swayRoll = glm::mix(swayRoll, parkourRollTarget, 1.f - std::exp(-dt * 11.0f));
    swayRoll += static_cast<float>(mx) * 0.00004f;
    if (std::abs(strafe) < 0.01f)
      swayRoll *= std::exp(-dt * 2.2f);

    if (slideActive && sp > 0.2f) {
      const float ge = groundEase;
      const float fpMul = thirdPersonTestMode ? 1.f : kFpSlideBobYMul;
      const float fpSideMul = thirdPersonTestMode ? 1.f : kFpSlideBobSideMul;
      const float fpPitchMul = thirdPersonTestMode ? 1.f : kFpSlidePitchMul;
      bobPhase += horizDist * (glm::two_pi<float>() / std::max(kBobSlideStrideM, 0.08f)) * kSlideBobFreq *
                  kSlideViewBobPhaseBoost;
      const float sLo = std::sin(bobPhase + 0.38f);
      const float sHi = std::sin(bobPhase * 1.68f + 0.15f);
      float raw = sLo * 0.58f + sHi * 0.42f;
      const float up = std::max(0.f, raw);
      const float dn = std::min(0.f, raw);
      const float slideVert = std::pow(up, 0.78f) + dn * std::pow(-dn, 0.42f) * 1.14f;
      bobOffsetY = slideVert * kBobAmp * kSlideBobYScale * ge * fpMul;
      bobSideOffset =
          std::sin(bobPhase * 0.74f + 0.55f) * kBobSideAmp * kSlideBobSideScale * ge * fpSideMul;
      walkPitchOsc = (sHi * 0.68f + std::sin(bobPhase * 2.05f + 0.72f) * 0.32f) * kWalkPitchAmp *
                     kSlidePitchScale * ge * fpPitchMul;
      const float runSideTarget =
          std::sin(bobPhase * 0.62f + 0.05f) * kRunSideSwayAmp * 1.35f * ge * fpSideMul;
      runSideSway = glm::mix(runSideSway, runSideTarget, 1.f - std::exp(-dt * 2.15f));
    } else if (groundedEnd && sp > 0.2f) {
      const float bobRunMult = glm::mix(1.f, kRunBobPhaseMult, runAnimBlend);
      bobPhase += horizDist * strideInv * glm::two_pi<float>() * kViewBobPhaseBoost * bobRunMult;
      const float sLo = std::sin(bobPhase + 0.46f);
      const float sHi = std::sin(bobPhase * 2.0f - 0.22f);
      const float sSk = std::sin(bobPhase * 3.0f + 0.95f);
      float raw = sLo * 0.52f + sHi * 0.41f + sSk * 0.07f;
      const float up = std::max(0.f, raw);
      const float dn = std::min(0.f, raw);
      const float shapedVert = std::pow(up, 0.76f) + dn * std::pow(-dn, 0.4f) * 1.12f;
      const float ge = groundEase;
      const float runScale = glm::mix(1.f, kRunBobAmpScale, runAnimBlend);
      const float runSideScale = glm::mix(1.f, kRunBobSideScale, runAnimBlend);
      const float runPitchScale = glm::mix(1.f, kRunPitchOscScale, runAnimBlend);
      bobOffsetY = shapedVert * kBobAmp * ge * runScale;
      bobSideOffset = std::sin(bobPhase * 0.5f + 0.38f) * kBobSideAmp * moveBlend * ge * runSideScale;
      walkPitchOsc =
          (sHi * 0.62f + std::sin(bobPhase * 2.0f + 0.88f) * 0.38f) * kWalkPitchAmp * moveBlend * ge * runPitchScale;
      const float runSideTarget =
          std::sin(bobPhase * 0.5f + 0.08f) * kRunSideSwayAmp * moveBlend * ge * runAnimBlend;
      runSideSway = glm::mix(runSideSway, runSideTarget, 1.f - std::exp(-dt * 3.2f));
    } else if (!groundedEnd && sp > 0.22f) {
      const float airT = std::clamp(sp / std::max(kAirSpeedCap * 0.42f, 0.01f), 0.f, 1.f);
      bobPhase += horizDist * strideInv * glm::two_pi<float>() * kViewBobPhaseBoost * kAirBobPhaseScale;
      const float sA = std::sin(bobPhase * 1.12f + 0.18f);
      const float sB = std::sin(bobPhase * 2.02f + 0.55f);
      const float airY = (sA * 0.64f + sB * 0.36f) * kBobAmp * kAirBobAmpMul * airT;
      const float airSide = std::sin(bobPhase * 0.92f + 0.4f) * kBobSideAmp * kAirBobSideMul * airT;
      const float airPitch = sB * kWalkPitchAmp * kAirBobPitchMul * airT;
      bobOffsetY = glm::mix(bobOffsetY, airY, 1.f - std::exp(-dt * 11.f));
      bobSideOffset = glm::mix(bobSideOffset, airSide, 1.f - std::exp(-dt * 11.f));
      walkPitchOsc = glm::mix(walkPitchOsc, airPitch, 1.f - std::exp(-dt * 9.f));
      runSideSway *= std::exp(-dt * 6.f);
    } else {
      bobOffsetY *= std::exp(-dt * 7.f);
      bobSideOffset *= std::exp(-dt * 7.f);
      walkPitchOsc *= std::exp(-dt * 7.f);
      runSideSway *= std::exp(-dt * 8.5f);
    }

    idlePhase += dt * kIdleSpeed;
    const float idleBlend = idleAnimBlend * groundEase;
    idlePitch = std::sin(idlePhase * 0.92f) * kIdlePitchAmp * idleBlend;
    idleRoll = std::sin(idlePhase * 0.64f + 2.1f) * kIdleRollAmp * idleBlend;
    idleBobY = std::sin(idlePhase * 1.18f + 0.4f) * kIdleBobAmp * idleBlend;
    idleSide = std::sin(idlePhase * 0.48f + 1.4f) * kIdleSideAmp * idleBlend;

    randomSwayPhase += dt * kRandomSwaySpeed;
    const float randomBlend = std::clamp(1.f - moveBlend * 0.45f - runAnimBlend * 0.22f, 0.4f, 1.f);
    const float n1 = std::sin(randomSwayPhase * 0.73f + 0.31f);
    const float n2 = std::sin(randomSwayPhase * 1.41f + 1.78f);
    const float n3 = std::sin(randomSwayPhase * 2.19f + 4.02f);
    const float n4 = std::sin(randomSwayPhase * 0.57f + 2.64f);
    randomSwayPitch = (n1 * 0.62f + n2 * 0.38f) * kRandomSwayPitchAmp * randomBlend;
    randomSwayRoll = (n2 * 0.54f + n3 * 0.46f) * kRandomSwayRollAmp * randomBlend;
    randomSwayBobY = (n3 * 0.58f + n4 * 0.42f) * kRandomSwayBobAmp * randomBlend;
    randomSwaySide = (n4 * 0.66f + n1 * 0.34f) * kRandomSwaySideAmp * randomBlend;

    if (slideActive && slideAnimClip >= 0) {
      slideAnimElapsed += dt;
      if (slideAnimElapsed >= slideAnimDurSec) {
        slideActive = false;
        slideClearClipNextFrame = true;
      }
    }

    if (kPlayerLedgeMantleEnabled) {
      // Ledge grab after full head/camera update so rays match drawFrame (bob, sway, roll, mouse roll, pos).
      // Let wall climb auto-chain into the existing mantle once the probe can see a valid top edge.
      const bool mantleInput = wallClimbActive || jumpBuffer > 0.f || down(SDL_SCANCODE_SPACE);
      float mantleMoveBoost = 0.f, mantleExtraFall = 0.f;
      mantleLedgeMovementAid(horizVel, groundedEnd, mantleMoveBoost, mantleExtraFall);
      const float runTEffMantle = glm::min(1.f, mantleRunT(horizVel) + mantleMoveBoost);
      const float maxUpGrab = mantleMaxVelUp(runTEffMantle);
      const float minFallMantle = kLedgeGrabMaxFallVelY - mantleExtraFall;
      const bool skipMantleSmallShelfHop =
          playerAirWalkSmallGap ||
          (playerTrackingAirFall && !playerJumpArchActive && !slideActive &&
           playerWalkOffWalkableGapDropCached <= kPlayerWalkOffSmallGapMaxDropM * 1.18f &&
           glm::length(horizVel) > 0.048f);
      if (ledgeClimbT < 0.f && !ledgeHangActive && !ladderClimbActive && !playerDeathActive &&
          eyeHeight >= kLedgeGrabMinEyeHeight && mantleInput && !dropKickActive &&
          !groundedEnd && !slideActive && !skipMantleSmallShelfHop && velY <= maxUpGrab &&
          velY >= minFallMantle) {
        if (tryStartLedgeHang()) {
          jumpBuffer = 0.f;
          coyoteTime = 0.f;
        } else if (tryStartLedgeClimb()) {
          jumpBuffer = 0.f;
          coyoteTime = 0.f;
        }
      }
    }

    const int clipBeforeSync = playerAvatarClip;
    avatarLocoGroundedSmoothed =
        glm::mix(avatarLocoGroundedSmoothed, groundedEnd ? 1.f : 0.f, 1.f - std::exp(-dt * 17.f));
    // Must use real groundedEnd for clip selection — groundedLocoClip stays “sticky” in air and was
    // letting sprint/walk/crouch branches win over pre-fall / jump mid / small-gap air walk.
    syncPlayerAvatarClip(groundedEnd, down);
    if (playerAvatarClip != clipBeforeSync) {
      if (avatarClipsAllowCrossfade(clipBeforeSync, playerAvatarClip)) {
        playerAvatarBlendFromClip = clipBeforeSync;
        playerAvatarClipBlend = 0.f;
      } else
        playerAvatarClipBlend = 1.f;
    }

    // Distance-sync locomotion animation to horizontal movement (stride ≈ 2× footstep distance).
    if (staffSkinnedActive && !staffRig.clips.empty() && !playerDeathActive) {
      const int c = playerAvatarClip;
      const float cycleM = glm::mix(2.f * kAvatarStrideWalkM, 2.f * kAvatarStrideRunM, runAnimBlend);
      const double kPlay = static_cast<double>(kAvatarAnimPlaybackScale);

      // Fall-mid pose skips distance sync only while the avatar is actually on a jump clip; otherwise
      // (no jump assets) sync can fall through to walk in air and must keep stride advancing.
      const bool jumpClipLoaded = (avClipJump >= 0 || avClipJumpRun >= 0);
      const bool onJumpClip = jumpClipLoaded && (c == avClipJump || c == avClipJumpRun);
      if (playerJumpAnimRemain > 1e-4f || playerJumpPostLandRemain > 1e-4f ||
          playerPushAnimRemain > 1e-4f || playerKickAnimRemain > 1e-4f || dropKickActive ||
          ledgeClimbT >= 0.f || ladderClimbActive ||
          playerPreFallAnimRemain > 1e-4f || (playerAvatarJumpFallMidPose() && onJumpClip)) {
      } else if (slideActive && slideAnimClip >= 0) {
      } else {
        const bool crouchMove = (avClipCrouchFwd >= 0 && c == avClipCrouchFwd) ||
                                (avClipCrouchBack >= 0 && c == avClipCrouchBack) ||
                                (avClipCrouchLeft >= 0 && c == avClipCrouchLeft) ||
                                (avClipCrouchRight >= 0 && c == avClipCrouchRight);
        const bool crouchIdleBow = avClipCrouchIdleBow >= 0 && c == avClipCrouchIdleBow;
        const bool idleLike = (avClipIdle >= 0 && c == avClipIdle) || crouchIdleBow;

        if (avClipWalk >= 0 && c == avClipWalk && sp > 0.032f) {
          const double d = staff_skin::clipDuration(staffRig, avClipWalk);
          const double m = std::max(static_cast<double>(cycleM), 0.15);
          avatarLocoPhaseSec += static_cast<double>(horizDist) * (d / m) * kPlay;
        } else if (avClipSprint >= 0 && c == avClipSprint && sp > 0.032f) {
          const double d = staff_skin::clipDuration(staffRig, avClipSprint);
          const double m = std::max(static_cast<double>(cycleM), 0.15);
          avatarLocoPhaseSec += static_cast<double>(horizDist) * (d / m) * kPlay;
        } else if (crouchMove) {
          const double d = staff_skin::clipDuration(staffRig, c);
          const double m = std::max(static_cast<double>(2.f * kAvatarStrideWalkM * 0.94f), 0.15);
          if (sp > 0.04f)
            avatarLocoPhaseSec += static_cast<double>(horizDist) * (d / m) * kPlay;
          else
            avatarLocoPhaseSec += static_cast<double>(dt) * 0.76 * kPlay;
        } else if (idleLike)
          avatarLocoPhaseSec += static_cast<double>(dt) * 0.82 * kPlay;
      }
    }

    }

    playerAvatarClipBlend = std::min(1.f, playerAvatarClipBlend + dt / kAvatarClipBlendSec);

    {
      const float sp = glm::length(horizVel);
      const float aH = 1.f - std::exp(-dt * kAvatarHorizSpeedSmoothHz);
      avatarHorizSpeedSmoothed = glm::mix(avatarHorizSpeedSmoothed, sp, aH);
    }

    if (std::getenv("VULKAN_GAME_DEBUG_AVATAR_ANIM")) {
      static float dbgAccum = 0.f;
      dbgAccum += dt;
      if (dbgAccum >= 0.22f) {
        dbgAccum = 0.f;
        std::fprintf(stderr,
                     "[avatar anim] clip=%d from=%d blend=%.2f |locoPh=%.3f| "
                     "spd=%.2f spdSm=%.2f grounded=%d\n",
                     playerAvatarClip, playerAvatarBlendFromClip, playerAvatarClipBlend,
                     avatarLocoPhaseSec, glm::length(horizVel), avatarHorizSpeedSmoothed,
                     isGrounded() ? 1 : 0);
      }
    }

    if (kPlayerLedgeMantleEnabled && (ledgeClimbT >= 0.f || ladderClimbActive)) {
      const float feetLift = camPos.y - eyeHeight;
      const float supLift = playerTerrainSupportY(camPos.x, camPos.z, feetLift);
      lastShadowGroundUnderY = supLift;
      wasGrounded = true;
      bobOffsetY *= std::exp(-dt * 11.f);
      bobSideOffset *= std::exp(-dt * 11.f);
      walkPitchOsc *= std::exp(-dt * 11.f);
      runSideSway *= std::exp(-dt * 12.f);
      swayPitch *= std::exp(-dt * 6.f);
    }
    if (ledgeHangActive) {
      const float feetLift = camPos.y - eyeHeight;
      const float supLift = playerTerrainSupportY(camPos.x, camPos.z, feetLift);
      lastShadowGroundUnderY = supLift;
      wasGrounded = false;
      bobOffsetY *= std::exp(-dt * 11.f);
      bobSideOffset *= std::exp(-dt * 11.f);
      walkPitchOsc *= std::exp(-dt * 11.f);
      runSideSway *= std::exp(-dt * 12.f);
    }

    {
      const bool mercyZone = playerHealth > 0.f && playerHealth < kPlayerHealthMercyCap;
      if (!mercyZone) {
        playerMercyHealDelayRemain = 0.f;
        playerInMercyHealthZone = false;
      } else {
        if (!playerInMercyHealthZone) {
          playerMercyHealDelayRemain = kPlayerHealthMercyHealDelaySec;
          playerInMercyHealthZone = true;
        }
        if (playerMercyHealDelayRemain > 0.f)
          playerMercyHealDelayRemain = std::max(0.f, playerMercyHealDelayRemain - dt);
        else
          playerHealth =
              std::min(kPlayerHealthMercyCap, playerHealth + dt * kPlayerHealthMercyHealPerSec);
      }
    }
    {
      const float hungerHealThreshold = kPlayerHungerMax * kPlayerHungerAutoHealThresholdFrac;
      const bool hungerCanHeal = playerHunger >= hungerHealThreshold;
      if (hungerCanHeal && playerHealth > 0.f && playerHealth < kPlayerHealthMax) {
        playerHealth = std::min(kPlayerHealthMax, playerHealth + dt * kPlayerHungerAutoHealPerSec);
      }
    }
    // Loop brvhrtz heartbeat while in mercy band (HP < cap); ≤15 alone was inaudible vs fast mercy heal.
    audioSetLowHealthHeartbeat(playerHealth > 0.f && playerHealth < kPlayerHealthMercyCap);

    tickPlayerDeathScene(dt);
    rebuildTerrainIfNeeded();
  }

  bool running = true;

  void shutdown() {
    vkDeviceWaitIdle(device);
    savePipelineCacheToDisk();
    audioShutdown();
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
      vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
      vkDestroyFence(device, inFlightFences[i], nullptr);
    }
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
      vkUnmapMemory(device, uniformBuffersMemory[i]);
      vkDestroyBuffer(device, uniformBuffers[i], nullptr);
      vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }
    vkDestroySampler(device, sceneTextureSampler, nullptr);
    vkDestroyImageView(device, sceneTextureView, nullptr);
    vkDestroyImage(device, sceneTextureImage, nullptr);
    vkFreeMemory(device, sceneTextureMemory, nullptr);
    vkDestroySampler(device, signTextureSampler, nullptr);
    vkDestroyImageView(device, signTextureView, nullptr);
    vkDestroyImage(device, signTextureImage, nullptr);
    vkFreeMemory(device, signTextureMemory, nullptr);
    vkDestroySampler(device, shelfRackTextureSampler, nullptr);
    vkDestroyImageView(device, shelfRackTextureView, nullptr);
    vkDestroyImage(device, shelfRackTextureImage, nullptr);
    vkFreeMemory(device, shelfRackTextureMemory, nullptr);
    vkDestroySampler(device, crateTextureSampler, nullptr);
    vkDestroyImageView(device, crateTextureView, nullptr);
    vkDestroyImage(device, crateTextureImage, nullptr);
    vkFreeMemory(device, crateTextureMemory, nullptr);
    vkDestroySampler(device, palletTextureSampler, nullptr);
    vkDestroyImageView(device, palletTextureView, nullptr);
    vkDestroyImage(device, palletTextureImage, nullptr);
    vkFreeMemory(device, palletTextureMemory, nullptr);
    vkDestroySampler(device, deliMetalTextureSampler, nullptr);
    vkDestroyImageView(device, deliMetalTextureView, nullptr);
    vkDestroyImage(device, deliMetalTextureImage, nullptr);
    vkFreeMemory(device, deliMetalTextureMemory, nullptr);
    vkDestroySampler(device, deliBaseTextureSampler, nullptr);
    vkDestroyImageView(device, deliBaseTextureView, nullptr);
    vkDestroyImage(device, deliBaseTextureImage, nullptr);
    vkFreeMemory(device, deliBaseTextureMemory, nullptr);
    if (deliPizzaVertexBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, deliPizzaVertexBuffer, nullptr);
    if (deliPizzaVertexBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, deliPizzaVertexBufferMemory, nullptr);
    if (deliPizzaInstanceBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, deliPizzaInstanceBuffer, nullptr);
    if (deliPizzaInstanceBufferMemory != VK_NULL_HANDLE) vkFreeMemory(device, deliPizzaInstanceBufferMemory, nullptr);
    if (deliMeatballVertexBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, deliMeatballVertexBuffer, nullptr);
    if (deliMeatballVertexBufferMemory != VK_NULL_HANDLE)
      vkFreeMemory(device, deliMeatballVertexBufferMemory, nullptr);
    if (deliMeatballInstanceBuffer != VK_NULL_HANDLE) vkDestroyBuffer(device, deliMeatballInstanceBuffer, nullptr);
    if (deliMeatballInstanceBufferMemory != VK_NULL_HANDLE)
      vkFreeMemory(device, deliMeatballInstanceBufferMemory, nullptr);
    if (staffGlbDiffuseSampler != VK_NULL_HANDLE)
      vkDestroySampler(device, staffGlbDiffuseSampler, nullptr);
    if (staffGlbDiffuseView != VK_NULL_HANDLE)
      vkDestroyImageView(device, staffGlbDiffuseView, nullptr);
    if (staffGlbDiffuseImage != VK_NULL_HANDLE)
      vkDestroyImage(device, staffGlbDiffuseImage, nullptr);
    if (staffGlbDiffuseMemory != VK_NULL_HANDLE)
      vkFreeMemory(device, staffGlbDiffuseMemory, nullptr);
    staffGlbDiffuseSampler = VK_NULL_HANDLE;
    staffGlbDiffuseView = VK_NULL_HANDLE;
    staffGlbDiffuseImage = VK_NULL_HANDLE;
    staffGlbDiffuseMemory = VK_NULL_HANDLE;
    if (hudFontTextureSampler != VK_NULL_HANDLE)
      vkDestroySampler(device, hudFontTextureSampler, nullptr);
    if (hudFontTextureView != VK_NULL_HANDLE)
      vkDestroyImageView(device, hudFontTextureView, nullptr);
    if (hudFontTextureImage != VK_NULL_HANDLE)
      vkDestroyImage(device, hudFontTextureImage, nullptr);
    if (hudFontTextureMemory != VK_NULL_HANDLE)
      vkFreeMemory(device, hudFontTextureMemory, nullptr);
    hudFontTextureSampler = VK_NULL_HANDLE;
    hudFontTextureView = VK_NULL_HANDLE;
    hudFontTextureImage = VK_NULL_HANDLE;
    hudFontTextureMemory = VK_NULL_HANDLE;
    gHudUiFontReady = false;
    if (titleIkeaLogoSampler != VK_NULL_HANDLE)
      vkDestroySampler(device, titleIkeaLogoSampler, nullptr);
    if (titleIkeaLogoView != VK_NULL_HANDLE)
      vkDestroyImageView(device, titleIkeaLogoView, nullptr);
    if (titleIkeaLogoImage != VK_NULL_HANDLE)
      vkDestroyImage(device, titleIkeaLogoImage, nullptr);
    if (titleIkeaLogoMemory != VK_NULL_HANDLE)
      vkFreeMemory(device, titleIkeaLogoMemory, nullptr);
    titleIkeaLogoSampler = VK_NULL_HANDLE;
    titleIkeaLogoView = VK_NULL_HANDLE;
    titleIkeaLogoImage = VK_NULL_HANDLE;
    titleIkeaLogoMemory = VK_NULL_HANDLE;
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
    if (shrekEggDiffuseSampler != VK_NULL_HANDLE)
      vkDestroySampler(device, shrekEggDiffuseSampler, nullptr);
    if (shrekEggDiffuseView != VK_NULL_HANDLE)
      vkDestroyImageView(device, shrekEggDiffuseView, nullptr);
    if (shrekEggDiffuseImage != VK_NULL_HANDLE)
      vkDestroyImage(device, shrekEggDiffuseImage, nullptr);
    if (shrekEggDiffuseMemory != VK_NULL_HANDLE)
      vkFreeMemory(device, shrekEggDiffuseMemory, nullptr);
    shrekEggDiffuseSampler = VK_NULL_HANDLE;
    shrekEggDiffuseView = VK_NULL_HANDLE;
    shrekEggDiffuseImage = VK_NULL_HANDLE;
    shrekEggDiffuseMemory = VK_NULL_HANDLE;
    shrekEggDiffuseLoaded = false;
#endif
    for (uint32_t i = 0; i < kMaxExtraTextures; ++i) {
      ExtraTexSlot& s = extraTexSlots[i];
      if (s.sampler != VK_NULL_HANDLE)
        vkDestroySampler(device, s.sampler, nullptr);
      if (s.view != VK_NULL_HANDLE)
        vkDestroyImageView(device, s.view, nullptr);
      if (s.image != VK_NULL_HANDLE)
        vkDestroyImage(device, s.image, nullptr);
      if (s.memory != VK_NULL_HANDLE)
        vkFreeMemory(device, s.memory, nullptr);
      s = {};
    }
    extraTexturesLoadedCount = 0;
    cleanupSwapchain();
    if (sceneRenderSampler != VK_NULL_HANDLE) {
      vkDestroySampler(device, sceneRenderSampler, nullptr);
      sceneRenderSampler = VK_NULL_HANDLE;
    }
    if (postPipelineLayout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(device, postPipelineLayout, nullptr);
      postPipelineLayout = VK_NULL_HANDLE;
    }
    if (postDescriptorSetLayout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(device, postDescriptorSetLayout, nullptr);
      postDescriptorSetLayout = VK_NULL_HANDLE;
    }
    postDescriptorSets.clear();
    vkUnmapMemory(device, groundVertexBufferMemory);
    vkDestroyBuffer(device, groundVertexBuffer, nullptr);
    vkFreeMemory(device, groundVertexBufferMemory, nullptr);
    vkUnmapMemory(device, ceilingVertexBufferMemory);
    vkDestroyBuffer(device, ceilingVertexBuffer, nullptr);
    vkFreeMemory(device, ceilingVertexBufferMemory, nullptr);
    if (pillarInstanceMapped) {
      vkUnmapMemory(device, pillarInstanceBufferMemory);
      pillarInstanceMapped = nullptr;
    }
    vkDestroyBuffer(device, pillarInstanceBuffer, nullptr);
    vkFreeMemory(device, pillarInstanceBufferMemory, nullptr);
    if (signInstanceMapped) {
      vkUnmapMemory(device, signInstanceBufferMemory);
      signInstanceMapped = nullptr;
    }
    vkDestroyBuffer(device, signInstanceBuffer, nullptr);
    vkFreeMemory(device, signInstanceBufferMemory, nullptr);
    vkDestroyBuffer(device, pillarVertexBuffer, nullptr);
    vkFreeMemory(device, pillarVertexBufferMemory, nullptr);
    vkDestroyBuffer(device, crosshairVertexBuffer, nullptr);
    vkFreeMemory(device, crosshairVertexBufferMemory, nullptr);
    vkDestroyBuffer(device, controlsHelpVertexBuffer, nullptr);
    vkFreeMemory(device, controlsHelpVertexBufferMemory, nullptr);
    vkDestroyBuffer(device, deathMenuVertexBuffer, nullptr);
    vkFreeMemory(device, deathMenuVertexBufferMemory, nullptr);
    vkDestroyBuffer(device, pauseMenuVertexBuffer, nullptr);
    vkFreeMemory(device, pauseMenuVertexBufferMemory, nullptr);
    if (inventoryMenuVertexMapped) {
      vkUnmapMemory(device, inventoryMenuVertexBufferMemory);
      inventoryMenuVertexMapped = nullptr;
    }
    vkDestroyBuffer(device, inventoryMenuVertexBuffer, nullptr);
    vkFreeMemory(device, inventoryMenuVertexBufferMemory, nullptr);
    destroyTitleMenuGpuMeshes();
    if (introSplashVertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, introSplashVertexBuffer, nullptr);
      vkFreeMemory(device, introSplashVertexBufferMemory, nullptr);
    }
    if (loadingScreenVertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, loadingScreenVertexBuffer, nullptr);
      vkFreeMemory(device, loadingScreenVertexBufferMemory, nullptr);
    }
    if (healthHudVertexMapped) {
      vkUnmapMemory(device, healthHudVertexBufferMemory);
      healthHudVertexMapped = nullptr;
    }
    vkDestroyBuffer(device, healthHudVertexBuffer, nullptr);
    vkFreeMemory(device, healthHudVertexBufferMemory, nullptr);
    vkDestroyBuffer(device, signVertexBuffer, nullptr);
    vkFreeMemory(device, signVertexBufferMemory, nullptr);
    vkDestroyBuffer(device, signStringVertexBuffer, nullptr);
    vkFreeMemory(device, signStringVertexBufferMemory, nullptr);
    vkDestroyBuffer(device, identityInstanceBuffer, nullptr);
    vkFreeMemory(device, identityInstanceBufferMemory, nullptr);
    vkUnmapMemory(device, shelfInstanceBufferMemory);
    vkDestroyBuffer(device, shelfInstanceBuffer, nullptr);
    vkFreeMemory(device, shelfInstanceBufferMemory, nullptr);
    vkUnmapMemory(device, shelfCrateInstanceBufferMemory);
    vkDestroyBuffer(device, shelfCrateInstanceBuffer, nullptr);
    vkFreeMemory(device, shelfCrateInstanceBufferMemory, nullptr);
    vkDestroyBuffer(device, shelfCrateVertexBuffer, nullptr);
    vkFreeMemory(device, shelfCrateVertexBufferMemory, nullptr);
    vkUnmapMemory(device, shelfPalletInstanceBufferMemory);
    vkDestroyBuffer(device, shelfPalletInstanceBuffer, nullptr);
    vkFreeMemory(device, shelfPalletInstanceBufferMemory, nullptr);
    vkDestroyBuffer(device, shelfPalletVertexBuffer, nullptr);
    vkFreeMemory(device, shelfPalletVertexBufferMemory, nullptr);
    vkUnmapMemory(device, shelfLadderInstanceBufferMemory);
    vkDestroyBuffer(device, shelfLadderInstanceBuffer, nullptr);
    vkFreeMemory(device, shelfLadderInstanceBufferMemory, nullptr);
    vkDestroyBuffer(device, shelfLadderVertexBuffer, nullptr);
    vkFreeMemory(device, shelfLadderVertexBufferMemory, nullptr);
    if (shelfBoxCutterInstanceMapped != nullptr) {
      vkUnmapMemory(device, shelfBoxCutterInstanceBufferMemory);
      shelfBoxCutterInstanceMapped = nullptr;
    }
    if (shelfBoxCutterInstanceBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, shelfBoxCutterInstanceBuffer, nullptr);
      vkFreeMemory(device, shelfBoxCutterInstanceBufferMemory, nullptr);
      shelfBoxCutterInstanceBuffer = VK_NULL_HANDLE;
      shelfBoxCutterInstanceBufferMemory = VK_NULL_HANDLE;
    }
    if (shelfBoxCutterVertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, shelfBoxCutterVertexBuffer, nullptr);
      vkFreeMemory(device, shelfBoxCutterVertexBufferMemory, nullptr);
      shelfBoxCutterVertexBuffer = VK_NULL_HANDLE;
      shelfBoxCutterVertexBufferMemory = VK_NULL_HANDLE;
    }
    shelfBoxCutterVertexCount = 0;
    gShelfBoxCutterMeshLoaded = false;
    if (shelfBoxCutterTextureSampler != VK_NULL_HANDLE)
      vkDestroySampler(device, shelfBoxCutterTextureSampler, nullptr);
    if (shelfBoxCutterTextureView != VK_NULL_HANDLE)
      vkDestroyImageView(device, shelfBoxCutterTextureView, nullptr);
    if (shelfBoxCutterTextureImage != VK_NULL_HANDLE)
      vkDestroyImage(device, shelfBoxCutterTextureImage, nullptr);
    if (shelfBoxCutterTextureMemory != VK_NULL_HANDLE)
      vkFreeMemory(device, shelfBoxCutterTextureMemory, nullptr);
    shelfBoxCutterTextureSampler = VK_NULL_HANDLE;
    shelfBoxCutterTextureView = VK_NULL_HANDLE;
    shelfBoxCutterTextureImage = VK_NULL_HANDLE;
    shelfBoxCutterTextureMemory = VK_NULL_HANDLE;
    if (shelfRustyPipeInstanceMapped != nullptr) {
      vkUnmapMemory(device, shelfRustyPipeInstanceBufferMemory);
      shelfRustyPipeInstanceMapped = nullptr;
    }
    if (shelfRustyPipeInstanceBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, shelfRustyPipeInstanceBuffer, nullptr);
      vkFreeMemory(device, shelfRustyPipeInstanceBufferMemory, nullptr);
      shelfRustyPipeInstanceBuffer = VK_NULL_HANDLE;
      shelfRustyPipeInstanceBufferMemory = VK_NULL_HANDLE;
    }
    if (shelfRustyPipeVertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, shelfRustyPipeVertexBuffer, nullptr);
      vkFreeMemory(device, shelfRustyPipeVertexBufferMemory, nullptr);
      shelfRustyPipeVertexBuffer = VK_NULL_HANDLE;
      shelfRustyPipeVertexBufferMemory = VK_NULL_HANDLE;
    }
    shelfRustyPipeVertexCount = 0;
    gShelfRustyPipeMeshLoaded = false;
    if (shelfRustyPipeTextureSampler != VK_NULL_HANDLE)
      vkDestroySampler(device, shelfRustyPipeTextureSampler, nullptr);
    if (shelfRustyPipeTextureView != VK_NULL_HANDLE)
      vkDestroyImageView(device, shelfRustyPipeTextureView, nullptr);
    if (shelfRustyPipeTextureImage != VK_NULL_HANDLE)
      vkDestroyImage(device, shelfRustyPipeTextureImage, nullptr);
    if (shelfRustyPipeTextureMemory != VK_NULL_HANDLE)
      vkFreeMemory(device, shelfRustyPipeTextureMemory, nullptr);
    shelfRustyPipeTextureSampler = VK_NULL_HANDLE;
    shelfRustyPipeTextureView = VK_NULL_HANDLE;
    shelfRustyPipeTextureImage = VK_NULL_HANDLE;
    shelfRustyPipeTextureMemory = VK_NULL_HANDLE;
    vkUnmapMemory(device, marketInstanceBufferMemory);
    vkDestroyBuffer(device, marketInstanceBuffer, nullptr);
    vkFreeMemory(device, marketInstanceBufferMemory, nullptr);
    vkDestroyBuffer(device, marketVertexBuffer, nullptr);
    vkFreeMemory(device, marketVertexBufferMemory, nullptr);
    vkDestroyBuffer(device, shelfVertexBuffer, nullptr);
    vkFreeMemory(device, shelfVertexBufferMemory, nullptr);
    if (fluorescentInstanceMapped) {
      vkUnmapMemory(device, fluorescentInstanceBufferMemory);
      fluorescentInstanceMapped = nullptr;
    }
    vkDestroyBuffer(device, fluorescentInstanceBuffer, nullptr);
    vkFreeMemory(device, fluorescentInstanceBufferMemory, nullptr);
    vkDestroyBuffer(device, fluorescentVertexBuffer, nullptr);
    vkFreeMemory(device, fluorescentVertexBufferMemory, nullptr);
    if (staffBoneSsbMapped)
      vkUnmapMemory(device, staffBoneSsbMemory);
    vkDestroyBuffer(device, staffBoneSsbBuffer, nullptr);
    vkFreeMemory(device, staffBoneSsbMemory, nullptr);
    staffBoneSsbMapped = nullptr;
    if (employeeInstanceMapped)
      vkUnmapMemory(device, employeeInstanceBufferMemory);
    vkDestroyBuffer(device, employeeInstanceBuffer, nullptr);
    vkFreeMemory(device, employeeInstanceBufferMemory, nullptr);
    vkDestroyBuffer(device, employeeVertexBuffer, nullptr);
    vkFreeMemory(device, employeeVertexBufferMemory, nullptr);
#if defined(VULKAN_GAME_SHREK_EGG_GLB)
    if (shrekEggVertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, shrekEggVertexBuffer, nullptr);
      vkFreeMemory(device, shrekEggVertexBufferMemory, nullptr);
      shrekEggVertexBuffer = VK_NULL_HANDLE;
    }
#endif
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    if (pipelineCache != VK_NULL_HANDLE) {
      vkDestroyPipelineCache(device, pipelineCache, nullptr);
      pipelineCache = VK_NULL_HANDLE;
    }
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    if (yellowMenuCursor) {
      SDL_FreeCursor(yellowMenuCursor);
      yellowMenuCursor = nullptr;
    }
    if (window) {
      SDL_DestroyWindow(window);
      window = nullptr;
    }
    IMG_Quit();
    SDL_Quit();
  }
};

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  try {
    App app;
    app.initWindow();
    if (!audioInit())
      std::cerr << "Footstep audio failed to load (check assets/audio/sfx_footstep_concrete.mp3).\n";
    else {
      // Boot into title menu: freeze store day/night sequencing and play title bed (WAV).
      audioSetStoreDayNightCyclePaused(true);
      audioSetTitleMenuMusicActive(true);
    }
    app.initVulkan();

    auto last = std::chrono::steady_clock::now();
    SDL_Event e;
    // Low-pass wall dt so one-off CPU/GPU spikes don’t jerk movement/animation as much.
    float dtSmooth = 1.f / static_cast<float>(kTargetFps);
    while (app.running) {
      const auto frameStart = std::chrono::steady_clock::now();
      SDL_PumpEvents();
      while (SDL_PollEvent(&e))
        app.handleEvent(e);
      auto now = std::chrono::steady_clock::now();
      float dtRaw = std::chrono::duration<float>(now - last).count();
      last = now;
      if (dtRaw > 0.1f)
        dtRaw = 0.1f;
      if (!std::isfinite(dtRaw) || dtRaw < 1e-5f)
        dtRaw = 1.f / static_cast<float>(kTargetFps);
      constexpr float kDtSmoothAlpha = 0.22f;
      dtSmooth = dtSmooth * (1.f - kDtSmoothAlpha) + dtRaw * kDtSmoothAlpha;
      app.update(dtSmooth);
      app.drawFrame(dtSmooth);
      gPerfTrace.endFrame();
      if (gGamePerf.fpsCap > 0) {
        const auto budget =
            std::chrono::nanoseconds(1'000'000'000ll / static_cast<int64_t>(gGamePerf.fpsCap));
        std::this_thread::sleep_until(frameStart + budget);
      }
    }
    app.gameSaveWrite();
    app.shutdown();
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
    return 1;
  }
  return 0;
}
