#include "audio.hpp"

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Night / day cues fire at this fraction through each switch SFX (0.5 = middle of the clip).
constexpr float kPowerSwitchBlackoutMidFrac = 0.5f;
constexpr float kLightSwitchRestoreMidFrac = 0.5f;
// If ma_sound_get_length_in_seconds fails (e.g. not decoded yet), use these guesses.
constexpr float kFallbackPowerSwitchSec = 0.45f;
constexpr float kFallbackLightSwitchOnSec = 0.12f;
constexpr float kFootstepVolWalk = 0.9f;
constexpr float kFootstepVolLand = 0.7f;
// Blackout horror bed, staff-chase layer, Shrek egg — shared nominal gain (crossfade scales horror vs chase).
constexpr float kGameplayMusicVol = 1.6f;
// Store fluorescent day loop only — hot gain so the day bed dominates the mix (miniaudio linear volume).
constexpr float kStoreDayMusicVol = 17.0f;
constexpr float kStorePowerSwitchVol = 1.4f;
constexpr float kStoreLightSwitchVol = 1.3f;
constexpr float kChaseMusicFadeSec = 2.6f;
constexpr float kChaseMusicFadeInPursuitSec = 0.95f;
constexpr float kChaseMusicFadeOutAfterPursuitSec = 1.85f;
constexpr float kChaseMusicPursuitVolMul = 1.18f;
constexpr float kStaffSpottedVol = 2.2f;
constexpr float kStaffChaseVoVol = 2.4f;
constexpr float kLowHealthHeartbeatVol = 1.0f;
constexpr float kBigFallImpactVol = 1.1f;
constexpr float kStaffMeleeImpactVol = 1.1f;
constexpr float kTitleMenuMusicVol = 52.0f;

static ma_engine gEngine{};
static ma_sound gFootstep[8]{};
static ma_sound gSlideSound{};
// Store music: two short feedback combs in series + LPF (dense tail, damped highs — hall-ish, not slap echo).
static ma_delay_node gStoreComb{};
static ma_delay_node gStoreComb2{};
static ma_lpf_node gStoreHallLpf{};
// Title menu bed: lighter hall so it feels distant/spacey without washing out detail.
static ma_delay_node gTitleComb{};
static ma_delay_node gTitleComb2{};
static ma_lpf_node gTitleHallLpf{};
static ma_sound gStoreMusic{};
// Daytime store bed: multiple MP3s; shuffle when a new “day” starts after lights return.
static std::vector<std::string> gStoreDayMusicPaths{};
static size_t gStoreDayMusicIdx = 0;
static ma_sound gPowerSwitchSfx{};
static ma_sound gHorrorMusic{};
static ma_sound gChaseMusic{};
static ma_sound gShrekEggMusic{};
static bool gShrekEggMusicReady = false;
static ma_sound gLightSwitchOnSfx{};
static bool gReady = false;
static bool gSlideReady = false;
static bool gSlideAudioActive = false;
static bool gStoreAmbienceReady = false;
static bool gPowerSwitchReady = false;
static bool gHorrorReady = false;
static bool gChaseReady = false;
static float gChaseCrossfade = 0.f;
// 0 = no duck; 1 = All Star at full proximity gain — scales other beds down (set from Shrek distance each frame).
static float gShrekEggDuckOtherMusic01 = 0.f;
// Nominal store loop level before Shrek duck (audioSetStoreAmbienceVolume / day restore).
static float gStoreMusicLinear01 = 1.f;
constexpr float kShrekEggDuckOtherMusicMax = 0.9f;
static bool gLightSwitchOnReady = false;
static ma_sound gStaffSpottedSfx[4]{};
static bool gStaffSpottedReady = false;
static int gStaffSpottedNext = 0;
static ma_sound gStaffChaseVoSfx[2]{};
static bool gStaffChaseVoReady = false;
static int gStaffChaseVoNext = 0;
static ma_sound gHeartbeatSfx{};
static bool gHeartbeatReady = false;
static bool gHeartbeatActive = false;
static ma_sound gBigFallSfx{};
static bool gBigFallReady = false;
static ma_sound gStaffMeleeImpactSfx{};
static bool gStaffMeleeImpactReady = false;
static ma_sound gTitleMenuMusic{};
static bool gTitleMenuMusicReady = false;
static bool gTitleMenuMusicActive = false;
static ma_sound gLoadingScreenSfx{};
static bool gLoadingScreenSfxReady = false;
static bool gLoadingScreenSfxActive = false;
static float gChaseVoCooldown = 0.f;
static float gChaseVoDiceAccum = 0.f;
static std::atomic<int> gStoreSeqEvent{0};
static std::atomic<bool> gStoreFluoroOn{true};
static std::atomic<bool> gStoreDayNightPaused{false};
static ma_uint64 gStoreCyclePauseStartMs = 0;
// Player death: stop gameplay beds without seeking so respawn resumes the same PCM position.
static bool gDeathPauseHadStoreMusic = false;
static bool gDeathPauseHadHorror = false;
static bool gDeathPauseHadChase = false;
static bool gDeathPauseHadShrekEgg = false;
static std::mt19937 gRng;
static float gFullLengthSec = 0.25f;
static float gPowerSwitchDurationSec = kFallbackPowerSwitchSec;
static float gLightSwitchOnDurationSec = kFallbackLightSwitchOnSec;

enum class StoreAnimPhase : std::uint8_t {
  DayOnce,
  SwitchOff,
  Horror,
  DayLoop,
};
static StoreAnimPhase gStorePhase = StoreAnimPhase::DayOnce;
static int gDayCount = 1;
// Fluorescents off/on at engine times (same timeline as miniaudio mix), not game dt.
static bool gBlackoutScheduled = false;
static ma_uint64 gBlackoutAtEngineMs = 0;
static bool gDayRestoreScheduled = false;
static ma_uint64 gDayRestoreAtEngineMs = 0;

static ma_uint64 delayMsFromSeconds(float sec) {
  const double ms = static_cast<double>(sec) * 1000.0;
  const long long rounded = std::llround(ms);
  return static_cast<ma_uint64>(std::max<long long>(0, rounded));
}

static void scheduleBlackoutFromNow() {
  const ma_uint64 now = ma_engine_get_time_in_milliseconds(&gEngine);
  const float delaySec =
      std::clamp(gPowerSwitchDurationSec * kPowerSwitchBlackoutMidFrac, 0.02f, 60.f);
  gBlackoutAtEngineMs = now + delayMsFromSeconds(delaySec);
  gBlackoutScheduled = true;
}

static void scheduleDayRestoreFromNow() {
  const ma_uint64 now = ma_engine_get_time_in_milliseconds(&gEngine);
  const float delaySec =
      std::clamp(gLightSwitchOnDurationSec * kLightSwitchRestoreMidFrac, 0.02f, 60.f);
  gDayRestoreAtEngineMs = now + delayMsFromSeconds(delaySec);
  gDayRestoreScheduled = true;
}

static void dayMusicOnceEndedCb(void* /*pUserData*/, ma_sound* /*pSound*/) {
  if (gStoreDayNightPaused.load(std::memory_order_relaxed))
    return;
  gStoreSeqEvent.store(1, std::memory_order_release);
}
static void powerSwitchOffEndedCb(void* /*pUserData*/, ma_sound* /*pSound*/) {
  if (gStoreDayNightPaused.load(std::memory_order_relaxed))
    return;
  gStoreSeqEvent.store(2, std::memory_order_release);
}
static void horrorEndedCb(void* /*pUserData*/, ma_sound* /*pSound*/) {
  if (gStoreDayNightPaused.load(std::memory_order_relaxed))
    return;
  gStoreSeqEvent.store(3, std::memory_order_release);
}

static void refreshStoreMusicVolumeWithDuck();

static void shutdownPowerSwitchSfx() {
  if (!gPowerSwitchReady)
    return;
  if (ma_sound_is_playing(&gPowerSwitchSfx))
    ma_sound_stop(&gPowerSwitchSfx);
  ma_sound_uninit(&gPowerSwitchSfx);
  gPowerSwitchReady = false;
}

static bool initPowerSwitchSfx() {
  static const char* const paths[] = {
#ifdef VULKAN_GAME_POWER_SWITCH_SFX
      VULKAN_GAME_POWER_SWITCH_SFX,
#endif
#ifdef VULKAN_GAME_ASSETS_DIR
      VULKAN_GAME_ASSETS_DIR "/audio/power_switch.mp3",
#endif
      nullptr,
  };
  for (const char* const* p = paths; *p != nullptr; ++p) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = *p;
    sc.flags = MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION;
    ma_result r = ma_sound_init_ex(&gEngine, &sc, &gPowerSwitchSfx);
    if (r == MA_SUCCESS) {
      ma_sound_set_volume(&gPowerSwitchSfx, kStorePowerSwitchVol);
      float len = 0.f;
      if (ma_sound_get_length_in_seconds(&gPowerSwitchSfx, &len) == MA_SUCCESS && std::isfinite(len) &&
          len > 0.05f)
        gPowerSwitchDurationSec = len;
      else
        gPowerSwitchDurationSec = kFallbackPowerSwitchSec;
      gPowerSwitchReady = true;
      return true;
    }
  }
  return false;
}

static void shutdownChaseMusic() {
  if (!gChaseReady)
    return;
  ma_sound_set_end_callback(&gChaseMusic, nullptr, nullptr);
  if (ma_sound_is_playing(&gChaseMusic))
    ma_sound_stop(&gChaseMusic);
  ma_sound_uninit(&gChaseMusic);
  gChaseReady = false;
}

static void shutdownHorrorAmbient() {
  if (!gHorrorReady)
    return;
  ma_sound_set_end_callback(&gHorrorMusic, nullptr, nullptr);
  if (ma_sound_is_playing(&gHorrorMusic))
    ma_sound_stop(&gHorrorMusic);
  ma_sound_uninit(&gHorrorMusic);
  gHorrorReady = false;
}

static bool initHorrorAmbient() {
  static const char* const paths[] = {
#ifdef VULKAN_GAME_HORROR_AMBIENT_MP3
      VULKAN_GAME_HORROR_AMBIENT_MP3,
#endif
#ifdef VULKAN_GAME_ASSETS_DIR
      VULKAN_GAME_ASSETS_DIR "/audio/horror_ambient.mp3",
#endif
      nullptr,
  };
  for (const char* const* p = paths; *p != nullptr; ++p) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = *p;
    sc.flags = MA_SOUND_FLAG_STREAM | MA_SOUND_FLAG_NO_SPATIALIZATION;
    ma_result r = ma_sound_init_ex(&gEngine, &sc, &gHorrorMusic);
    if (r == MA_SUCCESS) {
      gHorrorReady = true;
      return true;
    }
  }
  return false;
}

static bool initChaseMusic() {
  static const char* const paths[] = {
#ifdef VULKAN_GAME_CHASE_MUSIC_MP3
      VULKAN_GAME_CHASE_MUSIC_MP3,
#endif
#ifdef VULKAN_GAME_ASSETS_DIR
      VULKAN_GAME_ASSETS_DIR "/audio/chase_music.mp3",
#endif
      nullptr,
  };
  for (const char* const* p = paths; *p != nullptr; ++p) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = *p;
    sc.flags = MA_SOUND_FLAG_STREAM | MA_SOUND_FLAG_NO_SPATIALIZATION;
    ma_result r = ma_sound_init_ex(&gEngine, &sc, &gChaseMusic);
    if (r == MA_SUCCESS) {
      ma_sound_set_looping(&gChaseMusic, MA_TRUE);
      gChaseReady = true;
      return true;
    }
  }
  return false;
}

static void shutdownShrekEggMusic() {
  if (!gShrekEggMusicReady)
    return;
  if (ma_sound_is_playing(&gShrekEggMusic))
    ma_sound_stop(&gShrekEggMusic);
  ma_sound_uninit(&gShrekEggMusic);
  gShrekEggMusicReady = false;
}

static bool initShrekEggMusic() {
#ifdef VULKAN_GAME_SHREK_EGG_MUSIC
  static const char* const paths[] = {
      VULKAN_GAME_SHREK_EGG_MUSIC,
      nullptr,
  };
  for (const char* const* p = paths; *p != nullptr; ++p) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = *p;
    sc.flags = MA_SOUND_FLAG_STREAM | MA_SOUND_FLAG_NO_SPATIALIZATION;
    ma_result r = ma_sound_init_ex(&gEngine, &sc, &gShrekEggMusic);
    if (r == MA_SUCCESS) {
      ma_sound_set_looping(&gShrekEggMusic, MA_TRUE);
      ma_sound_set_volume(&gShrekEggMusic, kGameplayMusicVol);
      gShrekEggMusicReady = true;
      return true;
    }
  }
#endif
  return false;
}

static void shutdownLightSwitchOnSfx() {
  if (!gLightSwitchOnReady)
    return;
  ma_sound_set_end_callback(&gLightSwitchOnSfx, nullptr, nullptr);
  if (ma_sound_is_playing(&gLightSwitchOnSfx))
    ma_sound_stop(&gLightSwitchOnSfx);
  ma_sound_uninit(&gLightSwitchOnSfx);
  gLightSwitchOnReady = false;
}

static bool initLightSwitchOnSfx() {
  static const char* const paths[] = {
#ifdef VULKAN_GAME_LIGHT_SWITCH_ON_SFX
      VULKAN_GAME_LIGHT_SWITCH_ON_SFX,
#endif
#ifdef VULKAN_GAME_ASSETS_DIR
      VULKAN_GAME_ASSETS_DIR "/audio/light_switch_on.mp3",
#endif
      nullptr,
  };
  for (const char* const* p = paths; *p != nullptr; ++p) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = *p;
    sc.flags = MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION;
    ma_result r = ma_sound_init_ex(&gEngine, &sc, &gLightSwitchOnSfx);
    if (r == MA_SUCCESS) {
      float len = 0.f;
      if (ma_sound_get_length_in_seconds(&gLightSwitchOnSfx, &len) == MA_SUCCESS && std::isfinite(len) &&
          len > 0.04f)
        gLightSwitchOnDurationSec = len;
      else
        gLightSwitchOnDurationSec = kFallbackLightSwitchOnSec;
      gLightSwitchOnReady = true;
      return true;
    }
  }
  return false;
}

static void shutdownStaffSpottedSfx() {
  if (!gStaffSpottedReady)
    return;
  for (int i = 0; i < 4; ++i) {
    if (ma_sound_is_playing(&gStaffSpottedSfx[i]))
      ma_sound_stop(&gStaffSpottedSfx[i]);
    ma_sound_uninit(&gStaffSpottedSfx[i]);
  }
  gStaffSpottedReady = false;
  gStaffSpottedNext = 0;
}

static bool initStaffSpottedSfx() {
  static const char* const paths[] = {
#ifdef VULKAN_GAME_STAFF_SPOTTED_SFX
      VULKAN_GAME_STAFF_SPOTTED_SFX,
#endif
#ifdef VULKAN_GAME_ASSETS_DIR
      VULKAN_GAME_ASSETS_DIR "/audio/staff_spotted.mp3",
#endif
      nullptr,
  };
  const char* okPath = nullptr;
  for (const char* const* p = paths; *p != nullptr; ++p) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = *p;
    sc.flags = MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION;
    ma_result r = ma_sound_init_ex(&gEngine, &sc, &gStaffSpottedSfx[0]);
    if (r == MA_SUCCESS) {
      okPath = *p;
      ma_sound_set_volume(&gStaffSpottedSfx[0], kStaffSpottedVol);
      ma_sound_set_looping(&gStaffSpottedSfx[0], MA_FALSE);
      break;
    }
  }
  if (!okPath)
    return false;
  for (int i = 1; i < 4; ++i) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = okPath;
    sc.flags = MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION;
    ma_result r = ma_sound_init_ex(&gEngine, &sc, &gStaffSpottedSfx[i]);
    if (r != MA_SUCCESS) {
      for (int j = 0; j < i; ++j)
        ma_sound_uninit(&gStaffSpottedSfx[j]);
      return false;
    }
    ma_sound_set_volume(&gStaffSpottedSfx[i], kStaffSpottedVol);
    ma_sound_set_looping(&gStaffSpottedSfx[i], MA_FALSE);
  }
  gStaffSpottedReady = true;
  return true;
}

static void shutdownStaffChaseVoSfx() {
  if (!gStaffChaseVoReady)
    return;
  for (int i = 0; i < 2; ++i) {
    if (ma_sound_is_playing(&gStaffChaseVoSfx[i]))
      ma_sound_stop(&gStaffChaseVoSfx[i]);
    ma_sound_uninit(&gStaffChaseVoSfx[i]);
  }
  gStaffChaseVoReady = false;
  gStaffChaseVoNext = 0;
  gChaseVoCooldown = 0.f;
  gChaseVoDiceAccum = 0.f;
}

static void shutdownHeartbeatSfx() {
  if (!gHeartbeatReady)
    return;
  gHeartbeatActive = false;
  if (ma_sound_is_playing(&gHeartbeatSfx))
    ma_sound_stop(&gHeartbeatSfx);
  ma_sound_uninit(&gHeartbeatSfx);
  gHeartbeatReady = false;
}

static void shutdownBigFallSfx() {
  if (!gBigFallReady)
    return;
  if (ma_sound_is_playing(&gBigFallSfx))
    ma_sound_stop(&gBigFallSfx);
  ma_sound_uninit(&gBigFallSfx);
  gBigFallReady = false;
}

static void shutdownStaffMeleeImpactSfx() {
  if (!gStaffMeleeImpactReady)
    return;
  if (ma_sound_is_playing(&gStaffMeleeImpactSfx))
    ma_sound_stop(&gStaffMeleeImpactSfx);
  ma_sound_uninit(&gStaffMeleeImpactSfx);
  gStaffMeleeImpactReady = false;
}

static bool initStaffMeleeImpactSfx() {
  static const char* const paths[] = {
#ifdef VULKAN_GAME_STAFF_MELEE_IMPACT_MP3
      VULKAN_GAME_STAFF_MELEE_IMPACT_MP3,
#endif
#ifdef VULKAN_GAME_ASSETS_DIR
      VULKAN_GAME_ASSETS_DIR "/audio/sfx_staff_melee_impact.mp3",
#endif
      nullptr,
  };
  for (const char* const* p = paths; *p != nullptr; ++p) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = *p;
    sc.flags = MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION;
    ma_result r = ma_sound_init_ex(&gEngine, &sc, &gStaffMeleeImpactSfx);
    if (r == MA_SUCCESS) {
      ma_sound_set_looping(&gStaffMeleeImpactSfx, MA_FALSE);
      ma_sound_set_volume(&gStaffMeleeImpactSfx, kStaffMeleeImpactVol);
      gStaffMeleeImpactReady = true;
      return true;
    }
  }
  return false;
}

static bool initBigFallSfx() {
  static const char* const paths[] = {
#ifdef VULKAN_GAME_BIG_FALL_MP3
      VULKAN_GAME_BIG_FALL_MP3,
#endif
#ifdef VULKAN_GAME_ASSETS_DIR
      VULKAN_GAME_ASSETS_DIR "/audio/sfx_big_fall_impact.mp3",
#endif
      nullptr,
  };
  for (const char* const* p = paths; *p != nullptr; ++p) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = *p;
    sc.flags = MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION;
    ma_result r = ma_sound_init_ex(&gEngine, &sc, &gBigFallSfx);
    if (r == MA_SUCCESS) {
      ma_sound_set_looping(&gBigFallSfx, MA_FALSE);
      ma_sound_set_volume(&gBigFallSfx, kBigFallImpactVol);
      gBigFallReady = true;
      return true;
    }
  }
  return false;
}

static bool initHeartbeatSfx() {
  static const char* const paths[] = {
#ifdef VULKAN_GAME_HEARTBEAT_MP3
      VULKAN_GAME_HEARTBEAT_MP3,
#endif
#ifdef VULKAN_GAME_ASSETS_DIR
      VULKAN_GAME_ASSETS_DIR "/audio/ui_low_health_heartbeat.mp3",
#endif
      nullptr,
  };
  for (const char* const* p = paths; *p != nullptr; ++p) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = *p;
    sc.flags = MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION;
    ma_result r = ma_sound_init_ex(&gEngine, &sc, &gHeartbeatSfx);
    if (r == MA_SUCCESS) {
      ma_sound_set_looping(&gHeartbeatSfx, MA_TRUE);
      ma_sound_set_volume(&gHeartbeatSfx, kLowHealthHeartbeatVol);
      gHeartbeatReady = true;
      return true;
    }
  }
  return false;
}

static bool initStaffChaseVoSfx() {
  static const char* const paths[] = {
#ifdef VULKAN_GAME_STAFF_CHASE_VO_SFX
      VULKAN_GAME_STAFF_CHASE_VO_SFX,
#endif
#ifdef VULKAN_GAME_ASSETS_DIR
      VULKAN_GAME_ASSETS_DIR "/audio/staff_chase_vo.mp3",
#endif
      nullptr,
  };
  const char* okPath = nullptr;
  for (const char* const* p = paths; *p != nullptr; ++p) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = *p;
    sc.flags = MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION;
    ma_result r = ma_sound_init_ex(&gEngine, &sc, &gStaffChaseVoSfx[0]);
    if (r == MA_SUCCESS) {
      okPath = *p;
      ma_sound_set_volume(&gStaffChaseVoSfx[0], kStaffChaseVoVol);
      ma_sound_set_looping(&gStaffChaseVoSfx[0], MA_FALSE);
      break;
    }
  }
  if (!okPath)
    return false;
  ma_sound_config sc = ma_sound_config_init_2(&gEngine);
  sc.pFilePath = okPath;
  sc.flags = MA_SOUND_FLAG_DECODE | MA_SOUND_FLAG_NO_SPATIALIZATION;
  ma_result r = ma_sound_init_ex(&gEngine, &sc, &gStaffChaseVoSfx[1]);
  if (r != MA_SUCCESS) {
    ma_sound_uninit(&gStaffChaseVoSfx[0]);
    return false;
  }
  ma_sound_set_volume(&gStaffChaseVoSfx[1], kStaffChaseVoVol);
  ma_sound_set_looping(&gStaffChaseVoSfx[1], MA_FALSE);
  gStaffChaseVoReady = true;
  return true;
}

static void shutdownStoreAmbience() {
  if (!gStoreAmbienceReady)
    return;
  gBlackoutScheduled = false;
  gDayRestoreScheduled = false;
  ma_sound_set_end_callback(&gStoreMusic, nullptr, nullptr);
  if (ma_sound_is_playing(&gStoreMusic))
    ma_sound_stop(&gStoreMusic);
  ma_sound_uninit(&gStoreMusic);
  ma_lpf_node_uninit(&gStoreHallLpf, nullptr);
  ma_delay_node_uninit(&gStoreComb2, nullptr);
  ma_delay_node_uninit(&gStoreComb, nullptr);
  gStoreAmbienceReady = false;
  gStoreDayMusicPaths.clear();
  gStoreDayMusicIdx = 0;
}

static void appendEnvMusicPathList(const char* raw, std::vector<std::string>& out) {
  if (!raw || !*raw)
    return;
  std::string s(raw);
  size_t start = 0;
  while (start < s.size()) {
    const size_t sep = s.find_first_of(";|", start);
    std::string tok = sep == std::string::npos ? s.substr(start) : s.substr(start, sep - start);
    while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t'))
      tok.erase(0, 1);
    while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t'))
      tok.pop_back();
    if (!tok.empty())
      out.push_back(std::move(tok));
    if (sep == std::string::npos)
      break;
    start = sep + 1;
  }
}

// Optional bundled names (copy your Downloads MP3s here). YouTube sources must be exported to MP3 locally
// (e.g. https://www.youtube.com/watch?v=jZoFzZ8pBmg ) — the engine only loads files from disk.
static void gatherStoreDayMusicCandidates(std::vector<std::string>& out) {
  out.clear();
  if (const char* e = std::getenv("VULKAN_GAME_STORE_DAY_MUSIC_PATHS"))
    appendEnvMusicPathList(e, out);
#ifdef VULKAN_GAME_ASSETS_DIR
  {
    const std::string ad = VULKAN_GAME_ASSETS_DIR;
    out.push_back("/home/memesdudeguy/Downloads/80s Retrowave _ Synthwave Music - Hackers by Karl Casey __ Royalty Free Copyright Safe Music.mp3");
    out.push_back(ad + "/audio/store_ambient_loop.mp3");
    out.push_back(ad + "/audio/store_ambient_loop.wav");
  }
#endif
#ifdef VULKAN_GAME_STORE_AMBIENT_WAV
  out.emplace_back(VULKAN_GAME_STORE_AMBIENT_WAV);
#endif
}

static bool reinitStoreMusicToIndex(size_t idx) {
  if (!gStoreAmbienceReady || idx >= gStoreDayMusicPaths.size())
    return false;
  ma_sound_set_end_callback(&gStoreMusic, nullptr, nullptr);
  if (ma_sound_is_playing(&gStoreMusic))
    ma_sound_stop(&gStoreMusic);
  ma_sound_uninit(&gStoreMusic);

  ma_sound_config mc = ma_sound_config_init_2(&gEngine);
  mc.pFilePath = gStoreDayMusicPaths[idx].c_str();
  mc.flags = MA_SOUND_FLAG_STREAM | MA_SOUND_FLAG_NO_SPATIALIZATION;
  mc.pInitialAttachment = &gStoreComb.baseNode;
  mc.initialAttachmentInputBusIndex = 0;
  const ma_result r = ma_sound_init_ex(&gEngine, &mc, &gStoreMusic);
  if (r != MA_SUCCESS) {
    std::fprintf(stderr, "[audio] store day music: could not load \"%s\" (ma_result %d)\n",
                 gStoreDayMusicPaths[idx].c_str(), static_cast<int>(r));
    return false;
  }
  gStoreDayMusicIdx = idx;
  ma_sound_set_volume(&gStoreMusic, kStoreDayMusicVol * gStoreMusicLinear01);
  ma_sound_set_looping(&gStoreMusic, MA_FALSE);
  ma_sound_set_end_callback(&gStoreMusic, dayMusicOnceEndedCb, nullptr);
  refreshStoreMusicVolumeWithDuck();
  return true;
}

static void shuffleStoreDayMusicForNewDay() {
  if (!gStoreAmbienceReady || gStoreDayMusicPaths.empty())
    return;
  if (gStoreDayMusicPaths.size() == 1u) {
    reinitStoreMusicToIndex(0);
    return;
  }
  std::uniform_int_distribution<size_t> dist(0, gStoreDayMusicPaths.size() - 1);
  for (int attempt = 0; attempt < 48; ++attempt) {
    const size_t j = dist(gRng);
    if (j != gStoreDayMusicIdx && reinitStoreMusicToIndex(j))
      return;
  }
  reinitStoreMusicToIndex(gStoreDayMusicIdx);
}

static bool initStoreAmbience() {
  ma_uint32 sampleRate = ma_engine_get_sample_rate(&gEngine);
  ma_uint32 ch = ma_engine_get_channels(&gEngine);
  if (sampleRate == 0 || ch == 0)
    return false;

  ma_node_graph* ng = ma_engine_get_node_graph(&gEngine);

  const ma_uint32 comb1Frames =
      (ma_uint32)std::max(256u, (ma_uint32)(static_cast<float>(sampleRate) * 0.034f));
  ma_delay_node_config dc1 = ma_delay_node_config_init(ch, sampleRate, comb1Frames, 0.42f);
  ma_result r = ma_delay_node_init(ng, &dc1, nullptr, &gStoreComb);
  if (r != MA_SUCCESS)
    return false;

  const ma_uint32 comb2Frames =
      (ma_uint32)std::max(256u, (ma_uint32)(static_cast<float>(sampleRate) * 0.056f));
  ma_delay_node_config dc2 = ma_delay_node_config_init(ch, sampleRate, comb2Frames, 0.38f);
  r = ma_delay_node_init(ng, &dc2, nullptr, &gStoreComb2);
  if (r != MA_SUCCESS) {
    ma_delay_node_uninit(&gStoreComb, nullptr);
    return false;
  }

  ma_lpf_node_config lpfCfg = ma_lpf_node_config_init(ch, sampleRate, 4200.0, 2);
  r = ma_lpf_node_init(ng, &lpfCfg, nullptr, &gStoreHallLpf);
  if (r != MA_SUCCESS) {
    ma_delay_node_uninit(&gStoreComb2, nullptr);
    ma_delay_node_uninit(&gStoreComb, nullptr);
    return false;
  }

  auto uninitStoreFxChain = []() {
    ma_lpf_node_uninit(&gStoreHallLpf, nullptr);
    ma_delay_node_uninit(&gStoreComb2, nullptr);
    ma_delay_node_uninit(&gStoreComb, nullptr);
  };
  r = ma_node_attach_output_bus(&gStoreComb.baseNode, 0, &gStoreComb2.baseNode, 0);
  if (r != MA_SUCCESS) {
    uninitStoreFxChain();
    return false;
  }
  r = ma_node_attach_output_bus(&gStoreComb2.baseNode, 0, &gStoreHallLpf.baseNode, 0);
  if (r != MA_SUCCESS) {
    uninitStoreFxChain();
    return false;
  }
  r = ma_node_attach_output_bus(&gStoreHallLpf.baseNode, 0, ma_node_graph_get_endpoint(ng), 0);
  if (r != MA_SUCCESS) {
    uninitStoreFxChain();
    return false;
  }

  ma_delay_node_set_wet(&gStoreComb, 0.33f);
  ma_delay_node_set_dry(&gStoreComb, 0.66f);
  ma_delay_node_set_decay(&gStoreComb, 0.44f);

  ma_delay_node_set_wet(&gStoreComb2, 0.28f);
  ma_delay_node_set_dry(&gStoreComb2, 0.44f);
  ma_delay_node_set_decay(&gStoreComb2, 0.5f);

  std::vector<std::string> candidates;
  gatherStoreDayMusicCandidates(candidates);
  gStoreDayMusicPaths.clear();
  for (const std::string& p : candidates) {
    std::error_code ec;
    if (fs::is_regular_file(fs::path(p), ec))
      gStoreDayMusicPaths.push_back(p);
  }
  if (gStoreDayMusicPaths.empty()) {
    ma_lpf_node_uninit(&gStoreHallLpf, nullptr);
    ma_delay_node_uninit(&gStoreComb2, nullptr);
    ma_delay_node_uninit(&gStoreComb, nullptr);
    return false;
  }
  std::shuffle(gStoreDayMusicPaths.begin(), gStoreDayMusicPaths.end(), gRng);

  bool loaded = false;
  for (size_t i = 0; i < gStoreDayMusicPaths.size(); ++i) {
    ma_sound_config mc = ma_sound_config_init_2(&gEngine);
    mc.pFilePath = gStoreDayMusicPaths[i].c_str();
    mc.flags = MA_SOUND_FLAG_STREAM | MA_SOUND_FLAG_NO_SPATIALIZATION;
    mc.pInitialAttachment = &gStoreComb.baseNode;
    mc.initialAttachmentInputBusIndex = 0;
    r = ma_sound_init_ex(&gEngine, &mc, &gStoreMusic);
    if (r == MA_SUCCESS) {
      gStoreDayMusicIdx = i;
      loaded = true;
      break;
    }
    std::fprintf(stderr, "[audio] store day music: skip (decode/init failed) \"%s\"\n",
                 gStoreDayMusicPaths[i].c_str());
  }
  if (!loaded) {
    gStoreDayMusicPaths.clear();
    ma_lpf_node_uninit(&gStoreHallLpf, nullptr);
    ma_delay_node_uninit(&gStoreComb2, nullptr);
    ma_delay_node_uninit(&gStoreComb, nullptr);
    return false;
  }

  gStoreMusicLinear01 = 1.f;
  ma_sound_set_volume(&gStoreMusic, kStoreDayMusicVol * gStoreMusicLinear01);
  ma_sound_set_looping(&gStoreMusic, MA_FALSE);
  ma_sound_set_end_callback(&gStoreMusic, dayMusicOnceEndedCb, nullptr);
  gStorePhase = StoreAnimPhase::DayOnce;
  gStoreFluoroOn.store(true, std::memory_order_relaxed);
  gStoreSeqEvent.store(0, std::memory_order_relaxed);
  gBlackoutScheduled = false;
  gDayRestoreScheduled = false;
  // Do not start here: title menu / boot uses audioSetStoreDayNightCyclePaused(true) with no audible day bed.
  // Gameplay entry calls audioSetStoreDayNightCyclePaused(false), which cold-starts the day loop when needed.
  return true;
}

static void shutdownLoadingScreenSfx() {
  if (gLoadingScreenSfxReady) {
    if (ma_sound_is_playing(&gLoadingScreenSfx))
      ma_sound_stop(&gLoadingScreenSfx);
    ma_sound_uninit(&gLoadingScreenSfx);
    gLoadingScreenSfxReady = false;
    gLoadingScreenSfxActive = false;
  }
}

static void shutdownTitleMenuMusic() {
  if (gTitleMenuMusicReady) {
    if (ma_sound_is_playing(&gTitleMenuMusic))
      ma_sound_stop(&gTitleMenuMusic);
    ma_sound_uninit(&gTitleMenuMusic);
    gTitleMenuMusicReady = false;
    gTitleMenuMusicActive = false;
  }
  ma_lpf_node_uninit(&gTitleHallLpf, nullptr);
  ma_delay_node_uninit(&gTitleComb2, nullptr);
  ma_delay_node_uninit(&gTitleComb, nullptr);
}

static bool initTitleMenuMusic() {
  std::vector<std::string> pathStrs;
#ifdef VULKAN_GAME_TITLE_MENU_WAV
  pathStrs.emplace_back(VULKAN_GAME_TITLE_MENU_WAV);
#endif
#ifdef VULKAN_GAME_ASSETS_DIR
  pathStrs.emplace_back(std::string(VULKAN_GAME_ASSETS_DIR) + "/audio/New_Project.wav");
  pathStrs.emplace_back(std::string(VULKAN_GAME_ASSETS_DIR) + "/audio/the_long_hall.wav");
#endif
  if (const char* e = std::getenv("VULKAN_GAME_TITLE_MENU_WAV")) {
    if (e[0] != '\0')
      pathStrs.emplace_back(e);
  }
  if (const char* home = std::getenv("HOME")) {
    pathStrs.emplace_back(std::string(home) + "/Downloads/New_Project.wav");
    pathStrs.emplace_back(std::string(home) + "/Downloads/the_long_hall.wav");
  }
  if (const char* userProfile = std::getenv("USERPROFILE")) {
    pathStrs.emplace_back(std::string(userProfile) + "/Downloads/New_Project.wav");
    pathStrs.emplace_back(std::string(userProfile) + "/Downloads/the_long_hall.wav");
  }

  ma_node_graph* ng = ma_engine_get_node_graph(&gEngine);
  const ma_uint32 ch = ma_engine_get_channels(&gEngine);
  const ma_uint32 sampleRate = ma_engine_get_sample_rate(&gEngine);
  const ma_uint32 comb1Frames =
      static_cast<ma_uint32>(std::max(1.0, std::round(sampleRate * 0.083)));  // ~83 ms
  const ma_uint32 comb2Frames =
      static_cast<ma_uint32>(std::max(1.0, std::round(sampleRate * 0.109)));  // ~109 ms
  ma_delay_node_config tdc1 = ma_delay_node_config_init(ch, sampleRate, comb1Frames, 0.40f);
  ma_result r = ma_delay_node_init(ng, &tdc1, nullptr, &gTitleComb);
  if (r != MA_SUCCESS)
    return false;
  ma_delay_node_config tdc2 = ma_delay_node_config_init(ch, sampleRate, comb2Frames, 0.35f);
  r = ma_delay_node_init(ng, &tdc2, nullptr, &gTitleComb2);
  if (r != MA_SUCCESS) {
    ma_delay_node_uninit(&gTitleComb, nullptr);
    return false;
  }
  ma_lpf_node_config tlpfCfg = ma_lpf_node_config_init(ch, sampleRate, 3600.0, 2);
  r = ma_lpf_node_init(ng, &tlpfCfg, nullptr, &gTitleHallLpf);
  if (r != MA_SUCCESS) {
    ma_delay_node_uninit(&gTitleComb2, nullptr);
    ma_delay_node_uninit(&gTitleComb, nullptr);
    return false;
  }
  r = ma_node_attach_output_bus(&gTitleComb.baseNode, 0, &gTitleComb2.baseNode, 0);
  if (r != MA_SUCCESS) {
    ma_lpf_node_uninit(&gTitleHallLpf, nullptr);
    ma_delay_node_uninit(&gTitleComb2, nullptr);
    ma_delay_node_uninit(&gTitleComb, nullptr);
    return false;
  }
  r = ma_node_attach_output_bus(&gTitleComb2.baseNode, 0, &gTitleHallLpf.baseNode, 0);
  if (r != MA_SUCCESS) {
    ma_lpf_node_uninit(&gTitleHallLpf, nullptr);
    ma_delay_node_uninit(&gTitleComb2, nullptr);
    ma_delay_node_uninit(&gTitleComb, nullptr);
    return false;
  }
  r = ma_node_attach_output_bus(&gTitleHallLpf.baseNode, 0, ma_node_graph_get_endpoint(ng), 0);
  if (r != MA_SUCCESS) {
    ma_lpf_node_uninit(&gTitleHallLpf, nullptr);
    ma_delay_node_uninit(&gTitleComb2, nullptr);
    ma_delay_node_uninit(&gTitleComb, nullptr);
    return false;
  }
  ma_delay_node_set_wet(&gTitleComb, 0.26f);
  ma_delay_node_set_dry(&gTitleComb, 0.78f);
  ma_delay_node_set_decay(&gTitleComb, 0.40f);
  ma_delay_node_set_wet(&gTitleComb2, 0.22f);
  ma_delay_node_set_dry(&gTitleComb2, 0.54f);
  ma_delay_node_set_decay(&gTitleComb2, 0.46f);

  for (const auto& ps : pathStrs) {
    const char* p = ps.c_str();
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = p;
    sc.flags = MA_SOUND_FLAG_STREAM | MA_SOUND_FLAG_NO_SPATIALIZATION;
    sc.pInitialAttachment = &gTitleComb.baseNode;
    sc.initialAttachmentInputBusIndex = 0;
    r = ma_sound_init_ex(&gEngine, &sc, &gTitleMenuMusic);
    if (r == MA_SUCCESS) {
      ma_sound_set_looping(&gTitleMenuMusic, MA_TRUE);
      ma_sound_set_volume(&gTitleMenuMusic, kTitleMenuMusicVol);
      gTitleMenuMusicReady = true;
      gTitleMenuMusicActive = false;
      return true;
    }
  }
  ma_lpf_node_uninit(&gTitleHallLpf, nullptr);
  ma_delay_node_uninit(&gTitleComb2, nullptr);
  ma_delay_node_uninit(&gTitleComb, nullptr);
  return false;
}

static void initLoadingScreenSfx() {
  std::vector<std::string> paths;
#ifdef VULKAN_GAME_ASSETS_DIR
  paths.emplace_back(std::string(VULKAN_GAME_ASSETS_DIR) + "/audio/freesound_community-space-ambience-56265.mp3");
#endif
  if (const char* home = std::getenv("HOME"))
    paths.emplace_back(std::string(home) + "/Downloads/freesound_community-space-ambience-56265.mp3");
  if (const char* up = std::getenv("USERPROFILE"))
    paths.emplace_back(std::string(up) + "/Downloads/freesound_community-space-ambience-56265.mp3");
  for (const auto& p : paths) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = p.c_str();
    sc.flags = MA_SOUND_FLAG_STREAM | MA_SOUND_FLAG_NO_SPATIALIZATION;
    if (ma_sound_init_ex(&gEngine, &sc, &gLoadingScreenSfx) == MA_SUCCESS) {
      ma_sound_set_looping(&gLoadingScreenSfx, MA_FALSE);
      ma_sound_set_volume(&gLoadingScreenSfx, 2.0f);
      gLoadingScreenSfxReady = true;
      return;
    }
  }
}

bool audioInit() {
  ma_engine_config engCfg = ma_engine_config_init();
  ma_result r = ma_engine_init(&engCfg, &gEngine);
  if (r != MA_SUCCESS)
    return false;

  std::string footPath;
#ifdef VULKAN_GAME_ASSETS_DIR
  footPath = std::string(VULKAN_GAME_ASSETS_DIR) + "/audio/sfx_footstep_concrete.mp3";
#endif
  const char* footC = footPath.empty() ? nullptr : footPath.c_str();
  const ma_uint32 streamFlags = MA_SOUND_FLAG_STREAM | MA_SOUND_FLAG_NO_SPATIALIZATION;
  for (int i = 0; i < 8; ++i) {
    ma_sound_config sc = ma_sound_config_init_2(&gEngine);
    sc.pFilePath = footC;
    sc.flags = streamFlags;
    r = ma_sound_init_ex(&gEngine, &sc, &gFootstep[i]);
    if (r != MA_SUCCESS) {
      for (int j = 0; j < i; ++j)
        ma_sound_uninit(&gFootstep[j]);
      ma_engine_uninit(&gEngine);
      return false;
    }
    ma_sound_set_looping(&gFootstep[i], MA_FALSE);
  }

  gSlideReady = false;
#ifdef VULKAN_GAME_ASSETS_DIR
  {
    std::string slidePath = std::string(VULKAN_GAME_ASSETS_DIR) + "/audio/sfx_slide_body_fall.mp3";
    ma_sound_config scSlide = ma_sound_config_init_2(&gEngine);
    scSlide.pFilePath = slidePath.c_str();
    scSlide.flags = streamFlags;
    r = ma_sound_init_ex(&gEngine, &scSlide, &gSlideSound);
    if (r == MA_SUCCESS) {
      ma_sound_set_looping(&gSlideSound, MA_FALSE);
      ma_sound_set_volume(&gSlideSound, 0.68f);
      gSlideReady = true;
    }
  }
#endif

  {
    std::random_device rd;
    gRng.seed(rd());
  }

  gStoreAmbienceReady = initStoreAmbience();
  if (!gStoreAmbienceReady)
    std::fprintf(stderr, "Store ambience: could not load music (delay/reverb chain skipped).\n");

  if (!initPowerSwitchSfx())
    std::fprintf(stderr, "Power switch SFX: could not load (blackout will be silent).\n");

  if (!initHorrorAmbient())
    std::fprintf(stderr, "Horror ambient: could not load (will skip night segment).\n");

  if (!initChaseMusic())
    std::fprintf(stderr, "Chase music: could not load (night chase bed will be silent).\n");

  if (!initShrekEggMusic()) {
    // optional easter egg bed — silent if MP3 missing
  }

  if (!initLightSwitchOnSfx())
    std::fprintf(stderr, "Light switch on SFX: could not load (may skip straight to daytime music).\n");

  if (!initStaffSpottedSfx())
    std::fprintf(stderr, "Staff spotted SFX: could not load (night spot sting will be silent).\n");

  if (!initStaffChaseVoSfx())
    std::fprintf(stderr, "Staff chase VO: could not load (closed-store line will be silent).\n");

  if (!initHeartbeatSfx())
    std::fprintf(stderr, "Low-health heartbeat: could not load brvhrtz-heartbeat MP3 (mercy heal still runs).\n");

  if (!initBigFallSfx())
    std::fprintf(stderr, "Big fall impact: could not load universfield-falling MP3 (silent on hard landings).\n");

  if (!initStaffMeleeImpactSfx())
    std::fprintf(stderr,
                 "Staff/push impact: could not load fist-fight MP3 (silent on staff hits and shoves).\n");

  float lenSec = 0.f;
  if (ma_sound_get_length_in_seconds(&gFootstep[0], &lenSec) == MA_SUCCESS && lenSec > 0.01f)
    gFullLengthSec = lenSec;

  if (!initTitleMenuMusic())
    std::fprintf(stderr, "Title menu music: could not load the_long_hall.wav (title screen will be quiet).\n");

  initLoadingScreenSfx();

  gReady = true;
  return true;
}

void audioShutdown() {
  if (!gReady)
    return;
  shutdownLoadingScreenSfx();
  shutdownTitleMenuMusic();
  shutdownStaffMeleeImpactSfx();
  shutdownBigFallSfx();
  shutdownHeartbeatSfx();
  shutdownStoreAmbience();
  shutdownShrekEggMusic();
  shutdownChaseMusic();
  shutdownHorrorAmbient();
  shutdownLightSwitchOnSfx();
  shutdownStaffChaseVoSfx();
  shutdownStaffSpottedSfx();
  shutdownPowerSwitchSfx();
  if (gSlideReady) {
    ma_sound_uninit(&gSlideSound);
    gSlideReady = false;
  }
  for (int i = 0; i < 8; ++i)
    ma_sound_uninit(&gFootstep[i]);
  ma_engine_uninit(&gEngine);
  gStoreFluoroOn.store(true, std::memory_order_relaxed);
  gStoreSeqEvent.store(0, std::memory_order_relaxed);
  gBlackoutScheduled = false;
  gDayRestoreScheduled = false;
  gStorePhase = StoreAnimPhase::DayOnce;
  gSlideAudioActive = false;
  gReady = false;
}

void audioPlayFootstep(bool landing, float volumeMul) {
  if (!gReady)
    return;
  volumeMul = std::clamp(volumeMul, 0.f, 2.5f);
  if (volumeMul <= 0.002f)
    return;

  static int next = 0;
  ma_sound* s = &gFootstep[next];
  next = (next + 1) % 8;

  std::uniform_real_distribution<float> pitchDist(0.82f, 1.22f);
  std::uniform_real_distribution<float> lenFrac(0.38f, 1.0f);

  const float pitch = pitchDist(gRng);
  const float frac = lenFrac(gRng);

  ma_sound_stop(s);
  ma_sound_reset_stop_time_and_fade(s);
  ma_sound_seek_to_pcm_frame(s, 0);
  const float baseVol = landing ? kFootstepVolLand : kFootstepVolWalk;
  ma_sound_set_volume(s, baseVol * volumeMul);
  ma_sound_set_pitch(s, pitch);

  const ma_uint64 nowMs = ma_engine_get_time_in_milliseconds(&gEngine);
  const float wallSec = (gFullLengthSec * frac) / std::max(0.2f, pitch);
  const ma_uint64 durMs = static_cast<ma_uint64>(std::ceil(static_cast<double>(wallSec) * 1000.0));
  ma_sound_set_stop_time_in_milliseconds(s, nowMs + durMs);
  ma_sound_start(s);
}

void audioSetSlide(bool active) {
  if (!gReady || !gSlideReady)
    return;
  if (active) {
    if (!gSlideAudioActive) {
      gSlideAudioActive = true;
      ma_sound_stop(&gSlideSound);
      ma_sound_reset_stop_time_and_fade(&gSlideSound);
      ma_sound_seek_to_pcm_frame(&gSlideSound, 0);
      ma_sound_start(&gSlideSound);
    }
  } else if (gSlideAudioActive) {
    gSlideAudioActive = false;
    ma_sound_stop(&gSlideSound);
  }
}

static void refreshStoreMusicVolumeWithDuck() {
  if (!gStoreAmbienceReady || !ma_sound_is_playing(&gStoreMusic))
    return;
  const float duckMul = 1.f - kShrekEggDuckOtherMusicMax * gShrekEggDuckOtherMusic01;
  ma_sound_set_volume(&gStoreMusic, kStoreDayMusicVol * gStoreMusicLinear01 * duckMul);
}

void audioSetStoreAmbienceVolume(float linear01) {
  if (!gReady || !gStoreAmbienceReady)
    return;
  gStoreMusicLinear01 = std::clamp(linear01, 0.f, 1.f);
  refreshStoreMusicVolumeWithDuck();
}

static void stopChaseMusicLayer() {
  gChaseCrossfade = 0.f;
  if (!gChaseReady)
    return;
  if (ma_sound_is_playing(&gChaseMusic))
    ma_sound_stop(&gChaseMusic);
}

static void storeFluoroSet(bool on) {
  if (gStoreDayNightPaused.load(std::memory_order_relaxed))
    return;
  gStoreFluoroOn.store(on, std::memory_order_relaxed);
}

static void startDayLoopMusic() {
  stopChaseMusicLayer();
  storeFluoroSet(true);
  ++gDayCount;
  if (!gStoreAmbienceReady)
    return;
  shuffleStoreDayMusicForNewDay();
  ma_sound_set_end_callback(&gStoreMusic, dayMusicOnceEndedCb, nullptr);
  ma_sound_set_looping(&gStoreMusic, MA_FALSE);
  ma_sound_stop(&gStoreMusic);
  ma_sound_reset_stop_time_and_fade(&gStoreMusic);
  ma_sound_seek_to_pcm_frame(&gStoreMusic, 0);
  gStoreMusicLinear01 = 1.f;
  ma_sound_set_volume(&gStoreMusic, kStoreDayMusicVol * gStoreMusicLinear01);
  ma_sound_start(&gStoreMusic);
  gStorePhase = StoreAnimPhase::DayLoop;
  refreshStoreMusicVolumeWithDuck();
}

// Light-on SFX first; fluorescents + daytime music at midpoint of that clip (engine ms).
static void playLightSwitchOnAndRestoreDay() {
  if (gLightSwitchOnReady) {
    ma_sound_set_end_callback(&gLightSwitchOnSfx, nullptr, nullptr);
    ma_sound_stop(&gLightSwitchOnSfx);
    ma_sound_reset_stop_time_and_fade(&gLightSwitchOnSfx);
    ma_sound_seek_to_pcm_frame(&gLightSwitchOnSfx, 0);
    ma_sound_set_volume(&gLightSwitchOnSfx, kStoreLightSwitchVol);
    ma_sound_start(&gLightSwitchOnSfx);
    scheduleDayRestoreFromNow();
  } else {
    startDayLoopMusic();
  }
}

static void afterHorrorTrackComplete() {
  stopChaseMusicLayer();
  ma_sound_set_end_callback(&gHorrorMusic, nullptr, nullptr);
  if (ma_sound_is_playing(&gHorrorMusic))
    ma_sound_stop(&gHorrorMusic);
  playLightSwitchOnAndRestoreDay();
}

static void startHorrorMusic() {
  stopChaseMusicLayer();
  if (gHorrorReady) {
    ma_sound_set_end_callback(&gHorrorMusic, horrorEndedCb, nullptr);
    ma_sound_stop(&gHorrorMusic);
    ma_sound_reset_stop_time_and_fade(&gHorrorMusic);
    ma_sound_seek_to_pcm_frame(&gHorrorMusic, 0);
    ma_sound_set_volume(&gHorrorMusic, kGameplayMusicVol);
    ma_sound_start(&gHorrorMusic);
    gStorePhase = StoreAnimPhase::Horror;
  } else {
    playLightSwitchOnAndRestoreDay();
  }
}

static void afterPowerSwitchOffComplete() {
  ma_sound_set_end_callback(&gPowerSwitchSfx, nullptr, nullptr);
  startHorrorMusic();
}

static ma_uint64 soundCursorOrZero(ma_sound* s) {
  if (!s)
    return 0;
  ma_uint64 f = 0;
  if (ma_sound_get_cursor_in_pcm_frames(s, &f) != MA_SUCCESS)
    return 0;
  return f;
}

static void stopSeekMaybeStart(ma_sound* s, ma_uint64 frame, bool start) {
  if (!s)
    return;
  ma_sound_stop(s);
  ma_sound_reset_stop_time_and_fade(s);
  ma_sound_seek_to_pcm_frame(s, frame);
  if (start)
    ma_sound_start(s);
}

void audioSetTitleMenuMusicActive(bool active) {
  if (!gReady || !gTitleMenuMusicReady)
    return;
  if (active == gTitleMenuMusicActive)
    return;
  gTitleMenuMusicActive = active;
  if (active) {
    ma_sound_stop(&gTitleMenuMusic);
    ma_sound_reset_stop_time_and_fade(&gTitleMenuMusic);
    ma_sound_seek_to_pcm_frame(&gTitleMenuMusic, 0);
    ma_sound_set_fade_in_milliseconds(&gTitleMenuMusic, 0, 1, 800);
    ma_sound_start(&gTitleMenuMusic);
  } else {
    constexpr ma_uint64 kFadeOutMs = 2200;
    ma_sound_set_fade_in_milliseconds(&gTitleMenuMusic, -1, 0, kFadeOutMs);
    const ma_uint64 now = ma_engine_get_time_in_milliseconds(&gEngine);
    ma_sound_set_stop_time_in_milliseconds(&gTitleMenuMusic, now + kFadeOutMs);
  }
}

void audioSetLoadingScreenActive(bool active) {
  if (!gReady || !gLoadingScreenSfxReady)
    return;
  if (active == gLoadingScreenSfxActive)
    return;
  gLoadingScreenSfxActive = active;
  if (active) {
    ma_sound_stop(&gLoadingScreenSfx);
    ma_sound_reset_stop_time_and_fade(&gLoadingScreenSfx);
    ma_sound_seek_to_pcm_frame(&gLoadingScreenSfx, 0);
    ma_sound_set_fade_in_milliseconds(&gLoadingScreenSfx, 0, 1, 1400);
    ma_sound_start(&gLoadingScreenSfx);
  } else {
    constexpr ma_uint64 kFadeOutMs = 2400;
    ma_sound_set_fade_in_milliseconds(&gLoadingScreenSfx, -1, 0, kFadeOutMs);
    const ma_uint64 now = ma_engine_get_time_in_milliseconds(&gEngine);
    ma_sound_set_stop_time_in_milliseconds(&gLoadingScreenSfx, now + kFadeOutMs);
  }
}

void audioSetStoreDayNightCyclePaused(bool paused) {
  if (!gReady)
    return;
  const bool was = gStoreDayNightPaused.exchange(paused, std::memory_order_acq_rel);
  if (paused && !was) {
    gStoreCyclePauseStartMs = ma_engine_get_time_in_milliseconds(&gEngine);
    gDeathPauseHadStoreMusic = gStoreAmbienceReady && ma_sound_is_playing(&gStoreMusic);
    gDeathPauseHadHorror = gHorrorReady && ma_sound_is_playing(&gHorrorMusic);
    gDeathPauseHadChase = gChaseReady && ma_sound_is_playing(&gChaseMusic);
    gDeathPauseHadShrekEgg = gShrekEggMusicReady && ma_sound_is_playing(&gShrekEggMusic);
    if (gDeathPauseHadStoreMusic)
      ma_sound_stop(&gStoreMusic);
    if (gDeathPauseHadHorror)
      ma_sound_stop(&gHorrorMusic);
    if (gDeathPauseHadChase)
      ma_sound_stop(&gChaseMusic);
    if (gDeathPauseHadShrekEgg)
      ma_sound_stop(&gShrekEggMusic);
  } else if (!paused && was) {
    const ma_uint64 now = ma_engine_get_time_in_milliseconds(&gEngine);
    const ma_uint64 add = (now > gStoreCyclePauseStartMs) ? (now - gStoreCyclePauseStartMs) : 0;
    if (add > 0) {
      if (gBlackoutScheduled)
        gBlackoutAtEngineMs += add;
      if (gDayRestoreScheduled)
        gDayRestoreAtEngineMs += add;
    }
    if (gDeathPauseHadStoreMusic) {
      ma_sound_start(&gStoreMusic);
      refreshStoreMusicVolumeWithDuck();
    } else if (gStoreAmbienceReady && !ma_sound_is_playing(&gStoreMusic) &&
               (gStorePhase == StoreAnimPhase::DayOnce || gStorePhase == StoreAnimPhase::DayLoop) &&
               gStoreFluoroOn.load(std::memory_order_relaxed)) {
      // Start day bed at current cursor (0 after audioResetToNewGame; saved PCM after restore). Never seek to 0 here
      // or Continue would wipe the loaded store track position.
      ma_sound_set_end_callback(&gStoreMusic, dayMusicOnceEndedCb, nullptr);
      ma_sound_set_looping(&gStoreMusic, MA_FALSE);
      ma_sound_start(&gStoreMusic);
      refreshStoreMusicVolumeWithDuck();
    }
    if (gDeathPauseHadHorror)
      ma_sound_start(&gHorrorMusic);
    if (gDeathPauseHadChase)
      ma_sound_start(&gChaseMusic);
    if (gDeathPauseHadShrekEgg)
      ma_sound_start(&gShrekEggMusic);
    gDeathPauseHadStoreMusic = false;
    gDeathPauseHadHorror = false;
    gDeathPauseHadChase = false;
    gDeathPauseHadShrekEgg = false;
  }
}

void audioUpdateStore(float /*dt*/) {
  if (!gReady || !gStoreAmbienceReady)
    return;
  if (gStoreDayNightPaused.load(std::memory_order_relaxed))
    return;

  // Run events first so scheduleBlackoutFromNow / scheduleDayRestoreFromNow in this same call are
  // picked up below (checking deadlines before events delayed blackout by a full frame + felt wrong).
  for (;;) {
    const int ev = gStoreSeqEvent.exchange(0, std::memory_order_acq_rel);
    if (ev == 0)
      break;

    if (ev == 1 && (gStorePhase == StoreAnimPhase::DayOnce || gStorePhase == StoreAnimPhase::DayLoop)) {
      if (gPowerSwitchReady) {
        ma_sound_set_end_callback(&gPowerSwitchSfx, powerSwitchOffEndedCb, nullptr);
        ma_sound_stop(&gPowerSwitchSfx);
        ma_sound_reset_stop_time_and_fade(&gPowerSwitchSfx);
        ma_sound_seek_to_pcm_frame(&gPowerSwitchSfx, 0);
        ma_sound_start(&gPowerSwitchSfx);
        scheduleBlackoutFromNow();
        gStorePhase = StoreAnimPhase::SwitchOff;
      } else {
        storeFluoroSet(false);
        afterPowerSwitchOffComplete();
      }
      continue;
    }

    if (ev == 2 && gStorePhase == StoreAnimPhase::SwitchOff) {
      afterPowerSwitchOffComplete();
      continue;
    }

    if (ev == 3 && gStorePhase == StoreAnimPhase::Horror) {
      afterHorrorTrackComplete();
      continue;
    }
  }

  const ma_uint64 engNow = ma_engine_get_time_in_milliseconds(&gEngine);
  if (gBlackoutScheduled && engNow >= gBlackoutAtEngineMs) {
    gBlackoutScheduled = false;
    storeFluoroSet(false);
  }
  if (gDayRestoreScheduled && engNow >= gDayRestoreAtEngineMs) {
    gDayRestoreScheduled = false;
    startDayLoopMusic();
  }

  // Self-heal sequencing if an expected clip did not resume (e.g., state survived but stream stopped).
  // This keeps the day/night state machine from getting stuck silent.
  if (!gStoreDayNightPaused.load(std::memory_order_relaxed)) {
    if (gStorePhase == StoreAnimPhase::SwitchOff) {
      const bool swPlaying = gPowerSwitchReady && ma_sound_is_playing(&gPowerSwitchSfx);
      if (!gBlackoutScheduled && !swPlaying) {
        storeFluoroSet(false);
        afterPowerSwitchOffComplete();
      }
    } else if (gStorePhase == StoreAnimPhase::Horror) {
      const bool horrorPlaying = gHorrorReady && ma_sound_is_playing(&gHorrorMusic);
      if (!gDayRestoreScheduled && !horrorPlaying)
        afterHorrorTrackComplete();
    } else if (gStorePhase == StoreAnimPhase::DayOnce || gStorePhase == StoreAnimPhase::DayLoop) {
      if (gStoreAmbienceReady && gStoreFluoroOn.load(std::memory_order_relaxed) && !gBlackoutScheduled &&
          !gDayRestoreScheduled) {
        const bool swPlaying = gPowerSwitchReady && ma_sound_is_playing(&gPowerSwitchSfx);
        if (!swPlaying && !ma_sound_is_playing(&gStoreMusic)) {
          ma_sound_set_end_callback(&gStoreMusic, dayMusicOnceEndedCb, nullptr);
          ma_sound_set_looping(&gStoreMusic, MA_FALSE);
          ma_sound_start(&gStoreMusic);
          gStorePhase = StoreAnimPhase::DayLoop;
          refreshStoreMusicVolumeWithDuck();
        }
      }
    }
  }
}

void audioPlayStaffSpotted() {
  if (!gReady || !gStaffSpottedReady)
    return;
  ma_sound* s = &gStaffSpottedSfx[gStaffSpottedNext];
  gStaffSpottedNext = (gStaffSpottedNext + 1) % 4;
  ma_sound_stop(s);
  ma_sound_reset_stop_time_and_fade(s);
  ma_sound_seek_to_pcm_frame(s, 0);
  ma_sound_start(s);
}

static void updateStaffChaseMusicCrossfade(float dt, bool anyStaffChasing) {
  if (!gReady || dt <= 0.f)
    return;
  const bool night = !audioAreStoreFluorescentsOn();
  const float target = (night && anyStaffChasing && gChaseReady) ? 1.f : 0.f;
  const bool nightPursuit = night && anyStaffChasing;
  float fadeSec = kChaseMusicFadeSec;
  if (gChaseCrossfade < target)
    fadeSec = nightPursuit ? kChaseMusicFadeInPursuitSec : kChaseMusicFadeSec;
  else if (gChaseCrossfade > target)
    fadeSec = nightPursuit ? kChaseMusicFadeSec : kChaseMusicFadeOutAfterPursuitSec;
  const float rate = fadeSec > 1e-4f ? (dt / fadeSec) : 1.f;
  if (gChaseCrossfade < target)
    gChaseCrossfade = std::min(target, gChaseCrossfade + rate);
  else
    gChaseCrossfade = std::max(target, gChaseCrossfade - rate);

  const float shrekDuckMul = 1.f - kShrekEggDuckOtherMusicMax * gShrekEggDuckOtherMusic01;

  if (gChaseReady) {
    if (gChaseCrossfade > 0.002f || target > 0.f) {
      if (!ma_sound_is_playing(&gChaseMusic)) {
        ma_sound_stop(&gChaseMusic);
        ma_sound_reset_stop_time_and_fade(&gChaseMusic);
        ma_sound_seek_to_pcm_frame(&gChaseMusic, 0);
        ma_sound_set_volume(&gChaseMusic, 0.f);
        ma_sound_start(&gChaseMusic);
      }
      const float chaseVolMul = nightPursuit ? kChaseMusicPursuitVolMul : 1.f;
      ma_sound_set_volume(&gChaseMusic,
                          kGameplayMusicVol * gChaseCrossfade * chaseVolMul * shrekDuckMul);
    } else if (ma_sound_is_playing(&gChaseMusic))
      ma_sound_stop(&gChaseMusic);
  }

  if (gHorrorReady && ma_sound_is_playing(&gHorrorMusic))
    ma_sound_set_volume(&gHorrorMusic,
                        kGameplayMusicVol * (1.f - gChaseCrossfade) * shrekDuckMul);

  refreshStoreMusicVolumeWithDuck();
}

void audioUpdateStaffChaseTaunts(float dt, bool anyStaffChasing) {
  updateStaffChaseMusicCrossfade(dt, anyStaffChasing);
  if (!gReady || !gStaffChaseVoReady || dt <= 0.f)
    return;
  if (!anyStaffChasing) {
    gChaseVoDiceAccum = 0.f;
    if (gChaseVoCooldown > 0.f)
      gChaseVoCooldown = std::max(0.f, gChaseVoCooldown - dt);
    return;
  }
  if (gChaseVoCooldown > 0.f) {
    gChaseVoCooldown -= dt;
    return;
  }
  const bool nightPursuit = !audioAreStoreFluorescentsOn();
  const float dicePeriod = nightPursuit ? 0.42f : 0.7f;
  const float playChance = nightPursuit ? 0.29f : 0.16f;
  gChaseVoDiceAccum += dt;
  while (gChaseVoDiceAccum >= dicePeriod) {
    gChaseVoDiceAccum -= dicePeriod;
    std::uniform_real_distribution<float> u(0.f, 1.f);
    if (u(gRng) > playChance)
      continue;
    ma_sound* s = &gStaffChaseVoSfx[gStaffChaseVoNext];
    gStaffChaseVoNext = (gStaffChaseVoNext + 1) % 2;
    ma_sound_stop(s);
    ma_sound_reset_stop_time_and_fade(s);
    ma_sound_seek_to_pcm_frame(s, 0);
    const float voMul = nightPursuit ? 1.12f : 1.f;
    ma_sound_set_volume(s, kStaffChaseVoVol * voMul);
    ma_sound_start(s);
    gChaseVoCooldown = (nightPursuit ? 3.6f : 6.5f) + u(gRng) * (nightPursuit ? 6.5f : 10.f);
    gChaseVoDiceAccum = 0.f;
    break;
  }
}

void audioPlayStaffMeleeImpact() {
  if (!gReady || !gStaffMeleeImpactReady)
    return;
  std::uniform_real_distribution<float> pitchDist(0.94f, 1.06f);
  ma_sound_set_volume(&gStaffMeleeImpactSfx, kStaffMeleeImpactVol);
  ma_sound_stop(&gStaffMeleeImpactSfx);
  ma_sound_reset_stop_time_and_fade(&gStaffMeleeImpactSfx);
  ma_sound_seek_to_pcm_frame(&gStaffMeleeImpactSfx, 0);
  ma_sound_set_pitch(&gStaffMeleeImpactSfx, pitchDist(gRng));
  ma_sound_start(&gStaffMeleeImpactSfx);
}

bool audioCaptureStoreCycleSaveState(AudioStoreCycleSaveState* outState) {
  if (!outState || !gReady)
    return false;
  const bool paused = gStoreDayNightPaused.load(std::memory_order_relaxed);
  AudioStoreCycleSaveState st{};
  st.version = 1;
  st.storePhase = static_cast<uint32_t>(gStorePhase);
  if (gStoreFluoroOn.load(std::memory_order_relaxed))
    st.flags |= (1u << 0);
  if (gBlackoutScheduled)
    st.flags |= (1u << 1);
  if (gDayRestoreScheduled)
    st.flags |= (1u << 2);
  const bool storePlaying = gStoreAmbienceReady &&
                            (ma_sound_is_playing(&gStoreMusic) || (paused && gDeathPauseHadStoreMusic));
  const bool horrorPlaying = gHorrorReady &&
                             (ma_sound_is_playing(&gHorrorMusic) || (paused && gDeathPauseHadHorror));
  const bool chasePlaying = gChaseReady &&
                            (ma_sound_is_playing(&gChaseMusic) || (paused && gDeathPauseHadChase));
  const bool shrekPlaying = gShrekEggMusicReady &&
                            (ma_sound_is_playing(&gShrekEggMusic) || (paused && gDeathPauseHadShrekEgg));
  if (storePlaying)
    st.flags |= (1u << 3);
  if (horrorPlaying)
    st.flags |= (1u << 4);
  if (chasePlaying)
    st.flags |= (1u << 5);
  if (shrekPlaying)
    st.flags |= (1u << 6);
  st.flags |= (static_cast<uint32_t>(std::max(gDayCount, 1)) << 16);
  st.version = 2;
  st.storeDayMusicTrackIdx =
      gStoreAmbienceReady && !gStoreDayMusicPaths.empty()
          ? static_cast<uint32_t>(std::min(gStoreDayMusicIdx, gStoreDayMusicPaths.size() - 1u))
          : 0u;
  st.storeCursorFrames = gStoreAmbienceReady ? soundCursorOrZero(&gStoreMusic) : 0;
  st.horrorCursorFrames = gHorrorReady ? soundCursorOrZero(&gHorrorMusic) : 0;
  st.chaseCursorFrames = gChaseReady ? soundCursorOrZero(&gChaseMusic) : 0;
  st.shrekCursorFrames = gShrekEggMusicReady ? soundCursorOrZero(&gShrekEggMusic) : 0;
  const ma_uint64 now = ma_engine_get_time_in_milliseconds(&gEngine);
  st.blackoutRemainingMs = (gBlackoutScheduled && gBlackoutAtEngineMs > now) ? (gBlackoutAtEngineMs - now) : 0;
  st.dayRestoreRemainingMs = (gDayRestoreScheduled && gDayRestoreAtEngineMs > now) ? (gDayRestoreAtEngineMs - now) : 0;
  *outState = st;
  return true;
}

void audioRestoreStoreCycleSaveState(const AudioStoreCycleSaveState& state) {
  if (!gReady)
    return;
  const ma_uint64 now = ma_engine_get_time_in_milliseconds(&gEngine);
  gStorePhase = static_cast<StoreAnimPhase>(std::min<uint32_t>(state.storePhase, 3u));
  const int savedDay = static_cast<int>((state.flags >> 16) & 0xFFFFu);
  gDayCount = savedDay > 0 ? savedDay : 1;
  gStoreFluoroOn.store((state.flags & (1u << 0)) != 0, std::memory_order_relaxed);
  gBlackoutScheduled = (state.flags & (1u << 1)) != 0;
  gDayRestoreScheduled = (state.flags & (1u << 2)) != 0;
  gBlackoutAtEngineMs = now + state.blackoutRemainingMs;
  gDayRestoreAtEngineMs = now + state.dayRestoreRemainingMs;

  if (gStoreAmbienceReady) {
    if (state.version >= 2u && !gStoreDayMusicPaths.empty()) {
      const size_t want =
          static_cast<size_t>(state.storeDayMusicTrackIdx) % gStoreDayMusicPaths.size();
      if (want != gStoreDayMusicIdx)
        reinitStoreMusicToIndex(want);
    }
    ma_sound_set_looping(&gStoreMusic, MA_FALSE);
    ma_sound_set_end_callback(&gStoreMusic, dayMusicOnceEndedCb, nullptr);
    // Seek only while the cycle is still paused; audioSetStoreDayNightCyclePaused(false) then starts the day
    // bed at this cursor when phase is day + lights on (avoids relying on a fragile "was playing" save bit).
    stopSeekMaybeStart(&gStoreMusic, state.storeCursorFrames, false);
    refreshStoreMusicVolumeWithDuck();
  }
  if (gHorrorReady)
    stopSeekMaybeStart(&gHorrorMusic, state.horrorCursorFrames, (state.flags & (1u << 4)) != 0);
  if (gChaseReady)
    stopSeekMaybeStart(&gChaseMusic, state.chaseCursorFrames, (state.flags & (1u << 5)) != 0);
  if (gShrekEggMusicReady)
    stopSeekMaybeStart(&gShrekEggMusic, state.shrekCursorFrames, (state.flags & (1u << 6)) != 0);
}

void audioPlayBigFallImpact(float impactStrength01) {
  if (!gReady || !gBigFallReady)
    return;
  const float u = std::clamp(impactStrength01, 0.f, 1.f);
  // Louder for harder falls; cap above 1.f so very nasty drops still punch through mix a bit.
  const float shaped = std::pow(u, 0.82f);
  const float vol = kBigFallImpactVol * (0.24f + shaped * 1.05f);
  ma_sound_set_volume(&gBigFallSfx, std::min(vol, 1.38f));
  // Slight pitch drop when severity is high (heavier thud); keep variation in the random band.
  std::uniform_real_distribution<float> pitchDist(0.78f, 1.12f);
  const float heavyMul = 1.04f - u * 0.22f;
  ma_sound_stop(&gBigFallSfx);
  ma_sound_reset_stop_time_and_fade(&gBigFallSfx);
  ma_sound_seek_to_pcm_frame(&gBigFallSfx, 0);
  ma_sound_set_pitch(&gBigFallSfx, pitchDist(gRng) * heavyMul);
  ma_sound_start(&gBigFallSfx);
}

void audioSetLowHealthHeartbeat(bool active) {
  if (!gReady || !gHeartbeatReady)
    return;
  if (active) {
    if (!gHeartbeatActive) {
      gHeartbeatActive = true;
      ma_sound_stop(&gHeartbeatSfx);
      ma_sound_reset_stop_time_and_fade(&gHeartbeatSfx);
      ma_sound_seek_to_pcm_frame(&gHeartbeatSfx, 0);
      ma_sound_start(&gHeartbeatSfx);
    }
  } else if (gHeartbeatActive) {
    gHeartbeatActive = false;
    ma_sound_stop(&gHeartbeatSfx);
  }
}

void audioSetShrekEggDanceActive(bool active) {
  if (!gReady || !gShrekEggMusicReady)
    return;
  if (active) {
    if (!ma_sound_is_playing(&gShrekEggMusic)) {
      ma_sound_stop(&gShrekEggMusic);
      ma_sound_reset_stop_time_and_fade(&gShrekEggMusic);
      ma_sound_seek_to_pcm_frame(&gShrekEggMusic, 0);
      ma_sound_set_volume(&gShrekEggMusic, kGameplayMusicVol);
      ma_sound_start(&gShrekEggMusic);
    }
  } else {
    if (ma_sound_is_playing(&gShrekEggMusic))
      ma_sound_stop(&gShrekEggMusic);
    gShrekEggDuckOtherMusic01 = 0.f;
    refreshStoreMusicVolumeWithDuck();
  }
}

void audioUpdateShrekEggVolumeByDistance(float distanceM) {
  if (!gReady || !gShrekEggMusicReady || !ma_sound_is_playing(&gShrekEggMusic)) {
    gShrekEggDuckOtherMusic01 = 0.f;
    return;
  }
  const float d = std::max(0.f, distanceM);
  // Audible mainly within a few metres of him; fades out by ~40 m.
  constexpr float kFullM = 5.f;
  constexpr float kSilentM = 40.f;
  const float t = std::clamp((d - kFullM) / std::max(1e-3f, kSilentM - kFullM), 0.f, 1.f);
  const float gain = 1.f - t;
  ma_sound_set_volume(&gShrekEggMusic, kGameplayMusicVol * gain);
  // Duck store / horror / chase in proportion to how loud the egg track is (close = strong duck).
  gShrekEggDuckOtherMusic01 = gain;
}

bool audioAreStoreFluorescentsOn() {
  if (!gReady)
    return true;
  return gStoreFluoroOn.load(std::memory_order_relaxed);
}

int audioGetDayCount() {
  return gDayCount;
}

void audioResetToNewGame() {
  if (!gReady)
    return;
  gDayCount = 1;
  gStorePhase = StoreAnimPhase::DayOnce;
  gStoreFluoroOn.store(true, std::memory_order_relaxed);
  gStoreSeqEvent.store(0, std::memory_order_relaxed);
  gBlackoutScheduled = false;
  gDayRestoreScheduled = false;
  gDeathPauseHadStoreMusic = false;
  gDeathPauseHadHorror = false;
  gDeathPauseHadChase = false;
  gDeathPauseHadShrekEgg = false;
  if (gStoreAmbienceReady) {
    shuffleStoreDayMusicForNewDay();
    ma_sound_stop(&gStoreMusic);
    ma_sound_reset_stop_time_and_fade(&gStoreMusic);
    ma_sound_seek_to_pcm_frame(&gStoreMusic, 0);
    ma_sound_set_looping(&gStoreMusic, MA_FALSE);
    ma_sound_set_end_callback(&gStoreMusic, dayMusicOnceEndedCb, nullptr);
    gStoreMusicLinear01 = 1.f;
    ma_sound_set_volume(&gStoreMusic, kStoreDayMusicVol * gStoreMusicLinear01);
  }
  if (gHorrorReady)
    ma_sound_stop(&gHorrorMusic);
  if (gChaseReady)
    ma_sound_stop(&gChaseMusic);
  if (gShrekEggMusicReady)
    ma_sound_stop(&gShrekEggMusic);
}
