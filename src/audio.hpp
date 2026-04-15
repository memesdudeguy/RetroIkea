#pragma once

#include <cstdint>

// Packed for on-disk GameSaveFile (see main.cpp).
#pragma pack(push, 1)
struct AudioStoreCycleSaveState {
  uint32_t version = 2;
  uint32_t storePhase = 0;
  uint32_t flags = 0;
  uint64_t storeCursorFrames = 0;
  uint64_t horrorCursorFrames = 0;
  uint64_t chaseCursorFrames = 0;
  uint64_t shrekCursorFrames = 0;
  uint64_t blackoutRemainingMs = 0;
  uint64_t dayRestoreRemainingMs = 0;
  /// Index into runtime day-music path list (shuffle). Added v2; v1 saves treat as 0.
  uint32_t storeDayMusicTrackIdx = 0;
};
#pragma pack(pop)
static_assert(sizeof(AudioStoreCycleSaveState) == 64);

bool audioInit();
void audioShutdown();
void audioPlayFootstep(bool landing = false, float volumeMul = 1.f);
void audioSetSlide(bool active);
void audioSetStoreAmbienceVolume(float linear01);
// Call each frame (pass dt): store music / lights / switch SFX sequencing.
void audioUpdateStore(float dt);
bool audioAreStoreFluorescentsOn();
int audioGetDayCount();
void audioResetToNewGame();
// Freeze day↔night sequencing (fluorescents + store music events) while player is dead; store / horror / chase /
// Shrek beds pause in place (same PCM on respawn). Scheduled blackout/restore times shift by pause duration.
void audioSetStoreDayNightCyclePaused(bool paused);
// Title menu bed (WAV, looping). Mutually exclusive with in-game store beds when combined with
// audioSetStoreDayNightCyclePaused(true) on the title screen.
void audioSetTitleMenuMusicActive(bool active);
// Night: staff just noticed the player (first frame of “watching”).
void audioPlayStaffSpotted();
// Night: at least one staff is chasing — call each frame; line plays occasionally with cooldowns.
void audioUpdateStaffChaseTaunts(float dt, bool anyStaffChasing);
// Easter egg: loop dance bed while true; fades out when false.
void audioSetShrekEggDanceActive(bool active);
// While the egg is active: scale loop volume by distance from listener to the egg (full near him, silent far away).
void audioUpdateShrekEggVolumeByDistance(float distanceM);
// Looping heartbeat SFX while active (0 < HP < mercy cap, same band as screen-edge critical); silent if MP3 missing.
void audioSetLowHealthHeartbeat(bool active);
// One-shot hard-landing sting when fall damage applies. impactStrength01 ≈ 0..1 from height/speed — higher =
// louder + slightly deeper (bigger falls read heavier).
void audioPlayBigFallImpact(float impactStrength01 = 1.f);
// Staff punch/contact damage to player, and LMB shove / body-slam knockdown you apply to staff.
void audioPlayStaffMeleeImpact();
// Save/restore in-game music/day-cycle runtime so Continue resumes where the player left off.
bool audioCaptureStoreCycleSaveState(AudioStoreCycleSaveState* outState);
void audioRestoreStoreCycleSaveState(const AudioStoreCycleSaveState& state);
void audioSetLoadingScreenActive(bool active);
