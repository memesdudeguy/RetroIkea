#pragma once

#include <cstdint>
#include <deque>

#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include "audio.hpp"

// Minimal IP-to-IP multiplayer: two players, UDP, LAN-friendly (same as --mp-join target).
// Player pose snapshots are drawn for the remote peer; host sends deli/staff world state. Inventory is never sent — each copy of the game keeps its own bag and save data.

static constexpr uint32_t kRetroMpMagic = 0x52494B32u; // 'RIK2'
// Host → client world state (staff poses / deli counters). Separate datagram from player snapshots.
static constexpr uint32_t kRetroMpWorldMagic = 0x52495753u; // 'RIWS'
// Client → host: deli pickup (authoritative counter decrement).
static constexpr uint32_t kRetroMpDeliPickupMagic = 0x524B4450u; // 'RKDP'
// Client → host: melee vs staff (shove / kick / drop-kick hit).
static constexpr uint32_t kRetroMpStaffMeleeMagic = 0x524B4D43u; // 'RKMC'
// Either peer → other: player chose RETRY after death (keep both sessions in sync).
static constexpr uint32_t kRetroMpDeathRetryMagic = 0x524B4452u; // 'RKDR'
static constexpr uint8_t kRetroMpDeliPickupPizza = 0;
static constexpr uint8_t kRetroMpDeliPickupMeat = 1;
static constexpr uint8_t kRetroMpStaffMeleeShove = 0;
static constexpr uint8_t kRetroMpStaffMeleeKick = 1;
static constexpr uint8_t kRetroMpStaffMeleeDropKick = 2;
// Fit in typical LAN MTU with room below 1500-byte IP datagram.
static constexpr int kRetroMpWorldMaxStaff = 26;
static constexpr int kRetroMpWorldMaxDeli = 15;
static constexpr uint16_t kRetroMpDefaultPort = 27341;
// Bit flags in RetroMpWirePacket::emoteFlags (receiver maps to local clip indices).
static constexpr uint8_t kRetroMpEmoteDance = 1u << 0;
static constexpr uint8_t kRetroMpEmoteDeath = 1u << 1;
// emoteReserved1: semantic avatar motion tag for cross-build clip mapping.
static constexpr uint8_t kRetroMpAnimTagNone = 0u;
static constexpr uint8_t kRetroMpAnimTagJump = 1u;
static constexpr uint8_t kRetroMpAnimTagJumpRun = 2u;
static constexpr uint8_t kRetroMpAnimTagLedgeClimb = 3u;
static constexpr uint8_t kRetroMpAnimTagShimmyLeft = 4u;
static constexpr uint8_t kRetroMpAnimTagShimmyRight = 5u;
static constexpr uint8_t kRetroMpAnimTagKick = 6u;
static constexpr uint8_t kRetroMpAnimTagIdle = 7u;
static constexpr uint8_t kRetroMpAnimTagWalk = 8u;
static constexpr uint8_t kRetroMpAnimTagSprint = 9u;
static constexpr uint8_t kRetroMpAnimTagCrouchFwd = 10u;
static constexpr uint8_t kRetroMpAnimTagCrouchBack = 11u;
static constexpr uint8_t kRetroMpAnimTagCrouchLeft = 12u;
static constexpr uint8_t kRetroMpAnimTagCrouchRight = 13u;
static constexpr uint8_t kRetroMpAnimTagCrouchIdle = 14u;
static constexpr uint8_t kRetroMpAnimTagLand = 15u;
static constexpr uint8_t kRetroMpAnimTagLedgeGrab = 16u;
static constexpr uint8_t kRetroMpAnimTagStepPush = 17u;
static constexpr uint8_t kRetroMpAnimTagSlideRight = 18u;
static constexpr uint8_t kRetroMpAnimTagSlideLight = 19u;
// Sent alongside kRetroMpEmoteDance — peers remap to local proximity dance clip index.
static constexpr uint8_t kRetroMpAnimTagDanceProximity = 20u;
// When kRetroMpEmoteDeath: emoteReserved0 die payload (clip role + fall playback).
static constexpr uint8_t kRetroMpDeathAuxClipMask = 3u;
static constexpr uint8_t kRetroMpDeathAuxClipMeleeFall = 0u; // local staffClipMeleeFall
static constexpr uint8_t kRetroMpDeathAuxClipLand = 1u;      // local avClipLand
static constexpr uint8_t kRetroMpDeathAuxClipNone = 2u;      // no knockdown clip
static constexpr uint8_t kRetroMpDeathAuxPlayingFall = 4u;   // scrubbing fall; else hold at end pose

#pragma pack(push, 1)
struct RetroMpWirePacket {
  uint32_t magic = kRetroMpMagic;
  uint32_t seq = 0;
  float camX = 0.f;
  float camY = 0.f;
  float camZ = 0.f;
  float yaw = 0.f;
  float pitch = 0.f;
  float eyeHeight = 0.f;
  float horizVelX = 0.f;
  float horizVelZ = 0.f;
  int32_t avatarClip = 0;
  int32_t blendFromClip = 0;
  float clipBlend = 1.f;
  uint64_t locoPhaseBits = 0;
  uint8_t emoteFlags = 0;
  uint8_t emoteReserved0 = 0;
  uint8_t emoteReserved1 = 0;
  uint8_t emoteReserved2 = 0;
  // Host audio/day-cycle snapshot so clients can join in the same timeline/song phase.
  uint32_t audioVersion = 0;
  uint32_t audioStorePhase = 0;
  uint32_t audioFlags = 0;
  uint64_t audioStoreCursorFrames = 0;
  uint64_t audioHorrorCursorFrames = 0;
  uint64_t audioChaseCursorFrames = 0;
  uint64_t audioShrekCursorFrames = 0;
  uint64_t audioBlackoutRemainingMs = 0;
  uint64_t audioDayRestoreRemainingMs = 0;
  uint32_t audioStoreDayMusicTrackIdx = 0;
};
#pragma pack(pop)

static_assert(sizeof(RetroMpWirePacket) == 128, "RetroMpWirePacket size");

#pragma pack(push, 1)
struct RetroMpWorldHeader {
  uint32_t magic = kRetroMpWorldMagic;
  uint32_t seq = 0;
  uint8_t staffCount = 0;
  uint8_t deliCount = 0;
  uint8_t reserved[2]{};
};

struct RetroMpStaffWire {
  uint64_t key = 0;
  float posX = 0.f;
  float posZ = 0.f;
  float feetY = 0.f;
  float yaw = 0.f;
  uint8_t flags = 0; // bits 0-1 nightPhase, 2-4 meleeState, 5 staffDead, 6 staffRp3dCorpse, 7 meleeAttackPick (meleeState==1)
  uint8_t reserved = 0;
  int16_t drawClip = 0;
  float drawPhase = 0.f;
  uint16_t meleePhaseNorm = 0;
  uint8_t drawLoop = 0;
  uint8_t pad = 0;
};

struct RetroMpDeliWire {
  uint64_t key = 0;
  uint8_t pizzaCount = 0;
  uint8_t meatCount = 0;
  uint16_t pizzaReplenishCs = 0; // centiseconds; 0 = no timer
  uint16_t meatReplenishCs = 0;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct RetroMpDeliPickupPacket {
  uint32_t magic = kRetroMpDeliPickupMagic;
  uint32_t seq = 0;
  uint64_t deliSlotKey = 0;
  uint8_t foodKind = kRetroMpDeliPickupPizza;
  uint8_t pad[7]{};
  float camX = 0.f;
  float camZ = 0.f;
};
#pragma pack(pop)
static_assert(sizeof(RetroMpDeliPickupPacket) == 32, "RetroMpDeliPickupPacket size");

#pragma pack(push, 1)
struct RetroMpStaffMeleePacket {
  uint32_t magic = kRetroMpStaffMeleeMagic;
  uint32_t seq = 0;
  uint64_t staffNpcKey = 0;
  uint8_t meleeKind = kRetroMpStaffMeleeShove;
  uint8_t pad[7]{};
  float camX = 0.f;
  float camZ = 0.f;
  float fwdX = 0.f;
  float fwdZ = 0.f;
};
#pragma pack(pop)
static_assert(sizeof(RetroMpStaffMeleePacket) == 40, "RetroMpStaffMeleePacket size");

#pragma pack(push, 1)
struct RetroMpDeathRetryPacket {
  uint32_t magic = kRetroMpDeathRetryMagic;
  uint32_t seq = 0;
};
#pragma pack(pop)
static_assert(sizeof(RetroMpDeathRetryPacket) == 8, "RetroMpDeathRetryPacket size");

static_assert(sizeof(RetroMpWorldHeader) == 12, "RetroMpWorldHeader size");
static_assert(sizeof(RetroMpStaffWire) == 36, "RetroMpStaffWire size");
static_assert(sizeof(RetroMpDeliWire) == 14, "RetroMpDeliWire size");

struct RetroMpSession {
  bool active = false;
  bool isHost = false;
  uint16_t bindPort = kRetroMpDefaultPort;
  char peerHostUtf8[128]{};
  char publicHostUtf8[64]{};
  uint16_t peerPort = kRetroMpDefaultPort;

  uint32_t sendSeq = 0;
  uint32_t lastRemoteSeq = 0;
  double lastRecvMono = -1.0;
  uint32_t worldSendSeq = 0;
  double lastWorldRecvMono = -1.0;

  RetroMpWirePacket lastRemote{};
  std::vector<uint8_t> lastWorldPacket;
  std::deque<RetroMpDeliPickupPacket> hostDeliPickupQueue;
  std::deque<RetroMpStaffMeleePacket> hostStaffMeleeQueue;
  bool remoteDeathRetryPending = false;
  uint32_t deathRetrySendSeq = 0;

#if defined(_WIN32)
  uint64_t sock = UINT64_MAX;
#else
  int sock = -1;
#endif

  void shutdown();

  // Host: re-scan Tailscale-style CGNAT IPv4 (100.64–100.127.x.x) without hitting WAN check-ip again.
  // Call when pause menu opens so a late Tailscale handshake updates HOST WAITING.
  void refreshAnnounceJoinIpv4();

  // Runtime (pause menu / CLI). stop() clears session; host binds UDP port; join binds ephemeral port.
  bool startHost(uint16_t port = kRetroMpDefaultPort);
  bool startJoin(const char* ipv4, uint16_t port = kRetroMpDefaultPort);
  void stop() { shutdown(); }

  // Call once at startup. Returns true if multiplayer mode enabled.
  bool initFromArgs(int argc, char** argv);

  // Non-blocking receive + age in seconds since last packet (for ghost timeout).
  void pollReceive(float wallDt);

  void sendSnapshot(const glm::vec3& camPos, float yaw, float pitch, float eyeHeight,
                    const glm::vec2& horizVel, int avatarClip, int blendFromClip, float clipBlend,
                    double locoPhaseSec, uint8_t emoteFlags, uint8_t emoteDeathAux,
                    uint8_t emoteAnimTag, uint8_t emoteBlendFromTag,
                    const AudioStoreCycleSaveState* audioState);

  // Host only: variable-length payload (starts with RetroMpWorldHeader).
  void sendWorldSync(const void* payload, size_t payloadBytes);

  // Client → host RPCs (joining peer only).
  void sendDeliPickupRequest(const RetroMpDeliPickupPacket& p);
  void sendStaffMeleeRequest(const RetroMpStaffMeleePacket& p);

  void sendDeathRetry();

  // True once per remote "RETRY" UDP (cleared by caller). Only call when active.
  bool consumeRemoteDeathRetry();

  bool remoteValid(float timeoutSec) const;
};

double retroMpMonotonicSec();
