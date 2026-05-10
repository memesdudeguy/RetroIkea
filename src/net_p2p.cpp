#include "net_p2p.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
using SockT = SOCKET;
constexpr SockT kSockInvalid = INVALID_SOCKET;
#else
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/socket.h>
using SockT = int;
constexpr SockT kSockInvalid = -1;
#endif

#ifdef _WIN32
#define IOCALL(s) static_cast<SOCKET>(s)
#else
#define IOCALL(s) (s)
#endif

double retroMpMonotonicSec() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static bool setNonBlocking(SockT s) {
#ifdef _WIN32
  u_long mode = 1;
  return ioctlsocket(s, FIONBIO, &mode) == 0;
#else
  int fl = fcntl(s, F_GETFL, 0);
  if (fl < 0)
    return false;
  return fcntl(s, F_SETFL, fl | O_NONBLOCK) == 0;
#endif
}

static bool envFlagEnabled(const char* name) {
  const char* v = std::getenv(name);
  return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T');
}

static int envIntClamped(const char* name, int fallback, int lo, int hi) {
  if (const char* raw = std::getenv(name)) {
    char* end = nullptr;
    const long v = std::strtol(raw, &end, 10);
    if (end != raw)
      return static_cast<int>(std::clamp<long>(v, lo, hi));
  }
  return fallback;
}

static void tuneUdpLowLatency(SockT s) {
  int one = 1;
  int buf = 256 * 1024;
#ifdef _WIN32
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
  setsockopt(s, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char*>(&buf), sizeof(buf));
  setsockopt(s, SOL_SOCKET, SO_SNDBUF, reinterpret_cast<const char*>(&buf), sizeof(buf));
  int tos = 0x10; // IPTOS_LOWDELAY; keep literal for MinGW headers that omit it.
  setsockopt(s, IPPROTO_IP, IP_TOS, reinterpret_cast<const char*>(&tos), sizeof(tos));
#else
  setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  setsockopt(s, SOL_SOCKET, SO_RCVBUF, &buf, sizeof(buf));
  setsockopt(s, SOL_SOCKET, SO_SNDBUF, &buf, sizeof(buf));
  int tos = 0x10;
  setsockopt(s, IPPROTO_IP, IP_TOS, &tos, sizeof(tos));
#endif
}

static bool ipv4BytesAreTailscaleCgnat(const uint8_t* o) {
  return o != nullptr && o[0] == 100u && o[1] >= 64u && o[1] <= 127u;
}

static bool ipv4BytesArePrivateLan(const uint8_t* o) {
  if (!o)
    return false;
  if (o[0] == 10u)
    return true;
  if (o[0] == 172u && o[1] >= 16u && o[1] <= 31u)
    return true;
  if (o[0] == 192u && o[1] == 168u)
    return true;
  return false;
}

static FILE* retroMpPopenRead(const char* cmd) {
#ifdef _WIN32
  return _popen(cmd, "r");
#else
  return popen(cmd, "r");
#endif
}

static int retroMpPclose(FILE* f) {
  if (!f)
    return -1;
#ifdef _WIN32
  return _pclose(f);
#else
  return pclose(f);
#endif
}

static bool tailscaleIpv4FromCli(char* out, size_t outCap) {
  out[0] = '\0';
  if (!out || outCap < sizeof("100.64.255.255"))
    return false;
#ifdef _WIN32
  static const char* const kCli[] = {
      "\"C:\\Program Files\\Tailscale\\tailscale.exe\" ip -4",
      "\"C:\\Program Files (x86)\\Tailscale\\tailscale.exe\" ip -4",
      "tailscale ip -4"};
#else
  static const char* const kCli[] = {"tailscale ip -4"};
#endif
  for (const char* cmd : kCli) {
    FILE* f = retroMpPopenRead(cmd);
    if (!f)
      continue;
    std::vector<char> line(128);
    if (!std::fgets(line.data(), static_cast<int>(line.size()), f)) {
      retroMpPclose(f);
      continue;
    }
    retroMpPclose(f);
    char* nl = std::strchr(line.data(), '\r');
    if (!nl)
      nl = std::strchr(line.data(), '\n');
    if (nl)
      *nl = '\0';
    char tokBuf[INET_ADDRSTRLEN]{};
    const char* first = line.data();
    while (*first == ' ' || *first == '\t')
      ++first;
    // First token only (handles multiple IPs on separate lines elsewhere).
    {
      std::size_t i = 0;
      while (first[i] && first[i] != ' ' && first[i] != '\t' &&
             first[i] != ',' && i + 1 < sizeof(tokBuf))
        ++i;
      if (i == 0)
        continue;
      std::memcpy(tokBuf, first, i);
      tokBuf[i] = '\0';
    }
    in_addr a4{};
    if (inet_pton(AF_INET, tokBuf, &a4) != 1)
      continue;
    const auto* oct = reinterpret_cast<const uint8_t*>(&a4);
    if (!ipv4BytesAreTailscaleCgnat(oct))
      continue;
    char norm[INET_ADDRSTRLEN]{};
    if (inet_ntop(AF_INET, &a4, norm, sizeof(norm)) == nullptr)
      continue;
    std::strncpy(out, norm, outCap - 1);
    out[outCap - 1] = '\0';
    return true;
  }
  return false;
}

#ifdef _WIN32
static bool tailscaleIpv4FromWinAdapters(char* out, size_t outCap) {
  out[0] = '\0';
  if (!out || outCap < 8)
    return false;
  ULONG sz = static_cast<ULONG>(16u * 1024u);
  std::vector<uint8_t> buf(sz);
  PIP_ADAPTER_ADDRESSES aa = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data());

  ULONG err = GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST |
                                                GAA_FLAG_SKIP_DNS_SERVER,
                                   nullptr, aa, &sz);
  if (err == ERROR_BUFFER_OVERFLOW) {
    buf.resize(sz);
    aa = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data());
    err = GetAdaptersAddresses(AF_INET,
                               GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER,
                               nullptr, aa, &sz);
  }
  if (err != NO_ERROR || aa == nullptr)
    return false;

  char preferred[INET_ADDRSTRLEN]{};
  char fallback[INET_ADDRSTRLEN]{};

  for (PIP_ADAPTER_ADDRESSES a = aa; a != nullptr; a = a->Next) {
    const bool tailName =
        (a->FriendlyName && wcsstr(a->FriendlyName, L"Tailscale") != nullptr) ||
        (a->Description && wcsstr(a->Description, L"Tailscale") != nullptr);
    for (PIP_ADAPTER_UNICAST_ADDRESS u = a->FirstUnicastAddress; u != nullptr; u = u->Next) {
      if (u->Address.lpSockaddr == nullptr ||
          static_cast<size_t>(u->Address.lpSockaddr->sa_family) != AF_INET)
        continue;
      const auto* sin = reinterpret_cast<const sockaddr_in*>(u->Address.lpSockaddr);
      const auto* oc = reinterpret_cast<const uint8_t*>(&sin->sin_addr);
      if (!ipv4BytesAreTailscaleCgnat(oc))
        continue;
      char nb[INET_ADDRSTRLEN]{};
      if (inet_ntop(AF_INET, &sin->sin_addr, nb, sizeof(nb)) == nullptr)
        continue;
      if (tailName) {
        std::memcpy(preferred, nb, sizeof(preferred));
        break;
      }
      if (fallback[0] == '\0')
        std::memcpy(fallback, nb, sizeof(fallback));
    }
    if (preferred[0] != '\0')
      break;
  }

  const char* pick = preferred[0] != '\0' ? preferred : (fallback[0] != '\0' ? fallback : nullptr);
  if (pick == nullptr)
    return false;
  std::strncpy(out, pick, outCap - 1);
  out[outCap - 1] = '\0';
  return true;
}
#else
static bool tailscaleIpv4FromUnixIfaddrs(char* out, size_t outCap) {
  out[0] = '\0';
  if (!out || outCap < 8)
    return false;
  ifaddrs* ifap = nullptr;
  if (getifaddrs(&ifap) != 0 || ifap == nullptr)
    return false;
  char preferred[INET_ADDRSTRLEN]{};
  char fallback[INET_ADDRSTRLEN]{};

  for (ifaddrs* ifa = ifap; ifa != nullptr; ifa = ifa->ifa_next) {
    if (!ifa->ifa_addr || static_cast<size_t>(ifa->ifa_addr->sa_family) != AF_INET)
      continue;
    const char* n = ifa->ifa_name;
    if (!n || std::strcmp(n, "lo") == 0)
      continue;
    const auto* sin = reinterpret_cast<const sockaddr_in*>(ifa->ifa_addr);
    const auto* oc = reinterpret_cast<const uint8_t*>(&sin->sin_addr);
    if (!ipv4BytesAreTailscaleCgnat(oc))
      continue;
    char nb[INET_ADDRSTRLEN]{};
    if (inet_ntop(AF_INET, &sin->sin_addr, nb, sizeof(nb)) == nullptr)
      continue;
    if (strstr(n, "tailscale") != nullptr) {
      std::memcpy(preferred, nb, sizeof(preferred));
      break;
    }
    if (fallback[0] == '\0')
      std::memcpy(fallback, nb, sizeof(fallback));
  }
  freeifaddrs(ifap);
  const char* pick = preferred[0] != '\0' ? preferred : (fallback[0] != '\0' ? fallback : nullptr);
  if (pick == nullptr)
    return false;
  std::strncpy(out, pick, outCap - 1);
  out[outCap - 1] = '\0';
  return true;
}
#endif

static bool tryAnnounceTailscaleIpv4(char* out, size_t outCap) {
#ifdef _WIN32
  if (tailscaleIpv4FromWinAdapters(out, outCap))
    return true;
#else
  if (tailscaleIpv4FromUnixIfaddrs(out, outCap))
    return true;
#endif
  return tailscaleIpv4FromCli(out, outCap);
}

#ifdef _WIN32
static bool privateLanIpv4FromWinAdapters(char* out, size_t outCap) {
  out[0] = '\0';
  if (!out || outCap < 8)
    return false;
  ULONG sz = static_cast<ULONG>(16u * 1024u);
  std::vector<uint8_t> buf(sz);
  PIP_ADAPTER_ADDRESSES aa = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data());
  ULONG err = GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST |
                                                GAA_FLAG_SKIP_DNS_SERVER,
                                   nullptr, aa, &sz);
  if (err == ERROR_BUFFER_OVERFLOW) {
    buf.resize(sz);
    aa = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buf.data());
    err = GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST |
                                            GAA_FLAG_SKIP_DNS_SERVER,
                               nullptr, aa, &sz);
  }
  if (err != NO_ERROR || aa == nullptr)
    return false;
  for (PIP_ADAPTER_ADDRESSES a = aa; a != nullptr; a = a->Next) {
    for (PIP_ADAPTER_UNICAST_ADDRESS u = a->FirstUnicastAddress; u != nullptr; u = u->Next) {
      if (!u->Address.lpSockaddr || u->Address.lpSockaddr->sa_family != AF_INET)
        continue;
      const auto* sin = reinterpret_cast<const sockaddr_in*>(u->Address.lpSockaddr);
      const auto* oc = reinterpret_cast<const uint8_t*>(&sin->sin_addr);
      if (!ipv4BytesArePrivateLan(oc))
        continue;
      char nb[INET_ADDRSTRLEN]{};
      if (inet_ntop(AF_INET, &sin->sin_addr, nb, sizeof(nb)) == nullptr)
        continue;
      std::strncpy(out, nb, outCap - 1);
      out[outCap - 1] = '\0';
      return true;
    }
  }
  return false;
}
#else
static bool privateLanIpv4FromUnixIfaddrs(char* out, size_t outCap) {
  out[0] = '\0';
  if (!out || outCap < 8)
    return false;
  ifaddrs* ifap = nullptr;
  if (getifaddrs(&ifap) != 0 || ifap == nullptr)
    return false;
  bool ok = false;
  for (ifaddrs* ifa = ifap; ifa != nullptr && !ok; ifa = ifa->ifa_next) {
    if (!ifa->ifa_addr || static_cast<size_t>(ifa->ifa_addr->sa_family) != AF_INET)
      continue;
    const char* n = ifa->ifa_name;
    if (!n || std::strcmp(n, "lo") == 0)
      continue;
    const auto* sin = reinterpret_cast<const sockaddr_in*>(ifa->ifa_addr);
    const auto* oc = reinterpret_cast<const uint8_t*>(&sin->sin_addr);
    if (!ipv4BytesArePrivateLan(oc))
      continue;
    char nb[INET_ADDRSTRLEN]{};
    if (inet_ntop(AF_INET, &sin->sin_addr, nb, sizeof(nb)) == nullptr)
      continue;
    std::strncpy(out, nb, outCap - 1);
    out[outCap - 1] = '\0';
    ok = true;
  }
  freeifaddrs(ifap);
  return ok;
}
#endif

static bool tryAnnounceDirectIpv4(char* out, size_t outCap) {
#ifdef _WIN32
  if (privateLanIpv4FromWinAdapters(out, outCap))
    return true;
#else
  if (privateLanIpv4FromUnixIfaddrs(out, outCap))
    return true;
#endif
  return tryAnnounceTailscaleIpv4(out, outCap);
}

static bool fetchPublicIpv4(char* out, size_t outCap) {
  if (!out || outCap < 8)
    return false;
  out[0] = '\0';
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  if (getaddrinfo("checkip.amazonaws.com", "80", &hints, &res) != 0 || !res)
    return false;
  SockT s = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  if (s == kSockInvalid) {
    freeaddrinfo(res);
    return false;
  }
#ifdef _WIN32
  DWORD toMs = 1500;
  setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&toMs), sizeof(toMs));
  setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&toMs), sizeof(toMs));
#else
  timeval tv{};
  tv.tv_sec = 1;
  tv.tv_usec = 500000;
  setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
  bool ok = false;
  do {
    if (::connect(s, res->ai_addr, static_cast<int>(res->ai_addrlen)) != 0)
      break;
    static const char kReq[] =
        "GET / HTTP/1.1\r\nHost: checkip.amazonaws.com\r\nConnection: close\r\nUser-Agent: retro-ikea\r\n\r\n";
    const int sent = static_cast<int>(::send(IOCALL(s), kReq, static_cast<int>(sizeof(kReq) - 1), 0));
    if (sent <= 0)
      break;
    std::string resp;
    char buf[512];
    for (;;) {
      int r = static_cast<int>(::recv(IOCALL(s), buf, static_cast<int>(sizeof(buf)), 0));
      if (r <= 0)
        break;
      resp.append(buf, static_cast<size_t>(r));
      if (resp.size() > 4096)
        break;
    }
    const size_t bodyPos = resp.find("\r\n\r\n");
    if (bodyPos == std::string::npos)
      break;
    std::string body = resp.substr(bodyPos + 4);
    const size_t eol = body.find_first_of("\r\n");
    if (eol != std::string::npos)
      body.resize(eol);
    in_addr a4{};
    if (inet_pton(AF_INET, body.c_str(), &a4) != 1)
      break;
    std::strncpy(out, body.c_str(), outCap - 1);
    out[outCap - 1] = '\0';
    ok = true;
  } while (false);
#ifdef _WIN32
  closesocket(s);
#else
  ::close(s);
#endif
  freeaddrinfo(res);
  return ok;
}

void RetroMpSession::shutdown() {
#ifdef _WIN32
  if (sock != UINT64_MAX) {
    closesocket(static_cast<SOCKET>(sock));
    sock = UINT64_MAX;
  }
#else
  if (sock >= 0) {
    ::close(sock);
    sock = -1;
  }
#endif
  active = false;
  isHost = false;
  sendSeq = 0;
  lastRemoteSeq = 0;
  lastRecvMono = -1.0;
  lastRemote = RetroMpWirePacket{};
  worldSendSeq = 0;
  lastWorldRecvMono = -1.0;
  lastWorldPacket.clear();
  hostDeliPickupQueue.clear();
  hostStaffMeleeQueue.clear();
  hostFoodActionQueue.clear();
  remoteDeathRetryPending = false;
  deathRetrySendSeq = 0;
  std::memset(peerHostUtf8, 0, sizeof(peerHostUtf8));
  std::memset(publicHostUtf8, 0, sizeof(publicHostUtf8));
}

void RetroMpSession::refreshAnnounceJoinIpv4() {
  if (!active || !isHost)
    return;
  char ip[INET_ADDRSTRLEN]{};
  if (!tryAnnounceDirectIpv4(ip, sizeof(ip)) || ip[0] == '\0')
    return;
  std::strncpy(publicHostUtf8, ip, sizeof(publicHostUtf8) - 1);
  publicHostUtf8[sizeof(publicHostUtf8) - 1] = '\0';
}

static bool ensureWsa() {
#ifdef _WIN32
  static bool wsaStarted = false;
  if (!wsaStarted) {
    WSADATA w{};
    if (WSAStartup(MAKEWORD(2, 2), &w) != 0) {
      std::fprintf(stderr, "[mp] WSAStartup failed.\n");
      return false;
    }
    wsaStarted = true;
  }
#endif
  return true;
}

bool RetroMpSession::startHost(uint16_t port) {
  shutdown();
  if (!ensureWsa())
    return false;

  SockT s = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (s == kSockInvalid) {
    std::fprintf(stderr, "[mp] socket() failed.\n");
    return false;
  }
  tuneUdpLowLatency(s);
#ifdef _WIN32
  sock = static_cast<uint64_t>(s);
#else
  sock = static_cast<int>(s);
#endif
  sockaddr_in bindAddr{};
  bindAddr.sin_family = AF_INET;
  bindAddr.sin_addr.s_addr = INADDR_ANY;
  bindAddr.sin_port = htons(port);
  if (::bind(IOCALL(sock), reinterpret_cast<sockaddr*>(&bindAddr), sizeof(bindAddr)) != 0) {
    std::fprintf(stderr, "[mp] bind(host) failed (port in use?).\n");
#ifdef _WIN32
    closesocket(s);
#else
    ::close(s);
#endif
#ifdef _WIN32
    sock = UINT64_MAX;
#else
    sock = -1;
#endif
    return false;
  }
  if (!setNonBlocking(IOCALL(sock))) {
#ifdef _WIN32
    closesocket(s);
#else
    ::close(s);
#endif
#ifdef _WIN32
    sock = UINT64_MAX;
#else
    sock = -1;
#endif
    std::fprintf(stderr, "[mp] non-blocking failed.\n");
    return false;
  }
  active = true;
  isHost = true;
  bindPort = port;
  peerPort = port;
  std::fprintf(stderr, "[mp] Hosting UDP :%u (wait for client packets)\n", static_cast<unsigned>(port));
  publicHostUtf8[0] = '\0';
  if (tryAnnounceDirectIpv4(publicHostUtf8, sizeof(publicHostUtf8)))
    std::fprintf(stderr, "[mp] Direct join IP %s:%u\n", publicHostUtf8,
                 static_cast<unsigned>(port));
  else if (envFlagEnabled("VULKAN_GAME_MP_WAN_HINT") && fetchPublicIpv4(publicHostUtf8, sizeof(publicHostUtf8)))
    std::fprintf(stderr, "[mp] Public WAN join hint %s:%u\n", publicHostUtf8, static_cast<unsigned>(port));
  else
    std::fprintf(stderr, "[mp] Direct IP hint unavailable; enter this PC's LAN/Tailscale IP on the client.\n");
  return true;
}

bool RetroMpSession::startJoin(const char* ipv4, uint16_t port) {
  shutdown();
  if (!ipv4 || ipv4[0] == '\0') {
    std::fprintf(stderr, "[mp] join: empty IP.\n");
    return false;
  }
  if (!ensureWsa())
    return false;

  SockT s = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (s == kSockInvalid) {
    std::fprintf(stderr, "[mp] socket() failed.\n");
    return false;
  }
  tuneUdpLowLatency(s);
#ifdef _WIN32
  sock = static_cast<uint64_t>(s);
#else
  sock = static_cast<int>(s);
#endif
  sockaddr_in bindAddr{};
  bindAddr.sin_family = AF_INET;
  bindAddr.sin_addr.s_addr = INADDR_ANY;
  bindAddr.sin_port = htons(0);
  if (::bind(IOCALL(sock), reinterpret_cast<sockaddr*>(&bindAddr), sizeof(bindAddr)) != 0) {
    std::fprintf(stderr, "[mp] bind(client) failed.\n");
#ifdef _WIN32
    closesocket(s);
#else
    ::close(s);
#endif
#ifdef _WIN32
    sock = UINT64_MAX;
#else
    sock = -1;
#endif
    return false;
  }
  if (!setNonBlocking(IOCALL(sock))) {
#ifdef _WIN32
    closesocket(s);
#else
    ::close(s);
#endif
#ifdef _WIN32
    sock = UINT64_MAX;
#else
    sock = -1;
#endif
    return false;
  }
  sockaddr_in testPeer{};
  testPeer.sin_family = AF_INET;
  testPeer.sin_port = htons(port);
  if (inet_pton(AF_INET, ipv4, &testPeer.sin_addr) != 1) {
    std::fprintf(stderr, "[mp] Bad IP: %s\n", ipv4);
    shutdown();
    return false;
  }
  std::strncpy(peerHostUtf8, ipv4, sizeof(peerHostUtf8) - 1);
  peerPort = port;
  bindPort = port;
  active = true;
  isHost = false;
  std::fprintf(stderr, "[mp] Joining host %s:%u\n", peerHostUtf8, static_cast<unsigned>(port));
  return true;
}

static bool parseU16(const char* s, uint16_t& out) {
  char* end = nullptr;
  unsigned long v = std::strtoul(s, &end, 10);
  if (end == s || v > 65535u)
    return false;
  out = static_cast<uint16_t>(v);
  return true;
}

static void parseJoinHostPortInPlace(char* host, size_t hostCap, uint16_t& port) {
  if (!host || hostCap == 0)
    return;
  char* colon = std::strrchr(host, ':');
  if (!colon || colon == host)
    return;
  uint16_t parsed = port;
  if (!parseU16(colon + 1, parsed) || parsed == 0)
    return;
  *colon = '\0';
  port = parsed;
}

bool RetroMpSession::initFromArgs(int argc, char** argv) {
  shutdown();

  bool wantHost = false;
  bool wantJoin = false;
  uint16_t hostPort = kRetroMpDefaultPort;
  uint16_t joinPort = kRetroMpDefaultPort;
  char joinIp[128]{};

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mp-host") == 0) {
      wantHost = true;
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        uint16_t p = hostPort;
        if (parseU16(argv[i + 1], p)) {
          hostPort = p;
          ++i;
        }
      }
    } else if (std::strcmp(argv[i], "--mp-join") == 0 && i + 1 < argc) {
      wantJoin = true;
      std::strncpy(joinIp, argv[++i], sizeof(joinIp) - 1);
      parseJoinHostPortInPlace(joinIp, sizeof(joinIp), joinPort);
      if (i + 1 < argc && argv[i + 1][0] != '-') {
        uint16_t p = joinPort;
        if (parseU16(argv[i + 1], p)) {
          joinPort = p;
          ++i;
        }
      }
    }
  }

  if (const char* e = std::getenv("VULKAN_GAME_MP_HOST")) {
    if (e[0] == '1' || e[0] == 'y' || e[0] == 'Y')
      wantHost = true;
  }
  if (const char* e = std::getenv("VULKAN_GAME_MP_JOIN")) {
    if (e[0] != '\0') {
      wantJoin = true;
      std::strncpy(joinIp, e, sizeof(joinIp) - 1);
      parseJoinHostPortInPlace(joinIp, sizeof(joinIp), joinPort);
    }
  }
  if (const char* e = std::getenv("VULKAN_GAME_MP_PORT")) {
    uint16_t p = kRetroMpDefaultPort;
    if (parseU16(e, p)) {
      hostPort = p;
      joinPort = p;
    }
  }

  if (wantHost && wantJoin) {
    std::fprintf(stderr, "[mp] Both --mp-host and --mp-join set; using host mode.\n");
    wantJoin = false;
  }
  if (!wantHost && !wantJoin)
    return false;

  if (wantHost)
    return startHost(hostPort);
  return startJoin(joinIp, joinPort);
}

void RetroMpSession::pollReceive(float wallDt) {
  if (!active)
    return;

  alignas(RetroMpWirePacket) uint8_t dgram[2048]{};
  sockaddr_in from{};
  socklen_t fromLen = sizeof(from);

  int packetBudget = envIntClamped("VULKAN_GAME_MP_RECV_BUDGET", 192, 32, 512);
  if (std::isfinite(wallDt) && wallDt > (1.f / 35.f))
    packetBudget = std::min(packetBudget, envIntClamped("VULKAN_GAME_MP_SLOW_RECV_BUDGET", 96, 16, 256));

  for (int packetsRead = 0; packetsRead < packetBudget; ++packetsRead) {
#ifdef _WIN32
    int r = ::recvfrom(static_cast<SOCKET>(sock), reinterpret_cast<char*>(dgram), static_cast<int>(sizeof(dgram)), 0,
                       reinterpret_cast<sockaddr*>(&from), &fromLen);
#else
    ssize_t r =
        ::recvfrom(sock, reinterpret_cast<char*>(dgram), sizeof(dgram), 0,
                   reinterpret_cast<sockaddr*>(&from), &fromLen);
#endif
    if (r <= 0)
      break;
    if (r < 4)
      continue;
    uint32_t mag = 0;
    std::memcpy(&mag, dgram, sizeof(mag));
    if (mag == kRetroMpMagic) {
      if (r != static_cast<int>(sizeof(RetroMpWirePacket)))
        continue;
      RetroMpWirePacket buf{};
      std::memcpy(&buf, dgram, sizeof(buf));
      if (buf.magic != kRetroMpMagic)
        continue;

      if (isHost) {
        peerPort = ntohs(from.sin_port);
        char tmp[64]{};
        if (inet_ntop(AF_INET, &from.sin_addr, tmp, sizeof(tmp)))
          std::strncpy(peerHostUtf8, tmp, sizeof(peerHostUtf8) - 1);
      }

      lastRemote = buf;
      lastRemoteSeq = buf.seq;
      lastRecvMono = retroMpMonotonicSec();
    } else if (mag == kRetroMpWorldMagic) {
      if (r < static_cast<int>(sizeof(RetroMpWorldHeader)))
        continue;
      if (static_cast<size_t>(r) > sizeof(dgram))
        continue;
      if (!isHost) {
        lastWorldPacket.assign(dgram, dgram + static_cast<size_t>(r));
        lastWorldRecvMono = retroMpMonotonicSec();
      }
    } else if (mag == kRetroMpDeliPickupMagic) {
      if (isHost &&
          static_cast<size_t>(r) == sizeof(RetroMpDeliPickupPacket)) {
        RetroMpDeliPickupPacket p{};
        std::memcpy(&p, dgram, sizeof(p));
        if (p.magic == kRetroMpDeliPickupMagic) {
          hostDeliPickupQueue.push_back(p);
          while (hostDeliPickupQueue.size() > 64u)
            hostDeliPickupQueue.pop_front();
        }
      }
    } else if (mag == kRetroMpStaffMeleeMagic) {
      if (isHost &&
          static_cast<size_t>(r) == sizeof(RetroMpStaffMeleePacket)) {
        RetroMpStaffMeleePacket p{};
        std::memcpy(&p, dgram, sizeof(p));
        if (p.magic == kRetroMpStaffMeleeMagic) {
          hostStaffMeleeQueue.push_back(p);
          while (hostStaffMeleeQueue.size() > 64u)
            hostStaffMeleeQueue.pop_front();
        }
      }
    } else if (mag == kRetroMpFoodActionMagic) {
      if (isHost &&
          static_cast<size_t>(r) == sizeof(RetroMpFoodActionPacket)) {
        RetroMpFoodActionPacket p{};
        std::memcpy(&p, dgram, sizeof(p));
        if (p.magic == kRetroMpFoodActionMagic) {
          hostFoodActionQueue.push_back(p);
          while (hostFoodActionQueue.size() > 64u)
            hostFoodActionQueue.pop_front();
        }
      }
    } else if (mag == kRetroMpDeathRetryMagic) {
      if (static_cast<size_t>(r) != sizeof(RetroMpDeathRetryPacket))
        continue;
      RetroMpDeathRetryPacket pkt{};
      std::memcpy(&pkt, dgram, sizeof(pkt));
      if (pkt.magic != kRetroMpDeathRetryMagic || pkt.seq == 0)
        continue;
      if (isHost) {
        peerPort = ntohs(from.sin_port);
        char tmp[64]{};
        if (inet_ntop(AF_INET, &from.sin_addr, tmp, sizeof(tmp)))
          std::strncpy(peerHostUtf8, tmp, sizeof(peerHostUtf8) - 1);
      }
      remoteDeathRetryPending = true;
    }
  }
}

bool RetroMpSession::consumeRemoteDeathRetry() {
  if (!remoteDeathRetryPending)
    return false;
  remoteDeathRetryPending = false;
  return true;
}

void RetroMpSession::sendDeathRetry() {
  if (!active)
    return;
  if (isHost && peerHostUtf8[0] == '\0')
    return;
  if (!isHost && peerHostUtf8[0] == '\0')
    return;
  RetroMpDeathRetryPacket p{};
  p.magic = kRetroMpDeathRetryMagic;
  p.seq = ++deathRetrySendSeq;
  sockaddr_in to{};
  to.sin_family = AF_INET;
  to.sin_port = htons(peerPort);
  if (inet_pton(AF_INET, peerHostUtf8, &to.sin_addr) != 1)
    return;

#ifdef _WIN32
  ::sendto(static_cast<SOCKET>(sock), reinterpret_cast<const char*>(&p), sizeof(p), 0,
           reinterpret_cast<sockaddr*>(&to), sizeof(to));
#else
  ::sendto(sock, reinterpret_cast<const char*>(&p), sizeof(p), 0,
           reinterpret_cast<sockaddr*>(&to), sizeof(to));
#endif
}

void RetroMpSession::sendSnapshot(const glm::vec3& camPos, float yaw, float pitch, float eyeHeight,
                                  const glm::vec2& horizVel, int avatarClip, int blendFromClip,
                                  float clipBlend, double locoPhaseSec, uint8_t emoteFlags,
                                  uint8_t emoteDeathAux, uint8_t emoteAnimTag, uint8_t emoteBlendFromTag,
                                  const AudioStoreCycleSaveState* audioState) {
  if (!active)
    return;
  // Host learns peer address from the first inbound datagram; don't spam until then.
  if (isHost && peerHostUtf8[0] == '\0')
    return;
  if (!isHost && peerHostUtf8[0] == '\0')
    return;

  RetroMpWirePacket p{};
  p.magic = kRetroMpMagic;
  p.seq = ++sendSeq;
  p.camX = camPos.x;
  p.camY = camPos.y;
  p.camZ = camPos.z;
  p.yaw = yaw;
  p.pitch = pitch;
  p.eyeHeight = eyeHeight;
  p.horizVelX = horizVel.x;
  p.horizVelZ = horizVel.y; // horizontal plane uses vec2(x,z) in .x/.y
  p.avatarClip = avatarClip;
  p.blendFromClip = blendFromClip;
  p.clipBlend = clipBlend;
  std::memcpy(&p.locoPhaseBits, &locoPhaseSec, sizeof(double));
  p.emoteFlags = emoteFlags;
  p.emoteReserved0 = emoteDeathAux;
  p.emoteReserved1 = emoteAnimTag;
  p.emoteReserved2 = emoteBlendFromTag;
  if (audioState) {
    p.audioVersion = audioState->version;
    p.audioStorePhase = audioState->storePhase;
    p.audioFlags = audioState->flags;
    p.audioStoreCursorFrames = audioState->storeCursorFrames;
    p.audioHorrorCursorFrames = audioState->horrorCursorFrames;
    p.audioChaseCursorFrames = audioState->chaseCursorFrames;
    p.audioShrekCursorFrames = audioState->shrekCursorFrames;
    p.audioBlackoutRemainingMs = audioState->blackoutRemainingMs;
    p.audioDayRestoreRemainingMs = audioState->dayRestoreRemainingMs;
    p.audioStoreDayMusicTrackIdx = audioState->storeDayMusicTrackIdx;
  }

  sockaddr_in to{};
  to.sin_family = AF_INET;
  to.sin_port = htons(peerPort);
  if (inet_pton(AF_INET, peerHostUtf8, &to.sin_addr) != 1)
    return;

#ifdef _WIN32
  ::sendto(static_cast<SOCKET>(sock), reinterpret_cast<const char*>(&p), sizeof(p), 0,
           reinterpret_cast<sockaddr*>(&to), sizeof(to));
#else
  ::sendto(sock, reinterpret_cast<const char*>(&p), sizeof(p), 0,
           reinterpret_cast<sockaddr*>(&to), sizeof(to));
#endif
}

void RetroMpSession::sendDeliPickupRequest(const RetroMpDeliPickupPacket& p) {
  if (!active || isHost || peerHostUtf8[0] == '\0')
    return;
  RetroMpDeliPickupPacket out = p;
  out.magic = kRetroMpDeliPickupMagic;

  sockaddr_in to{};
  to.sin_family = AF_INET;
  to.sin_port = htons(peerPort);
  if (inet_pton(AF_INET, peerHostUtf8, &to.sin_addr) != 1)
    return;

#ifdef _WIN32
  ::sendto(static_cast<SOCKET>(sock), reinterpret_cast<const char*>(&out), sizeof(out), 0,
           reinterpret_cast<sockaddr*>(&to), sizeof(to));
#else
  ::sendto(sock, reinterpret_cast<const char*>(&out), sizeof(out), 0,
           reinterpret_cast<sockaddr*>(&to), sizeof(to));
#endif
}

void RetroMpSession::sendStaffMeleeRequest(const RetroMpStaffMeleePacket& p) {
  if (!active || isHost || peerHostUtf8[0] == '\0')
    return;
  RetroMpStaffMeleePacket out = p;
  out.magic = kRetroMpStaffMeleeMagic;

  sockaddr_in to{};
  to.sin_family = AF_INET;
  to.sin_port = htons(peerPort);
  if (inet_pton(AF_INET, peerHostUtf8, &to.sin_addr) != 1)
    return;

#ifdef _WIN32
  ::sendto(static_cast<SOCKET>(sock), reinterpret_cast<const char*>(&out), sizeof(out), 0,
           reinterpret_cast<sockaddr*>(&to), sizeof(to));
#else
  ::sendto(sock, reinterpret_cast<const char*>(&out), sizeof(out), 0,
           reinterpret_cast<sockaddr*>(&to), sizeof(to));
#endif
}

void RetroMpSession::sendFoodActionRequest(const RetroMpFoodActionPacket& p) {
  if (!active || isHost || peerHostUtf8[0] == '\0')
    return;
  RetroMpFoodActionPacket out = p;
  out.magic = kRetroMpFoodActionMagic;

  sockaddr_in to{};
  to.sin_family = AF_INET;
  to.sin_port = htons(peerPort);
  if (inet_pton(AF_INET, peerHostUtf8, &to.sin_addr) != 1)
    return;

#ifdef _WIN32
  ::sendto(static_cast<SOCKET>(sock), reinterpret_cast<const char*>(&out), sizeof(out), 0,
           reinterpret_cast<sockaddr*>(&to), sizeof(to));
#else
  ::sendto(sock, reinterpret_cast<const char*>(&out), sizeof(out), 0,
           reinterpret_cast<sockaddr*>(&to), sizeof(to));
#endif
}

void RetroMpSession::sendWorldSync(const void* payload, size_t payloadBytes) {
  if (!active || !isHost || !payload || payloadBytes < sizeof(RetroMpWorldHeader))
    return;
  if (payloadBytes > 1300)
    return;
  if (peerHostUtf8[0] == '\0')
    return;

  sockaddr_in to{};
  to.sin_family = AF_INET;
  to.sin_port = htons(peerPort);
  if (inet_pton(AF_INET, peerHostUtf8, &to.sin_addr) != 1)
    return;

#ifdef _WIN32
  ::sendto(static_cast<SOCKET>(sock), reinterpret_cast<const char*>(payload), static_cast<int>(payloadBytes), 0,
           reinterpret_cast<sockaddr*>(&to), sizeof(to));
#else
  ::sendto(sock, reinterpret_cast<const char*>(payload), payloadBytes, 0,
           reinterpret_cast<sockaddr*>(&to), sizeof(to));
#endif
}

bool RetroMpSession::remoteValid(float timeoutSec) const {
  if (!active || lastRecvMono < 0.0)
    return false;
  return (retroMpMonotonicSec() - lastRecvMono) < static_cast<double>(timeoutSec);
}

#undef IOCALL
