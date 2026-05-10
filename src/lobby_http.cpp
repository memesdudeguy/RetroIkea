#include "lobby_http.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include <httplib.h>

namespace {

struct UrlParts {
  bool tls = false;
  std::string host;
  int port = 80;
};

// Parsed "github://owner/repo[/branch]" form. branch defaults to "main" so the
// canonical default URL stays compact: github://memesdudeguy/RetroIkea.
struct GhRepoTarget {
  std::string owner;
  std::string repo;
  std::string branch;
};

static bool parseGithubLobbyUrl(const char* raw, GhRepoTarget& out, std::string& err) {
  out = GhRepoTarget{};
  if (!raw)
    return false;
  std::string u(raw);
  while (!u.empty() && (u.back() == '/' || u.back() == ' '))
    u.pop_back();
  constexpr const char* kPrefix = "github://";
  constexpr size_t kPrefixLen = 9;  // strlen("github://")
  if (u.size() < kPrefixLen || u.compare(0, kPrefixLen, kPrefix) != 0)
    return false;  // Not a github URL; let the caller try regular http(s).
  u.erase(0, kPrefixLen);
  const size_t firstSlash = u.find('/');
  if (firstSlash == std::string::npos || firstSlash == 0) {
    err = "github lobby URL needs owner/repo (e.g. github://memesdudeguy/RetroIkea)";
    return false;
  }
  out.owner = u.substr(0, firstSlash);
  std::string rest = u.substr(firstSlash + 1);
  const size_t secondSlash = rest.find('/');
  if (secondSlash == std::string::npos) {
    out.repo = rest;
    out.branch = "main";
  } else {
    out.repo = rest.substr(0, secondSlash);
    out.branch = rest.substr(secondSlash + 1);
    if (out.branch.empty())
      out.branch = "main";
  }
  if (out.owner.empty() || out.repo.empty()) {
    err = "github lobby URL is missing owner or repo";
    return false;
  }
  return true;
}

static bool isGithubLobbyUrl(const char* raw) {
  if (!raw)
    return false;
  return std::strncmp(raw, "github://", 9) == 0;
}

static std::string describeLobbyRequest(const UrlParts& u, const char* path) {
  std::string s = u.tls ? "https://" : "http://";
  s += u.host;
  const bool defaultPort = (u.tls && u.port == 443) || (!u.tls && u.port == 80);
  if (!defaultPort)
    s += ":" + std::to_string(u.port);
  s += path ? path : "/";
  return s;
}

static int lobbyTimeoutSec(const char* name, int fallback, int lo, int hi) {
  if (const char* raw = std::getenv(name)) {
    char* end = nullptr;
    const long v = std::strtol(raw, &end, 10);
    if (end != raw)
      return static_cast<int>(std::max<long>(lo, std::min<long>(hi, v)));
  }
  return fallback;
}

static bool parseLobbyOrigin(const char* raw, UrlParts& out, std::string& err) {
  out = UrlParts{};
  if (!raw || raw[0] == '\0') {
    err = "Empty lobby URL";
    return false;
  }
  std::string u(raw);
  while (!u.empty() && (u.back() == '/' || u.back() == ' '))
    u.pop_back();
  if (u.size() >= 8 && u.compare(0, 8, "https://") == 0) {
    out.tls = true;
    out.port = 443;
    u.erase(0, 8);
  } else if (u.size() >= 7 && u.compare(0, 7, "http://") == 0) {
    out.tls = false;
    out.port = 80;
    u.erase(0, 7);
  } else {
    err = "Lobby URL must start with http:// or https://";
    return false;
  }

  const size_t colon = u.find(':');
  const size_t slash = u.find('/');
  if (colon != std::string::npos &&
      (slash == std::string::npos || colon < slash)) {
    out.host = u.substr(0, colon);
    std::string portStr = u.substr(colon + 1);
    if (slash != std::string::npos && slash < u.size())
      portStr = u.substr(colon + 1, slash - (colon + 1));
    char* end = nullptr;
    long p = std::strtol(portStr.c_str(), &end, 10);
    if (!end || end == portStr.c_str() || p < 1 || p > 65535) {
      err = "Bad port in lobby URL";
      return false;
    }
    out.port = static_cast<int>(p);
  } else {
    if (slash != std::string::npos)
      out.host = u.substr(0, slash);
    else
      out.host = std::move(u);
  }
  if (out.host.empty()) {
    err = "Missing host in lobby URL";
    return false;
  }
  return true;
}

static bool httplibGet(const UrlParts& u, const char* path, std::string& bodyOut, std::string& err) {
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
  if (u.tls) {
    httplib::SSLClient cli(u.host.c_str(), u.port);
    cli.set_connection_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_CONNECT_TIMEOUT_SEC", 1, 1, 15), 0);
    cli.set_read_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_READ_TIMEOUT_SEC", 2, 1, 30), 0);
    cli.enable_server_certificate_verification(true);
    auto res = cli.Get(path);
    if (!res) {
      err = "HTTPS GET failed (network)";
      return false;
    }
    if (res->status < 200 || res->status >= 300) {
      err = "Lobby HTTP " + std::to_string(res->status) + " at " + describeLobbyRequest(u, path);
      if (res->status == 404)
        err += " (wrong lobby URL or deploy)";
      return false;
    }
    bodyOut = res->body;
    return true;
  }
#else
  if (u.tls) {
    err = "HTTPS lobby URL needs an OpenSSL-enabled build (install OpenSSL dev, rebuild)";
    return false;
  }
#endif
  httplib::Client cli(u.host.c_str(), u.port);
  cli.set_connection_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_CONNECT_TIMEOUT_SEC", 1, 1, 15), 0);
  cli.set_read_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_READ_TIMEOUT_SEC", 2, 1, 30), 0);
  auto res = cli.Get(path);
  if (!res) {
    err = "HTTP GET failed (network)";
    return false;
  }
  if (res->status < 200 || res->status >= 300) {
    err = "Lobby HTTP " + std::to_string(res->status) + " at " + describeLobbyRequest(u, path);
    if (res->status == 404)
      err += " (wrong lobby URL or deploy)";
    return false;
  }
  bodyOut = res->body;
  return true;
}

static bool httplibPostJson(const UrlParts& u, const char* path, const std::string& json, std::string& err) {
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
  if (u.tls) {
    httplib::SSLClient cli(u.host.c_str(), u.port);
    cli.set_connection_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_CONNECT_TIMEOUT_SEC", 1, 1, 15), 0);
    cli.set_read_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_READ_TIMEOUT_SEC", 2, 1, 30), 0);
    cli.enable_server_certificate_verification(true);
    httplib::Headers h{{"Content-Type", "application/json"}};
    auto res = cli.Post(path, h, json, "application/json");
    if (!res) {
      err = "HTTPS POST failed (network)";
      return false;
    }
    if (res->status < 200 || res->status >= 300) {
      err = "Lobby HTTP " + std::to_string(res->status) + " at " + describeLobbyRequest(u, path);
      if (res->status == 404)
        err += " (wrong lobby URL or deploy)";
      return false;
    }
    return true;
  }
#else
  if (u.tls) {
    err = "HTTPS lobby URL needs an OpenSSL-enabled build (install OpenSSL dev, rebuild)";
    return false;
  }
#endif
  httplib::Client cli(u.host.c_str(), u.port);
  cli.set_connection_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_CONNECT_TIMEOUT_SEC", 1, 1, 15), 0);
  cli.set_read_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_READ_TIMEOUT_SEC", 2, 1, 30), 0);
  httplib::Headers h{{"Content-Type", "application/json"}};
  auto res = cli.Post(path, h, json, "application/json");
  if (!res) {
    err = "HTTP POST failed (network)";
    return false;
  }
  if (res->status < 200 || res->status >= 300) {
    err = "Lobby HTTP " + std::to_string(res->status) + " at " + describeLobbyRequest(u, path);
    if (res->status == 404)
      err += " (wrong lobby URL or deploy)";
    return false;
  }
  return true;
}

static bool httplibDelete(const UrlParts& u, const char* path, std::string& err) {
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
  if (u.tls) {
    httplib::SSLClient cli(u.host.c_str(), u.port);
    cli.set_connection_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_CONNECT_TIMEOUT_SEC", 1, 1, 15), 0);
    cli.set_read_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_READ_TIMEOUT_SEC", 2, 1, 30), 0);
    cli.enable_server_certificate_verification(true);
    auto res = cli.Delete(path);
    if (!res) {
      err = "HTTPS DELETE failed (network)";
      return false;
    }
    if (res->status < 200 || res->status >= 300) {
      err = "Lobby HTTP " + std::to_string(res->status) + " at " + describeLobbyRequest(u, path);
      return false;
    }
    return true;
  }
#else
  if (u.tls) {
    err = "HTTPS lobby URL needs an OpenSSL-enabled build (install OpenSSL dev, rebuild)";
    return false;
  }
#endif
  httplib::Client cli(u.host.c_str(), u.port);
  cli.set_connection_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_CONNECT_TIMEOUT_SEC", 1, 1, 15), 0);
  cli.set_read_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_READ_TIMEOUT_SEC", 2, 1, 30), 0);
  auto res = cli.Delete(path);
  if (!res) {
    err = "HTTP DELETE failed (network)";
    return false;
  }
  if (res->status < 200 || res->status >= 300) {
    err = "Lobby HTTP " + std::to_string(res->status) + " at " + describeLobbyRequest(u, path);
    return false;
  }
  return true;
}

// Read https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}. Anonymous; no token
// needed. Subject to GitHub's 60-req/hour/IP unauth rate limit, which is plenty for periodic
// browser refreshes (the title menu typically only refreshes when the user pushes REFRESH).
static bool githubRawGet(const GhRepoTarget& gh, const char* path, std::string& bodyOut,
                         std::string& err) {
#if !defined(CPPHTTPLIB_OPENSSL_SUPPORT)
  err = "GitHub-only lobby needs an OpenSSL-enabled build (HTTPS to raw.githubusercontent.com)";
  return false;
#else
  httplib::SSLClient cli("raw.githubusercontent.com", 443);
  cli.set_connection_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_CONNECT_TIMEOUT_SEC", 4, 1, 30), 0);
  cli.set_read_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_READ_TIMEOUT_SEC", 8, 1, 60), 0);
  cli.enable_server_certificate_verification(true);
  std::string p = "/" + gh.owner + "/" + gh.repo + "/" + gh.branch + "/" + (path ? path : "");
  // raw.githubusercontent.com follows the same `User-Agent: ...` etiquette as the API.
  httplib::Headers h{{"User-Agent", "RetroIkea-Lobby/1"}};
  auto res = cli.Get(p.c_str(), h);
  if (!res) {
    err = "raw.githubusercontent.com unreachable (network)";
    return false;
  }
  if (res->status == 404) {
    err = "GitHub lobby registry not found at " + p +
          " — has the lobby workflow ever run? (it auto-creates the file).";
    return false;
  }
  if (res->status < 200 || res->status >= 300) {
    err = "GitHub raw HTTP " + std::to_string(res->status) + " at " + p;
    return false;
  }
  bodyOut = res->body;
  return true;
#endif
}

// POST https://api.github.com/repos/{owner}/{repo}/dispatches with a personal access token to
// trigger the lobby workflow. Token comes from RETRO_IKEA_GH_TOKEN at runtime — there is no
// safe way to bake a write token into the public game binary. The token only needs the
// "Contents: Read & Write" fine-grained scope on the lobby repo (so the workflow can commit
// the registry update) — it never touches user files.
static bool githubDispatchEvent(const GhRepoTarget& gh, const char* eventType,
                                const std::string& clientPayloadJson, std::string& err) {
#if !defined(CPPHTTPLIB_OPENSSL_SUPPORT)
  err = "GitHub-only lobby needs an OpenSSL-enabled build (HTTPS to api.github.com)";
  return false;
#else
  const char* token = std::getenv("RETRO_IKEA_GH_TOKEN");
  if (!token || token[0] == '\0') {
    err = "Set RETRO_IKEA_GH_TOKEN to a fine-grained PAT (Contents: Read & Write on " +
          gh.owner + "/" + gh.repo + ") so this host can publish to the lobby.";
    return false;
  }
  httplib::SSLClient cli("api.github.com", 443);
  cli.set_connection_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_CONNECT_TIMEOUT_SEC", 4, 1, 30), 0);
  cli.set_read_timeout(lobbyTimeoutSec("RETRO_IKEA_LOBBY_READ_TIMEOUT_SEC", 8, 1, 60), 0);
  cli.enable_server_certificate_verification(true);
  std::string body = std::string("{\"event_type\":\"") + eventType +
                     "\",\"client_payload\":" + clientPayloadJson + "}";
  std::string auth = std::string("Bearer ") + token;
  httplib::Headers h{
      {"User-Agent", "RetroIkea-Lobby/1"},
      {"Accept", "application/vnd.github+json"},
      {"X-GitHub-Api-Version", "2022-11-28"},
      {"Authorization", auth},
  };
  std::string path = "/repos/" + gh.owner + "/" + gh.repo + "/dispatches";
  auto res = cli.Post(path.c_str(), h, body, "application/json");
  if (!res) {
    err = "api.github.com unreachable (network)";
    return false;
  }
  // 204 No Content is the documented success for repository_dispatch.
  if (res->status < 200 || res->status >= 300) {
    err = "GitHub dispatch HTTP " + std::to_string(res->status) + " at " + path;
    if (res->status == 401)
      err += " — RETRO_IKEA_GH_TOKEN missing or wrong";
    else if (res->status == 403)
      err += " — token lacks Contents write on " + gh.owner + "/" + gh.repo;
    else if (res->status == 404)
      err += " — repo not found / token has no access";
    return false;
  }
  return true;
#endif
}

static bool extractJsonStringField(const std::string& obj, const char* key, std::string& out) {
  // FastAPI / Python json.dumps uses spaces: "host": "1.2.3.4" — not "host":"1.2.3.4"
  const std::string pat = std::string("\"") + key + "\":";
  size_t p = obj.find(pat);
  if (p == std::string::npos)
    return false;
  p += pat.size();
  while (p < obj.size() && std::isspace(static_cast<unsigned char>(obj[p])))
    ++p;
  if (p >= obj.size() || obj[p] != '"')
    return false;
  ++p;
  const size_t start = p;
  while (p < obj.size()) {
    if (obj[p] == '\\' && p + 1 < obj.size()) {
      p += 2;
      continue;
    }
    if (obj[p] == '"') {
      out.assign(obj, start, p - start);
      return true;
    }
    ++p;
  }
  return false;
}

static bool extractJsonIntField(const std::string& obj, const char* key, int& out) {
  const std::string pat = std::string("\"") + key + "\":";
  size_t p = obj.find(pat);
  if (p == std::string::npos)
    return false;
  p += pat.size();
  while (p < obj.size() && std::isspace(static_cast<unsigned char>(obj[p])))
    ++p;
  char* end = nullptr;
  long v = std::strtol(obj.c_str() + static_cast<ptrdiff_t>(p), &end, 10);
  if (!end || end == obj.c_str() + static_cast<ptrdiff_t>(p))
    return false;
  out = static_cast<int>(v);
  return true;
}

static std::vector<std::string> splitTopLevelJsonObjects(const std::string& arrBody) {
  std::vector<std::string> out;
  size_t i = arrBody.find('[');
  if (i == std::string::npos)
    return out;
  ++i;
  while (i < arrBody.size()) {
    while (i < arrBody.size() && std::isspace(static_cast<unsigned char>(arrBody[i])))
      ++i;
    if (i >= arrBody.size() || arrBody[i] != '{')
      break;
    size_t depth = 0;
    const size_t start = i;
    for (; i < arrBody.size(); ++i) {
      if (arrBody[i] == '{')
        ++depth;
      else if (arrBody[i] == '}') {
        if (depth > 0)
          --depth;
        if (depth == 0) {
          ++i;
          out.emplace_back(arrBody.substr(start, i - start));
          break;
        }
      }
    }
    while (i < arrBody.size() && (arrBody[i] == ',' || std::isspace(static_cast<unsigned char>(arrBody[i]))))
      ++i;
  }
  return out;
}

}  // namespace

const char* lobbyEnvUrl() {
  if (const char* e = std::getenv("RETRO_IKEA_LOBBY_URL"))
    if (e[0] != '\0')
      return e;
#ifdef RETRO_IKEA_DEFAULT_LOBBY_URL
  return RETRO_IKEA_DEFAULT_LOBBY_URL;
#else
  return "";
#endif
}

bool lobbyFetchServerList(const char* lobbyBaseUrl, std::vector<LobbyListedServer>& out, std::string& errMsg) {
  out.clear();
  if (isGithubLobbyUrl(lobbyBaseUrl)) {
    GhRepoTarget gh;
    if (!parseGithubLobbyUrl(lobbyBaseUrl, gh, errMsg))
      return false;
    std::string body;
    if (!githubRawGet(gh, "lobby/registry.json", body, errMsg))
      return false;
    // The GitHub-only lobby JSON has the form
    //   { "version": 1, "ttl_sec": 90, "servers": [ {id,host,port,name,expires_at}, ... ] }
    // Locate the "servers" array, then split top-level objects out of it.
    const std::string serversKey = "\"servers\":";
    size_t kp = body.find(serversKey);
    if (kp == std::string::npos)
      return true;  // No servers field yet — empty list, not an error.
    kp += serversKey.size();
    while (kp < body.size() && std::isspace(static_cast<unsigned char>(body[kp])))
      ++kp;
    if (kp >= body.size() || body[kp] != '[')
      return true;
    const auto objs = splitTopLevelJsonObjects(body.substr(kp));
    for (const std::string& seg : objs) {
      LobbyListedServer row{};
      extractJsonStringField(seg, "id", row.id);
      extractJsonStringField(seg, "host", row.host);
      int port = 27341;
      extractJsonIntField(seg, "port", port);
      if (port < 1 || port > 65535)
        port = 27341;
      row.port = static_cast<uint16_t>(port);
      extractJsonStringField(seg, "name", row.name);
      if (!row.host.empty())
        out.push_back(std::move(row));
    }
    return true;
  }
  UrlParts u;
  if (!parseLobbyOrigin(lobbyBaseUrl, u, errMsg))
    return false;
  std::string body;
  if (!httplibGet(u, "/api/v1/servers", body, errMsg))
    return false;
  const auto objs = splitTopLevelJsonObjects(body);
  for (const std::string& seg : objs) {
    LobbyListedServer row{};
    extractJsonStringField(seg, "id", row.id);
    extractJsonStringField(seg, "host", row.host);
    int port = 27341;
    extractJsonIntField(seg, "port", port);
    if (port < 1 || port > 65535)
      port = 27341;
    row.port = static_cast<uint16_t>(port);
    extractJsonStringField(seg, "name", row.name);
    if (!row.host.empty())
      out.push_back(std::move(row));
  }
  return true;
}

bool lobbyRegisterHeartbeat(const char* lobbyBaseUrl, const char* sessionId, const char* hostIpv4, uint16_t port,
                            const char* displayName, std::string& errMsg) {
  if (!sessionId || sessionId[0] == '\0' || !hostIpv4 || hostIpv4[0] == '\0') {
    errMsg = "Missing session or host for lobby register";
    return false;
  }
  std::string escName;
  if (displayName && displayName[0] != '\0') {
    escName = displayName;
    for (char& c : escName) {
      if (c == '"' || c == '\\')
        c = '_';
    }
  }
  std::string json = std::string("{\"id\":\"") + sessionId + "\",\"host\":\"" + hostIpv4 + "\",\"port\":" +
                     std::to_string(static_cast<unsigned>(port)) + ",\"name\":\"" + escName + "\"}";
  if (isGithubLobbyUrl(lobbyBaseUrl)) {
    GhRepoTarget gh;
    if (!parseGithubLobbyUrl(lobbyBaseUrl, gh, errMsg))
      return false;
    return githubDispatchEvent(gh, "lobby_register", json, errMsg);
  }
  UrlParts u;
  if (!parseLobbyOrigin(lobbyBaseUrl, u, errMsg))
    return false;
  return httplibPostJson(u, "/api/v1/servers/register", json, errMsg);
}

bool lobbyUnregisterServer(const char* lobbyBaseUrl, const char* sessionId, std::string& errMsg) {
  if (!sessionId || sessionId[0] == '\0') {
    errMsg = "Missing session id for lobby unregister";
    return false;
  }
  if (isGithubLobbyUrl(lobbyBaseUrl)) {
    GhRepoTarget gh;
    if (!parseGithubLobbyUrl(lobbyBaseUrl, gh, errMsg))
      return false;
    std::string json = std::string("{\"id\":\"") + sessionId + "\"}";
    return githubDispatchEvent(gh, "lobby_unregister", json, errMsg);
  }
  UrlParts u;
  if (!parseLobbyOrigin(lobbyBaseUrl, u, errMsg))
    return false;
  std::string path = std::string("/api/v1/servers/") + sessionId;
  return httplibDelete(u, path.c_str(), errMsg);
}

namespace {
std::chrono::seconds lobbyHeartbeatPeriod(const std::string& url) {
  if (const char* raw = std::getenv("RETRO_IKEA_LOBBY_HEARTBEAT_SEC")) {
    char* end = nullptr;
    long v = std::strtol(raw, &end, 10);
    if (end != raw && v >= 5 && v <= 300)
      return std::chrono::seconds(v);
  }
  // GitHub Actions is paced in ~30s units (queue + runner spin-up); heartbeat slower than the
  // FastAPI default to avoid filling the workflow queue with redundant register events while
  // still beating the 90s registry TTL comfortably.
  if (url.size() >= 9 && url.compare(0, 9, "github://") == 0)
    return std::chrono::seconds(30);
  return std::chrono::seconds(15);
}
}  // namespace

LobbyHostPublisher::~LobbyHostPublisher() {
  stop();
}

void LobbyHostPublisher::start(const std::string& url, const std::string& sessionId, const std::string& host,
                               uint16_t port, const std::string& name) {
  std::unique_lock<std::mutex> lk(mu_);
  url_ = url;
  sessionId_ = sessionId;
  host_ = host;
  port_ = port;
  name_ = name;
  fieldsDirty_ = true;
  firstHeartbeatSent_ = false;
  if (url.empty()) {
    state_.store(State::kDisabled, std::memory_order_release);
    lastErr_ = "Set RETRO_IKEA_LOBBY_URL to publish this host";
  } else if (host.empty() || sessionId.empty()) {
    state_.store(State::kRegistering, std::memory_order_release);
    lastErr_.clear();
  } else {
    state_.store(State::kRegistering, std::memory_order_release);
    lastErr_.clear();
  }
  if (!worker_.joinable()) {
    stopRequested_.store(false, std::memory_order_release);
    active_.store(true, std::memory_order_release);
    worker_ = std::thread(&LobbyHostPublisher::workerLoop, this);
  } else {
    cv_.notify_all();
  }
}

void LobbyHostPublisher::stop() {
  std::thread joiner;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!worker_.joinable())
      return;
    stopRequested_.store(true, std::memory_order_release);
    cv_.notify_all();
    joiner = std::move(worker_);
  }
  if (joiner.joinable())
    joiner.join();
  active_.store(false, std::memory_order_release);
  state_.store(State::kIdle, std::memory_order_release);
}

bool LobbyHostPublisher::postHeartbeatLocked(std::string& errOut) {
  std::string url, sessionId, host, name;
  uint16_t port = 0;
  {
    std::lock_guard<std::mutex> lk(mu_);
    url = url_;
    sessionId = sessionId_;
    host = host_;
    name = name_;
    port = port_;
    fieldsDirty_ = false;
  }
  if (url.empty() || host.empty() || sessionId.empty()) {
    errOut = host.empty() ? "no public IP yet" : "publisher not configured";
    return false;
  }
  return lobbyRegisterHeartbeat(url.c_str(), sessionId.c_str(), host.c_str(), port,
                                name.empty() ? "RetroIkea" : name.c_str(), errOut);
}

void LobbyHostPublisher::workerLoop() {
  std::string snapshotUrl;
  {
    std::lock_guard<std::mutex> lk(mu_);
    snapshotUrl = url_;
  }
  const auto period = lobbyHeartbeatPeriod(snapshotUrl);
  while (!stopRequested_.load(std::memory_order_acquire)) {
    std::string err;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!url_.empty() && !host_.empty() && !sessionId_.empty()) {
        if (state_.load(std::memory_order_acquire) != State::kRegistered)
          state_.store(State::kRegistering, std::memory_order_release);
      }
    }
    const bool ok = postHeartbeatLocked(err);
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (ok) {
        if (!firstHeartbeatSent_)
          std::fprintf(stderr, "[lobby] host listed as \"%s\" at %s:%u\n",
                       (name_.empty() ? "RetroIkea" : name_.c_str()), host_.c_str(),
                       static_cast<unsigned>(port_));
        firstHeartbeatSent_ = true;
        lastNamePublished_ = name_;
        lastErr_.clear();
        state_.store(State::kRegistered, std::memory_order_release);
      } else {
        lastErr_ = err;
        if (url_.empty())
          state_.store(State::kDisabled, std::memory_order_release);
        else
          state_.store(State::kError, std::memory_order_release);
      }
    }
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait_for(lk, period, [this] {
      return stopRequested_.load(std::memory_order_acquire) || fieldsDirty_;
    });
  }

  // Final DELETE so the lobby drops the entry without waiting for TTL.
  std::string url, sessionId;
  {
    std::lock_guard<std::mutex> lk(mu_);
    url = url_;
    sessionId = sessionId_;
  }
  if (!url.empty() && !sessionId.empty() && firstHeartbeatSent_) {
    std::string err;
    if (lobbyUnregisterServer(url.c_str(), sessionId.c_str(), err))
      std::fprintf(stderr, "[lobby] unregistered host\n");
    else
      std::fprintf(stderr, "[lobby] unregister failed: %s\n", err.c_str());
  }
}

std::string LobbyHostPublisher::statusText() const {
  std::lock_guard<std::mutex> lk(mu_);
  switch (state_.load(std::memory_order_acquire)) {
    case State::kIdle:
      return "OFF";
    case State::kRegistering:
      if (host_.empty())
        return "WAITING FOR PUBLIC IP";
      return "REGISTERING\xE2\x80\xA6";
    case State::kRegistered: {
      std::string s = "LISTED AS ";
      s += lastNamePublished_.empty() ? std::string("RETROIKEA") : lastNamePublished_;
      return s;
    }
    case State::kError:
      return std::string("ERR ") + lastErr_;
    case State::kDisabled:
      return lastErr_.empty() ? std::string("LOBBY URL NOT SET") : lastErr_;
  }
  return "?";
}
