#include "lobby_http.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>

#include <httplib.h>

namespace {

struct UrlParts {
  bool tls = false;
  std::string host;
  int port = 80;
};

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
  UrlParts u;
  if (!parseLobbyOrigin(lobbyBaseUrl, u, errMsg))
    return false;
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
  return httplibPostJson(u, "/api/v1/servers/register", json, errMsg);
}
