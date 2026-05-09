#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct LobbyListedServer {
  std::string id;
  std::string host;
  uint16_t port = 27341;
  std::string name;
};

// lobbyBaseUrl: origin only, no trailing slash (e.g. https://lobby.example.com or http://127.0.0.1:8765)
bool lobbyFetchServerList(const char* lobbyBaseUrl, std::vector<LobbyListedServer>& out, std::string& errMsg);
bool lobbyRegisterHeartbeat(const char* lobbyBaseUrl, const char* sessionId, const char* hostIpv4, uint16_t port,
                            const char* displayName, std::string& errMsg);

const char* lobbyEnvUrl();
