#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
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
bool lobbyUnregisterServer(const char* lobbyBaseUrl, const char* sessionId, std::string& errMsg);

const char* lobbyEnvUrl();

// Background publisher: keeps the lobby entry alive via periodic POST heartbeats so the host shows up
// in clients' server lists immediately. Synchronous network calls run on a worker thread to avoid main-loop hitches.
class LobbyHostPublisher {
 public:
  enum class State {
    kIdle,
    kRegistering,
    kRegistered,
    kError,
    kDisabled,
  };

  ~LobbyHostPublisher();

  // Begin publishing. Safe to call again to refresh fields (host/port/name) without dropping the existing entry.
  // Pass an empty url to mark the publisher as disabled (status text reports the reason).
  void start(const std::string& url, const std::string& sessionId, const std::string& host, uint16_t port,
             const std::string& name);

  // Send a final DELETE so the host disappears from the lobby quickly, then join the worker.
  void stop();

  bool active() const { return active_.load(std::memory_order_acquire); }
  State state() const { return state_.load(std::memory_order_acquire); }

  // Short status string for the pause menu / logs (e.g. "REGISTERED <name>", "REGISTERING…", "ERR <message>").
  std::string statusText() const;

 private:
  void workerLoop();
  bool postHeartbeatLocked(std::string& errOut);

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::thread worker_;
  std::atomic<bool> active_{false};
  std::atomic<bool> stopRequested_{false};
  std::atomic<State> state_{State::kIdle};

  std::string url_;
  std::string sessionId_;
  std::string host_;
  std::string name_;
  uint16_t port_ = 27341;
  bool fieldsDirty_ = false;
  bool firstHeartbeatSent_ = false;
  std::string lastErr_;
  std::string lastNamePublished_;
};
