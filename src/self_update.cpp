#include "self_update.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include <httplib.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#ifndef RETRO_IKEA_RELEASE_TAG
#define RETRO_IKEA_RELEASE_TAG "beta-0"
#endif

namespace {

constexpr const char* kRepoOwner = "memesdudeguy";
constexpr const char* kRepoName = "RetroIkea";
constexpr const char* kInstallerAssetName = "RetroIkea-Beta-Setup.exe";

static bool envFlagEnabled(const char* name) {
  const char* v = std::getenv(name);
  return v && (v[0] == '1' || v[0] == 'y' || v[0] == 'Y' || v[0] == 't' || v[0] == 'T');
}

static int betaNumberFromTag(const std::string& tag) {
  const char* p = tag.c_str();
  while (*p && !std::isdigit(static_cast<unsigned char>(*p)))
    ++p;
  if (!*p)
    return -1;
  char* end = nullptr;
  const long n = std::strtol(p, &end, 10);
  if (end == p || n < 0 || n > 100000)
    return -1;
  return static_cast<int>(n);
}

static bool extractJsonStringAt(const std::string& s, size_t keyPos, const char* key, std::string& out) {
  const std::string pat = std::string("\"") + key + "\":";
  size_t p = s.find(pat, keyPos);
  if (p == std::string::npos)
    return false;
  p += pat.size();
  while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p])))
    ++p;
  if (p >= s.size() || s[p] != '"')
    return false;
  ++p;
  std::string val;
  while (p < s.size()) {
    if (s[p] == '\\' && p + 1 < s.size()) {
      val.push_back(s[p + 1]);
      p += 2;
      continue;
    }
    if (s[p] == '"') {
      out = std::move(val);
      return true;
    }
    val.push_back(s[p++]);
  }
  return false;
}

static bool findNewestReleaseTag(const std::string& releasesJson, int currentBeta, std::string& tagOut) {
  size_t pos = 0;
  while ((pos = releasesJson.find("\"tag_name\":", pos)) != std::string::npos) {
    std::string tag;
    if (!extractJsonStringAt(releasesJson, pos, "tag_name", tag)) {
      pos += 11;
      continue;
    }
    const int beta = betaNumberFromTag(tag);
    const size_t nextRelease = releasesJson.find("\"tag_name\":", pos + 11);
    const size_t assetPos = releasesJson.find(kInstallerAssetName, pos);
    if (beta > currentBeta && assetPos != std::string::npos &&
        (nextRelease == std::string::npos || assetPos < nextRelease)) {
      tagOut = std::move(tag);
      return true;
    }
    pos += 11;
  }
  return false;
}

static std::string installerPathForTag(const std::string& tag) {
  return std::string("/") + kRepoOwner + "/" + kRepoName + "/releases/download/" + tag + "/" + kInstallerAssetName;
}

static bool httpsGetBody(const char* host, const char* path, std::string& bodyOut, std::string& errOut,
                         int readTimeoutSec) {
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
  httplib::SSLClient cli(host, 443);
  cli.set_connection_timeout(2, 0);
  cli.set_read_timeout(readTimeoutSec, 0);
  cli.enable_server_certificate_verification(true);
  cli.set_follow_location(true);
  httplib::Headers headers{{"User-Agent", "RetroIkea-SelfUpdater"}, {"Accept", "*/*"}};
  auto res = cli.Get(path, headers);
  if (!res) {
    errOut = "network request failed";
    return false;
  }
  if (res->status < 200 || res->status >= 300) {
    errOut = "HTTP " + std::to_string(res->status);
    return false;
  }
  bodyOut = std::move(res->body);
  return true;
#else
  (void)host;
  (void)path;
  (void)bodyOut;
  (void)readTimeoutSec;
  errOut = "HTTPS updater needs OpenSSL build";
  return false;
#endif
}

static bool writeFileBytes(const std::string& path, const std::string& bytes, std::string& errOut) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    errOut = "could not create installer file";
    return false;
  }
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!f) {
    errOut = "could not write installer file";
    return false;
  }
  return true;
}

static std::string tempInstallerPath(const std::string& tag) {
#if defined(_WIN32)
  char dir[MAX_PATH]{};
  DWORD n = GetTempPathA(static_cast<DWORD>(sizeof(dir)), dir);
  if (n == 0 || n >= sizeof(dir))
    return std::string(kInstallerAssetName);
  std::string safeTag = tag;
  for (char& c : safeTag)
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_')
      c = '-';
  return std::string(dir) + "RetroIkea-" + safeTag + "-Setup.exe";
#else
  return std::string("/tmp/RetroIkea-") + tag + "-Setup.exe";
#endif
}

static bool launchInstallerAndExit(const std::string& path, std::string& errOut) {
#if defined(_WIN32)
  const char* args = "/SP- /SILENT /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS";
  HINSTANCE r = ShellExecuteA(nullptr, "open", path.c_str(), args, nullptr, SW_SHOWNORMAL);
  if (reinterpret_cast<intptr_t>(r) <= 32) {
    errOut = "could not launch installer";
    return false;
  }
  return true;
#else
  (void)path;
  errOut = "self-install is only enabled on Windows";
  return false;
#endif
}

}  // namespace

bool retroIkeaAutoUpdateMaybeLaunch(std::string& statusOut) {
  statusOut.clear();
#if !defined(_WIN32)
  return false;
#else
  if (envFlagEnabled("RETRO_IKEA_DISABLE_AUTO_UPDATE"))
    return false;

  const std::string currentTag = RETRO_IKEA_RELEASE_TAG;
  const int currentBeta = betaNumberFromTag(currentTag);
  if (currentBeta < 0) {
    statusOut = "auto-update skipped: bad current release tag " + currentTag;
    return false;
  }

  std::string releases;
  std::string err;
  const std::string apiPath = std::string("/repos/") + kRepoOwner + "/" + kRepoName + "/releases?per_page=10";
  if (!httpsGetBody("api.github.com", apiPath.c_str(), releases, err, 5)) {
    statusOut = "auto-update check failed: " + err;
    return false;
  }

  std::string nextTag;
  if (!findNewestReleaseTag(releases, currentBeta, nextTag)) {
    statusOut = "auto-update: already current (" + currentTag + ")";
    return false;
  }

  statusOut = "auto-update: downloading " + nextTag;
  std::fprintf(stderr, "[update] %s\n", statusOut.c_str());

  std::string installerBytes;
  if (!httpsGetBody("github.com", installerPathForTag(nextTag).c_str(), installerBytes, err, 120)) {
    statusOut = "auto-update download failed: " + err;
    return false;
  }
  if (installerBytes.size() < 1024 * 1024) {
    statusOut = "auto-update download failed: installer too small";
    return false;
  }

  const std::string installerPath = tempInstallerPath(nextTag);
  if (!writeFileBytes(installerPath, installerBytes, err)) {
    statusOut = "auto-update save failed: " + err;
    return false;
  }

  if (!launchInstallerAndExit(installerPath, err)) {
    statusOut = "auto-update launch failed: " + err;
    return false;
  }

  statusOut = "auto-update launched " + nextTag;
  return true;
#endif
}

