#pragma once

#include <string>

// Returns true when a newer installer was launched and the caller should exit.
bool retroIkeaAutoUpdateMaybeLaunch(std::string& statusOut);

