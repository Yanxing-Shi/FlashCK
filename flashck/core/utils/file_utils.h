#pragma once

#include <filesystem>

namespace flashck {

bool CheckWithRetries(const std::filesystem::path& exe_path, const int max_attempts = 3, const int delay_seconds = 5);

std::filesystem::path CreateTemporaryDirectory(const std::string& prefix);

}  // namespace flashck