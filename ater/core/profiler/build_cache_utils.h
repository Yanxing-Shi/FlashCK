#pragma once

#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "ater/core/utils/enforce.h"
#include "ater/core/utils/flags.h"
#include "ater/core/utils/log.h"
#include "ater/core/utils/printf.h"
#include "ater/core/utils/string_utils.h"

ATER_DECLARE_bool(ATER_FORCE_PROFILER_CACHE);
ATER_DECLARE_int32(ATER_BUILD_CACHE_SKIP_PERCENTAGE);

namespace ater {

const static std::vector<std::string> g_source_extensions = {
    "cpp",
    "h",
    "cu",
    "cuh",
    "c",
    "hpp",
    "hxx",
    "inl",
    "py",
    "cxx",
    "cc",
    "version",
    "binhash",
    "hash",
};

const static std::vector<std::string> g_source_filenames = {"makefile"};

const static std::vector<std::string> g_source_filenames_prefix = {"makefile"};

// File extensions of files to be considered cache artifacts(unless they are considered source files)
// note : we're not caching .obj files anymore as these are not strictly necessary to keep.
const static std::vector<std::string> g_cache_extensions = {"so", "dll", "exe", ""};

/*
Splits filename into basename and extension
    and lowercases results to enable simple lookup
    in a case-insensitive manner.

    Args:
        filename (str): Filename/Path to split

    Returns:
        Tuple[str,str]: file basename, file extension
*/
inline std::tuple<std::string, std::string> FileNameNormSplit(std::string& file_name)
{
    std::string base_name = std::filesystem::path(file_name).stem();
    std::string extension = std::filesystem::path(file_name).extension();

    std::transform(
        base_name.begin(), base_name.end(), base_name.begin(), [](unsigned char c) { return std::tolower(c); });
    std::transform(
        extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
    return std::make_tuple(base_name, extension);
}

/*
Simple filter function, returns true if the passed filename is considered
    to be a source file (used to build the cache key) for the purpose of caching

    Args:
        filename (str): File path as a string

    Returns:
        bool: Whether the filename is a source file
*/
inline bool IsSource(std::string& file_name)
{
    std::string base_name, extension;
    std::tie(base_name, extension) = FileNameNormSplit(file_name);
    return std::find(std::begin(g_source_filenames), std::end(g_source_filenames), base_name)
               != std::end(g_source_filenames)
           || std::find(std::begin(g_source_extensions), std::end(g_source_extensions), extension)
                  != std::end(g_source_extensions)
           || std::any_of(std::begin(g_source_filenames_prefix),
                          std::end(g_source_filenames_prefix),
                          [&](const std::string& prefix) { return StartsWith(base_name, prefix); });
}

/*
Simple filter function, returns true if the passed filename is considered
    to be a cacheable artifact (not used to build cache key, but stored in cache)
    for the purpose of caching

    Args:
        filename (str): File path as a string

    Returns:
        bool: Whether the filename is a cache artifact
*/
inline bool IsCacheArtifact(std::string& file_name)
{
    std::string base_name, extension;
    std::tie(base_name, extension) = FileNameNormSplit(file_name);
    return std::find(std::begin(g_cache_extensions), std::end(g_cache_extensions), extension)
               != std::end(g_cache_extensions)
           && !IsSource(file_name);
}

/*
Simple filter function, returns true if the passed filename is considered
    to be a bin file which needs to be considered for the purpose of creating
    a cache-key, but may be deleted after an initial build.

    bin files are hashed, and their hashes are kept in a small separete file
    for future use when building the cache key. So the hash is not lost, even if the binary
    file is deleted.

    Args:
        filename (str): File path as a string

    Returns:
        bool: Whether the filename is a binary file in the above sense
*/
inline bool IsBinFile(std::string& filename)
{
    std::transform(filename.begin(), filename.end(), filename.begin(), [](unsigned char c) { return std::tolower(c); });
    return EndsWith(filename, ".bin");
}

/*
Create a hash of the (source file) contents of a build directory, used for
    creating a cache key of an entire directory along with the build commands.

    Args:
        cmds (List[str]): Build commands to be incorporated in hash key computation
        build_dir (str): Path to build directory ( not part of hash )
        filter_func (Callable[[str], bool], optional): Filter function which determines whether a given file is
considered a source file or not. Defaults to is_source(path). debug (bool, optional): Whether to write a
'cache_key.log' file into the build directory, so that cache misses can be debugged more easily. Defaults to False.
content_replacer (Callable[[Path], Optional[bytes]], optional): Content replacer is an optional function that may
replace content of a file for hashing purposes. If None, or if this function returns None, then no content
replacement is done   on the file. Returns: str: SHA256 Hash of the build directory contents in the form of a
hexdigest string.
*/
inline const std::string CreateDirHash(const std::vector<std::string>&                       cmds,
                                       const std::filesystem::path&                          build_dir,
                                       const std::function<bool(std::string&)>&              filter_func = IsSource,
                                       bool                                                  debug       = false,
                                       const std::function<std::string(const std::string&)>& content_replacer = nullptr)
{
    if (!std::filesystem::is_directory(build_dir)) {
        return "empty_dir";
    }

    std::ofstream hash_log;

    if (debug) {
        std::filesystem::path log_file = build_dir / "cache_key.log";
        hash_log                       = std::ofstream(log_file);
        if (hash_log.is_open()) {
            hash_log << Sprintf("Building dir hash of {}\n", build_dir.string());
        }
        else {
            ATER_THROW(Unavailable("Failed to write hash log file: {}", log_file.string()));
        }
    }

    std::set<std::filesystem::path> files;
    for (const auto& path : std::filesystem::recursive_directory_iterator(build_dir)) {
        files.emplace(std::filesystem::relative(path, build_dir));
    }

    std::string cmd_final_hash_string;
    for (auto cmd : cmds) {
        // Make sure we can cache regardless of the build directory location.
        Replace(cmd, build_dir.string(), "${BUILD_DIR}");
        // std::string cmd_replace     = std::regex_replace(cmd, std::regex(build_dir), );
        auto cmd_hash_string = SHA256ToHexString(cmd);
        if (debug) {
            hash_log << Sprintf("COMMAND: {} -> {}\n", cmd, cmd_hash_string);
        }
        cmd_final_hash_string += cmd_hash_string;
    }

    std::string file_path_hash_string;
    std::string replaced_content_hash_string;
    std::string full_path_hash_string;
    std::string file_total_hash_string;

    for (auto& file_path : files) {
        std::string file = file_path.string();
        if (!filter_func(file)) {
            continue;
        }
        file_path_hash_string           = SHA256ToHexString(file);
        std::filesystem::path full_path = build_dir / file;
        if (content_replacer != nullptr) {
            std::string replaced_content = content_replacer(full_path);
            if (replaced_content.length() > 0) {
                replaced_content_hash_string = SHA256ToHexString(replaced_content);
            }
            else {
                // https://stackoverflow.com/questions/20911584/how-to-read-a-file-in-multiple-chunks-until-eof-c
                std::ifstream full_path_file(full_path, std::ios::in | std::ios::binary);
                // read file in chunks of 32kb in order to support large files ( constants.obj )
                const int         buf_size = 1024 * 32;
                std::vector<char> buffer(buf_size, 0);
                while (!full_path_file.eof()) {
                    full_path_file.read(buffer.data(), buffer.size());
                    std::streamsize data_size = full_path_file.gcount();
                    if (data_size > 0) {
                        std::string buffer_data(buffer.begin(), buffer.end() + data_size);
                        full_path_hash_string = SHA256ToHexString(buffer_data);
                    }
                }
            }
        }
        std::string file_total_hash_string =
            file_path_hash_string + replaced_content_hash_string + full_path_hash_string;
        if (debug) {
            hash_log << Sprintf("FILE: {} -> {}\n", file, file_total_hash_string);
        }
    }
    std::string return_hash_string = cmd_final_hash_string + file_total_hash_string;
    if (debug) {
        hash_log << Sprintf("Final hash of {} is {}\n", build_dir.string(), return_hash_string);
    }
    return return_hash_string;
}

inline void WriteBinHash(const std::filesystem::path&             build_dir,
                         const std::string&                       binhash_filename = "constants.hash",
                         const std::function<bool(std::string&)>& filter_func      = IsBinFile)
{
    const std::string bin_hash = CreateDirHash({binhash_filename}, build_dir, filter_func);

    std::filesystem::path bin_hash_file_path = build_dir / binhash_filename;
    std::ofstream         bin_hash_file(bin_hash_file_path);
    if (bin_hash_file.is_open()) {
        bin_hash_file << bin_hash;
        bin_hash_file.close();
    }
    else {
        ATER_THROW(Unavailable("Failed to write bin hash file: {}", bin_hash_file_path.string()));
    }
}

inline bool ShouldSkipBuildCache()
{
    if (FLAGS_ATER_FORCE_PROFILER_CACHE) {
        return false;
    }

    int skip_percentage = FLAGS_ATER_BUILD_CACHE_SKIP_PERCENTAGE;
    if (skip_percentage == 0) {
        return false;
    }
    else {
        if (skip_percentage == 100) {
            return true;
        }
        else if (skip_percentage == 0) {
            return false;
        }
        else {
            std::random_device              rd;
            std::mt19937                    gen(rd());
            std::uniform_int_distribution<> dis(1, 100);
            int                             rand_num = dis(gen);
            if (rand_num <= skip_percentage) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace ater