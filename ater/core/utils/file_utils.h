#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>

#include <sqlite3.h>

namespace ater {

inline std::filesystem::file_time_type SystemToFileTime(std::chrono::system_clock::time_point sys_tp)
{
    using namespace std::literals;
    return std::filesystem::file_time_type{sys_tp.time_since_epoch() + 3'234'576h};
}

inline std::chrono::system_clock::time_point FileToSystemTime(std::filesystem::file_time_type f_tp)
{
    using namespace std::literals;
    return std::chrono::system_clock::time_point{f_tp.time_since_epoch() - 3'234'576h};
}

/*
Emulates the Linux 'touch' command by creating an empty file if it doesn't exist, or updating the modified timestamp
if it does.

    :param file_path: str: The path to the file to be created or updated.
    :return: None
    */
inline void TouchFile(const std::filesystem::path& file_path)
{
    if (!std::filesystem::exists(file_path)) {
        std::filesystem::create_directories(file_path.parent_path());
        std::ofstream file(file_path.c_str());
        file.close();
    }

    const auto sys_now = std::chrono::system_clock::now();
    std::filesystem::last_write_time(file_path, SystemToFileTime(sys_now));
}

/*
Returns the age of a file in seconds since its last modified timestamp.

    :param file_path: str: The path to the file.
    :return: float: The age of the file in seconds.
*/
inline double GetFileAge(const std::filesystem::path& file_path)
{
    if (!std::filesystem::is_regular_file(file_path)) {
        return 3600 * 24 * 1000.0;
    }

    const auto start = std::chrono::system_clock::now();
    auto       ftime = std::filesystem::last_write_time(file_path);

    // Calculate the file age in seconds
    const std::chrono::duration<double> file_age_seconds = start - FileToSystemTime(ftime);
    return file_age_seconds.count();
}

inline std::size_t GetCountLines(const std::string& filename)
{
    std::ifstream ifs(filename.c_str());
    if (!ifs) {
        throw std::runtime_error("Failed open file ");
    }

    std::string line;
    size_t      counter = 0;
    while (std::getline(ifs, line))
        counter++;
    return counter;
}

inline std::size_t GetFileSize(const std::string& filename)
{
    std::ifstream ifs(filename, std::ifstream::ate | std::ifstream::binary);
    if (!ifs) {
        throw std::runtime_error("Failed open file ");
    }

    return static_cast<std::size_t>(ifs.tellg());
}

inline std::filesystem::path CreateTemporaryDirectory(const std::string& folder_name,
                                                      unsigned long long max_tries = 1000)
{
    auto                                    tmp_dir = std::filesystem::temp_directory_path() / folder_name;
    unsigned long long                      i       = 0;
    std::random_device                      dev;
    std::mt19937                            prng(dev());
    std::uniform_int_distribution<uint64_t> rand(0);
    std::filesystem::path                   path;
    while (true) {
        std::stringstream ss;
        ss << std::hex << rand(prng);
        path = tmp_dir / ss.str();
        // true if the directory was created.
        if (std::filesystem::create_directories(path)) {
            break;
        }
        if (i == max_tries) {
            throw std::runtime_error("could not find non-existing directory");
        }
        i++;
    }
    return path;
}

inline void check_sqlite(int result, sqlite3* cache_db, const char* file, const int line)
{
    std::ostringstream ss;
    if (result != SQLITE_OK && result != SQLITE_DONE && result != SQLITE_ROW) {
        ss << "[ATER][ERROR][SQLITE] " << sqlite3_errmsg(cache_db) << ", "
           << "File: " << file << ", "
           << "Line: " << line << " \n";
        sqlite3_close_v2(cache_db);
        throw std::runtime_error(ss.str());
    }
}

#define ATER_CHECK_SQLITE(val, cache_db) check_sqlite(val, cache_db, __FILE__, __LINE__)

// Get lambda function pointer
template<typename Function>
struct FunctionTraits;

template<typename Ret, typename... Args>
struct FunctionTraits<Ret(Args...)> {
    typedef Ret (*ptr)(Args...);
};

template<typename Ret, typename... Args>
struct FunctionTraits<Ret (*const)(Args...)>: FunctionTraits<Ret(Args...)> {};

template<typename Cls, typename Ret, typename... Args>
struct FunctionTraits<Ret (Cls::*)(Args...) const>: FunctionTraits<Ret(Args...)> {};

// requires C++20
template<typename F>
inline auto LambdaToPointer(F lambda) -> typename FunctionTraits<decltype(&F::operator())>::ptr
{
    static auto lambda_copy = lambda;

    return []<typename... Args>(Args... args) { return lambda_copy(args...); };
}

// class AterFileStream {
// public:
//     explicit AterFileStream(const const std::string& filename, std::ios_base::openmode mode = std::ios_base::in):
//         ifs_(filename, mode)

// private:
//     std::ifstream ifs_;
//     std::ofstream ofs_;
// };

// }

// inline void CheckFileOpen(const std::ofstream& file)
// {
//     if (!file.is_open())
//         throw std::runtime_error("Unable to open file");
// }

// #define ATER_CHECK_FILE_OPEN(file) CheckFileOpen(file)

// inline void CheckFileCreate(bool val)
// {
//     if (val) {
//         fmt::print("CACHE: Created cache directory", fmt::arg("tmp_cache_dir", tmp_cache_dir));
//     }
//     else {
//         fmt::print("CACHE: Failed to create cache directory", fmt::arg("tmp_cache_dir", tmp_cache_dir));
//     }
// }

// #define ATER_CHECK_FILE_CREATE(val, ) CheckFileCreate(val)

// inline AterHipDataType GetModelFileType(std::string ini_file, std::string section_name)
// {
//     AterHipDataType model_file_type;
//     INIReader       reader = INIReader(ini_file);
//     if (reader.ParseError() < 0) {
//         Ater::warn("Can't load {0}. Use FP32 as default", ini_file.c_str());
//         model_file_type = AterHipDataType::FP32;
//     }
//     else {
//         std::string weight_data_type_str = std::string(reader.Get(section_name, "weight_data_type"));
//         if (weight_data_type_str.find("fp32") != std::string::npos) {
//             model_file_type = AterHipDataType::FP32;
//         }
//         else if (weight_data_type_str.find("fp16") != std::string::npos) {
//             model_file_type = AterHipDataType::FP16;
//         }
//         else if (weight_data_type_str.find("bf16") != std::string::npos) {
//             model_file_type = AterHipDataType::BF16;
//         }
//         else {
//             Ater::warn("Invalid type {0}. Use FP32 as default", weight_data_type_str.c_str());
//             model_file_type = AterHipDataType::FP32;
//         }
//     }
//     return model_file_type;
// }

}  // namespace ater