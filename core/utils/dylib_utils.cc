#include "dylib_utils.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

namespace flashck {

// ==============================================================================
// dylib class implementation
// ==============================================================================

dylib::dylib(dylib&& other) noexcept: m_handle(other.m_handle), m_path(std::move(other.m_path))
{
    other.m_handle = nullptr;
    other.m_path.clear();
}

dylib& dylib::operator=(dylib&& other) noexcept
{
    if (this != &other) {
        if (m_handle) {
            close_library(m_handle);
        }
        m_handle       = other.m_handle;
        m_path         = std::move(other.m_path);
        other.m_handle = nullptr;
        other.m_path.clear();
    }
    return *this;
}

dylib::dylib(const char* dir_path, const char* lib_name, bool decorations)
{
    if (!dir_path) {
        throw std::invalid_argument("The directory path is null");
    }
    if (!lib_name) {
        throw std::invalid_argument("The library name is null");
    }

    const std::string full_path = build_library_path(dir_path, lib_name, decorations);
    load_library(full_path);
}

dylib::dylib(const std::string& dir_path, const std::string& lib_name, bool decorations):
    dylib(dir_path.c_str(), lib_name.c_str(), decorations)
{
}

dylib::dylib(const std::string& dir_path, const char* lib_name, bool decorations):
    dylib(dir_path.c_str(), lib_name, decorations)
{
}

dylib::dylib(const char* dir_path, const std::string& lib_name, bool decorations):
    dylib(dir_path, lib_name.c_str(), decorations)
{
}

dylib::dylib(const std::string& lib_name, bool decorations): dylib("", lib_name.c_str(), decorations) {}

dylib::dylib(const char* lib_name, bool decorations): dylib("", lib_name, decorations) {}

dylib::dylib(const std::filesystem::path& lib_path): dylib("", lib_path.string().c_str(), NO_DECORATIONS) {}

dylib::dylib(const std::filesystem::path& dir_path, const std::string& lib_name, bool decorations):
    dylib(dir_path.string().c_str(), lib_name.c_str(), decorations)
{
}

dylib::dylib(const std::filesystem::path& dir_path, const char* lib_name, bool decorations):
    dylib(dir_path.string().c_str(), lib_name, decorations)
{
}

dylib::~dylib()
{
    if (m_handle) {
        close_library(m_handle);
    }
}

dylib::native_symbol_type dylib::get_symbol(const char* symbol_name) const
{
    if (!symbol_name) {
        throw std::invalid_argument("The symbol name to lookup is null");
    }
    if (!m_handle) {
        throw std::logic_error("The dynamic library handle is null. This object may have been moved from.");
    }

    auto symbol = locate_symbol(m_handle, symbol_name);
    if (symbol == nullptr) {
        throw symbol_error("Could not get symbol \"" + std::string(symbol_name) + "\"\n" + get_error_description());
    }
    return symbol;
}

dylib::native_symbol_type dylib::get_symbol(const std::string& symbol_name) const
{
    return get_symbol(symbol_name.c_str());
}

bool dylib::has_symbol(const char* symbol_name) const noexcept
{
    if (!m_handle || !symbol_name) {
        return false;
    }
    return locate_symbol(m_handle, symbol_name) != nullptr;
}

bool dylib::has_symbol(const std::string& symbol_name) const noexcept
{
    return has_symbol(symbol_name.c_str());
}

dylib::native_handle_type dylib::native_handle() noexcept
{
    return m_handle;
}

bool dylib::is_loaded() const noexcept
{
    return m_handle != nullptr;
}

std::string dylib::get_path() const
{
    return m_path;
}

// ==============================================================================
// Private helper methods
// ==============================================================================

void dylib::load_library(const std::string& full_path)
{
    m_handle = open_library(full_path.c_str());
    if (!m_handle) {
        throw load_error("Could not load library \"" + full_path + "\"\n" + get_error_description());
    }
    m_path = full_path;
}

std::string dylib::build_library_path(const std::string& dir_path, const std::string& lib_name, bool decorations)
{
    std::string final_name = lib_name;
    std::string final_path = dir_path;

    if (decorations) {
        final_name = std::string(LIB_PREFIX) + final_name + std::string(LIB_SUFFIX);
    }

    if (!final_path.empty() && final_path.back() != '/') {
        final_path += '/';
    }

    return final_path + final_name;
}

// ==============================================================================
// Static helper methods
// ==============================================================================

dylib::native_handle_type dylib::open_library(const char* path) noexcept
{
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}

dylib::native_symbol_type dylib::locate_symbol(native_handle_type lib, const char* name) noexcept
{
    return dlsym(lib, name);
}

void dylib::close_library(native_handle_type lib) noexcept
{
    dlclose(lib);
}

std::string dylib::get_error_description() noexcept
{
    const auto description = dlerror();
    return (description == nullptr) ? "No error reported by dlerror" : description;
}

// ==============================================================================
// Utility functions implementation
// ==============================================================================

namespace dylib_utils {

bool library_exists(const std::string& lib_path)
{
    std::ifstream file(lib_path);
    return file.good();
}

std::string get_library_name(const std::string& base_name)
{
    return std::string(dylib::LIB_PREFIX) + base_name + std::string(dylib::LIB_SUFFIX);
}

std::string find_library(const std::string& lib_name)
{
    // Standard library search paths
    std::vector<std::string> search_paths = {
        "/lib", "/usr/lib", "/usr/local/lib", "/lib64", "/usr/lib64", "/usr/local/lib64"};

    // Check LD_LIBRARY_PATH
    const char* ld_library_path = std::getenv("LD_LIBRARY_PATH");
    if (ld_library_path) {
        std::istringstream iss(ld_library_path);
        std::string        path;
        while (std::getline(iss, path, ':')) {
            if (!path.empty()) {
                search_paths.push_back(path);
            }
        }
    }

    // Search for the library
    const std::string decorated_name = get_library_name(lib_name);

    for (const auto& search_path : search_paths) {
        std::string full_path = search_path + "/" + decorated_name;
        if (library_exists(full_path)) {
            return full_path;
        }
    }

    return "";  // Not found
}

std::vector<std::string> get_dependencies(const std::string& lib_path)
{
    std::vector<std::string> dependencies;

    // Use ldd command to get dependencies
    std::string command = "ldd " + lib_path + " 2>/dev/null";
    FILE*       pipe    = popen(command.c_str(), "r");
    if (!pipe) {
        return dependencies;
    }

    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string line(buffer);
        // Parse ldd output format: "libname.so => /path/to/libname.so (0x...)"
        size_t arrow_pos = line.find(" => ");
        if (arrow_pos != std::string::npos) {
            size_t space_pos = line.find(' ', arrow_pos + 4);
            if (space_pos != std::string::npos) {
                std::string dep_path = line.substr(arrow_pos + 4, space_pos - arrow_pos - 4);
                if (!dep_path.empty() && dep_path != "(0x") {
                    dependencies.push_back(dep_path);
                }
            }
        }
    }

    pclose(pipe);
    return dependencies;
}

}  // namespace dylib_utils

}  // namespace flashck