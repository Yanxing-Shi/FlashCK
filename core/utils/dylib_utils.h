#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace flashck {

/**
 * @brief Dynamic library loader
 * Provides a clean interface for loading and managing shared libraries
 */
class dylib {
public:
    // Linux-specific constants
    static constexpr const char* LIB_PREFIX      = "lib";
    static constexpr const char* LIB_SUFFIX      = ".so";
    static constexpr bool        ADD_DECORATIONS = true;
    static constexpr bool        NO_DECORATIONS  = false;

    using native_handle_type = void*;
    using native_symbol_type = void*;

    /**
     * Exception types for error handling
     */
    class exception: public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    class load_error: public exception {
        using exception::exception;
    };

    class symbol_error: public exception {
        using exception::exception;
    };

    // Non-copyable but movable
    dylib(const dylib&)            = delete;
    dylib& operator=(const dylib&) = delete;

    dylib(dylib&& other) noexcept;
    dylib& operator=(dylib&& other) noexcept;

    /**
     * Constructors for various loading scenarios
     */
    dylib() = default;

    dylib(const char* dir_path, const char* lib_name, bool decorations = ADD_DECORATIONS);
    dylib(const std::string& dir_path, const std::string& lib_name, bool decorations = ADD_DECORATIONS);
    dylib(const std::string& dir_path, const char* lib_name, bool decorations = ADD_DECORATIONS);
    dylib(const char* dir_path, const std::string& lib_name, bool decorations = ADD_DECORATIONS);

    explicit dylib(const std::string& lib_name, bool decorations = ADD_DECORATIONS);
    explicit dylib(const char* lib_name, bool decorations = ADD_DECORATIONS);
    explicit dylib(const std::filesystem::path& lib_path);

    dylib(const std::filesystem::path& dir_path, const std::string& lib_name, bool decorations = ADD_DECORATIONS);
    dylib(const std::filesystem::path& dir_path, const char* lib_name, bool decorations = ADD_DECORATIONS);

    ~dylib();

    /**
     * Symbol resolution methods
     */
    native_symbol_type get_symbol(const char* symbol_name) const;
    native_symbol_type get_symbol(const std::string& symbol_name) const;

    /**
     * Template methods for function loading
     */
    template<typename T>
    T* get_function(const char* symbol_name) const
    {
        return reinterpret_cast<T*>(get_symbol(symbol_name));
    }

    template<typename T>
    T get_function_ptr(const char* symbol_name) const
    {
        return reinterpret_cast<T>(get_symbol(symbol_name));
    }

    template<typename T>
    T* get_function(const std::string& symbol_name) const
    {
        return get_function<T>(symbol_name.c_str());
    }

    template<typename T>
    T get_function_ptr(const std::string& symbol_name) const
    {
        return get_function_ptr<T>(symbol_name.c_str());
    }

    /**
     * Template methods for variable access
     */
    template<typename T>
    T& get_variable(const char* symbol_name) const
    {
        return *reinterpret_cast<T*>(get_symbol(symbol_name));
    }

    template<typename T>
    T& get_variable(const std::string& symbol_name) const
    {
        return get_variable<T>(symbol_name.c_str());
    }

    /**
     * Symbol existence checking
     */
    bool has_symbol(const char* symbol_name) const noexcept;
    bool has_symbol(const std::string& symbol_name) const noexcept;

    /**
     * Native handle access
     */
    native_handle_type native_handle() noexcept;

    /**
     * Library information
     */
    bool        is_loaded() const noexcept;
    std::string get_path() const;

private:
    native_handle_type m_handle{nullptr};
    std::string        m_path;

    // Private helper methods (implemented in .cc file)
    void        load_library(const std::string& full_path);
    std::string build_library_path(const std::string& dir_path, const std::string& lib_name, bool decorations);

    // Static helper methods
    static native_handle_type open_library(const char* path) noexcept;
    static native_symbol_type locate_symbol(native_handle_type lib, const char* name) noexcept;
    static void               close_library(native_handle_type lib) noexcept;
    static std::string        get_error_description() noexcept;
};

/**
 * Utility functions for library management
 */
namespace dylib_utils {
/**
 * Check if a library exists and can be loaded
 */
bool library_exists(const std::string& lib_path);

/**
 * Get the standard library name with decorations
 */
std::string get_library_name(const std::string& base_name);

/**
 * Find library in standard system paths
 */
std::string find_library(const std::string& lib_name);

/**
 * Get library dependencies
 */
std::vector<std::string> get_dependencies(const std::string& lib_path);
}  // namespace dylib_utils

}  // namespace flashck