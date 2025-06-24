#pragma once

#include <map>
#include <variant>

#include "gflags/gflags.h"

/*
This file is designed to define all public FLAGS.
*/

/** @brief Declare boolean CFC flag (header) */
#define FC_DECLARE_bool(name) DECLARE_bool(name)

/** @brief Declare 32-bit int CFC flag (header) */
#define FC_DECLARE_int32(name) DECLARE_int32(name)

/** @brief Declare 64-bit int CFC flag (header) */
#define FC_DECLARE_int64(name) DECLARE_int64(name)

/** @brief Declare unsigned 64-bit int CFC flag (header) */
#define FC_DECLARE_uint64(name) DECLARE_uint64(name)

/** @brief Declare double CFC flag (header) */
#define FC_DECLARE_double(name) DECLARE_double(name)

/** @brief Declare string CFC flag (header) */
#define FC_DECLARE_string(name) DECLARE_string(name)

/** @brief Define bool flag with default (source)
    @param val Default value @param txt Help text */
#define FC_DEFINE_bool(name, value, txt) DEFINE_bool(name, value, txt)

/** @brief Define 32-bit int flag (source)
    @param val Default value @param txt Help text */
#define FC_DEFINE_int32(name, val, txt) DEFINE_int32(name, val, txt)

/** @brief Define unsigned 32-bit int flag (source)
    @param val Default value @param txt Help text */
#define FC_DEFINE_uint32(name, val, txt) DEFINE_uint32(name, val, txt)

/** @brief Define 64-bit int flag (source)
    @param val Default value @param txt Help text */
#define FC_DEFINE_int64(name, val, txt) DEFINE_int64(name, val, txt)

/** @brief Define unsigned 64-bit int flag (source)
    @param val Default value @param txt Help text */
#define FC_DEFINE_uint64(name, val, txt) DEFINE_uint64(name, val, txt)

/** @brief Define double flag (source)
    @param val Default value @param txt Help text */
#define FC_DEFINE_double(name, val, txt) DEFINE_double(name, val, txt)

/** @brief Define string flag (source)
    @param val Default value @param txt Help text */
#define FC_DEFINE_string(name, val, txt) DEFINE_string(name, val, txt)

namespace flashck {

/**
 * @brief Metadata container for command-line flag information
 *
 * Stores runtime information about registered flags including type-erased value
 * pointers and documentation. Used for reflection and dynamic flag handling.
 */
struct FlagInfo {
    /// Type-erased value storage supporting multiple flag types
    using ValueType = std::variant<bool, int32_t, int64_t, uint64_t, double, std::string>;

    std::string   name;          /**< Flag name (e.g. "verbose") */
    mutable void* value_ptr;     /**< Type-erased pointer to flag variable (mutable for const access) */
    ValueType     default_value; /**< Default value preserving original type information */
    std::string   doc;           /**< Help text describing flag purpose */
    bool          is_writable;   /**< Flag modification permission (false for internal flags) */
};

/// Registry mapping of flag names to their metadata (name -> FlagInfo)
using ExportedFlagInfoMap = std::map<std::string, FlagInfo>;

/**
 * @brief Access mutable global flag registry (singleton pattern)
 * @return Pointer to thread-local flag metadata instance
 *
 * @note 1. Initializes on first call (C++11 thread-safe init)
 *       2. Returned pointer valid for registry lifetime
 * @warning Requires external synchronization for concurrent writes
 */
ExportedFlagInfoMap* GetMutableExportedFlagInfoMap()
{
    static ExportedFlagInfoMap g_exported_flag_info_map;
    return &g_exported_flag_info_map;
}

/**
 * @brief Access read-only global flag registry
 * @return Immutable reference to singleton flag metadata
 *
 * @note Internally invokes mutable version for singleton access
 * @warning Returned references invalid if registry modified
 */
const ExportedFlagInfoMap& GetExportedFlagInfoMap()
{
    return *GetMutableExportedFlagInfoMap();
}

/**
 * @brief Dynamically sets the value of a registered command-line flag
 *
 * @param name Name of the flag to modify (case-sensitive)
 * @param value New value to set (must be parseable to flag's type)
 * @return true If the value was successfully updated
 * @return false If the flag doesn't exist or value parsing failed
 *
 * @warning This modifies global state - thread safety depends on gflags' implementation
 * @note Changes may not affect components that already read the flag value
 * @see gflags::SetCommandLineOption()
 */
bool SetFlagValue(const char* name, const char* value)
{
    return !gflags::SetCommandLineOption(name, value).empty();
}

/**
 * @brief Checks if a command-line flag exists in the registry
 *
 * @param name Name of the flag to search for (case-sensitive)
 * @return true If the flag exists (regardless of whether it was explicitly set)
 * @return false If the flag is not registered
 *
 * @note This checks registration status, not whether the flag's value is default
 * @warning Does not verify access permissions or mutability status
 * @see gflags::GetCommandLineOption()
 */
bool FindFlag(const char* name)
{
    std::string dummy;
    return gflags::GetCommandLineOption(name, &dummy);
}

/**
 * @brief Internal macro for defining and exporting CFC flags
 * @param __name Flag identifier (UPPER_CASE naming)
 * @param __is_writable User modification permission flag
 * @param __cpp_type Native C++ type for type checking
 * @param __gflag_type Underlying gflags type suffix (e.g. bool/int32)
 * @param __default_value Default value of identifier
 * @param __doc Documentation string for help output
 *
 * @note 1. Generates static registration object per flag
 *       2. Enforces type safety via static_assert
 *       3. Automatically registers to global flag registry
 * @warning Reserved for internal use - use FC_DEFINE_* macros instead
 */
#define __FC_DEFINE_EXPORTED_FLAG(__name, __is_writable, __cpp_type, __gflag_type, __default_value, __doc)             \
    FC_DEFINE_##__gflag_type(__name, __default_value, __doc);                                                          \
    struct __FCRegisterFlag_##__name {                                                                                 \
        __FCRegisterFlag_##__name()                                                                                    \
        {                                                                                                              \
            /* Type validation for supported flag types */                                                             \
            using FlagDeclaredType = typename std::remove_reference<decltype(FLAGS_##__name)>::type;                   \
            static_assert(std::is_same<FlagDeclaredType, std::string>::value                                           \
                              || std::is_arithmetic<FlagDeclaredType>::value,                                          \
                          "FLAGS should be std::string or arithmetic type");                                           \
            /* Register to global flag registry */                                                                     \
            auto* instance     = ::flashck::GetMutableExportedFlagInfoMap();                                           \
            auto& info         = (*instance)[#__name];                                                                 \
            info.name          = #__name;                                                                              \
            info.value_ptr     = &(FLAGS_##__name);                                                                    \
            info.default_value = static_cast<__cpp_type>(__default_value);                                             \
            info.doc           = __doc;                                                                                \
            info.is_writable   = __is_writable;                                                                        \
        }                                                                                                              \
        /* Force initialization through dummy method */                                                                \
        const int Touch() const                                                                                        \
        {                                                                                                              \
            return 0;                                                                                                  \
        }                                                                                                              \
    };                                                                                                                 \
    /* Static instance ensures registration during startup */                                                          \
    static __FCRegisterFlag_##__name __FCRegisterFlagInstance_##__name;                                                \
    /* Public access point for registration validation */                                                              \
    const int TouchFCFlagRegister_##__name()                                                                           \
    {                                                                                                                  \
        return __FCRegisterFlagInstance_##__name.Touch();                                                              \
    }

}  // namespace flashck

#define FC_FORCE_FCNK_FLAG(__name)                                                                                     \
    extern int TouchFCFlagRegister_##__name();                                                                         \
    static int __FC_use_flag_##__name = TouchFCFlagRegister_##__name()

#define FC_DEFINE_EXPORTED_bool(name, default_value, doc)                                                              \
    __FC_DEFINE_EXPORTED_FLAG(name, true, bool, bool, default_value, doc)

#define FC_DEFINE_EXPORTED_READONLY_bool(name, default_value, doc)                                                     \
    __FC_DEFINE_EXPORTED_FLAG(name, false, bool, bool, default_value, doc)

#define FC_DEFINE_EXPORTED_int32(name, default_value, doc)                                                             \
    __FC_DEFINE_EXPORTED_FLAG(name, true, int32_t, int32, default_value, doc)

#define FC_DEFINE_EXPORTED_int64(name, default_value, doc)                                                             \
    __FC_DEFINE_EXPORTED_FLAG(name, true, int64_t, int64, default_value, doc)

#define PHI_DEFINE_EXPORTED_uint64(name, default_value, doc)                                                           \
    __FC_DEFINE_EXPORTED_FLAG(name, true, uint64_t, uint64, default_value, doc)

#define FC_DEFINE_EXPORTED_double(name, default_value, doc)                                                            \
    __FC_DEFINE_EXPORTED_FLAG(name, true, double, double, default_value, doc)

#define FC_DEFINE_EXPORTED_string(name, default_value, doc)                                                            \
    __FC_DEFINE_EXPORTED_FLAG(name, true, std::string, string, default_value, doc)

namespace flashck {

bool SetFlagValue(const char* name, const char* value);

bool FindFlag(const char* name);

}  // namespace flashck