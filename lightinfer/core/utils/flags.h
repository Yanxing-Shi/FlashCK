#pragma once

#include <map>
#include <variant>

#include "gflags/gflags.h"

/*
This file is designed to define all public FLAGS.
*/

// ----------------------------DECLARE FLAGS----------------------------

#define LI_DECLARE_bool(name) DECLARE_bool(name)
#define LI_DECLARE_int32(name) DECLARE_int32(name)
#define LI_DECLARE_int64(name) DECLARE_int64(name)
#define LI_DECLARE_uint64(name) DECLARE_uint64(name)
#define LI_DECLARE_double(name) DECLARE_double(name)
#define LI_DECLARE_string(name) DECLARE_string(name)

// ----------------------------DEFINE FLAGS----------------------------
#define LI_DEFINE_bool(name, value, txt) DEFINE_bool(name, value, txt)
#define LI_DEFINE_int32(name, val, txt) DEFINE_int32(name, val, txt)
#define LI_DEFINE_uint32(name, val, txt) DEFINE_uint32(name, val, txt)
#define LI_DEFINE_int64(name, val, txt) DEFINE_int64(name, val, txt)
#define LI_DEFINE_uint64(name, val, txt) DEFINE_uint64(name, val, txt)
#define LI_DEFINE_double(name, val, txt) DEFINE_double(name, val, txt)
#define LI_DEFINE_string(name, val, txt) DEFINE_string(name, val, txt)

// --------------------------------define exported flags--------------------------------
namespace lightinfer {

struct FlagInfo {
    using ValueType = std::variant<bool, int32_t, int64_t, uint64_t, double, std::string>;

    std::string   name;
    mutable void* value_ptr;
    ValueType     default_value;
    std::string   doc;
    bool          is_writable;  // if public for users
};

using ExportedFlagInfoMap = std::map<std::string, FlagInfo>;
const ExportedFlagInfoMap& GetExportedFlagInfoMap();
ExportedFlagInfoMap*       GetMutableExportedFlagInfoMap();

#define __LI_DEFINE_EXPORTED_FLAG(__name, __is_writable, __cpp_type, __gflag_type, __default_value, __doc)             \
    LI_DEFINE_##__gflag_type(__name, __default_value, __doc);                                                          \
    struct __LIRegisterFlag_##__name {                                                                                 \
        __LIRegisterFlag_##__name()                                                                                    \
        {                                                                                                              \
            using FlagDeclaredType = typename std::remove_reference<decltype(FLAGS_##__name)>::type;                   \
            static_assert(std::is_same<FlagDeclaredType, std::string>::value                                           \
                              || std::is_arithmetic<FlagDeclaredType>::value,                                          \
                          "FLAGS should be std::string or arithmetic type");                                           \
            auto* instance     = ::lightinfer::GetMutableExportedFlagInfoMap();                                        \
            auto& info         = (*instance)[#__name];                                                                 \
            info.name          = #__name;                                                                              \
            info.value_ptr     = &(FLAGS_##__name);                                                                    \
            info.default_value = static_cast<__cpp_type>(__default_value);                                             \
            info.doc           = __doc;                                                                                \
            info.is_writable   = __is_writable;                                                                        \
        }                                                                                                              \
        const int Touch() const                                                                                        \
        {                                                                                                              \
            return 0;                                                                                                  \
        }                                                                                                              \
    };                                                                                                                 \
    static __LIRegisterFlag_##__name __LIRegisterFlagInstance_##__name;                                                \
    const int                        TouchLIFlagRegister_##__name()                                                    \
    {                                                                                                                  \
        return __LIRegisterFlagInstance_##__name.Touch();                                                              \
    }

}  // namespace lightinfer

#define LI_FORCE_LINK_FLAG(__name)                                                                                     \
    extern int TouchLIFlagRegister_##__name();                                                                         \
    static int __LI_use_flag_##__name = TouchLIFlagRegister_##__name()

#define LI_DEFINE_EXPORTED_bool(name, default_value, doc)                                                              \
    __LI_DEFINE_EXPORTED_FLAG(name, true, bool, bool, default_value, doc)

#define LI_DEFINE_EXPORTED_READONLY_bool(name, default_value, doc)                                                     \
    __LI_DEFINE_EXPORTED_FLAG(name, false, bool, bool, default_value, doc)

#define LI_DEFINE_EXPORTED_int32(name, default_value, doc)                                                             \
    __LI_DEFINE_EXPORTED_FLAG(name, true, int32_t, int32, default_value, doc)

#define LI_DEFINE_EXPORTED_int64(name, default_value, doc)                                                             \
    __LI_DEFINE_EXPORTED_FLAG(name, true, int64_t, int64, default_value, doc)

#define PHI_DEFINE_EXPORTED_uint64(name, default_value, doc)                                                           \
    __LI_DEFINE_EXPORTED_FLAG(name, true, uint64_t, uint64, default_value, doc)

#define LI_DEFINE_EXPORTED_double(name, default_value, doc)                                                            \
    __LI_DEFINE_EXPORTED_FLAG(name, true, double, double, default_value, doc)

#define LI_DEFINE_EXPORTED_string(name, default_value, doc)                                                            \
    __LI_DEFINE_EXPORTED_FLAG(name, true, std::string, string, default_value, doc)

namespace lightinfer {

bool SetFlagValue(const char* name, const char* value);

bool FindFlag(const char* name);

void InitGflags(int argc, char** argv, bool remove_flags = true);

}  // namespace lightinfer