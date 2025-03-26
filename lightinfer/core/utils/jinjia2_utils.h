#pragma once

#include "jinja2cpp/user_callable.h"
#include <jinja2cpp/reflected_value.h>
#include <jinja2cpp/template.h>

#include "lightinfer/core/utils/enforce.h"

namespace lightinfer {

template<typename T = void>
inline void CheckJinJia2(T result, const char* file, const int line)
{
    std::ostringstream ss;
    if (!result) {
        ss << "[LI][JINJIA2] " << result.error() << "File: " << file << ", "
           << "Line: " << line << " \n";
        LI_THROW(External("{}", ss.str()));
    }
}
#define CHECK_JINJIA2(val) CheckJinJia2(val, __FILE__, __LINE__)

inline std::string TemplateLoadAndRender(const std::string& source, const jinja2::ValuesMap& value_map)
{
    jinja2::Template tpl;
    CHECK_JINJIA2(tpl.Load(source));
    auto render_result = tpl.RenderAsString(value_map);
    CHECK_JINJIA2(render_result);
    return render_result.value();
}

}  // namespace lightinfer