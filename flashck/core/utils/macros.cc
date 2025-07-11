#include "flashck/core/utils/macros.h"

#include "flashck/core/utils/printf.h"

namespace flashck {

template<typename Result>
void CheckJinjaResult(Result result, const char* file, int line)
{
    if (!result) {
        LOG(FATAL) << "[JINJA2] Error: " << result.error() << "\nFile: " << file << ":" << line;
    }
}

#define CHECK_JINJA(expr) CheckJinjaResult((expr), __FILE__, __LINE__)

std::string TemplateLoadAndRender(const std::string& source, const jinja2::ValuesMap& params)
{
    jinja2::Template tpl;
    CHECK_JINJA(tpl.Load(source));
    auto render_result = tpl.RenderAsString(params);
    CHECK_JINJA(render_result);
    return render_result.value();
}

std::string GetHipErrorMessage(hipError_t err, const char* call)
{
    return Sprintf("HIP Error {}, Code {}: {}", call, static_cast<int>(err), hipGetErrorString(err));
}

}  // namespace flashck
