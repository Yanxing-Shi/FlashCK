#include "flashck/core/utils/macros.h"

#include "flashck/core/utils/printf.h"

namespace flashck {

/**
 * @brief Validates Jinja2 template operation results
 * @tparam Result Type containing error information (must implement error())
 * @param[in] result Result object to validate
 * @param[in] file Source file name (auto-captured)
 * @param[in] line Source line number (auto-captured)
 *
 * @throws Does not throw, but terminates program through glog on error
 */
template<typename Result>
void CheckJinjaResult(Result result, const char* file, int line)
{
    if (!result) {
        LOG(FATAL) << "[JINJA2] Error: " << result.error() << "\nFile: " << file << ":" << line;
    }
}

/**
 * @brief Macro wrapper for Jinja2 result validation
 * @param[in] expr Jinja2 operation expression to validate
 *
 * @par Usage:
 * @code
 * CHECK_JINJA(tpl.Load("template.html"));
 * @endcode
 *
 * @see CheckJinjaResult
 */
#define CHECK_JINJA(expr) CheckJinjaResult((expr), __FILE__, __LINE__)

/**
 * @brief Loads and renders Jinja2 template with parameters
 * @param[in] source Template source content
 * @param[in] params Template parameters as key-value pairs
 * @return Rendered template string
 *
 * @throws Does not throw, but may terminate program through CHECK_JINJA
 *
 * @par Example:
 * @code
 * jinja2::ValuesMap params{{"name", "John"}};
 * auto html = TemplateLoadAndRender("Hello {{name}}!", params);
 * @endcode
 */
std::string TemplateLoadAndRender(const std::string& source, const jinja2::ValuesMap& params)
{
    jinja2::Template tpl;
    CHECK_JINJA(tpl.Load(source));
    auto render_result = tpl.RenderAsString(params);
    CHECK_JINJA(render_result);
    return render_result.value();
}

/**
 * @brief Generates formatted HIP error message
 * @param[in] err HIP error code
 * @param[in] call API call expression
 * @return Formatted error string without location info
 */
[[nodiscard]] std::string GetHipErrorMessage(hipError_t err, const char* call)
{
    return Sprintf("HIP Error {}, Code {}: {}", call, static_cast<int>(err), hipGetErrorString(err));
}

}  // namespace flashck
