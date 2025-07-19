#include "flashck/core/utils/macros.h"

#include <sstream>
#include <stdexcept>

#include "flashck/core/utils/printf.h"

namespace flashck {

// ==============================================================================
// Internal Helper Functions
// ==============================================================================

/*!
 * @brief Internal helper for Jinja2 error checking
 * @tparam Result Jinja2 result type
 * @param result Jinja2 operation result
 * @param file Source file name
 * @param line Source line number
 * @throws std::runtime_error if result indicates failure
 */
template<typename Result>
void CheckJinjaResult(Result result, const char* file, int line)
{
    if (!result) {
        const std::string error_msg = Sprintf("Jinja2 Error at {}:{} - {}", file, line, result.error().ToString());
        LOG(ERROR) << error_msg;
        throw std::runtime_error(error_msg);
    }
}

// ==============================================================================
// Template Processing Functions
// ==============================================================================

std::string TemplateLoadAndRender(const std::string& source, const jinja2::ValuesMap& params)
{
    if (source.empty()) {
        return {};
    }

    try {
        // Load template
        jinja2::Template tpl;
        CheckJinjaResult(tpl.Load(source), __FILE__, __LINE__);

        // Render template
        auto render_result = tpl.RenderAsString(params);
        CheckJinjaResult(render_result, __FILE__, __LINE__);

        return render_result.value();
    }
    catch (const std::exception& e) {
        const std::string error_msg = Sprintf("Template processing failed: {}", e.what());
        LOG(ERROR) << error_msg;
        throw std::runtime_error(error_msg);
    }
}

jinja2::Template TemplateLoad(const std::string& source)
{
    if (source.empty()) {
        throw std::invalid_argument("Template source cannot be empty");
    }

    try {
        jinja2::Template tpl;
        CheckJinjaResult(tpl.Load(source), __FILE__, __LINE__);
        return tpl;
    }
    catch (const std::exception& e) {
        const std::string error_msg = Sprintf("Template loading failed: {}", e.what());
        LOG(ERROR) << error_msg;
        throw std::runtime_error(error_msg);
    }
}

std::string TemplateRender(jinja2::Template& template_obj, const jinja2::ValuesMap& params)
{
    try {
        auto render_result = template_obj.RenderAsString(params);
        CheckJinjaResult(render_result, __FILE__, __LINE__);
        return render_result.value();
    }
    catch (const std::exception& e) {
        const std::string error_msg = Sprintf("Template rendering failed: {}", e.what());
        LOG(ERROR) << error_msg;
        throw std::runtime_error(error_msg);
    }
}

// ==============================================================================
// HIP Error Handling Functions
// ==============================================================================

std::string GetHipErrorMessage(hipError_t err, const char* call) noexcept
{
    try {
        const char* error_str = hipGetErrorString(err);
        return Sprintf("HIP Error in {}: Code {} ({})",
                       call ? call : "unknown",
                       static_cast<int>(err),
                       error_str ? error_str : "unknown error");
    }
    catch (...) {
        // Fallback for cases where Sprintf might fail
        return std::string("HIP Error: Code ") + std::to_string(static_cast<int>(err));
    }
}

void ThrowHipError(hipError_t err, const char* call, const char* file, int line)
{
    try {
        const char*       error_str = hipGetErrorString(err);
        const std::string error_msg = Sprintf("HIP Error in {} at {}:{} - Code {}: {}",
                                              call ? call : "unknown",
                                              file ? file : "unknown",
                                              line,
                                              static_cast<int>(err),
                                              error_str ? error_str : "unknown error");

        LOG(ERROR) << error_msg;
        throw std::runtime_error(error_msg);
    }
    catch (const std::runtime_error&) {
        // Re-throw runtime errors
        throw;
    }
    catch (...) {
        // Fallback error message
        const std::string fallback_msg = "HIP Error: Code " + std::to_string(static_cast<int>(err));
        LOG(ERROR) << fallback_msg;
        throw std::runtime_error(fallback_msg);
    }
}

void LogHipWarning(hipError_t err, const char* call, const char* file, int line) noexcept
{
    try {
        const char*       error_str   = hipGetErrorString(err);
        const std::string warning_msg = Sprintf("HIP Warning in {} at {}:{} - Code {}: {}",
                                                call ? call : "unknown",
                                                file ? file : "unknown",
                                                line,
                                                static_cast<int>(err),
                                                error_str ? error_str : "unknown error");

        LOG(WARNING) << warning_msg;
    }
    catch (...) {
        // Fallback logging if formatting fails
        try {
            LOG(WARNING) << "HIP Warning: Error code " << static_cast<int>(err);
        }
        catch (...) {
            // Last resort: do nothing to avoid throwing in noexcept function
        }
    }
}

}  // namespace flashck
