#include "flashck/core/utils/macros.h"

#include <functional>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "flashck/core/utils/printf.h"

namespace flashck {

namespace {
// Internal helper for Jinja2 error checking
template<typename Result>
void CheckJinjaResult(Result result, const char* file, int line)
{
    if (!result) [[unlikely]] {
        LOG(FATAL) << "[JINJA2] Error: " << result.error() << "\nFile: " << file << ":" << line;
    }
}
}  // anonymous namespace

std::string TemplateLoadAndRender(const std::string& source, const jinja2::ValuesMap& params)
{
    jinja2::Template tpl;
    CheckJinjaResult(tpl.Load(source), __FILE__, __LINE__);
    auto render_result = tpl.RenderAsString(params);
    CheckJinjaResult(render_result, __FILE__, __LINE__);
    return render_result.value();
}

std::string GetHipErrorMessage(hipError_t err, const char* call) noexcept
{
    try {
        return Sprintf("HIP Error in {}: Code {} ({})", call, static_cast<int>(err), hipGetErrorString(err));
    }
    catch (...) {
        // Fallback for cases where Sprintf might fail
        return std::string("HIP Error: ") + std::to_string(static_cast<int>(err));
    }
}

void ThrowHipError(hipError_t err, const char* call, const char* file, int line)
{
    const std::string error_msg = Sprintf(
        "HIP Error in {} at {}:{} - Code {}: {}", call, file, line, static_cast<int>(err), hipGetErrorString(err));
    LOG(ERROR) << error_msg;
    throw std::runtime_error(error_msg);
}

void LogHipWarning(hipError_t err, const char* call, const char* file, int line) noexcept
{
    try {
        const std::string warning_msg = Sprintf("HIP Warning in {} at {}:{} - Code {}: {}",
                                                call,
                                                file,
                                                line,
                                                static_cast<int>(err),
                                                hipGetErrorString(err));
        LOG(WARNING) << warning_msg;
    }
    catch (...) {
        // Fallback logging if formatting fails
        LOG(WARNING) << "HIP Warning: Error code " << static_cast<int>(err);
    }
}

void ThrowJinjaError(const std::string& error, const char* expr, const char* context, const char* file, int line)
{
    const std::string error_msg = Sprintf("Jinja2 Error in {} at {}:{} - Expression: {}, Context: {}, Error: {}",
                                          expr,
                                          file,
                                          line,
                                          expr,
                                          context,
                                          error.c_str());
    LOG(ERROR) << error_msg;
    throw std::runtime_error(error_msg);
}

// Performance optimization: Thread-local cache for frequently used templates
thread_local std::unordered_map<std::string, jinja2::Template> g_template_cache;

std::string TemplateLoadAndRenderCached(const std::string& source, const jinja2::ValuesMap& params)
{
    // Use hash of source as cache key
    const std::string cache_key = std::to_string(std::hash<std::string>{}(source));

    auto it = g_template_cache.find(cache_key);
    if (it == g_template_cache.end()) {
        jinja2::Template tpl;
        CheckJinjaResult(tpl.Load(source), __FILE__, __LINE__);
        g_template_cache[cache_key] = std::move(tpl);
        it                          = g_template_cache.find(cache_key);
    }

    auto render_result = it->second.RenderAsString(params);
    CheckJinjaResult(render_result, __FILE__, __LINE__);
    return render_result.value();
}

void ClearTemplateCache() noexcept
{
    try {
        g_template_cache.clear();
    }
    catch (...) {
        // Ignore errors during cleanup
    }
}

// Debug mode error checking (only active in debug builds)
#ifdef NDEBUG
#define HIP_DEBUG_CHECK(call) (call)
#else
#define HIP_DEBUG_CHECK(call) HIP_ERROR_CHECK(call)
#endif

}  // namespace flashck
