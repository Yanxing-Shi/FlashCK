#pragma once

#include <string>

namespace flashck {

/**
 * @class ProblemBase
 * @brief Base template for kernel optimization problems using CRTP
 * @tparam T Derived problem class type
 *
 * Provides a unified interface for all kernel optimization problems in FlashCK.
 * Uses Curiously Recurring Template Pattern (CRTP) for zero-overhead static
 * polymorphism while maintaining consistent API across problem types.
 *
 * Derived classes must implement:
 * - GetTypeImpl(): Return problem type identifier string
 * - SerializeImpl(): Return JSON serialization of problem parameters
 *
 * Usage:
 * ```cpp
 * class MyProblem : public ProblemBase<MyProblem> {
 *     std::string GetTypeImpl() { return "MyProblem"; }
 *     std::string SerializeImpl() { return "{}"; }
 * };
 * ```
 */
template<typename T>
class ProblemBase {
public:
    /**
     * @brief Get problem type identifier
     * @return Type name string for database storage and categorization
     */
    std::string GetType()
    {
        return static_cast<T*>(this)->GetTypeImpl();
    }

    /**
     * @brief Serialize problem configuration
     * @return JSON string containing all problem parameters for caching and comparison
     */
    std::string Serialize()
    {
        return static_cast<T*>(this)->SerializeImpl();
    }

protected:
    /// Protected destructor to prevent deletion through base pointer
    ~ProblemBase() = default;
};

}  // namespace flashck