#pragma once

namespace flashck {

template<typename T>
class ProblemBase {
public:
    std::string GetType()
    {
        return static_cast<T*>(this)->GetTypeImpl();
    }

    std::string Serialize()
    {
        return static_cast<T*>(this)->SerializeImpl();
    }
};

}  // namespace flashck