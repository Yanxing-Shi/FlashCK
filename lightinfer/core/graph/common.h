#pragma once

#include <memory>

#include "lightinfer/core/utils/enforce.h"
#include "lightinfer/core/utils/log.h"

namespace lightinfer {

class Node;

class Operation;

class Variable;

class Layer;

class Context;

class MemoryManager;

class Tensor;

class Shape;

class Allocator;

enum class MemoryType {
    Fixed     = 0,
    Shared    = 1,
    Offset    = 2,
    Undefined = 3,
};

inline std::string MemoryTypeToStr(MemoryType type)
{
    switch (type) {
        case MemoryType::Fixed:
            return "FixedMemory";
        case MemoryType::Shared:
            return "SharedMemory";
        case MemoryType::Offset:
            return "OffsetMemory";
        default:
            LI_THROW(Unimplemented("Invalid memory type {}", static_cast<int>(type)));
    }
}

enum class Mode {
    // Train,
    Undefined = 0,
    Inference = 1
    // TrainAndValidiation,
};

inline std::string ModeToStr(Mode mode)
{
    switch (mode) {
        case Mode::Inference:
            return "Inference";
        default:
            LI_THROW(Unimplemented("Mode {} not support!", static_cast<int>(mode)));
    }
}

enum class NodeType {
    Undefined = 0,
    Variable  = 1,
    Operation = 2,
};

inline std::string NodeTypeToString(NodeType type)
{
    switch (type) {
        case NodeType::Variable:
            return "Variable";
        case NodeType::Operation:
            return "Operation";
        default:
            LI_THROW(Unimplemented("not support NodeType {}", static_cast<int>(type)));
    }
}

enum class VarType {
    Undefined     = 0,
    FixedVar      = 1,
    SharedVar     = 2,
    OffsetVar     = 3,
    RegressiveVar = 4,
};

inline std::string VarTypeToString(VarType type)
{
    switch (type) {
        case VarType::FixedVar:
            return "FixedVar";
        case VarType::SharedVar:
            return "SharedVar";
        case VarType::OffsetVar:
            return "OffsetVar";
        case VarType::RegressiveVar:
            return "RegressiveVar";
        default:
            LI_THROW(Unimplemented("not support VarType {}", static_cast<int>(type)));
    }
}
}  // namespace lightinfer