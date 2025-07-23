#pragma once

#include <memory>

#include "core/utils/common.h"

namespace flashck {

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
    Fixed  = 0,
    Shared = 1,
    Offset = 2,
};

enum class NodeType {
    Undefined = 0,
    Variable  = 1,
    Operation = 2,
};

enum class VarType {
    Undefined = 0,
    FixedVar  = 1,
    SharedVar = 2,
    OffsetVar = 3,
};

}  // namespace flashck