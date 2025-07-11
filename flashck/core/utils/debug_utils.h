#pragma once

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace flashck {

template<typename T>
void ResultChecker(const T* tensor, int64_t elem_cnt, const std::string& tensor_name, hipStream_t stream = 0);

template<typename T>
void PrintToFile(const T* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode);

template<typename T>
void CheckMaxVal(const T* result, const int size, hipStream_t stream = nullptr);

}  // namespace flashck