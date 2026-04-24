#pragma once

#include "infini_train/include/checkpoint.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/optimizer.h"

#include "gflags/gflags.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace infini_train {

float ConvertBF16ToFloat(void *ptr);

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs);

void ReadMatrixAllFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols);

void ReadMatrixRowShardFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols, int64_t row_start,
                             int64_t row_cnt);

void ReadMatrixColShardFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols, int64_t col_start,
                             int64_t col_cnt);

void ReadVectorAllFloat(std::ifstream &ifs, float *dst, int64_t len);

void ReadVectorShardFloat(std::ifstream &ifs, float *dst, int64_t len, int64_t start, int64_t cnt);

} // namespace infini_train
