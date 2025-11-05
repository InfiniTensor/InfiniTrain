#include "example/common/utils.h"

namespace infini_train {

float ConvertBF16ToFloat(void *ptr) {
    uint16_t *raw_data = reinterpret_cast<uint16_t *>(ptr);
    uint32_t f32_bits = static_cast<uint32_t>(raw_data[0]) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(f));
    return f;
}

// Model Reader Helper Function
std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

void ReadMatrixAllFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols) {
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
    ifs.read(reinterpret_cast<char *>(dst), bytes);
}

// Shard Reader Functions
// Read Row Shard: [row_start : row_start+row_cnt) × [0:cols]
void ReadMatrixRowShardFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols, int64_t row_start,
                             int64_t row_cnt) {
    std::streampos base = ifs.tellg();
    const size_t row_bytes = static_cast<size_t>(cols) * sizeof(float);
    ifs.seekg(base + std::streamoff(row_start * row_bytes));
    // assume row-major
    ifs.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(row_cnt * row_bytes));
    ifs.seekg(base + std::streamoff(rows * row_bytes));
}

// Read Column Shard: [0:rows) × [col_start : col_start+col_cnt)
void ReadMatrixColShardFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols, int64_t col_start,
                             int64_t col_cnt) {
    std::streampos base = ifs.tellg();
    const size_t row_bytes = static_cast<size_t>(cols) * sizeof(float);
    const size_t pick_bytes = static_cast<size_t>(col_cnt) * sizeof(float);
    // assume row-major, need loop
    for (int64_t r = 0; r < rows; ++r) {
        ifs.seekg(base + std::streamoff(r * row_bytes + col_start * sizeof(float)));
        ifs.read(reinterpret_cast<char *>(dst + r * col_cnt), static_cast<std::streamsize>(pick_bytes));
    }
    ifs.seekg(base + std::streamoff(rows * row_bytes));
}

// Read Whole Array
void ReadVectorAllFloat(std::ifstream &ifs, float *dst, int64_t len) {
    ifs.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(len * sizeof(float)));
}

// Read Array Shard: [start : start+cnt)
void ReadVectorShardFloat(std::ifstream &ifs, float *dst, int64_t len, int64_t start, int64_t cnt) {
    std::streampos base = ifs.tellg();
    ifs.seekg(base + std::streamoff(start * sizeof(float)));
    ifs.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(cnt * sizeof(float)));
    ifs.seekg(base + std::streamoff(len * sizeof(float)));
}

std::vector<int> GetPipelineParallelGroupRanks(int pp_world_size) {
    std::vector<int> ranks;
    ranks.reserve(pp_world_size);
    for (int i = 0; i < pp_world_size; ++i) { ranks.push_back(i); }
    return ranks;
}

} // namespace infini_train
