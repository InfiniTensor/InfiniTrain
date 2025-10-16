#include "utils.h"

#include <cstdint>
#include <cstring>

#include "infini_train/include/nn/parallel/global.h"

namespace infini_train {

float ConvertBF16ToFloat(void *ptr) {
    uint16_t *raw_data = reinterpret_cast<uint16_t *>(ptr);
    uint32_t f32_bits = static_cast<uint32_t>(raw_data[0]) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(f));
    return f;
}

std::vector<int> GetDataParallelGroupRanks(int rank) {
    std::vector<int> ranks;

    int world_size = nn::parallel::global::GetWorldSize();
    int tp_size = nn::parallel::global::GetTensorParallelSize();
    int dp_size = nn::parallel::global::GetDataParallelSize();

    ranks.reserve(dp_size);
    int dp_group_id = rank % tp_size;

    for (int r = 0; r < world_size; ++r) {
        if (r % tp_size == dp_group_id) {
            ranks.push_back(r);
        }
    }

    return ranks;
}

std::vector<int> GetTensorParallelGroupRanks(int rank) {
    std::vector<int> ranks;

    int world_size = nn::parallel::global::GetWorldSize();
    int tp_size = nn::parallel::global::GetTensorParallelSize();

    ranks.reserve(tp_size);
    int tp_group_id = rank / tp_size;

    for (int r = 0; r < world_size; ++r) {
        if (r / tp_size == tp_group_id) {
            ranks.push_back(r);
        }
    }

    return ranks;
}

} // namespace infini_train
