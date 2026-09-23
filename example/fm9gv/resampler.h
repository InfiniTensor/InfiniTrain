#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace fm9gv {

class Resampler : public infini_train::nn::CloneableModule<Resampler> {
public:
    static constexpr char kType[] = "FM9GVResampler";

    Resampler(int64_t num_queries = 64, int64_t embed_dim = 2560, int64_t num_heads = 16,
              int64_t kv_dim = 1152);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;
    void LoadFromBin(const std::string &path);

private:
    int64_t num_queries_;
    int64_t embed_dim_;
    int64_t num_heads_;
    int64_t kv_dim_;
    int64_t grid_size_;
};

} // namespace fm9gv
