#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

class NewGELU : public infini_train::nn::Module {
public:
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class TensorParallelMLP : public infini_train::nn::Module {
public:
    static constexpr char kCFcLayerName[] = "c_fc";
    static constexpr char kGeluLayerName[] = "gelu";
    static constexpr char kCProjLayerName[] = "c_proj";
    
    TensorParallelMLP(int64_t n_embd,
                      int64_t hidden_dim,
                      const infini_train::nn::parallel::TensorParallelGroup &tp_group,
                      bool fc_bias   = true,
                      bool proj_bias = true);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    int64_t hidden_dim_ = 0;
    infini_train::nn::parallel::TensorParallelGroup tp_group_;
};
