#pragma once

#include <memory>

#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/reduce_op_type.h"

namespace infini_train::nn::parallel {

class DistributedDataParallel : public nn::Module {
public:
    class Rank {
    public:
        Rank(int process_rank, int thread_rank, int process_size, int thread_size);

        int process_rank() const;
        int thread_rank() const;
        int process_size() const;
        int thread_size() const;

        int WorldSize() const;

        bool IsDDP() const;

        bool IsMainRank() const;

    private:
        const int process_rank_ = 0;
        const int thread_rank_ = 0;
        const int process_size_ = 1;
        const int thread_size_ = 1;
    };

    DistributedDataParallel(std::shared_ptr<nn::Module> module, int device_id);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};

} // namespace infini_train::nn::parallel
