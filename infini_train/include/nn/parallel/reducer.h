#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "infini_train/include/datatype.h"
#include "infini_train/include/nn/parallel/distributed_data_parallel_config.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"

namespace infini_train {
class Tensor;
class Device;
namespace autograd {
class PostAccumulateGradHook;
} // namespace autograd
namespace nn::parallel {
class Work;
} // namespace nn::parallel
} // namespace infini_train

namespace infini_train::nn::parallel {
namespace {
constexpr size_t kBytesPerMB = 1024ULL * 1024ULL;
} // namespace

// GradBucket passes bucket contents tensor to DDP communication hook.
// ref: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/comm.hpp
class GradBucket {
public:
    explicit GradBucket(const std::vector<std::shared_ptr<Tensor>> &tensors) : tensors_(tensors) {}
    const std::vector<std::shared_ptr<Tensor>> &tensors() const { return tensors_; }

private:
    std::vector<std::shared_ptr<Tensor>> tensors_;
};

// Compute bucket assignment according to the size of each tensors and bucket capacity.
// Returns the indices of tensors in the corrsponding bucket, i.e. output[bucket_i] = {tensor_j, tensor_k, ...}
// The index of tensors[idx] assigned to bucket(j and k above) is tensor_indices[idx].
// When tensor_indices is empty, the index of tensors[idx] assigned to bucket(j and k above) is idx itself.
std::vector<std::vector<size_t>> ComputeBucketAssignmentBySize(const std::vector<std::shared_ptr<Tensor>> &tensors,
                                                               const std::vector<size_t> &bucket_size_limits,
                                                               const std::vector<size_t> &tensor_indices = {});

// DDP Reducer that handles gradient bucketing in backward
// ref: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/reducer.hpp
class Reducer : public std::enable_shared_from_this<Reducer> {
public:
    /** @brief Constructor of Reducer
     *
     * @param parameters A list of parameters for this process's single model replica
     * @param bucket_indices The bucket assignment for this reducer
     * @param ddp_config DDP related options, see definition of DistributedDataParallelConfig
     */
    explicit Reducer(std::vector<std::shared_ptr<Tensor>> parameters, std::vector<std::vector<size_t>> bucket_indices,
                     const DistributedDataParallelConfig ddp_config = DistributedDataParallelConfig());

    // Attach PostAllReduceHooks to params
    void AttachHooksToParameters();

    // Prepare bucket info for next step
    void PrepareForBackward();

    // For custom DDP hook to overwrite the default AllReduce.
    // This can be used for algorithms like Gradient Compression/GossipGrad.
    // Hook is registered using `Reducer::RegisterCommHook()`.
    // TODO(zbl): Leave the placeholder for the moment
    void RegisterCommHook(std::shared_ptr<autograd::PostAccumulateGradHook> hook);

    // Return every tensor in bucket's flat buffer
    std::vector<std::vector<std::shared_ptr<Tensor>>> GetBucketTensors() const;

private:
    // A variable locator locates a particular variable in the reducer's buckets
    struct VariableLocator {
        // Index of the bucket containing the variable in the `buckets_` vector
        size_t bucket_index = 0;
        // Index of the variable in the bucket
        size_t intra_bucket_index = 0;
    };

    // Bucket used in DDP backward
    struct Bucket {
        // Gradients of the bucket flattened into a 1-dimensional tensor
        std::shared_ptr<Tensor> contents;
        DataType dtype;
        int device_rank = 0;

        // Variables whose gradients are held in this bucket
        std::vector<std::shared_ptr<Tensor>> variables;

        // Per-variable offset/length into the flattened `gradients` tensor and
        // the corresponding `GradBucket` instance for communication hooks
        // In terms of element count, not bytes
        std::vector<size_t> offsets;
        std::vector<size_t> lengths;

        // Views into the `gradients` tensor for each individual gradient
        std::vector<std::shared_ptr<Tensor>> bucket_views_in;
        // TODO(zbl): reserved for occasions where grads have different stride/layout
        std::vector<std::shared_ptr<Tensor>> bucket_views_out;

        // Number of gradients left to be computed before the bucket is ready to be reduced
        size_t pending;

        // Global indices of participating variables in the bucket
        std::vector<size_t> variable_indices;

        // If this bucket should expect a single sparse gradient
        // If `true`, then this implies that `bucket.variables.size() == 1`.
        // TODO(zbl): support logics for sparse gradient later
        bool expect_sparse_gradient = false;

        // The result of async communication op
        std::shared_ptr<Work> work = nullptr;
    };

private:
    void InitializeBuckets(const std::vector<std::vector<size_t>> &bucket_indices);

    // NOTE(zbl): all grads are assumed dense and stored continously in bucket for now
    void MarkVariableReadyDense(size_t variable_index);
    void MarkBucketReady(size_t bucket_index);
    void FinalizeBucketDense(size_t bucket_index);
    void FinalizeBackward();

    void BuildBuckets(const std::vector<std::vector<size_t>> &bucket_indices);
    void InitializeBucketViews(Bucket &bucket);
    void RebuildBuckets();

private:
    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<Tensor>> params_;
    std::vector<Bucket> buckets_;
    std::vector<VariableLocator> locators_;

    std::atomic<size_t> buckets_finished_{0};
    std::shared_ptr<autograd::PostAccumulateGradHook> comm_hook_ = nullptr;
    DistributedDataParallelConfig ddp_config_;

    // Next bucket to be reduced
    // This is to make sure that all-reduce of buckets be launched in the order we expect
    size_t next_bucket_ = 0;
    // To record the order of params getting ready on first step
    std::vector<size_t> grad_ready_order_indices_;
    // To record whether each param is ready on first step
    std::vector<uint8_t> ready_seen_this_iter_;
    // Whether to rebuild buckets on next train step
    bool need_rebuild_ = false;
    // Whether to buckets have already been rebuilt on the second step
    bool has_rebuilt_bucket_ = false;
    // Whether all buckets are ready and backward can be finalized
    bool all_buckets_ready_this_iter_ = false;
};

} // namespace infini_train::nn::parallel
