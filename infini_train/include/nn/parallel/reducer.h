#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

// GradBucket passes bucket contents tensor to DDP communication hook.
// ref: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/comm.hpp
class GradBucket {
public:
    explicit GradBucket(const std::vector<std::shared_ptr<Tensor>> &tensors) : tensors_(tensors) {}
    const std::vector<std::shared_ptr<Tensor>> &getTensors() const { return tensors_; }

private:
    std::vector<std::shared_ptr<Tensor>> tensors_;
};

struct CommHookInterface {
    virtual ~CommHookInterface() = default;
    virtual std::vector<std::shared_ptr<Tensor>> runHook(const GradBucket &bucket) = 0;
};

// Compute bucket assignment according to the size of each tensors and bucket capacity.
// Returns the indices of tensors in the corrsponding bucket, i.e. output[bucket_i] = {tensor_j, tensor_k, ...}
// The index of tensors[idx] assigned to bucket(j and k above) is tensor_indices[idx].
// When tensor_indices is empty, the index of tensors[idx] assigned to bucket(j and k above) is idx itself.
std::vector<std::vector<size_t>> ComputeBucketAssignmentBySize(const std::vector<std::shared_ptr<Tensor>> &tensors,
                                                               const std::vector<size_t> &bucket_size_limits,
                                                               const std::vector<size_t> &tensor_indices = {});

struct ReducerOptions {
    // Max capacity for each bucket(in MB)
    size_t first_bucket_cap_mb = 128;
    size_t normal_bucket_cap_mb = 512;

    // When set true, map param.grad directly to the slice of bucket.flat(same address in memory) instead of memcpy
    bool gradient_as_bucket_view = true;
};

// DDP Reducer that handles gradient bucketing in backward
// ref: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/reducer.hpp
class Reducer : public std::enable_shared_from_this<Reducer> {
public:
    /** @brief Constructor of Reducer
     *
     * @param parameters A list of parameters for this process's single model replica
     * @param bucket_indices The bucket assignment for this reducer
     * @param comm_hook Communication hook(PostAccumulateGradHook)
     * @param opts Other options, see definition of ReducerOptions
     */
    explicit Reducer(std::vector<std::shared_ptr<Tensor>> parameters, std::vector<std::vector<size_t>> bucket_indices,
                     std::shared_ptr<CommHookInterface> comm_hook, const ReducerOptions &opts);
    ~Reducer();

    void InitializeBuckets(const std::vector<std::vector<size_t>> &bucket_indices);

    // Prepare bucket info for next step
    void PrepareForBackward();

    // For check use
    void FinalizeBackward();

    // Register hook (If not set, will use built-in AllReduce by default)
    void RegisterCommHook(std::shared_ptr<CommHookInterface> hook);

    // Attach PostAccumulateGradHook to params
    void AttachHooksToParameters();

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
        int device_index = 0;

        // Variables whose gradients are held in this bucket
        std::vector<std::shared_ptr<Tensor>> variables;

        // Per-variable offset/length into the flattened `gradients` tensor and
        // the corresponding `GradBucket` instance for communication hooks
        // In terms of element count, not bytes
        std::vector<size_t> offsets;
        std::vector<size_t> lengths;

        // Views into the `gradients` tensor for each individual gradient
        std::vector<std::shared_ptr<Tensor>> bucket_views_in;
        std::vector<std::shared_ptr<Tensor>> bucket_views_out;

        // Number of gradients left to be computed before the bucket is ready to be reduced
        size_t pending;

        // Global indices of participating variables in the bucket
        std::vector<size_t> variable_indices;

        // If this bucket should expect a single sparse gradient
        // If `true`, then this implies that `bucket.variables.size() == 1`.
        bool expect_sparse_gradient = false;

#ifdef USE_CUDA
        // Event to mark that AllReduce is completed
        cudaEvent_t allreduce_done = nullptr;
        // Event to mark that all tensors' grad in bucket are ready
        cudaEvent_t bucket_ready = nullptr;

        cudaStream_t compute_stream = nullptr;
        cudaStream_t comm_stream = nullptr;
#endif
    };

private:
    void MarkVariableReadyDense(size_t variable_index);
    void MarkBucketReady(size_t bucket_index);
    void FinalizeBucketDense(size_t bucket_index);

    void BuildBuckets(const std::vector<std::vector<size_t>> &bucket_indices);
    void InitializeBucketViews(Bucket &bucket);
    void RebuildBuckets();

    void PopulateBucketViewsOut(Bucket &bucket, const std::shared_ptr<Tensor> &tensor);

private:
    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<Tensor>> params_;
    std::vector<Bucket> buckets_;
    std::vector<VariableLocator> locators_;

    std::atomic<size_t> buckets_finished_{0};
    std::shared_ptr<CommHookInterface> comm_hook_;
    ReducerOptions opts_;

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
};

} // namespace infini_train::nn::parallel
