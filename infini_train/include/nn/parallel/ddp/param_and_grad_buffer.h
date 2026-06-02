#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "infini_train/include/datatype.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel_config.h"

namespace infini_train {
class Tensor;
namespace nn::parallel {
class ProcessGroup;
class Work;
} // namespace nn::parallel
} // namespace infini_train

namespace infini_train::nn::parallel {
class ParamAndGradBucket {
public:
    /**
     * @brief Create bucket metadata and flat-buffer views.
     *
     * @param params                  Parameters in bucket-local order.
     * @param param_data              View of this bucket in the flat parameter buffer, or nullptr if unused.
     * @param param_dtype             Parameter storage dtype.
     * @param grad_data               View of this bucket in the flat gradient buffer; nullptr for ZeRO-2.
     * @param grad_dtype              Gradient storage dtype.
     * @param offset                  Bucket start offset in the owning flat buffer.
     * @param num_elements_unpadded   Bucket element count before padding.
     * @param gradient_scaling_factor Pre-collective gradient scale factor.
     * @param bucket_id               Bucket index in the owning ParamAndGradBuffer.
     */
    ParamAndGradBucket(const std::vector<std::shared_ptr<Tensor>> &params, const std::shared_ptr<Tensor> &param_data,
                       DataType param_dtype, const std::shared_ptr<Tensor> &grad_data, DataType grad_dtype,
                       size_t offset, size_t num_elements_unpadded, float gradient_scaling_factor, size_t bucket_id);

    size_t bucket_id() const { return bucket_id_; }

    const std::vector<std::shared_ptr<Tensor>> &params() const { return params_; }

    const std::shared_ptr<Tensor> &param_data() const { return param_data_; }

    const std::shared_ptr<Tensor> &grad_data() const { return grad_data_; }

    DataType param_dtype() const { return param_dtype_; }

    DataType grad_dtype() const { return grad_dtype_; }

    size_t offset() const { return offset_; }

    size_t num_elements_unpadded() const { return num_elements_unpadded_; }

    float gradient_scaling_factor() const { return gradient_scaling_factor_; }

    bool GetTensorLocInBucket(const std::shared_ptr<Tensor> &parameter, size_t &start_in_bucket,
                              size_t &end_in_bucket) const;

    void ScaleGradients(float scaling_factor);

private:
    int64_t bucket_id_ = 0;
    std::vector<std::shared_ptr<Tensor>> params_;
    std::shared_ptr<Tensor> param_data_;
    std::shared_ptr<Tensor> grad_data_;
    DataType param_dtype_;
    DataType grad_dtype_;

    size_t offset_ = 0;
    size_t num_elements_unpadded_ = 0;
    float gradient_scaling_factor_ = 1.f;

    std::unordered_map<Tensor *, std::pair<size_t, size_t>> param_to_range_;
};

class ParamAndGradBucketGroup {
public:
    /**
     * @brief Group buckets that synchronize gradients and parameters together.
     *
     * @param buckets            Buckets owned by this group.
     * @param collective_pg      Process group for gradient and parameter collectives.
     * @param process_group_size Number of ranks in collective_pg.
     * @param ddp_config         DDP/DistributedOptimizer behavior config.
     */
    ParamAndGradBucketGroup(const std::vector<std::shared_ptr<ParamAndGradBucket>> &buckets,
                            const ProcessGroup *collective_pg, size_t process_group_size,
                            DistributedDataParallelConfig ddp_config);

    // Reset the state of this bucket group for the next training iter
    void Reset();

    // Register that the gradient of a parameter is ready, usually called in backward hook
    // When all params in a bucket group are ready, will call StartGradSync()
    void RegisterGradReady(const std::shared_ptr<Tensor> &parameter);

    // Start grad reduce
    void StartGradSync();

    // Accumulate a parameter grad into bucket storage for the ZeRO-2 pre-accumulate hook.
    void AccumulateParamGrad(const std::shared_ptr<Tensor> &parameter, const std::shared_ptr<Tensor> &grad,
                             bool overwrite, float learning_rate);

    // Wait for gradient reduce to complete
    void FinishGradSync();

    // Start parameter all-gather
    void StartParamSync(bool force_sync = false);

    // Wait for parameter all-gather to complete
    void FinishParamSync(bool skip_next_bucket_dispatch = false);

    // TODO(zbl): For PP, set the next bucket group used for parameter all-gather.
    void SetNextParamGatherBucketGroup(std::shared_ptr<ParamAndGradBucketGroup> next_group);

    const std::vector<std::shared_ptr<ParamAndGradBucket>> &buckets() const { return buckets_; }

    // ZeRO-2: Get a bucket's local grad shard buffer
    std::shared_ptr<Tensor> GetLocalGradShardBuffer(size_t bucket_idx) const;

    const DistributedDataParallelConfig &config() const { return ddp_config_; }

private:
    std::vector<std::shared_ptr<ParamAndGradBucket>> buckets_;
    const ProcessGroup *collective_pg_ = nullptr;
    size_t collective_pg_size_ = 1;
    int rank_in_collective_pg_ = -1;
    DistributedDataParallelConfig ddp_config_;

    std::unordered_set<Tensor *> params_;
    std::unordered_set<Tensor *> params_with_grad_;
    // Tensor -> (Bucket, Bucket Index)
    std::unordered_map<Tensor *, std::pair<std::shared_ptr<ParamAndGradBucket>, size_t>> param_to_bucket_;

    // TODO(zbl): Implement CoalescedWork for aggregate works
    //            According to Megatron-LM's _coalescing_manager
    std::vector<std::shared_ptr<Work>> grad_reduce_work_list_;
    std::vector<size_t> grad_reduce_bucket_indices_;
    std::vector<std::shared_ptr<Work>> param_gather_work_list_;

    // ZeRO-2: persistent grad shard buffers and temporary full grad buffers
    std::vector<std::shared_ptr<Tensor>> grad_shard_buffer_list_;
    std::vector<std::shared_ptr<Tensor>> temp_full_grad_buffer_list_;

    std::shared_ptr<ParamAndGradBucketGroup> next_param_gather_bucket_group_ = nullptr;

    std::vector<std::vector<std::shared_ptr<Tensor>>> param_buffer_shard_list_;
    std::vector<std::vector<std::shared_ptr<Tensor>>> grad_buffer_shard_list_;

    bool is_last_microbatch_ = true;

    bool grad_reduce_dispatched_ = false;
    bool param_gather_dispatched_ = false;
};

class ParamAndGradBuffer {
public:
    /**
     * @brief Own flat buffers and bucket metadata for one dtype group.
     *
     * @param params      Parameters with the same parameter/gradient dtype pair.
     * @param param_dtype Flat parameter-buffer dtype.
     * @param grad_dtype  Gradient storage dtype.
     * @param ddp_pg      Data-parallel process group used by derived bucket groups.
     * @param ddp_config  DDP/DistributedOptimizer bucketing and padding config.
     */
    ParamAndGradBuffer(const std::vector<std::shared_ptr<Tensor>> &params, DataType &param_dtype, DataType &grad_dtype,
                       const ProcessGroup *ddp_pg, DistributedDataParallelConfig ddp_config);

    DistributedDataParallelConfig ddp_config() const { return ddp_config_; }

    std::shared_ptr<Tensor> param_buffer() const { return param_buffer_; }

    std::shared_ptr<Tensor> grad_buffer() const { return grad_buffer_; }

    const ProcessGroup *ddp_pg() const { return ddp_pg_; }

    size_t ddp_world_size() const { return ddp_world_size_; }

    std::vector<std::shared_ptr<ParamAndGradBucket>> buckets() const { return buckets_; }

    void ScaleGradients(float scaling_factor);

    void Reset(bool need_rebind = true);

    void RebindGradViews();

private:
    void BuildBuckets(DataType param_dtype, DataType grad_dtype);

private:
    DistributedDataParallelConfig ddp_config_;
    std::vector<std::shared_ptr<Tensor>> params_;
    std::vector<std::shared_ptr<Tensor>> grads_;
    std::shared_ptr<Tensor> param_buffer_;
    std::shared_ptr<Tensor> grad_buffer_;

    size_t numel_ = 0;
    size_t numel_unpadded_ = 0;

    const ProcessGroup *ddp_pg_ = nullptr;
    size_t ddp_world_size_ = 1;
    std::vector<std::shared_ptr<ParamAndGradBucket>> buckets_;

    bool need_rebind_grad_views_ = true;

    std::vector<std::pair<size_t, size_t>> bucket_indices_;
    // Param to (start, end, bucket_id)
    std::unordered_map<Tensor *, std::tuple<size_t, size_t, size_t>> param_index_map_;
    // Param to bucket
    std::unordered_map<Tensor *, std::shared_ptr<ParamAndGradBucket>> param_bucket_map_;
};

std::vector<std::shared_ptr<ParamAndGradBucketGroup>>
PartitionBuckets(const std::vector<std::shared_ptr<ParamAndGradBuffer>> &buffers, bool force_single_bucket_group);

} // namespace infini_train::nn::parallel
