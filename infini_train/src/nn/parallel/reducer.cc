#include "infini_train/include/nn/parallel/reducer.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <numeric>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "glog/logging.h"

#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/nn/parallel/work.h"

namespace infini_train::nn::parallel {
namespace {
void CopyGradToBucket(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &flat, size_t dst_elem_offset,
                      void *stream = nullptr) {
    CHECK(grad && flat);
    const size_t element_size_in_bytes = kDataTypeToSize.at(grad->Dtype());
    const size_t bytes = grad->NumElements() * element_size_in_bytes;
    char *dst = static_cast<char *>(flat->DataPtr()) + dst_elem_offset * element_size_in_bytes;
    const void *src = grad->DataPtr();

    const auto dev_type = grad->GetDevice()->Type();
    if (dev_type == DeviceType::kCPU) {
        std::memcpy(dst, src, bytes);
        return;
    }
#ifdef USE_CUDA
    if (dev_type == DeviceType::kCUDA) {
        auto *cuda_dev = dynamic_cast<const CudaDevice *>(flat->GetDevice());
        CHECK(cuda_dev);
        cuda_dev->SetDevice();
        auto comm_stream = stream ? reinterpret_cast<cudaStream_t>(stream) : cuda_dev->Stream();
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, comm_stream);
        return;
    }
#endif
    LOG(FATAL) << "Unsupported device type in CopyGradToBucket";
}

void CopyBucketToGrad(const std::shared_ptr<Tensor> &flat, const std::shared_ptr<Tensor> &grad, size_t src_elem_offset,
                      void *stream = nullptr) {
    CHECK(grad && flat);
    const size_t element_size_in_bytes = kDataTypeToSize.at(grad->Dtype());
    const size_t bytes = grad->NumElements() * element_size_in_bytes;
    const char *src = static_cast<const char *>(flat->DataPtr()) + src_elem_offset * element_size_in_bytes;
    void *dst = grad->DataPtr();

    const auto dev_type = grad->GetDevice()->Type();
    if (dev_type == DeviceType::kCPU) {
        std::memcpy(dst, src, bytes);
        return;
    }
#ifdef USE_CUDA
    if (dev_type == DeviceType::kCUDA) {
        auto *cuda_dev = dynamic_cast<const CudaDevice *>(flat->GetDevice());
        CHECK(cuda_dev);
        cuda_dev->SetDevice();
        auto comm_stream = stream ? reinterpret_cast<cudaStream_t>(stream) : cuda_dev->Stream();
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, comm_stream);
        return;
    }
#endif
    LOG(FATAL) << "Unsupported device type in CopyBucketToGrad";
}

std::shared_ptr<Tensor> MakeGradView(const std::shared_ptr<Tensor> &contents, size_t offset_elems,
                                     const std::vector<int64_t> &dims) {
    // Return a view of contents (same chunk of memory)
    auto view = std::make_shared<Tensor>(*contents, offset_elems * kDataTypeToSize.at(contents->Dtype()), dims);
    return view;
}
} // namespace

std::vector<std::vector<size_t>> ComputeBucketAssignmentBySize(const std::vector<std::shared_ptr<Tensor>> &tensors,
                                                               const std::vector<size_t> &bucket_size_limits,
                                                               const std::vector<size_t> &tensor_indices) {

    CHECK(!tensors.empty());
    CHECK(!bucket_size_limits.empty());
    // By default, tensors are bucketed in reverse order, closer to the order that grad goes ready in backward
    auto ReverseOrder = [](size_t n) {
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::reverse(idx.begin(), idx.end());
        return idx;
    };
    std::vector<size_t> order = tensor_indices.empty() ? ReverseOrder(tensors.size()) : tensor_indices;

    // Group tensors by device/dtype, make sure that device and dtype is the same in a single bucket
    struct Key {
        int dev;
        DataType dtype;
        bool operator==(const Key &o) const { return dev == o.dev && dtype == o.dtype; }
    };
    struct KeyHash {
        size_t operator()(const Key &k) const {
            return (std::hash<int>()(k.dev) << 1) ^ std::hash<int>()(static_cast<int>(k.dtype));
        }
    };
    auto key_of = [&](size_t i) -> Key { return Key{tensors[i]->GetDevice()->Index(), tensors[i]->Dtype()}; };

    // Maintain the current state of each bucket
    struct State {
        std::vector<size_t> current_tensors; // Indices of tensors in the bucket
        size_t current_bytes = 0;            // Total bytes used by the bucket
        size_t limit_idx = 0;                // The index of bucket_size_limits used by the bucket
    };

    std::unordered_map<Key, State, KeyHash> states;
    std::vector<Key> key_order;
    // NOTE(zbl): Assume combinations of (device, dtype) <= 8
    states.reserve(8);

    std::vector<std::vector<size_t>> buckets_all;
    buckets_all.reserve(tensors.size());

    auto advance_limit = [&](State &s) {
        // Iterate along bucket_size_limits till the last one everytime a bucket is completed
        if (s.limit_idx + 1 < bucket_size_limits.size()) {
            ++s.limit_idx;
        }
    };

    auto current_cap = [&](const State &s) -> size_t { return bucket_size_limits[s.limit_idx]; };

    auto flush_current_bucket = [&](State &s) {
        if (!s.current_tensors.empty()) {
            buckets_all.push_back(std::move(s.current_tensors));
            s.current_tensors.clear();
            s.current_bytes = 0;
            advance_limit(s);
        }
    };

    for (size_t idx_in_order : order) {
        CHECK_LT(idx_in_order, tensors.size());
        const auto &tensor = tensors[idx_in_order];
        CHECK(tensor);

        const Key k = key_of(idx_in_order);
        auto it = states.find(k);
        if (it == states.end()) {
            it = states.emplace(k, State{}).first;
            key_order.push_back(k);
        }
        auto &state = it->second;

        const size_t element_size_in_bytes = kDataTypeToSize.at(tensor->Dtype());
        const size_t bytes = tensor->NumElements() * element_size_in_bytes;
        const size_t cap = current_cap(state);

        // Assign current tensor to current bucket first
        state.current_tensors.push_back(idx_in_order);
        state.current_bytes += bytes;

        // If current bucket is out of capacity, then flush and move on to the next bucket
        if (state.current_bytes >= cap) {
            flush_current_bucket(state);
        }
    }

    // Flush the last bucket of each group manually
    for (auto &key : key_order) { flush_current_bucket(states[key]); }

    return buckets_all;
}

Reducer::Reducer(std::vector<std::shared_ptr<Tensor>> parameters, std::vector<std::vector<size_t>> bucket_indices,
                 const ReducerOptions &opts)
    : params_(std::move(parameters)), opts_(opts) {
    BuildBuckets(bucket_indices);
    ready_seen_this_iter_.assign(params_.size(), 0);
}

void Reducer::InitializeBuckets(const std::vector<std::vector<size_t>> &bucket_indices) {
    buckets_.clear();
    locators_.clear();
    next_bucket_ = 0;
    BuildBuckets(bucket_indices);
}

void Reducer::InitializeBucketViews(Bucket &bucket) {
    bucket.bucket_views_in.clear();
    bucket.bucket_views_out.clear();
    bucket.bucket_views_in.reserve(bucket.variables.size());
    bucket.bucket_views_out.reserve(bucket.variables.size());

    for (size_t i = 0; i < bucket.variables.size(); ++i) {
        const auto &v = bucket.variables[i];
        const size_t offset_elems = bucket.offsets[i];
        auto view_in = MakeGradView(bucket.contents, offset_elems, v->Dims());
        bucket.bucket_views_in.push_back(view_in);
    }
    // Set (out == in) by default when all grads are dense
    bucket.bucket_views_out = bucket.bucket_views_in;
}

void Reducer::BuildBuckets(const std::vector<std::vector<size_t>> &bucket_indices) {
    locators_.resize(params_.size());
    buckets_.clear();
    buckets_.reserve(bucket_indices.size());

    for (size_t bucket_idx = 0; bucket_idx < bucket_indices.size(); ++bucket_idx) {
        Bucket bucket;

        CHECK(!bucket_indices[bucket_idx].empty());
        const auto &first_param = params_[bucket_indices[bucket_idx][0]];
        bucket.dtype = first_param->Dtype();
        bucket.device_rank = first_param->GetDevice()->rank().thread_rank();

        size_t total_elems = 0;

        for (auto param_idx : bucket_indices[bucket_idx]) {
            const auto &param = params_.at(param_idx);
            CHECK(param);
            CHECK(param->GetDevice() == first_param->GetDevice()) << "Bucket cannot span devices";
            CHECK(param->Dtype() == first_param->Dtype()) << "Bucket cannot span dtypes";

            bucket.variables.push_back(param);
            bucket.offsets.push_back(total_elems);
            bucket.lengths.push_back(param->NumElements());
            total_elems += param->NumElements();

            locators_[param_idx] = {bucket_idx, bucket.variables.size() - 1};
        }

        // Assgin 1D (flat) contents
        auto dev = bucket.variables.front()->GetDevice();
        bucket.contents
            = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(total_elems)}, bucket.dtype, dev);
        bucket.pending = bucket.variables.size();

        bucket.variable_indices = bucket_indices[bucket_idx];
        InitializeBucketViews(bucket);
        buckets_.push_back(std::move(bucket));
    }
}

void Reducer::RebuildBuckets() {
    // NOTE(zbl): Assume mutex is on when entering this function
    // If no order is recorded then skip the rebuild
    if (grad_ready_order_indices_.empty()) {
        return;
    }

    // full_order = real ready order + missing index
    std::vector<uint8_t> seen(params_.size(), 0);
    for (auto idx : grad_ready_order_indices_) {
        if (idx < params_.size()) {
            seen[idx] = 1;
        }
    }
    std::vector<size_t> full_order = grad_ready_order_indices_;
    full_order.reserve(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
        if (!seen[i]) {
            full_order.push_back(i);
        }
    }

    std::vector<std::shared_ptr<Tensor>> tensors_in_order;
    tensors_in_order.reserve(full_order.size());
    for (auto global_idx : full_order) {
        CHECK_LT(global_idx, params_.size());
        tensors_in_order.push_back(params_[global_idx]);
    }

    const size_t first_cap_bytes = opts_.first_bucket_cap_mb * 1024ULL * 1024ULL;
    const size_t normal_cap_bytes = opts_.normal_bucket_cap_mb * 1024ULL * 1024ULL;
    std::vector<size_t> bucket_size_limits = {first_cap_bytes, normal_cap_bytes};
    auto new_bucket_indices = ComputeBucketAssignmentBySize(tensors_in_order, bucket_size_limits, full_order);

    InitializeBuckets(new_bucket_indices);
}

std::vector<std::vector<std::shared_ptr<Tensor>>> Reducer::GetBucketTensors() const {
    std::lock_guard<std::mutex> g(mutex_);
    std::vector<std::vector<std::shared_ptr<Tensor>>> out;
    out.reserve(buckets_.size());
    for (auto const &b : buckets_) { out.push_back({b.contents}); }
    return out;
}

void Reducer::RegisterCommHook(std::shared_ptr<autograd::PostAccumulateGradHook> hook) {
    std::lock_guard<std::mutex> g(mutex_);
    comm_hook_ = std::move(hook);
}

void Reducer::PrepareForBackward() {
    std::lock_guard<std::mutex> g(mutex_);
    buckets_finished_.store(0, std::memory_order_relaxed);

    if (need_rebuild_ && !has_rebuilt_bucket_) {
        RebuildBuckets();
        has_rebuilt_bucket_ = true;
        need_rebuild_ = false;
    }
    next_bucket_ = 0;
    grad_ready_order_indices_.clear();
    ready_seen_this_iter_.assign(params_.size(), 0);

    for (auto &bucket : buckets_) {
        bucket.pending = bucket.variables.size();
        if (opts_.gradient_as_bucket_view) {
            for (size_t i = 0; i < bucket.variables.size(); ++i) {
                // Tie each param.grad to slice of contents
                const auto &param = bucket.variables[i];
                auto view = bucket.bucket_views_in[i];
                auto grad = param->grad();

                // NOTE(zbl): This will affect behaviors in `infini_train::autograd::AccumulateGrad::Backward()`
                // If ZeroGrad(set_to_none=True), grad is nullptr at this point
                // If ZeroGrad(set_to_none=False), grad is set to view of bucket.contents (or modified by user)
                // Either way, we reset grad to view of bucket.contents
                // Since bucket.contents might not be zeroed, we need to overwrite it on next grad accumulation
                if (!grad || (grad.get() != view.get())) {
                    if (grad) {
                        LOG(WARNING) << "gradient_as_bucket_view is enabled, but param " << param
                                     << " has a non-view grad tensor. Automatically overwriting it with bucket view.";
                    }
                    param->set_grad(view);
                    param->MarkGradOverwriteOnNextAccum();
                }
            }
        }
    }
}

void Reducer::AttachHooksToParameters() {
    for (size_t param_idx = 0; param_idx < params_.size(); ++param_idx) {
        class BucketHook final : public autograd::PostAccumulateGradHook {
        public:
            BucketHook(std::weak_ptr<Reducer> reducer, size_t var_index)
                : reducer_(std::move(reducer)), var_index_(var_index) {}

            void operator()(const std::shared_ptr<Tensor> &) override {
                if (auto r = reducer_.lock()) {
                    r->MarkVariableReadyDense(var_index_);
                }
            }

        private:
            std::weak_ptr<Reducer> reducer_;
            size_t var_index_;
        };

        auto hook = std::make_unique<BucketHook>(weak_from_this(), param_idx);
        params_[param_idx]->RegisterPostAccumulateGradHook(std::move(hook));
    }
}

void Reducer::MarkVariableReadyDense(size_t variable_index) {
    std::lock_guard<std::mutex> g(mutex_);
    const auto loc = locators_.at(variable_index);
    auto &bucket = buckets_.at(loc.bucket_index);

    // Record real order of bucket being ready
    if (!has_rebuilt_bucket_ && variable_index < ready_seen_this_iter_.size()
        && !ready_seen_this_iter_[variable_index]) {
        grad_ready_order_indices_.push_back(variable_index);
        ready_seen_this_iter_[variable_index] = 1;
    }

    if (!opts_.gradient_as_bucket_view) {
        auto grad = bucket.variables[loc.intra_bucket_index]->grad();
        CHECK(grad && grad->Dtype() == bucket.dtype && grad->GetDevice() == bucket.contents->GetDevice());
        CopyGradToBucket(grad, bucket.contents, bucket.offsets[loc.intra_bucket_index]);
    }

    CHECK(bucket.pending > 0);
    bucket.pending -= 1;

    bool should_launch_next = (bucket.pending == 0);

    if (should_launch_next) {
        MarkBucketReady(loc.bucket_index);
    }
}

void Reducer::MarkBucketReady(size_t bucket_index) {
    // NOTE(zbl): Assume mutex is on when entering this function
    if (bucket_index > next_bucket_) {
        // Only when bucket_index == next_bucket_ will we launch all-reduce
        // bucket_index > next_bucket_ means that there are still buckets before that are not ready
        return;
    }
    // From next_bucket_, launch ready buckets(pending==0) in turn
    while (next_bucket_ < buckets_.size() && buckets_[next_bucket_].pending == 0) {
        auto &bucket = buckets_[next_bucket_];
        FinalizeBucketDense(next_bucket_);
        ++next_bucket_;
    }

    // If all buckets are ready, then try to rebuild them in real order
    if (next_bucket_ == buckets_.size() && !has_rebuilt_bucket_) {
        if (!grad_ready_order_indices_.empty()) {
            need_rebuild_ = true;
        }
    }
}

void Reducer::FinalizeBucketDense(size_t bucket_index) {
    auto &bucket = buckets_.at(bucket_index);
    auto ddp_pg = ProcessGroupFactory::Instance()->Get(GetDataParallelProcessGroupName(bucket.device_rank));

    std::shared_ptr<Work> work;
    if (comm_hook_) {
        std::vector<std::shared_ptr<Tensor>> bucket_view{bucket.contents};
        // NOTE(zbl): Custom hook should do in-place operations
        //            e.g. comm_hook_(GradBucket{bucket_view})[0];
        // FIXME(zbl): support custom hook later
        LOG(FATAL) << "Custom hook is not supported now";
    } else {
        work = ddp_pg->AllReduceAsync(bucket.contents, function::ReduceOpType::kAvg);
    }

    if (!opts_.gradient_as_bucket_view) {
        for (size_t i = 0; i < bucket.variables.size(); ++i) {
            // NOTE(zbl): For better performance, try `bucket.variables[i]->set_grad(bucket.bucket_views_in[i]);`
            //            to directly assgin bucket slice to grad instead of copying
            CopyBucketToGrad(bucket.contents, bucket.variables[i]->grad(), bucket.offsets[i]);
            // bucket.variables[i]->set_grad(bucket.bucket_views_in[i]);
        }
    }
}
} // namespace infini_train::nn::parallel
