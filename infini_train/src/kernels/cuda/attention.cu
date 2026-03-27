#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"
#include <cuda_runtime.h>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

constexpr int kFusedBlockSizeSmall = 128;
constexpr int kFusedBlockSizeLarge = 256;
constexpr int kFusedKvTileSizeSmall = 8;
constexpr int kFusedKvTileSizeMedium = 16;
constexpr int kFusedKvTileSizeLarge = 32;

struct FusedKernelConfig {
    int block_size;
    int kv_tile_size;
};

struct FusedKernelConfigKey {
    int64_t head_dim;
    int64_t value_dim;
    int64_t kv_len;
    int64_t query_heads;
    int64_t key_heads;
    bool is_causal;
    bool use_dropout;

    bool operator==(const FusedKernelConfigKey &other) const {
        return head_dim == other.head_dim && value_dim == other.value_dim && kv_len == other.kv_len
            && query_heads == other.query_heads && key_heads == other.key_heads && is_causal == other.is_causal
            && use_dropout == other.use_dropout;
    }
};

struct FusedKernelConfigKeyHash {
    size_t operator()(const FusedKernelConfigKey &key) const {
        const size_t h1 = std::hash<int64_t>{}(key.head_dim);
        const size_t h2 = std::hash<int64_t>{}(key.value_dim);
        const size_t h3 = std::hash<int64_t>{}(key.kv_len);
        const size_t h4 = std::hash<int64_t>{}(key.query_heads);
        const size_t h5 = std::hash<int64_t>{}(key.key_heads);
        const size_t h6 = std::hash<int>{}(key.is_causal ? 1 : 0);
        const size_t h7 = std::hash<int>{}(key.use_dropout ? 1 : 0);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6);
    }
};

template <typename T>
void FillTensor(const std::shared_ptr<Tensor> &tensor, T value, std::string_view context_identifier) {
    DispatchFunc<INFINI_ALL_TYPES>(
        tensor->Dtype(), [=]<typename U>() { tensor->Fill<U>(static_cast<U>(value)); }, context_identifier);
}

std::shared_ptr<Tensor> OnesLikeShape(const std::vector<int64_t> &dims, DataType dtype, const Device &device,
                                      std::string_view context_identifier) {
    auto output = std::make_shared<Tensor>(dims, dtype, device);
    FillTensor(output, 1.0f, context_identifier);
    return output;
}

struct CausalMaskCacheKey {
    int64_t q_len;
    int64_t kv_len;
    Device::DeviceType device_type;
    int8_t device_index;

    bool operator==(const CausalMaskCacheKey &other) const {
        return q_len == other.q_len && kv_len == other.kv_len && device_type == other.device_type
            && device_index == other.device_index;
    }
};

struct CausalMaskCacheKeyHash {
    size_t operator()(const CausalMaskCacheKey &key) const {
        const size_t h1 = std::hash<int64_t>{}(key.q_len);
        const size_t h2 = std::hash<int64_t>{}(key.kv_len);
        const size_t h3 = std::hash<int>{}(static_cast<int>(key.device_type));
        const size_t h4 = std::hash<int>{}(static_cast<int>(key.device_index));
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};

std::shared_ptr<Tensor> GetCachedCausalMask(int64_t q_len, int64_t kv_len, const Device &device) {
    static std::mutex cache_mutex;
    static std::unordered_map<CausalMaskCacheKey, std::shared_ptr<Tensor>, CausalMaskCacheKeyHash> causal_mask_cache;
    constexpr size_t kCausalMaskCacheMaxEntries = 64;

    const CausalMaskCacheKey cache_key{q_len, kv_len, device.type(), device.index()};
    {
        std::lock_guard<std::mutex> guard(cache_mutex);
        auto it = causal_mask_cache.find(cache_key);
        if (it != causal_mask_cache.end()) {
            return it->second;
        }
    }

    auto lower_tri = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device.type(), "TrilForward"},
        OnesLikeShape({q_len, kv_len}, DataType::kFLOAT32, device, "CUDA BuildCausalMask"), 0);
    auto causal_mask = (lower_tri->View({1, 1, q_len, kv_len}) == 0.0f);

    {
        std::lock_guard<std::mutex> guard(cache_mutex);
        if (causal_mask_cache.size() >= kCausalMaskCacheMaxEntries) {
            causal_mask_cache.clear();
        }
        auto [it, inserted] = causal_mask_cache.emplace(cache_key, causal_mask);
        if (!inserted) {
            return it->second;
        }
    }
    return causal_mask;
}

std::shared_ptr<Tensor> BuildCausalMask(const std::shared_ptr<Tensor> &scores) {
    CHECK_EQ(scores->Dims().size(), 4);
    return GetCachedCausalMask(scores->Dims()[2], scores->Dims()[3], scores->GetDevice());
}

std::shared_ptr<Tensor> ApplyMasks(const std::shared_ptr<Tensor> &scores, const std::shared_ptr<Tensor> &attn_mask,
                                   bool is_causal) {
    auto masked_scores = scores;
    if (attn_mask) {
        masked_scores = masked_scores->MaskedFill(attn_mask, std::numeric_limits<float>::lowest());
    }
    if (is_causal) {
        masked_scores = masked_scores->MaskedFill(BuildCausalMask(masked_scores), std::numeric_limits<float>::lowest());
    }
    return masked_scores;
}

std::shared_ptr<Tensor> RecomputeAttentionProbabilitiesWithKeyTranspose(const std::shared_ptr<Tensor> &query,
                                                                        const std::shared_ptr<Tensor> &key_transposed,
                                                                        const std::shared_ptr<Tensor> &attn_mask,
                                                                        bool is_causal, double scale) {
    auto scores = query->Matmul(key_transposed) * static_cast<float>(scale);
    if (!attn_mask && !is_causal) {
        return nn::function::Softmax(scores, -1);
    }
    scores = ApplyMasks(scores, attn_mask, is_causal);
    return nn::function::Softmax(scores, -1);
}

std::shared_ptr<Tensor> SliceHeadRange(const std::shared_ptr<Tensor> &tensor, int64_t head_start, int64_t head_end) {
    CHECK(tensor);
    CHECK_EQ(tensor->Dims().size(), 4);
    CHECK_GE(head_start, 0);
    CHECK_LE(head_end, tensor->Dims()[1]);
    CHECK_LT(head_start, head_end);
    const auto &dims = tensor->Dims();
    return tensor->Slice({0, head_start, 0, 0}, {dims[0], head_end, dims[2], dims[3]}, {1, 1, 1, 1});
}

std::vector<std::shared_ptr<Tensor>> BuildPerHeadMaskViews(const std::shared_ptr<Tensor> &attn_mask,
                                                           int64_t query_heads) {
    // Pre-build one mask view per query head to avoid repeated Slice() calls
    // in fallback GQA loops.
    std::vector<std::shared_ptr<Tensor>> per_head_masks;
    if (!attn_mask) {
        return per_head_masks;
    }

    CHECK_EQ(attn_mask->Dims().size(), 4);
    const int64_t mask_heads = attn_mask->Dims()[1];
    if (mask_heads == 1) {
        per_head_masks.resize(static_cast<size_t>(query_heads), attn_mask);
        return per_head_masks;
    }

    CHECK_EQ(mask_heads, query_heads);
    per_head_masks.reserve(static_cast<size_t>(query_heads));
    for (int64_t h = 0; h < query_heads; ++h) { per_head_masks.push_back(SliceHeadRange(attn_mask, h, h + 1)); }
    return per_head_masks;
}

std::shared_ptr<Tensor> SliceQueryRange(const std::shared_ptr<Tensor> &tensor, int64_t q_start, int64_t q_end) {
    CHECK(tensor);
    CHECK_EQ(tensor->Dims().size(), 4);
    CHECK_GE(q_start, 0);
    CHECK_LE(q_end, tensor->Dims()[2]);
    CHECK_LT(q_start, q_end);
    const auto &dims = tensor->Dims();
    return tensor->Slice({0, 0, q_start, 0}, {dims[0], dims[1], q_end, dims[3]}, {1, 1, 1, 1});
}

std::shared_ptr<Tensor> SelectMaskForQueryRange(const std::shared_ptr<Tensor> &attn_mask, int64_t q_start,
                                                int64_t q_end) {
    if (!attn_mask) {
        return nullptr;
    }
    CHECK_EQ(attn_mask->Dims().size(), 4);
    const int64_t mask_q_len = attn_mask->Dims()[2];
    if (mask_q_len == 1) {
        return attn_mask;
    }
    CHECK_LE(q_end, mask_q_len);
    const auto &dims = attn_mask->Dims();
    return attn_mask->Slice({0, 0, q_start, 0}, {dims[0], dims[1], q_end, dims[3]}, {1, 1, 1, 1});
}

inline int64_t SelectFallbackQChunkSize(int64_t q_len, int64_t kv_len) {
    if (q_len >= 4096 || kv_len >= 4096) {
        return 64;
    }
    if (q_len >= 2048 || kv_len >= 2048) {
        return 128;
    }
    return 256;
}

std::shared_ptr<Tensor> RunChunkedFallbackForward(const std::shared_ptr<Tensor> &query,
                                                  const std::shared_ptr<Tensor> &key,
                                                  const std::shared_ptr<Tensor> &value,
                                                  const std::shared_ptr<Tensor> &attn_mask, bool is_causal,
                                                  double scale, int64_t q_chunk_size) {
    const int64_t q_len = query->Dims()[2];
    const int64_t kv_len = key->Dims()[2];
    auto key_t = key->Transpose(-2, -1);
    const auto causal_mask_full = is_causal ? GetCachedCausalMask(q_len, kv_len, query->GetDevice()) : nullptr;

    std::vector<std::shared_ptr<Tensor>> output_chunks;
    output_chunks.reserve(static_cast<size_t>((q_len + q_chunk_size - 1) / q_chunk_size));

    for (int64_t q_start = 0; q_start < q_len; q_start += q_chunk_size) {
        const int64_t q_end = std::min(q_start + q_chunk_size, q_len);
        auto q_chunk = SliceQueryRange(query, q_start, q_end);
        auto scores = q_chunk->Matmul(key_t) * static_cast<float>(scale);

        auto mask_chunk = SelectMaskForQueryRange(attn_mask, q_start, q_end);
        if (mask_chunk) {
            scores = scores->MaskedFill(mask_chunk, std::numeric_limits<float>::lowest());
        }
        if (causal_mask_full) {
            auto causal_chunk = SelectMaskForQueryRange(causal_mask_full, q_start, q_end);
            scores = scores->MaskedFill(causal_chunk, std::numeric_limits<float>::lowest());
        }

        auto probs = nn::function::Softmax(scores, -1);
        output_chunks.push_back(probs->Matmul(value));
    }
    return nn::function::Concat(output_chunks, 2);
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RunChunkedFallbackBackward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                           const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask,
                           const std::shared_ptr<Tensor> &grad_output, bool is_causal, double scale,
                           int64_t q_chunk_size) {
    const int64_t q_len = query->Dims()[2];
    const int64_t kv_len = key->Dims()[2];
    auto key_t = key->Transpose(-2, -1);
    auto value_t = value->Transpose(-2, -1);
    const auto causal_mask_full = is_causal ? GetCachedCausalMask(q_len, kv_len, query->GetDevice()) : nullptr;

    auto grad_key = std::make_shared<Tensor>(key->Dims(), DataType::kFLOAT32, key->GetDevice());
    auto grad_value = std::make_shared<Tensor>(value->Dims(), DataType::kFLOAT32, value->GetDevice());
    FillTensor(grad_key, 0.0f, "CUDA FallbackChunkedBackward");
    FillTensor(grad_value, 0.0f, "CUDA FallbackChunkedBackward");

    std::vector<std::shared_ptr<Tensor>> grad_query_chunks;
    grad_query_chunks.reserve(static_cast<size_t>((q_len + q_chunk_size - 1) / q_chunk_size));

    for (int64_t q_start = 0; q_start < q_len; q_start += q_chunk_size) {
        const int64_t q_end = std::min(q_start + q_chunk_size, q_len);
        auto q_chunk = SliceQueryRange(query, q_start, q_end);
        auto go_chunk = SliceQueryRange(grad_output, q_start, q_end);
        auto scores = q_chunk->Matmul(key_t) * static_cast<float>(scale);

        auto mask_chunk = SelectMaskForQueryRange(attn_mask, q_start, q_end);
        if (mask_chunk) {
            scores = scores->MaskedFill(mask_chunk, std::numeric_limits<float>::lowest());
        }
        if (causal_mask_full) {
            auto causal_chunk = SelectMaskForQueryRange(causal_mask_full, q_start, q_end);
            scores = scores->MaskedFill(causal_chunk, std::numeric_limits<float>::lowest());
        }

        auto probs = nn::function::Softmax(scores, -1);
        auto grad_value_chunk = probs->Transpose(-2, -1)->Matmul(go_chunk);
        grad_value = grad_value + grad_value_chunk;

        auto grad_probs = go_chunk->Matmul(value_t);
        auto sum_term = (grad_probs * probs)->Sum(-1, true);
        auto grad_scores = (grad_probs - sum_term) * probs;
        grad_probs.reset();
        sum_term.reset();
        probs.reset();

        if (mask_chunk) {
            grad_scores = grad_scores->MaskedFill(mask_chunk, 0.0f);
        }
        if (causal_mask_full) {
            auto causal_chunk = SelectMaskForQueryRange(causal_mask_full, q_start, q_end);
            grad_scores = grad_scores->MaskedFill(causal_chunk, 0.0f);
        }

        auto grad_query_chunk = grad_scores->Matmul(key) * static_cast<float>(scale);
        auto grad_key_chunk = grad_scores->Transpose(-2, -1)->Matmul(q_chunk) * static_cast<float>(scale);
        grad_key = grad_key + grad_key_chunk;
        grad_scores.reset();

        grad_query_chunks.push_back(grad_query_chunk);
    }

    auto grad_query = nn::function::Concat(grad_query_chunks, 2);
    return {grad_query, grad_key, grad_value};
}

inline cudaStream_t GetCudaStream(const Device &device) {
    auto *stream_impl = dynamic_cast<infini_train::core::cuda::CudaStream *>(
        infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device));
    CHECK(stream_impl) << "Expected CUDA stream implementation";
    return stream_impl->cuda_stream();
}

inline size_t GetMaxDynamicSharedMemoryBytes(const Device &device) {
    int dev = static_cast<int>(device.index());
    if (dev < 0) {
        CHECK_EQ(cudaGetDevice(&dev), cudaSuccess) << "cudaGetDevice failed";
    }
    int max_optin = 0;
    if (cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) == cudaSuccess
        && max_optin > 0) {
        return static_cast<size_t>(max_optin);
    }
    int max_default = 0;
    CHECK_EQ(cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, dev), cudaSuccess)
        << "cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlock) failed";
    return static_cast<size_t>(max_default);
}

inline bool IsSharedMemFeasible(int64_t head_dim, int64_t value_dim, int kv_tile_size, const Device &device) {
    const size_t required_bytes
        = static_cast<size_t>(head_dim + value_dim + static_cast<int64_t>(kv_tile_size) * (head_dim + value_dim))
        * sizeof(float);
    return required_bytes <= GetMaxDynamicSharedMemoryBytes(device);
}

inline bool IsFlashSupportedDtype(const std::shared_ptr<Tensor> &tensor) {
    if (!tensor) {
        return false;
    }
    return tensor->Dtype() == DataType::kFLOAT32 || tensor->Dtype() == DataType::kBFLOAT16;
}

inline std::shared_ptr<Tensor> ToFloatTensor(const std::shared_ptr<Tensor> &tensor) {
    if (!tensor || tensor->Dtype() == DataType::kFLOAT32) {
        return tensor;
    }
    return std::make_shared<Tensor>(tensor->To(DataType::kFLOAT32));
}

inline std::shared_ptr<Tensor> CastTensorTo(const std::shared_ptr<Tensor> &tensor, DataType dtype) {
    if (!tensor || tensor->Dtype() == dtype) {
        return tensor;
    }
    return std::make_shared<Tensor>(tensor->To(dtype));
}

inline bool CanUseFusedPath(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                            const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask) {
    if (attn_mask) {
        return false;
    }
    if (!IsFlashSupportedDtype(query) || !IsFlashSupportedDtype(key) || !IsFlashSupportedDtype(value)) {
        return false;
    }
    if (query->Dtype() != key->Dtype() || query->Dtype() != value->Dtype()) {
        return false;
    }
    if (query->Dims().size() != 4 || key->Dims().size() != 4 || value->Dims().size() != 4) {
        return false;
    }
    if (query->Dims()[0] != key->Dims()[0] || query->Dims()[0] != value->Dims()[0]) {
        return false;
    }
    if (key->Dims()[2] != value->Dims()[2]) {
        return false;
    }
    if (query->Dims()[3] != key->Dims()[3]) {
        return false;
    }
    if (query->Dims()[1] < key->Dims()[1]) {
        return false;
    }
    if (query->Dims()[1] % key->Dims()[1] != 0) {
        return false;
    }
    if (query->Dims()[3] > 256 || value->Dims()[3] > 256) {
        return false;
    }
    if (!IsSharedMemFeasible(query->Dims()[3], value->Dims()[3], kFusedKvTileSizeSmall, query->GetDevice())) {
        return false;
    }
    return true;
}

inline std::string GetFusedPathRejectionReason(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                                               const std::shared_ptr<Tensor> &value,
                                               const std::shared_ptr<Tensor> &attn_mask) {
    if (attn_mask) {
        return "attn_mask is not nullptr";
    }
    if (!IsFlashSupportedDtype(query) || !IsFlashSupportedDtype(key) || !IsFlashSupportedDtype(value)) {
        return "dtype is not FLOAT32/BFLOAT16";
    }
    if (query->Dtype() != key->Dtype() || query->Dtype() != value->Dtype()) {
        return "q/k/v dtype mismatch";
    }
    if (query->Dims().size() != 4 || key->Dims().size() != 4 || value->Dims().size() != 4) {
        return "q/k/v rank is not 4";
    }
    if (query->Dims()[0] != key->Dims()[0] || query->Dims()[0] != value->Dims()[0]) {
        return "batch mismatch among q/k/v";
    }
    if (key->Dims()[2] != value->Dims()[2]) {
        return "kv_len mismatch between k/v";
    }
    if (query->Dims()[3] != key->Dims()[3]) {
        return "q_head_dim != k_head_dim";
    }
    if (query->Dims()[1] < key->Dims()[1]) {
        return "q_heads < kv_heads";
    }
    if (query->Dims()[1] % key->Dims()[1] != 0) {
        return "q_heads % kv_heads != 0";
    }
    if (query->Dims()[3] > 256 || value->Dims()[3] > 256) {
        return "head_dim/value_dim > 256";
    }
    return "unknown";
}

inline bool CanUseFusedBackwardPath(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                                    const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask,
                                    const std::shared_ptr<Tensor> &lse, const std::shared_ptr<Tensor> &grad_output,
                                    double dropout_p) {
    if (dropout_p > 0.0) {
        return false;
    }
    if (!CanUseFusedPath(query, key, value, attn_mask)) {
        return false;
    }
    if (!lse || !grad_output) {
        return false;
    }
    if (lse->Dtype() != DataType::kFLOAT32) {
        return false;
    }
    if (grad_output->Dtype() != query->Dtype()) {
        return false;
    }
    if (lse->Dims().size() != 3) {
        return false;
    }
    if (grad_output->Dims().size() != 4) {
        return false;
    }
    if (lse->Dims()[0] != query->Dims()[0] || lse->Dims()[1] != query->Dims()[1]
        || lse->Dims()[2] != query->Dims()[2]) {
        return false;
    }
    if (grad_output->Dims()
        != std::vector<int64_t>{query->Dims()[0], query->Dims()[1], query->Dims()[2], value->Dims()[3]}) {
        return false;
    }
    return true;
}

inline FusedKernelConfig SelectFusedKernelConfig(int64_t head_dim, int64_t value_dim, int64_t kv_len,
                                                 int64_t query_heads, int64_t key_heads, bool is_causal,
                                                 bool use_dropout, const Device &device) {
    static std::mutex config_mutex;
    static std::unordered_map<FusedKernelConfigKey, FusedKernelConfig, FusedKernelConfigKeyHash> config_cache;

    const FusedKernelConfigKey cache_key{head_dim, value_dim, kv_len, query_heads, key_heads, is_causal, use_dropout};
    {
        std::lock_guard<std::mutex> guard(config_mutex);
        auto it = config_cache.find(cache_key);
        if (it != config_cache.end()) {
            return it->second;
        }
    }

    const int64_t group_size = key_heads > 0 ? (query_heads / key_heads) : 1;

    int block_size = kFusedBlockSizeLarge;
    if (head_dim > 128 || value_dim > 128) {
        block_size = kFusedBlockSizeSmall;
    } else if (head_dim <= 64 && value_dim <= 64 && kv_len <= 1024 && !use_dropout) {
        block_size = kFusedBlockSizeSmall;
    }
    if (group_size > 1 && kv_len >= 2048) {
        block_size = kFusedBlockSizeSmall;
    }

    int kv_tile_size = kFusedKvTileSizeMedium;
    if ((head_dim >= 128 || value_dim >= 128) || use_dropout) {
        kv_tile_size = kFusedKvTileSizeSmall;
    } else if (kv_len >= 4096 && head_dim <= 64 && value_dim <= 64 && !is_causal) {
        kv_tile_size = kFusedKvTileSizeLarge;
    }

    if (!IsSharedMemFeasible(head_dim, value_dim, kv_tile_size, device)) {
        if (kv_tile_size == kFusedKvTileSizeLarge
            && IsSharedMemFeasible(head_dim, value_dim, kFusedKvTileSizeMedium, device)) {
            kv_tile_size = kFusedKvTileSizeMedium;
        } else if (IsSharedMemFeasible(head_dim, value_dim, kFusedKvTileSizeSmall, device)) {
            kv_tile_size = kFusedKvTileSizeSmall;
        }
    }
    CHECK(IsSharedMemFeasible(head_dim, value_dim, kv_tile_size, device))
        << "SelectFusedKernelConfig returned infeasible shared memory config."
        << " head_dim=" << head_dim << " value_dim=" << value_dim << " kv_tile_size=" << kv_tile_size;

    const FusedKernelConfig cfg{block_size, kv_tile_size};
    {
        std::lock_guard<std::mutex> guard(config_mutex);
        config_cache.emplace(cache_key, cfg);
    }
    return cfg;
}

template <int BLOCK_SIZE, int KV_TILE_SIZE, bool USE_DROPOUT, typename T>
__global__ void FusedAttentionForwardKernel(const T *query, const T *key, const T *value, T *output, float *lse,
                                            int64_t batch, int64_t query_heads, int64_t key_heads, int64_t q_len,
                                            int64_t kv_len, int64_t head_dim, int64_t value_dim, int64_t group_size,
                                            bool is_causal, float scale, float dropout_p, uint64_t rng_seed,
                                            uint64_t rng_offset);

template <int BLOCK_SIZE, int KV_TILE_SIZE, bool USE_DROPOUT, typename T>
__global__ void FusedAttentionBackwardKernel(const T *query, const T *key, const T *value, const T *grad_output,
                                             const float *lse, float *grad_query, float *grad_key, float *grad_value,
                                             int64_t batch, int64_t query_heads, int64_t key_heads, int64_t q_len,
                                             int64_t kv_len, int64_t head_dim, int64_t value_dim, int64_t group_size,
                                             bool is_causal, float scale, float dropout_p, uint64_t rng_seed,
                                             uint64_t rng_offset);

template <int BLOCK_SIZE, int KV_TILE_SIZE, bool USE_DROPOUT, typename T>
void ConfigureFusedForwardKernelSharedMem(size_t shared_mem) {
    // Cache the configured shared-memory size per kernel template instance so
    // we only call cudaFuncSetAttribute when a larger requirement appears.
    static std::mutex configure_mutex;
    static size_t configured_bytes = 0;
    {
        std::lock_guard<std::mutex> guard(configure_mutex);
        if (shared_mem <= configured_bytes) {
            return;
        }
    }
    if (shared_mem > static_cast<size_t>(std::numeric_limits<int>::max())) {
        LOG_FIRST_N(WARNING, 1) << "Skip setting dynamic shared memory attribute for forward kernel: shared_mem "
                                << shared_mem << " exceeds int max.";
        return;
    }
    const cudaError_t err
        = cudaFuncSetAttribute(FusedAttentionForwardKernel<BLOCK_SIZE, KV_TILE_SIZE, USE_DROPOUT, T>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_mem));
    if (err != cudaSuccess) {
        LOG_FIRST_N(WARNING, 1) << "cudaFuncSetAttribute(forward) failed: " << cudaGetErrorString(err)
                                << ", shared_mem=" << shared_mem << ", block_size=" << BLOCK_SIZE
                                << ", kv_tile_size=" << KV_TILE_SIZE;
        return;
    }
    {
        std::lock_guard<std::mutex> guard(configure_mutex);
        if (shared_mem > configured_bytes) {
            configured_bytes = shared_mem;
        }
    }
}

template <int BLOCK_SIZE, int KV_TILE_SIZE, bool USE_DROPOUT, typename T>
void ConfigureFusedBackwardKernelSharedMem(size_t shared_mem) {
    // Cache the configured shared-memory size per kernel template instance so
    // we only call cudaFuncSetAttribute when a larger requirement appears.
    static std::mutex configure_mutex;
    static size_t configured_bytes = 0;
    {
        std::lock_guard<std::mutex> guard(configure_mutex);
        if (shared_mem <= configured_bytes) {
            return;
        }
    }
    if (shared_mem > static_cast<size_t>(std::numeric_limits<int>::max())) {
        LOG_FIRST_N(WARNING, 1) << "Skip setting dynamic shared memory attribute for backward kernel: shared_mem "
                                << shared_mem << " exceeds int max.";
        return;
    }
    const cudaError_t err
        = cudaFuncSetAttribute(FusedAttentionBackwardKernel<BLOCK_SIZE, KV_TILE_SIZE, USE_DROPOUT, T>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_mem));
    if (err != cudaSuccess) {
        LOG_FIRST_N(WARNING, 1) << "cudaFuncSetAttribute(backward) failed: " << cudaGetErrorString(err)
                                << ", shared_mem=" << shared_mem << ", block_size=" << BLOCK_SIZE
                                << ", kv_tile_size=" << KV_TILE_SIZE;
        return;
    }
    {
        std::lock_guard<std::mutex> guard(configure_mutex);
        if (shared_mem > configured_bytes) {
            configured_bytes = shared_mem;
        }
    }
}

template <int BLOCK_SIZE, int KV_TILE_SIZE, bool USE_DROPOUT, typename T>
void LaunchFusedForwardKernel(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                              const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &output,
                              const std::shared_ptr<Tensor> &lse, int64_t batch, int64_t query_heads, int64_t key_heads,
                              int64_t q_len, int64_t kv_len, int64_t head_dim, int64_t value_dim, int64_t group_size,
                              bool is_causal, float scale, float dropout_p, uint64_t rng_seed, uint64_t rng_offset,
                              cudaStream_t cuda_stream) {
    const int64_t rows = batch * query_heads * q_len;
    const size_t shared_mem
        = static_cast<size_t>(head_dim + value_dim + KV_TILE_SIZE * (head_dim + value_dim)) * sizeof(float);
    ConfigureFusedForwardKernelSharedMem<BLOCK_SIZE, KV_TILE_SIZE, USE_DROPOUT, T>(shared_mem);
    FusedAttentionForwardKernel<BLOCK_SIZE, KV_TILE_SIZE, USE_DROPOUT, T>
        <<<rows, BLOCK_SIZE, shared_mem, cuda_stream>>>(
            static_cast<const T *>(query->DataPtr()), static_cast<const T *>(key->DataPtr()),
            static_cast<const T *>(value->DataPtr()), static_cast<T *>(output->DataPtr()),
            static_cast<float *>(lse->DataPtr()), batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
            group_size, is_causal, scale, dropout_p, rng_seed, rng_offset);
}

template <int BLOCK_SIZE, int KV_TILE_SIZE, bool USE_DROPOUT, typename T>
void LaunchFusedBackwardKernel(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                               const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &grad_output,
                               const std::shared_ptr<Tensor> &lse, const std::shared_ptr<Tensor> &grad_query,
                               const std::shared_ptr<Tensor> &grad_key, const std::shared_ptr<Tensor> &grad_value,
                               int64_t batch, int64_t query_heads, int64_t key_heads, int64_t q_len, int64_t kv_len,
                               int64_t head_dim, int64_t value_dim, int64_t group_size, bool is_causal, float scale,
                               float dropout_p, uint64_t rng_seed, uint64_t rng_offset, cudaStream_t cuda_stream) {
    const int64_t rows = batch * query_heads * q_len;
    const size_t shared_mem
        = static_cast<size_t>(head_dim + value_dim + KV_TILE_SIZE * (head_dim + value_dim)) * sizeof(float);
    ConfigureFusedBackwardKernelSharedMem<BLOCK_SIZE, KV_TILE_SIZE, USE_DROPOUT, T>(shared_mem);
    FusedAttentionBackwardKernel<BLOCK_SIZE, KV_TILE_SIZE, USE_DROPOUT, T>
        <<<rows, BLOCK_SIZE, shared_mem, cuda_stream>>>(
            static_cast<const T *>(query->DataPtr()), static_cast<const T *>(key->DataPtr()),
            static_cast<const T *>(value->DataPtr()), static_cast<const T *>(grad_output->DataPtr()),
            static_cast<const float *>(lse->DataPtr()), static_cast<float *>(grad_query->DataPtr()),
            static_cast<float *>(grad_key->DataPtr()), static_cast<float *>(grad_value->DataPtr()), batch, query_heads,
            key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, scale, dropout_p, rng_seed,
            rng_offset);
}

__device__ inline float DeterministicUniform(uint64_t seed, uint64_t offset, uint64_t idx) {
    uint64_t x = seed ^ (offset + idx + 0x9e3779b97f4a7c15ULL);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x *= 0x2545F4914F6CDD1DULL;
    return static_cast<float>((x >> 40) & 0xFFFFFF) / static_cast<float>(1 << 24);
}

__device__ inline float WarpReduceSum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_down_sync(0xFFFFFFFF, value, offset); }
    return value;
}

template <int BLOCK_SIZE> __device__ inline float BlockReduceSum(float value, float *shared_warp_sums) {
    static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size.");
    constexpr int kWarpSize = 32;
    constexpr int kWarpsPerBlock = BLOCK_SIZE / kWarpSize;

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;

    value = WarpReduceSum(value);
    if (lane == 0) {
        shared_warp_sums[warp_id] = value;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        block_sum = lane < kWarpsPerBlock ? shared_warp_sums[lane] : 0.0f;
        block_sum = WarpReduceSum(block_sum);
        if (lane == 0) {
            shared_warp_sums[0] = block_sum;
        }
    }
    __syncthreads();
    return shared_warp_sums[0];
}

template <typename T> __device__ inline float ConvertToFloat(T value) { return static_cast<float>(value); }

template <> __device__ inline float ConvertToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T> __device__ inline T ConvertFromFloat(float value) { return static_cast<T>(value); }

template <> __device__ inline __nv_bfloat16 ConvertFromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <int BLOCK_SIZE, int KV_TILE_SIZE, bool USE_DROPOUT, typename T>
__global__ void FusedAttentionForwardKernel(const T *query, const T *key, const T *value, T *output, float *lse,
                                            int64_t batch, int64_t query_heads, int64_t key_heads, int64_t q_len,
                                            int64_t kv_len, int64_t head_dim, int64_t value_dim, int64_t group_size,
                                            bool is_causal, float scale, float dropout_p, uint64_t rng_seed,
                                            uint64_t rng_offset) {
    const int64_t row = blockIdx.x;
    if (row >= batch * query_heads * q_len) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t q_pos = row % q_len;
    const int64_t q_head = (row / q_len) % query_heads;
    const int64_t batch_idx = row / (query_heads * q_len);
    const int64_t kv_head = q_head / group_size;
    const float keep_prob = USE_DROPOUT ? (1.0f - dropout_p) : 1.0f;

    __shared__ float s_reduce[BLOCK_SIZE / 32];
    __shared__ float s_m;
    __shared__ float s_l;
    __shared__ float s_alpha;
    __shared__ float s_beta;
    extern __shared__ float s_mem[];
    float *s_q = s_mem;
    float *s_acc = s_q + head_dim;
    float *s_k_tile = s_acc + value_dim;
    float *s_v_tile = s_k_tile + static_cast<int64_t>(KV_TILE_SIZE) * head_dim;

    const int64_t q_base = (((batch_idx * query_heads + q_head) * q_len + q_pos) * head_dim);
    for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE) { s_q[d] = ConvertToFloat(query[q_base + d]); }

    for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) { s_acc[dv] = 0.0f; }
    if (tid == 0) {
        s_m = -INFINITY;
        s_l = 0.0f;
    }
    __syncthreads();

    const int64_t kv_upper = is_causal ? (q_pos + 1) : kv_len;
    for (int64_t kv_tile_start = 0; kv_tile_start < kv_upper; kv_tile_start += KV_TILE_SIZE) {
        const int64_t kv_tile_size = min(static_cast<int64_t>(KV_TILE_SIZE), kv_upper - kv_tile_start);

        const int64_t k_tile_elems = kv_tile_size * head_dim;
        for (int64_t idx = tid; idx < k_tile_elems; idx += BLOCK_SIZE) {
            const int64_t tile_kv = idx / head_dim;
            const int64_t d = idx % head_dim;
            const int64_t kv_pos = kv_tile_start + tile_kv;
            const int64_t k_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * head_dim + d);
            s_k_tile[idx] = ConvertToFloat(key[k_idx]);
        }

        const int64_t v_tile_elems = kv_tile_size * value_dim;
        for (int64_t idx = tid; idx < v_tile_elems; idx += BLOCK_SIZE) {
            const int64_t tile_kv = idx / value_dim;
            const int64_t dv = idx % value_dim;
            const int64_t kv_pos = kv_tile_start + tile_kv;
            const int64_t v_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * value_dim + dv);
            s_v_tile[idx] = ConvertToFloat(value[v_idx]);
        }
        __syncthreads();

        for (int64_t tile_kv = 0; tile_kv < kv_tile_size; ++tile_kv) {
            const int64_t kv_pos = kv_tile_start + tile_kv;

            float partial = 0.0f;
            for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE) {
                partial = fmaf(s_q[d], s_k_tile[tile_kv * head_dim + d], partial);
            }

            const float qk = BlockReduceSum<BLOCK_SIZE>(partial, s_reduce);

            if (tid == 0) {
                const float score = qk * scale;
                const uint64_t dropout_idx = ((row * kv_len) + kv_pos);
                const bool keep = !USE_DROPOUT || (DeterministicUniform(rng_seed, rng_offset, dropout_idx) < keep_prob);

                if (!keep) {
                    s_alpha = 1.0f;
                    s_beta = 0.0f;
                } else {
                    const float m_new = fmaxf(s_m, score);
                    s_alpha = __expf(s_m - m_new);
                    s_beta = __expf(score - m_new);
                    s_l = s_l * s_alpha + s_beta;
                    s_m = m_new;
                    if (USE_DROPOUT) {
                        s_beta /= keep_prob;
                    }
                }
            }
            __syncthreads();

            for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) {
                s_acc[dv] = fmaf(s_beta, s_v_tile[tile_kv * value_dim + dv], s_acc[dv] * s_alpha);
            }
            __syncthreads();
        }
    }

    for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) {
        const int64_t out_idx = (((batch_idx * query_heads + q_head) * q_len + q_pos) * value_dim + dv);
        output[out_idx] = ConvertFromFloat<T>(s_l > 0.0f ? s_acc[dv] / s_l : 0.0f);
    }
    if (tid == 0) {
        lse[row] = s_l > 0.0f ? (__logf(s_l) + s_m) : -INFINITY;
    }
}

template <int BLOCK_SIZE, int KV_TILE_SIZE, bool USE_DROPOUT, typename T>
__global__ void FusedAttentionBackwardKernel(const T *query, const T *key, const T *value, const T *grad_output,
                                             const float *lse, float *grad_query, float *grad_key, float *grad_value,
                                             int64_t batch, int64_t query_heads, int64_t key_heads, int64_t q_len,
                                             int64_t kv_len, int64_t head_dim, int64_t value_dim, int64_t group_size,
                                             bool is_causal, float scale, float dropout_p, uint64_t rng_seed,
                                             uint64_t rng_offset) {
    const int64_t row = blockIdx.x;
    if (row >= batch * query_heads * q_len) {
        return;
    }

    const int tid = threadIdx.x;
    const int64_t q_pos = row % q_len;
    const int64_t q_head = (row / q_len) % query_heads;
    const int64_t batch_idx = row / (query_heads * q_len);
    const int64_t kv_head = q_head / group_size;
    const float keep_prob = USE_DROPOUT ? (1.0f - dropout_p) : 1.0f;
    const float lse_row = lse[row];
    const int64_t q_base = (((batch_idx * query_heads + q_head) * q_len + q_pos) * head_dim);
    const int64_t go_base = (((batch_idx * query_heads + q_head) * q_len + q_pos) * value_dim);

    __shared__ float s_reduce[BLOCK_SIZE / 32];
    __shared__ float s_sum_term;
    extern __shared__ float s_mem[];
    float *s_q = s_mem;
    float *s_go = s_q + head_dim;
    float *s_k_tile = s_go + value_dim;
    float *s_v_tile = s_k_tile + static_cast<int64_t>(KV_TILE_SIZE) * head_dim;

    for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE) { s_q[d] = ConvertToFloat(query[q_base + d]); }
    for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) { s_go[dv] = ConvertToFloat(grad_output[go_base + dv]); }
    __syncthreads();

    const int64_t kv_upper = is_causal ? (q_pos + 1) : kv_len;
    float local_sum_term = 0.0f;
    for (int64_t kv_tile_start = 0; kv_tile_start < kv_upper; kv_tile_start += KV_TILE_SIZE) {
        const int64_t kv_tile_size = min(static_cast<int64_t>(KV_TILE_SIZE), kv_upper - kv_tile_start);

        const int64_t k_tile_elems = kv_tile_size * head_dim;
        for (int64_t idx = tid; idx < k_tile_elems; idx += BLOCK_SIZE) {
            const int64_t tile_kv = idx / head_dim;
            const int64_t d = idx % head_dim;
            const int64_t kv_pos = kv_tile_start + tile_kv;
            const int64_t k_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * head_dim + d);
            s_k_tile[idx] = ConvertToFloat(key[k_idx]);
        }

        const int64_t v_tile_elems = kv_tile_size * value_dim;
        for (int64_t idx = tid; idx < v_tile_elems; idx += BLOCK_SIZE) {
            const int64_t tile_kv = idx / value_dim;
            const int64_t dv = idx % value_dim;
            const int64_t kv_pos = kv_tile_start + tile_kv;
            const int64_t v_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * value_dim + dv);
            s_v_tile[idx] = ConvertToFloat(value[v_idx]);
        }
        __syncthreads();

        for (int64_t tile_kv = 0; tile_kv < kv_tile_size; ++tile_kv) {
            const int64_t kv_pos = kv_tile_start + tile_kv;

            float qk_partial = 0.0f;
            for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE) {
                qk_partial = fmaf(s_q[d], s_k_tile[tile_kv * head_dim + d], qk_partial);
            }
            const float qk = BlockReduceSum<BLOCK_SIZE>(qk_partial, s_reduce);
            const float score = qk * scale;
            const float prob_logit = fminf(score - lse_row, 0.0f);
            const float prob = __expf(prob_logit);

            float dprob_partial = 0.0f;
            for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) {
                dprob_partial = fmaf(s_go[dv], s_v_tile[tile_kv * value_dim + dv], dprob_partial);
            }
            const float dprob_sum = BlockReduceSum<BLOCK_SIZE>(dprob_partial, s_reduce);

            if (tid == 0) {
                float dprob = dprob_sum;
                if (!isfinite(dprob)) {
                    dprob = 0.0f;
                }
                if (USE_DROPOUT) {
                    const uint64_t dropout_idx = ((row * kv_len) + kv_pos);
                    const bool keep = DeterministicUniform(rng_seed, rng_offset, dropout_idx) < keep_prob;
                    dprob = keep ? (dprob / keep_prob) : 0.0f;
                }
                const float contrib = dprob * prob;
                local_sum_term += isfinite(contrib) ? contrib : 0.0f;
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        // local_sum_term is accumulated only by lane 0 in this block.
        s_sum_term = local_sum_term;
    }
    __syncthreads();

    constexpr int kMaxHeadDim = 256;
    constexpr int kMaxDPerThread = (kMaxHeadDim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Accumulate dQ in registers and write back once to reduce global memory
    // traffic in the inner KV loop.
    float dq_accum[kMaxDPerThread];
#pragma unroll
    for (int i = 0; i < kMaxDPerThread; ++i) { dq_accum[i] = 0.0f; }

    for (int64_t kv_tile_start = 0; kv_tile_start < kv_upper; kv_tile_start += KV_TILE_SIZE) {
        const int64_t kv_tile_size = min(static_cast<int64_t>(KV_TILE_SIZE), kv_upper - kv_tile_start);

        const int64_t k_tile_elems = kv_tile_size * head_dim;
        for (int64_t idx = tid; idx < k_tile_elems; idx += BLOCK_SIZE) {
            const int64_t tile_kv = idx / head_dim;
            const int64_t d = idx % head_dim;
            const int64_t kv_pos = kv_tile_start + tile_kv;
            const int64_t k_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * head_dim + d);
            s_k_tile[idx] = ConvertToFloat(key[k_idx]);
        }

        const int64_t v_tile_elems = kv_tile_size * value_dim;
        for (int64_t idx = tid; idx < v_tile_elems; idx += BLOCK_SIZE) {
            const int64_t tile_kv = idx / value_dim;
            const int64_t dv = idx % value_dim;
            const int64_t kv_pos = kv_tile_start + tile_kv;
            const int64_t v_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * value_dim + dv);
            s_v_tile[idx] = ConvertToFloat(value[v_idx]);
        }
        __syncthreads();

        for (int64_t tile_kv = 0; tile_kv < kv_tile_size; ++tile_kv) {
            const int64_t kv_pos = kv_tile_start + tile_kv;

            float qk_partial = 0.0f;
            for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE) {
                qk_partial = fmaf(s_q[d], s_k_tile[tile_kv * head_dim + d], qk_partial);
            }
            const float qk = BlockReduceSum<BLOCK_SIZE>(qk_partial, s_reduce);
            const float score = qk * scale;
            const float prob_logit = fminf(score - lse_row, 0.0f);
            const float prob = __expf(prob_logit);

            float dprob_partial = 0.0f;
            for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) {
                dprob_partial = fmaf(s_go[dv], s_v_tile[tile_kv * value_dim + dv], dprob_partial);
            }
            float dprob = BlockReduceSum<BLOCK_SIZE>(dprob_partial, s_reduce);
            if (!isfinite(dprob)) {
                dprob = 0.0f;
            }
            float keep_scale = 1.0f;
            if (USE_DROPOUT) {
                const uint64_t dropout_idx = ((row * kv_len) + kv_pos);
                const bool keep = DeterministicUniform(rng_seed, rng_offset, dropout_idx) < keep_prob;
                keep_scale = keep ? (1.0f / keep_prob) : 0.0f;
                dprob *= keep_scale;
            }

            float ds = prob * (dprob - s_sum_term);
            if (!isfinite(ds)) {
                ds = 0.0f;
            }
            const float ds_scaled = ds * scale;
            int d_slot = 0;
            for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE, ++d_slot) {
                const int64_t k_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * head_dim + d);
                dq_accum[d_slot] += ds_scaled * s_k_tile[tile_kv * head_dim + d];
                atomicAdd(&grad_key[k_idx], ds_scaled * s_q[d]);
            }
            for (int64_t dv = tid; dv < value_dim; dv += BLOCK_SIZE) {
                const int64_t v_idx = (((batch_idx * key_heads + kv_head) * kv_len + kv_pos) * value_dim + dv);
                atomicAdd(&grad_value[v_idx], prob * keep_scale * s_go[dv]);
            }
            __syncthreads();
        }
    }

    int d_slot = 0;
    for (int64_t d = tid; d < head_dim; d += BLOCK_SIZE, ++d_slot) { grad_query[q_base + d] = dq_accum[d_slot]; }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, uint64_t, uint64_t>
RunFallbackForward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                   const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask, double dropout_p,
                   bool is_causal, double scale, bool enable_gqa) {
    // Fallback keeps the mathematical behavior of SDPA and is used as the
    // numerically stable path when fused-kernel constraints are not satisfied.
    CHECK_EQ(dropout_p, 0.0) << "Fallback path supports dropout_p == 0 only.";
    auto query_math = ToFloatTensor(query);
    auto key_math = ToFloatTensor(key);
    auto value_math = ToFloatTensor(value);
    auto attn_mask_math = ToFloatTensor(attn_mask);
    const int64_t q_chunk_size = SelectFallbackQChunkSize(query_math->Dims()[2], key_math->Dims()[2]);

    const int64_t query_heads = query_math->Dims()[1];
    const int64_t key_heads = key_math->Dims()[1];
    CHECK_EQ(key_heads, value_math->Dims()[1]);
    if (!enable_gqa) {
        CHECK_EQ(query_heads, key_heads);
    }
    CHECK_GE(query_heads, key_heads);
    CHECK_EQ(query_heads % key_heads, 0);
    const int64_t n_rep = query_heads / key_heads;
    const auto per_head_masks = BuildPerHeadMaskViews(attn_mask_math, query_heads);

    std::vector<std::shared_ptr<Tensor>> output_heads;
    output_heads.reserve(static_cast<size_t>(query_heads));

    for (int64_t kv_head = 0; kv_head < key_heads; ++kv_head) {
        const int64_t q_head_start = kv_head * n_rep;
        auto k_single = SliceHeadRange(key_math, kv_head, kv_head + 1);
        auto v_single = SliceHeadRange(value_math, kv_head, kv_head + 1);

        for (int64_t rep = 0; rep < n_rep; ++rep) {
            const int64_t q_head = q_head_start + rep;
            auto q_single = SliceHeadRange(query_math, q_head, q_head + 1);
            auto mask_single = per_head_masks.empty() ? nullptr : per_head_masks[static_cast<size_t>(q_head)];
            output_heads.push_back(
                RunChunkedFallbackForward(q_single, k_single, v_single, mask_single, is_causal, scale, q_chunk_size));
        }
    }

    auto output = CastTensorTo(nn::function::Concat(output_heads, 1), query->Dtype());
    return {output, nullptr, 0, 0};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RunFallbackBackward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                    const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask,
                    const std::shared_ptr<Tensor> &grad_output, double dropout_p, bool is_causal, double scale,
                    bool enable_gqa) {
    // Fallback backward avoids custom-kernel atomic contention and supports
    // both MHA and GQA while preserving numerical behavior.
    CHECK_EQ(dropout_p, 0.0) << "Fallback path supports dropout_p == 0 only.";

    auto query_math = ToFloatTensor(query);
    auto key_math = ToFloatTensor(key);
    auto value_math = ToFloatTensor(value);
    auto grad_output_math = ToFloatTensor(grad_output);
    auto attn_mask_math = ToFloatTensor(attn_mask);
    const int64_t q_chunk_size = SelectFallbackQChunkSize(query_math->Dims()[2], key_math->Dims()[2]);

    const int64_t query_heads = query_math->Dims()[1];
    const int64_t key_heads = key_math->Dims()[1];
    CHECK_EQ(key_heads, value_math->Dims()[1]);
    if (!enable_gqa) {
        CHECK_EQ(query_heads, key_heads);
    }
    CHECK_GE(query_heads, key_heads);
    CHECK_EQ(query_heads % key_heads, 0);
    const int64_t n_rep = query_heads / key_heads;
    const auto per_head_masks = BuildPerHeadMaskViews(attn_mask_math, query_heads);

    std::vector<std::shared_ptr<Tensor>> grad_query_heads;
    std::vector<std::shared_ptr<Tensor>> grad_key_heads;
    std::vector<std::shared_ptr<Tensor>> grad_value_heads;
    grad_query_heads.reserve(static_cast<size_t>(query_heads));
    grad_key_heads.reserve(static_cast<size_t>(key_heads));
    grad_value_heads.reserve(static_cast<size_t>(key_heads));

    for (int64_t kv_head = 0; kv_head < key_heads; ++kv_head) {
        const int64_t q_head_start = kv_head * n_rep;
        auto k_single = SliceHeadRange(key_math, kv_head, kv_head + 1);
        auto v_single = SliceHeadRange(value_math, kv_head, kv_head + 1);

        std::shared_ptr<Tensor> grad_key_accum;
        std::shared_ptr<Tensor> grad_value_accum;

        for (int64_t rep = 0; rep < n_rep; ++rep) {
            const int64_t q_head = q_head_start + rep;
            auto q_single = SliceHeadRange(query_math, q_head, q_head + 1);
            auto go_single = SliceHeadRange(grad_output_math, q_head, q_head + 1);
            auto mask_single = per_head_masks.empty() ? nullptr : per_head_masks[static_cast<size_t>(q_head)];

            auto [grad_query_single, grad_key_single, grad_value_single] = RunChunkedFallbackBackward(
                q_single, k_single, v_single, mask_single, go_single, is_causal, scale, q_chunk_size);

            grad_query_heads.push_back(grad_query_single);
            grad_key_accum = grad_key_accum ? (grad_key_accum + grad_key_single) : grad_key_single;
            grad_value_accum = grad_value_accum ? (grad_value_accum + grad_value_single) : grad_value_single;
        }

        grad_key_heads.push_back(grad_key_accum);
        grad_value_heads.push_back(grad_value_accum);
    }

    auto grad_query = CastTensorTo(nn::function::Concat(grad_query_heads, 1), query->Dtype());
    auto grad_key = CastTensorTo(nn::function::Concat(grad_key_heads, 1), key->Dtype());
    auto grad_value = CastTensorTo(nn::function::Concat(grad_value_heads, 1), value->Dtype());
    return {grad_query, grad_key, grad_value};
}
} // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, uint64_t, uint64_t>
FlashAttentionForward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                      const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask, double dropout_p,
                      bool is_causal, double scale, bool enable_gqa) {
    CHECK(query);
    CHECK(key);
    CHECK(value);

    CHECK(query->GetDevice().IsCUDA());
    CHECK(key->GetDevice().IsCUDA());
    CHECK(value->GetDevice().IsCUDA());
    if (attn_mask) {
        CHECK(attn_mask->GetDevice().IsCUDA());
    }

    CHECK_GE(dropout_p, 0.0);
    CHECK_LT(dropout_p, 1.0);
    CHECK_GT(scale, 0.0);
    CHECK_EQ(query->Dims().size(), 4);
    CHECK_EQ(key->Dims().size(), 4);
    CHECK_EQ(value->Dims().size(), 4);
    CHECK_EQ(query->Dims()[0], key->Dims()[0]);
    CHECK_EQ(query->Dims()[0], value->Dims()[0]);
    CHECK_EQ(key->Dims()[2], value->Dims()[2]);
    CHECK_EQ(query->Dims()[3], key->Dims()[3]);

    const bool can_use_fused = CanUseFusedPath(query, key, value, attn_mask);
    if (dropout_p > 0.0 && !can_use_fused) {
        CHECK(false) << "dropout is only supported on fused CUDA path in current implementation.";
    }

    if (!can_use_fused) {
        LOG_FIRST_N(INFO, 1) << "FlashAttentionForward fallback path selected. reason="
                             << GetFusedPathRejectionReason(query, key, value, attn_mask)
                             << " q_heads=" << query->Dims()[1] << " kv_heads=" << key->Dims()[1]
                             << " q_len=" << query->Dims()[2] << " kv_len=" << key->Dims()[2]
                             << " head_dim=" << query->Dims()[3] << " value_dim=" << value->Dims()[3]
                             << " has_attn_mask=" << (attn_mask != nullptr) << " enable_gqa=" << enable_gqa;
        return RunFallbackForward(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
    }
    LOG_FIRST_N(INFO, 1) << "FlashAttentionForward fused path selected."
                         << " q_heads=" << query->Dims()[1] << " kv_heads=" << key->Dims()[1]
                         << " q_len=" << query->Dims()[2] << " kv_len=" << key->Dims()[2]
                         << " head_dim=" << query->Dims()[3] << " value_dim=" << value->Dims()[3]
                         << " enable_gqa=" << enable_gqa;

    const int64_t batch = query->Dims()[0];
    const int64_t query_heads = query->Dims()[1];
    const int64_t q_len = query->Dims()[2];
    const int64_t head_dim = query->Dims()[3];
    const int64_t key_heads = key->Dims()[1];
    const int64_t kv_len = key->Dims()[2];
    const int64_t value_dim = value->Dims()[3];

    int64_t group_size = 1;
    if (enable_gqa) {
        CHECK_EQ(query_heads % key_heads, 0);
        group_size = query_heads / key_heads;
    } else {
        CHECK_EQ(query_heads, key_heads);
    }

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch, query_heads, q_len, value_dim}, query->Dtype(),
                                           query->GetDevice());
    auto lse = std::make_shared<Tensor>(std::vector<int64_t>{batch, query_heads, q_len}, DataType::kFLOAT32,
                                        query->GetDevice());

    const uint64_t seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const uint64_t offset = 0;

    auto cuda_stream = GetCudaStream(query->GetDevice());
    const bool use_dropout = dropout_p > 0.0;
    const auto kernel_cfg = SelectFusedKernelConfig(head_dim, value_dim, kv_len, query_heads, key_heads, is_causal,
                                                    use_dropout, query->GetDevice());
    const int block_size = kernel_cfg.block_size;
    const int kv_tile_size = kernel_cfg.kv_tile_size;
    const bool use_bf16 = query->Dtype() == DataType::kBFLOAT16;

    auto launch_with_tile = [&](auto block_tag, auto tile_tag) {
        constexpr int BLOCK = decltype(block_tag)::value;
        constexpr int TILE = decltype(tile_tag)::value;
        if (use_dropout) {
            if (use_bf16) {
                LaunchFusedForwardKernel<BLOCK, TILE, true, __nv_bfloat16>(
                    query, key, value, output, lse, batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
                    group_size, is_causal, static_cast<float>(scale), static_cast<float>(dropout_p), seed, offset,
                    cuda_stream);
            } else {
                LaunchFusedForwardKernel<BLOCK, TILE, true, float>(
                    query, key, value, output, lse, batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
                    group_size, is_causal, static_cast<float>(scale), static_cast<float>(dropout_p), seed, offset,
                    cuda_stream);
            }
        } else {
            if (use_bf16) {
                LaunchFusedForwardKernel<BLOCK, TILE, false, __nv_bfloat16>(
                    query, key, value, output, lse, batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
                    group_size, is_causal, static_cast<float>(scale), static_cast<float>(dropout_p), seed, offset,
                    cuda_stream);
            } else {
                LaunchFusedForwardKernel<BLOCK, TILE, false, float>(
                    query, key, value, output, lse, batch, query_heads, key_heads, q_len, kv_len, head_dim, value_dim,
                    group_size, is_causal, static_cast<float>(scale), static_cast<float>(dropout_p), seed, offset,
                    cuda_stream);
            }
        }
    };

    if (block_size == kFusedBlockSizeSmall) {
        if (kv_tile_size == kFusedKvTileSizeSmall) {
            launch_with_tile(std::integral_constant<int, kFusedBlockSizeSmall>{},
                             std::integral_constant<int, kFusedKvTileSizeSmall>{});
        } else if (kv_tile_size == kFusedKvTileSizeLarge) {
            launch_with_tile(std::integral_constant<int, kFusedBlockSizeSmall>{},
                             std::integral_constant<int, kFusedKvTileSizeLarge>{});
        } else {
            launch_with_tile(std::integral_constant<int, kFusedBlockSizeSmall>{},
                             std::integral_constant<int, kFusedKvTileSizeMedium>{});
        }
    } else {
        if (kv_tile_size == kFusedKvTileSizeSmall) {
            launch_with_tile(std::integral_constant<int, kFusedBlockSizeLarge>{},
                             std::integral_constant<int, kFusedKvTileSizeSmall>{});
        } else if (kv_tile_size == kFusedKvTileSizeLarge) {
            launch_with_tile(std::integral_constant<int, kFusedBlockSizeLarge>{},
                             std::integral_constant<int, kFusedKvTileSizeLarge>{});
        } else {
            launch_with_tile(std::integral_constant<int, kFusedBlockSizeLarge>{},
                             std::integral_constant<int, kFusedKvTileSizeMedium>{});
        }
    }
    const auto forward_launch_error = cudaGetLastError();
    if (forward_launch_error != cudaSuccess) {
        LOG(ERROR) << "FusedAttentionForwardKernel launch failed, fallback to baseline path. error="
                   << cudaGetErrorString(forward_launch_error);
        if (dropout_p > 0.0) {
            CHECK(false) << "dropout fallback is not supported when fused forward launch fails.";
        }
        return RunFallbackForward(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
    }
    return {output, lse, seed, offset};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
FlashAttentionBackward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                       const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask,
                       const std::shared_ptr<Tensor> &lse, const std::shared_ptr<Tensor> &grad_output, double dropout_p,
                       bool is_causal, double scale, bool enable_gqa, uint64_t rng_seed, uint64_t rng_offset) {
    CHECK(query);
    CHECK(key);
    CHECK(value);
    CHECK(grad_output);

    CHECK(query->GetDevice().IsCUDA());
    CHECK(key->GetDevice().IsCUDA());
    CHECK(value->GetDevice().IsCUDA());
    CHECK(grad_output->GetDevice().IsCUDA());
    if (attn_mask) {
        CHECK(attn_mask->GetDevice().IsCUDA());
    }

    CHECK_GE(dropout_p, 0.0);
    CHECK_LT(dropout_p, 1.0);
    CHECK_GT(scale, 0.0);

    if (dropout_p > 0.0) {
        CHECK(false) << "Backward with dropout is currently unsupported (both fused and fallback). Set dropout_p=0.";
    }

    auto grad_output_kernel = grad_output;
    if (grad_output->Dtype() != query->Dtype()) {
        grad_output_kernel = CastTensorTo(grad_output, query->Dtype());
    }

    if (!CanUseFusedBackwardPath(query, key, value, attn_mask, lse, grad_output_kernel, dropout_p)) {
        LOG_FIRST_N(INFO, 1) << "FlashAttentionBackward fallback path selected."
                             << " q_heads=" << query->Dims()[1] << " kv_heads=" << key->Dims()[1]
                             << " q_len=" << query->Dims()[2] << " kv_len=" << key->Dims()[2]
                             << " head_dim=" << query->Dims()[3] << " value_dim=" << value->Dims()[3]
                             << " has_attn_mask=" << (attn_mask != nullptr) << " enable_gqa=" << enable_gqa;
        return RunFallbackBackward(query, key, value, attn_mask, grad_output, dropout_p, is_causal, scale, enable_gqa);
    }
    LOG_FIRST_N(INFO, 1) << "FlashAttentionBackward fused path selected."
                         << " q_heads=" << query->Dims()[1] << " kv_heads=" << key->Dims()[1]
                         << " q_len=" << query->Dims()[2] << " kv_len=" << key->Dims()[2]
                         << " head_dim=" << query->Dims()[3] << " value_dim=" << value->Dims()[3]
                         << " enable_gqa=" << enable_gqa;

    const int64_t batch = query->Dims()[0];
    const int64_t query_heads = query->Dims()[1];
    const int64_t q_len = query->Dims()[2];
    const int64_t head_dim = query->Dims()[3];
    const int64_t key_heads = key->Dims()[1];
    const int64_t kv_len = key->Dims()[2];
    const int64_t value_dim = value->Dims()[3];

    int64_t group_size = 1;
    if (enable_gqa) {
        CHECK_EQ(query_heads % key_heads, 0);
        group_size = query_heads / key_heads;
    } else {
        CHECK_EQ(query_heads, key_heads);
    }

    auto grad_query = std::make_shared<Tensor>(query->Dims(), DataType::kFLOAT32, query->GetDevice());
    auto grad_key = std::make_shared<Tensor>(key->Dims(), DataType::kFLOAT32, key->GetDevice());
    auto grad_value = std::make_shared<Tensor>(value->Dims(), DataType::kFLOAT32, value->GetDevice());
    FillTensor(grad_key, 0.0f, "CUDA FlashAttentionBackward");
    FillTensor(grad_value, 0.0f, "CUDA FlashAttentionBackward");

    auto cuda_stream = GetCudaStream(query->GetDevice());
    const bool use_dropout = dropout_p > 0.0;
    const auto kernel_cfg = SelectFusedKernelConfig(head_dim, value_dim, kv_len, query_heads, key_heads, is_causal,
                                                    use_dropout, query->GetDevice());
    const int block_size = kernel_cfg.block_size;
    const int kv_tile_size = kernel_cfg.kv_tile_size;
    const bool use_bf16 = query->Dtype() == DataType::kBFLOAT16;

    auto launch_with_tile = [&](auto block_tag, auto tile_tag) {
        constexpr int BLOCK = decltype(block_tag)::value;
        constexpr int TILE = decltype(tile_tag)::value;
        if (use_dropout) {
            if (use_bf16) {
                LaunchFusedBackwardKernel<BLOCK, TILE, true, __nv_bfloat16>(
                    query, key, value, grad_output_kernel, lse, grad_query, grad_key, grad_value, batch, query_heads,
                    key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, static_cast<float>(scale),
                    static_cast<float>(dropout_p), rng_seed, rng_offset, cuda_stream);
            } else {
                LaunchFusedBackwardKernel<BLOCK, TILE, true, float>(
                    query, key, value, grad_output_kernel, lse, grad_query, grad_key, grad_value, batch, query_heads,
                    key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, static_cast<float>(scale),
                    static_cast<float>(dropout_p), rng_seed, rng_offset, cuda_stream);
            }
        } else {
            if (use_bf16) {
                LaunchFusedBackwardKernel<BLOCK, TILE, false, __nv_bfloat16>(
                    query, key, value, grad_output_kernel, lse, grad_query, grad_key, grad_value, batch, query_heads,
                    key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, static_cast<float>(scale),
                    static_cast<float>(dropout_p), rng_seed, rng_offset, cuda_stream);
            } else {
                LaunchFusedBackwardKernel<BLOCK, TILE, false, float>(
                    query, key, value, grad_output_kernel, lse, grad_query, grad_key, grad_value, batch, query_heads,
                    key_heads, q_len, kv_len, head_dim, value_dim, group_size, is_causal, static_cast<float>(scale),
                    static_cast<float>(dropout_p), rng_seed, rng_offset, cuda_stream);
            }
        }
    };

    if (block_size == kFusedBlockSizeSmall) {
        if (kv_tile_size == kFusedKvTileSizeSmall) {
            launch_with_tile(std::integral_constant<int, kFusedBlockSizeSmall>{},
                             std::integral_constant<int, kFusedKvTileSizeSmall>{});
        } else if (kv_tile_size == kFusedKvTileSizeLarge) {
            launch_with_tile(std::integral_constant<int, kFusedBlockSizeSmall>{},
                             std::integral_constant<int, kFusedKvTileSizeLarge>{});
        } else {
            launch_with_tile(std::integral_constant<int, kFusedBlockSizeSmall>{},
                             std::integral_constant<int, kFusedKvTileSizeMedium>{});
        }
    } else {
        if (kv_tile_size == kFusedKvTileSizeSmall) {
            launch_with_tile(std::integral_constant<int, kFusedBlockSizeLarge>{},
                             std::integral_constant<int, kFusedKvTileSizeSmall>{});
        } else if (kv_tile_size == kFusedKvTileSizeLarge) {
            launch_with_tile(std::integral_constant<int, kFusedBlockSizeLarge>{},
                             std::integral_constant<int, kFusedKvTileSizeLarge>{});
        } else {
            launch_with_tile(std::integral_constant<int, kFusedBlockSizeLarge>{},
                             std::integral_constant<int, kFusedKvTileSizeMedium>{});
        }
    }
    const auto backward_launch_error = cudaGetLastError();
    if (backward_launch_error != cudaSuccess) {
        LOG(ERROR) << "FusedAttentionBackwardKernel launch failed, fallback to baseline path. error="
                   << cudaGetErrorString(backward_launch_error);
        return RunFallbackBackward(query, key, value, attn_mask, grad_output, dropout_p, is_causal, scale, enable_gqa);
    }
    return {CastTensorTo(grad_query, query->Dtype()), CastTensorTo(grad_key, key->Dtype()),
            CastTensorTo(grad_value, value->Dtype())};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ATTENTION_KERNEL(kernel_name)                                                                    \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ATTENTION_KERNEL(FlashAttentionForward)
REGISTER_CUDA_ATTENTION_KERNEL(FlashAttentionBackward)

#undef REGISTER_CUDA_ATTENTION_KERNEL
