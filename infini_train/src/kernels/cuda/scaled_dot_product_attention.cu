//------modify-start------------------------------------------
#include <cstdint>
#include <cstring>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include "glog/logging.h"

//------modify-start------------------------------------------
// Minimal cuDNN status checker (avoid multi-line macros).
inline void CudnnCheck(cudnnStatus_t status, const char *file, int line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        LOG(FATAL) << "CUDNN Error: " << cudnnGetErrorString(status) << " at " << file << ":" << line;
    }
}

#define CUDNN_CHECK(call) CudnnCheck((call), __FILE__, __LINE__)
//---------modify-end-----------------------------------------

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace fe = cudnn_frontend;

namespace {

struct SdpaKey {
    int64_t b = 0;
    int64_t h = 0;
    int64_t s = 0;
    int64_t d = 0;
    bool is_causal = false;
    uint32_t scale_bits = 0;

    bool operator==(const SdpaKey &other) const {
        return b == other.b && h == other.h && s == other.s && d == other.d && is_causal == other.is_causal
            && scale_bits == other.scale_bits;
    }
};

struct SdpaKeyHash {
    size_t operator()(const SdpaKey &k) const noexcept {
        // A simple 64-bit mix.
        size_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) {
            h ^= static_cast<size_t>(v);
            h *= 1099511628211ull;
        };
        mix(static_cast<uint64_t>(k.b));
        mix(static_cast<uint64_t>(k.h));
        mix(static_cast<uint64_t>(k.s));
        mix(static_cast<uint64_t>(k.d));
        mix(static_cast<uint64_t>(k.is_causal));
        mix(static_cast<uint64_t>(k.scale_bits));
        return h;
    }
};

struct CachedGraph {
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace_size = 0;
};

static std::mutex g_cache_mu;
static std::unordered_map<SdpaKey, CachedGraph, SdpaKeyHash> g_fwd_cache;
static std::unordered_map<SdpaKey, CachedGraph, SdpaKeyHash> g_bwd_cache;

static thread_local cudnnHandle_t tls_cudnn_handle = nullptr;

cudnnHandle_t GetCudnnHandle(cudaStream_t stream) {
    if (tls_cudnn_handle == nullptr) {
        CUDNN_CHECK(cudnnCreate(&tls_cudnn_handle));
    }
    CUDNN_CHECK(cudnnSetStream(tls_cudnn_handle, stream));
    return tls_cudnn_handle;
}

uint32_t FloatToBits(float x) {
    uint32_t u = 0;
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::memcpy(&u, &x, sizeof(uint32_t));
    return u;
}

std::vector<int64_t> ContigStrideBHSD(int64_t b, int64_t h, int64_t s, int64_t d) {
    (void)b;
    // Layout: (B, H, S, D) contiguous
    return {h * s * d, s * d, d, 1};
}

CachedGraph BuildFwdGraph(cudnnHandle_t handle, int64_t b, int64_t h, int64_t s, int64_t d, float attn_scale,
                          bool is_causal) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto stride = ContigStrideBHSD(b, h, s, d);

    constexpr int Q_UID = 1;
    constexpr int K_UID = 2;
    constexpr int V_UID = 3;
    constexpr int O_UID = 4;
    constexpr int STATS_UID = 5;

    auto Q = graph->tensor(
        fe::graph::Tensor_attributes().set_name("Q").set_uid(Q_UID).set_dim({b, h, s, d}).set_stride(stride));

    auto K = graph->tensor(
        fe::graph::Tensor_attributes().set_name("K").set_uid(K_UID).set_dim({b, h, s, d}).set_stride(stride));

    auto V = graph->tensor(
        fe::graph::Tensor_attributes().set_name("V").set_uid(V_UID).set_dim({b, h, s, d}).set_stride(stride));

    auto sdpa_options
        = fe::graph::SDPA_attributes().set_name("sdpa").set_generate_stats(true).set_attn_scale(attn_scale);

    if (is_causal) {
        sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
            .set_diagonal_band_right_bound(0);
    }

    auto [O, Stats] = graph->sdpa(Q, K, V, sdpa_options);

    O->set_output(true).set_uid(O_UID).set_dim({b, h, s, d}).set_stride(stride);
    Stats->set_output(true)
        .set_uid(STATS_UID)
        .set_dim({b, h, s, 1})
        .set_stride({h * s, s, 1, 1})
        .set_data_type(fe::DataType_t::FLOAT);

    auto build_status = graph->build(handle, {fe::HeurMode_t::A});
    CHECK(build_status.is_good()) << "cudnn-frontend SDPA forward graph build failed: " << build_status.get_message();

    int64_t workspace_size = 0;
    auto ws_status = graph->get_workspace_size(workspace_size);
    CHECK(ws_status.is_good()) << "cudnn-frontend get_workspace_size failed: " << ws_status.get_message();

    return {.graph = std::move(graph), .workspace_size = workspace_size};
}

CachedGraph BuildBwdGraph(cudnnHandle_t handle, int64_t b, int64_t h, int64_t s, int64_t d, float attn_scale,
                          bool is_causal) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto stride = ContigStrideBHSD(b, h, s, d);

    constexpr int Q_UID = 1;
    constexpr int K_UID = 2;
    constexpr int V_UID = 3;
    constexpr int O_UID = 4;
    constexpr int STATS_UID = 5;
    constexpr int DO_UID = 101;

    constexpr int DQ_UID = 102;
    constexpr int DK_UID = 103;
    constexpr int DV_UID = 104;

    auto Q = graph->tensor(
        fe::graph::Tensor_attributes().set_name("Q").set_uid(Q_UID).set_dim({b, h, s, d}).set_stride(stride));

    auto K = graph->tensor(
        fe::graph::Tensor_attributes().set_name("K").set_uid(K_UID).set_dim({b, h, s, d}).set_stride(stride));

    auto V = graph->tensor(
        fe::graph::Tensor_attributes().set_name("V").set_uid(V_UID).set_dim({b, h, s, d}).set_stride(stride));

    auto O = graph->tensor(
        fe::graph::Tensor_attributes().set_name("O").set_uid(O_UID).set_dim({b, h, s, d}).set_stride(stride));

    auto dO = graph->tensor(
        fe::graph::Tensor_attributes().set_name("dO").set_uid(DO_UID).set_dim({b, h, s, d}).set_stride(stride));

    auto Stats = graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("Stats")
                                   .set_uid(STATS_UID)
                                   .set_dim({b, h, s, 1})
                                   .set_stride({h * s, s, 1, 1})
                                   .set_data_type(fe::DataType_t::FLOAT));

    auto bwd_options = fe::graph::SDPA_backward_attributes().set_name("sdpa_backward").set_attn_scale(attn_scale);

    if (is_causal) {
        bwd_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
            .set_diagonal_band_right_bound(0);
    }

    auto [dQ, dK, dV] = graph->sdpa_backward(Q, K, V, O, dO, Stats, bwd_options);

    dQ->set_output(true).set_uid(DQ_UID).set_dim({b, h, s, d}).set_stride(stride);
    dK->set_output(true).set_uid(DK_UID).set_dim({b, h, s, d}).set_stride(stride);
    dV->set_output(true).set_uid(DV_UID).set_dim({b, h, s, d}).set_stride(stride);

    auto build_status = graph->build(handle, {fe::HeurMode_t::A});
    CHECK(build_status.is_good()) << "cudnn-frontend SDPA backward graph build failed: " << build_status.get_message();

    int64_t workspace_size = 0;
    auto ws_status = graph->get_workspace_size(workspace_size);
    CHECK(ws_status.is_good()) << "cudnn-frontend get_workspace_size failed: " << ws_status.get_message();

    return {.graph = std::move(graph), .workspace_size = workspace_size};
}

CachedGraph GetOrCreateFwdGraph(cudnnHandle_t handle, int64_t b, int64_t h, int64_t s, int64_t d, float attn_scale,
                                bool is_causal) {
    SdpaKey key{.b = b, .h = h, .s = s, .d = d, .is_causal = is_causal, .scale_bits = FloatToBits(attn_scale)};
    std::lock_guard<std::mutex> lock(g_cache_mu);
    auto it = g_fwd_cache.find(key);
    if (it != g_fwd_cache.end()) {
        return it->second;
    }
    auto cached = BuildFwdGraph(handle, b, h, s, d, attn_scale, is_causal);
    g_fwd_cache.emplace(key, cached);
    return cached;
}

CachedGraph GetOrCreateBwdGraph(cudnnHandle_t handle, int64_t b, int64_t h, int64_t s, int64_t d, float attn_scale,
                                bool is_causal) {
    SdpaKey key{.b = b, .h = h, .s = s, .d = d, .is_causal = is_causal, .scale_bits = FloatToBits(attn_scale)};
    std::lock_guard<std::mutex> lock(g_cache_mu);
    auto it = g_bwd_cache.find(key);
    if (it != g_bwd_cache.end()) {
        return it->second;
    }
    auto cached = BuildBwdGraph(handle, b, h, s, d, attn_scale, is_causal);
    g_bwd_cache.emplace(key, cached);
    return cached;
}

} // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
ScaledDotProductAttentionForward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                                 const std::shared_ptr<Tensor> &value, float attn_scale, bool is_causal) {
    CHECK(query->GetDevice().type() == Device::DeviceType::kCUDA);
    //------modify-start------------------------------------------
    // Avoid CHECK_EQ on enum class (would require operator<< overload).
    CHECK(query->Dtype() == DataType::kBFLOAT16) << "SDPA forward only supports BF16";
    CHECK(key->Dtype() == DataType::kBFLOAT16);
    CHECK(value->Dtype() == DataType::kBFLOAT16);
    //---------modify-end-----------------------------------------

    const auto &q_dims = query->Dims();
    CHECK_EQ(q_dims.size(), 4);
    const int64_t b = q_dims[0];
    const int64_t h = q_dims[1];
    const int64_t s = q_dims[2];
    const int64_t d = q_dims[3];

    //------modify-start------------------------------------------
    // Avoid CHECK_EQ on std::vector (requires operator<< overload).
    CHECK(key->Dims() == q_dims);
    CHECK(value->Dims() == q_dims);
    //---------modify-end-----------------------------------------

    auto device = query->GetDevice();
    core::DeviceGuard guard(device);

    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    auto handle = GetCudnnHandle(cuda_stream);

    //------modify-start------------------------------------------
    // Cache cuDNN-frontend graphs by shape/scale for performance.
    auto cached = GetOrCreateFwdGraph(handle, b, h, s, d, attn_scale, is_causal);
    //---------modify-end-----------------------------------------

    auto output = std::make_shared<Tensor>(q_dims, DataType::kBFLOAT16, device);
    auto stats = std::make_shared<Tensor>(std::vector<int64_t>{b, h, s, 1}, DataType::kFLOAT32, device);

    //------modify-start------------------------------------------
    // Defensive initialization: if the selected cuDNN plan does not fully overwrite outputs
    // (e.g., due to masked/unused lanes), this prevents stale NaNs from propagating.
    CHECK(cudaMemsetAsync(output->DataPtr(), 0, output->SizeInBytes(), cuda_stream) == cudaSuccess);
    CHECK(cudaMemsetAsync(stats->DataPtr(), 0, stats->SizeInBytes(), cuda_stream) == cudaSuccess);
    //---------modify-end-----------------------------------------

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> variant_pack;
    variant_pack.reserve(5);
    variant_pack[1] = query->DataPtr();
    variant_pack[2] = key->DataPtr();
    variant_pack[3] = value->DataPtr();
    variant_pack[4] = output->DataPtr();
    variant_pack[5] = stats->DataPtr();

    void *workspace_ptr = nullptr;
    std::shared_ptr<Tensor> workspace = nullptr;
    if (cached.workspace_size > 0) {
        workspace = std::make_shared<Tensor>(std::vector<int64_t>{cached.workspace_size}, DataType::kUINT8, device);
        workspace_ptr = workspace->DataPtr();

        //------modify-start------------------------------------------
        // Defensive initialization: some cuDNN plans may read workspace without full overwrite.
        CHECK(cudaMemsetAsync(workspace_ptr, 0, static_cast<size_t>(cached.workspace_size), cuda_stream)
              == cudaSuccess);
        //---------modify-end-----------------------------------------
    }

    auto exec_status = cached.graph->execute(handle, variant_pack, workspace_ptr);
    CHECK(exec_status.is_good()) << "cudnn-frontend SDPA forward execute failed: " << exec_status.get_message();

    return {output, stats};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
ScaledDotProductAttentionBackward(const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
                                  const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &output,
                                  const std::shared_ptr<Tensor> &stats, const std::shared_ptr<Tensor> &grad_output,
                                  float attn_scale, bool is_causal) {
    CHECK(query->GetDevice().type() == Device::DeviceType::kCUDA);
    //------modify-start------------------------------------------
    // Avoid CHECK_EQ on enum class (would require operator<< overload).
    CHECK(query->Dtype() == DataType::kBFLOAT16) << "SDPA backward only supports BF16";
    //---------modify-end-----------------------------------------

    const auto &q_dims = query->Dims();
    CHECK_EQ(q_dims.size(), 4);
    const int64_t b = q_dims[0];
    const int64_t h = q_dims[1];
    const int64_t s = q_dims[2];
    const int64_t d = q_dims[3];

    //------modify-start------------------------------------------
    // Avoid CHECK_EQ on std::vector (requires operator<< overload).
    CHECK(key->Dims() == q_dims);
    CHECK(value->Dims() == q_dims);
    CHECK(output->Dims() == q_dims);
    CHECK(grad_output->Dims() == q_dims);
    CHECK(stats->Dims() == (std::vector<int64_t>{b, h, s, 1}));
    //---------modify-end-----------------------------------------

    auto device = query->GetDevice();
    core::DeviceGuard guard(device);

    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    auto handle = GetCudnnHandle(cuda_stream);

    //------modify-start------------------------------------------
    // Cache cuDNN-frontend graphs by shape/scale for performance.
    auto cached = GetOrCreateBwdGraph(handle, b, h, s, d, attn_scale, is_causal);
    //---------modify-end-----------------------------------------

    auto grad_query = std::make_shared<Tensor>(q_dims, DataType::kBFLOAT16, device);
    auto grad_key = std::make_shared<Tensor>(q_dims, DataType::kBFLOAT16, device);
    auto grad_value = std::make_shared<Tensor>(q_dims, DataType::kBFLOAT16, device);

    //------modify-start------------------------------------------
    // Defensive initialization: ensure gradients are fully overwritten by cuDNN SDPA backward.
    CHECK(cudaMemsetAsync(grad_query->DataPtr(), 0, grad_query->SizeInBytes(), cuda_stream) == cudaSuccess);
    CHECK(cudaMemsetAsync(grad_key->DataPtr(), 0, grad_key->SizeInBytes(), cuda_stream) == cudaSuccess);
    CHECK(cudaMemsetAsync(grad_value->DataPtr(), 0, grad_value->SizeInBytes(), cuda_stream) == cudaSuccess);
    //---------modify-end-----------------------------------------

    std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> variant_pack;
    variant_pack.reserve(9);
    // inputs
    variant_pack[1] = query->DataPtr();
    variant_pack[2] = key->DataPtr();
    variant_pack[3] = value->DataPtr();
    variant_pack[4] = output->DataPtr();
    variant_pack[5] = stats->DataPtr();
    variant_pack[101] = grad_output->DataPtr();
    // outputs
    variant_pack[102] = grad_query->DataPtr();
    variant_pack[103] = grad_key->DataPtr();
    variant_pack[104] = grad_value->DataPtr();

    void *workspace_ptr = nullptr;
    std::shared_ptr<Tensor> workspace = nullptr;
    if (cached.workspace_size > 0) {
        workspace = std::make_shared<Tensor>(std::vector<int64_t>{cached.workspace_size}, DataType::kUINT8, device);
        workspace_ptr = workspace->DataPtr();

        //------modify-start------------------------------------------
        // Defensive initialization: some cuDNN plans may read workspace without full overwrite.
        CHECK(cudaMemsetAsync(workspace_ptr, 0, static_cast<size_t>(cached.workspace_size), cuda_stream)
              == cudaSuccess);
        //---------modify-end-----------------------------------------
    }

    auto exec_status = cached.graph->execute(handle, variant_pack, workspace_ptr);
    CHECK(exec_status.is_good()) << "cudnn-frontend SDPA backward execute failed: " << exec_status.get_message();

    return {grad_query, grad_key, grad_value};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_SDPA_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_SDPA_KERNEL(ScaledDotProductAttentionForward)
REGISTER_CUDA_SDPA_KERNEL(ScaledDotProductAttentionBackward)

#undef REGISTER_CUDA_SDPA_KERNEL
//---------modify-end-----------------------------------------
