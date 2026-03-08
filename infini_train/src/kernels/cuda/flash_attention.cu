#include <functional>
#include <numeric>
#include <vector>
#include <optional>
#include <tuple>
#include <memory>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include <cub/block/block_reduce.cuh>
#include <cublas_v2.h>
#include <cudnn.h>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/cuda/cuda_blas_handle.h"
#include "infini_train/src/core/cuda/cuda_stream.h"
#include "infini_train/include/common/common.h"   // ComputeStrides
#include <cuda_runtime.h>  // cudaStream_t

// 强烈建议使用 NVIDIA 提供的 frontend 库，否则原始 API 会写到手软
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;


namespace infini_train::kernels::cuda {

namespace {
constexpr int64_t Q_UID     = 101;
constexpr int64_t K_UID     = 102;
constexpr int64_t V_UID     = 103;
constexpr int64_t MASK_UID  = 104;
constexpr int64_t O_UID     = 201;
constexpr int64_t STATS_UID = 202;

constexpr int64_t dO_UID = 301;
constexpr int64_t dQ_UID = 401;
constexpr int64_t dK_UID = 402;
constexpr int64_t dV_UID = 403;

struct WorkspaceCache {
    void *ptr = nullptr;
    size_t size = 0;
};
}

// helpers for cuDNN frontend path
static cudaStream_t get_cuda_stream(const ::infini_train::Device &device) {
    auto impl = ::infini_train::core::GetDeviceGuardImpl(device.type());
    auto stream_obj = impl->GetStream(device);
    auto cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(stream_obj)->cuda_stream();
    return cuda_stream;
}

static cudnnHandle_t get_cudnn_handle(const ::infini_train::Device &device) {
    int cuda_device = 0;
    CUDA_CHECK(cudaGetDevice(&cuda_device));

    static thread_local std::unordered_map<int, cudnnHandle_t> handles;
    auto it = handles.find(cuda_device);
    if (it == handles.end()) {
        cudnnHandle_t handle;
        cudnnCreate(&handle);
        it = handles.emplace(cuda_device, handle).first;
    }

    auto cuda_stream = get_cuda_stream(device);
    cudnnSetStream(it->second, cuda_stream);

    return it->second;
}

static void *acquire_workspace(WorkspaceCache &cache, size_t requested_bytes) {
    if (requested_bytes == 0) {
        return nullptr;
    }
    if (cache.ptr == nullptr || cache.size < requested_bytes) {
        if (cache.ptr != nullptr) {
            CUDA_CHECK(cudaFree(cache.ptr));
        }
        CUDA_CHECK(cudaMalloc(&cache.ptr, requested_bytes));
        cache.size = requested_bytes;
    }
    return cache.ptr;
}

static WorkspaceCache &forward_workspace_cache() {
    static thread_local WorkspaceCache cache;
    return cache;
}

static WorkspaceCache &backward_workspace_cache() {
    static thread_local WorkspaceCache cache;
    return cache;
}

static fe::DataType_t get_cudnn_dtype(const ::infini_train::DataType dtype);
static std::shared_ptr<fe::graph::Tensor_attributes> make_graph_tensor(
    const std::shared_ptr<fe::graph::Graph> &graph,
    const std::shared_ptr<Tensor> &tensor,
    const std::string &name,
    int64_t uid);
static void check_fe_status(fe::error_t status, const char *stage);

static std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> ExecuteSdpaForwardWithLse(
        const std::shared_ptr<Tensor> &q,
        const std::shared_ptr<Tensor> &k,
        const std::shared_ptr<Tensor> &v,
        const std::shared_ptr<Tensor> &attn_mask,
        double dropout_p,
        bool is_causal,
        std::optional<double> scale,
        bool /*enable_gqa*/) {
    if (dropout_p > 0.0) {
        throw std::runtime_error("cuDNN frontend SDPA path currently does not support dropout in this minimal kernel");
    }

    auto out = std::make_shared<Tensor>(q->Dims(), q->Dtype(), q->GetDevice());

    auto q_dims = q->Dims();
    CHECK_EQ(q_dims.size(), 4) << "SDPA expects 4D Q/K/V tensor layout [B, H, S, D]";
    std::vector<int64_t> lse_dims = {q_dims[0], q_dims[1], q_dims[2], 1};
    auto lse = std::make_shared<Tensor>(lse_dims, DataType::kFLOAT32, q->GetDevice());

    cudnnHandle_t handle = get_cudnn_handle(q->GetDevice());

    float attn_scale = scale.has_value() ? static_cast<float>(scale.value())
                                          : 1.0f / std::sqrt(static_cast<float>(q->Dims().back()));

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(get_cudnn_dtype(q->Dtype()))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto q_tensor = make_graph_tensor(graph, q, "Q", Q_UID);
    auto k_tensor = make_graph_tensor(graph, k, "K", K_UID);
    auto v_tensor = make_graph_tensor(graph, v, "V", V_UID);

    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_name("flash_attention")
                            .set_generate_stats(true)
                            .set_attn_scale(attn_scale);

    if (is_causal) {
        sdpa_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
            .set_diagonal_band_right_bound(0);
    }

    if (attn_mask) {
        auto mask_tensor = make_graph_tensor(graph, attn_mask, "Bias", MASK_UID);
        sdpa_options.set_bias(mask_tensor);
    }

    auto [out_tensor, stats_tensor] = graph->sdpa(q_tensor, k_tensor, v_tensor, sdpa_options);
    out_tensor->set_output(true)
        .set_uid(O_UID)
        .set_dim(out->Dims())
        .set_stride(ComputeStrides(out->Dims()));

    stats_tensor->set_output(true)
        .set_uid(STATS_UID)
        .set_dim(lse_dims)
        .set_stride(ComputeStrides(lse_dims))
        .set_data_type(fe::DataType_t::FLOAT);

    check_fe_status(graph->build(handle, {fe::HeurMode_t::A}), "graph->build");

    int64_t workspace_size = 0;
    check_fe_status(graph->get_workspace_size(workspace_size), "graph->get_workspace_size");
    void *workspace = acquire_workspace(forward_workspace_cache(), static_cast<size_t>(workspace_size));

    std::unordered_map<int64_t, void *> variant_pack = {
        {Q_UID, q->DataPtr()},
        {K_UID, k->DataPtr()},
        {V_UID, v->DataPtr()},
        {O_UID, out->DataPtr()},
        {STATS_UID, lse->DataPtr()},
    };
    if (attn_mask) {
        variant_pack[MASK_UID] = attn_mask->DataPtr();
    }

    auto exec_status = graph->execute(handle, variant_pack, workspace);
    check_fe_status(exec_status, "graph->execute");

    return {out, lse};
}

static fe::DataType_t get_cudnn_dtype(const ::infini_train::DataType dtype) {
    switch (dtype) {
        case ::infini_train::DataType::kFLOAT32:
            return fe::DataType_t::FLOAT;
        case ::infini_train::DataType::kFLOAT16:
            return fe::DataType_t::HALF;
        case ::infini_train::DataType::kBFLOAT16:
            return fe::DataType_t::BFLOAT16;
        default:
            throw std::runtime_error("unsupported dtype for cuDNN SDP");
    }
}

static std::shared_ptr<fe::graph::Tensor_attributes> make_graph_tensor(
    const std::shared_ptr<fe::graph::Graph> &graph,
    const std::shared_ptr<Tensor> &tensor,
    const std::string &name,
    int64_t uid) {
    return graph->tensor(fe::graph::Tensor_attributes()
                             .set_name(name)
                             .set_uid(uid)
                             .set_dim(tensor->Dims())
                             .set_stride(ComputeStrides(tensor->Dims()))
                             .set_data_type(get_cudnn_dtype(tensor->Dtype())));
}

static void check_fe_status(fe::error_t status, const char *stage) {
    if (status.is_bad()) {
        throw std::runtime_error(std::string(stage) + ": " + status.get_message());
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> ScaledDotProductAttentionForward(
        const std::shared_ptr<Tensor> &q,
        const std::shared_ptr<Tensor> &k,
        const std::shared_ptr<Tensor> &v,
        const std::shared_ptr<Tensor> &attn_mask,
        double dropout_p,
        bool is_causal,
        std::optional<double> scale,
        bool enable_gqa) {
    return ExecuteSdpaForwardWithLse(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa);
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
ScaledDotProductAttentionBackward(
        const std::shared_ptr<Tensor> &grad_out,
        const std::shared_ptr<Tensor> &q,
        const std::shared_ptr<Tensor> &k,
        const std::shared_ptr<Tensor> &v,
        const std::shared_ptr<Tensor> &attn_mask,
        const std::shared_ptr<Tensor> &out,
        const std::shared_ptr<Tensor> &lse,
        double dropout_p,
        bool is_causal,
        std::optional<double> scale,
        bool enable_gqa) {

    auto dq = std::make_shared<Tensor>(q->Dims(), q->Dtype(), q->GetDevice());
    auto dk = std::make_shared<Tensor>(k->Dims(), k->Dtype(), k->GetDevice());
    auto dv = std::make_shared<Tensor>(v->Dims(), v->Dtype(), v->GetDevice());

    if (dropout_p > 0.0) {
        throw std::runtime_error("cuDNN frontend SDPA path currently does not support dropout in this minimal kernel");
    }
    (void)enable_gqa;


    // ---------- cuDNN frontend implementation ----------
    cudnnHandle_t handle = get_cudnn_handle(grad_out->GetDevice());

    float attn_scale = scale.has_value() ? static_cast<float>(scale.value())
                                          : 1.0f / std::sqrt(static_cast<float>(q->Dims().back()));

    auto graph = std::make_shared<fe::graph::Graph>();

    graph->set_io_data_type(get_cudnn_dtype(q->Dtype()))
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto q_tensor = make_graph_tensor(graph, q, "Q", Q_UID);
    auto k_tensor = make_graph_tensor(graph, k, "K", K_UID);
    auto v_tensor = make_graph_tensor(graph, v, "V", V_UID);
    auto o_tensor = make_graph_tensor(graph, out, "O", O_UID);
    auto dO_tensor = make_graph_tensor(graph, grad_out, "dO", dO_UID);
    auto lse_tensor = make_graph_tensor(graph, lse, "Stats", STATS_UID);

    auto sdpa_bwd_options = fe::graph::SDPA_backward_attributes()
                                .set_name("flash_attention_backward")
                                .set_attn_scale(attn_scale)
                                .set_deterministic_algorithm(true);

    if (is_causal) {
        sdpa_bwd_options.set_diagonal_alignment(cudnn_frontend::DiagonalAlignment_t::TOP_LEFT)
            .set_diagonal_band_right_bound(0);
    }

    if (attn_mask) {
        auto mask_tensor = make_graph_tensor(graph, attn_mask, "Bias", MASK_UID);
        sdpa_bwd_options.set_bias(mask_tensor);
    }

    auto [dQ_tensor, dK_tensor, dV_tensor] = graph->sdpa_backward(
        q_tensor, k_tensor, v_tensor, o_tensor, dO_tensor, lse_tensor, sdpa_bwd_options);

    dQ_tensor->set_output(true)
        .set_uid(dQ_UID)
        .set_dim(dq->Dims())
        .set_stride(ComputeStrides(dq->Dims()));
    dK_tensor->set_output(true)
        .set_uid(dK_UID)
        .set_dim(dk->Dims())
        .set_stride(ComputeStrides(dk->Dims()));
    dV_tensor->set_output(true)
        .set_uid(dV_UID)
        .set_dim(dv->Dims())
        .set_stride(ComputeStrides(dv->Dims()));

    check_fe_status(graph->build(handle, {fe::HeurMode_t::A}), "graph->build (backward)");

    int64_t workspace_size = 0;
    check_fe_status(graph->get_workspace_size(workspace_size), "graph->get_workspace_size (backward)");
    void *workspace = acquire_workspace(backward_workspace_cache(), static_cast<size_t>(workspace_size));

    std::unordered_map<int64_t, void *> variant_pack = {
        {Q_UID, q->DataPtr()},
        {K_UID, k->DataPtr()},
        {V_UID, v->DataPtr()},
        {O_UID, out->DataPtr()},
        {dO_UID, grad_out->DataPtr()},
        {STATS_UID, lse->DataPtr()},
        {dQ_UID, dq->DataPtr()},
        {dK_UID, dk->DataPtr()},
        {dV_UID, dv->DataPtr()},
    };
    if (attn_mask) {
        variant_pack[MASK_UID] = attn_mask->DataPtr();
    }

    auto exec_status = graph->execute(handle, variant_pack, workspace);
    check_fe_status(exec_status, "graph->execute (backward)");

    return {dq, dk, dv};
}

}


#define REGISTER_CUDA_LINEAR_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LINEAR_KERNEL(ScaledDotProductAttentionBackward)
REGISTER_CUDA_LINEAR_KERNEL(ScaledDotProductAttentionForward)

#undef REGISTER_CUDA_LINEAR_KERNEL
