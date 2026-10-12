#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "example/fm9gv/checkpoint_loader.h"
#include "example/fm9gv/model.h"
#include "example/fm9gv/multimodal_model.h"
#include "example/fm9gv/resampler.h"
#include "example/fm9gv/vision_encoder.h"

DEFINE_string(fm9gv_filepath, "", "converted FM9G text checkpoint");
DEFINE_string(sample_bin, "", "multimodal alignment sample produced by prepare_fm9gv_multimodal_alignment.py");
DEFINE_string(device, "dcu", "device type (cpu/cuda/dcu)");
DEFINE_string(dtype, "bfloat16", "precision (float32/bfloat16)");
DEFINE_bool(backward, true, "compute the gradient with respect to visual embeddings");
DEFINE_string(vision_grad_bin, "", "optional path for the FP32 visual-embedding gradient");
DEFINE_string(resampler_filepath, "", "optional converted FM9GV Resampler checkpoint");
DEFINE_string(vpm_sample_bin, "", "optional VPM-output sample to run through the Resampler");
DEFINE_string(vpm_grad_bin, "", "optional path for the FP32 VPM-output gradient");
DEFINE_string(vision_filepath, "", "optional converted FM9GV Vision Transformer checkpoint");
DEFINE_string(vpm_input_bin, "", "optional raw VPM pixel-input sample");
DEFINE_string(pixel_grad_bin, "", "optional path for the FP32 VPM pixel-input gradient");
DEFINE_string(vision_output_bin, "", "optional path for the FP32 final visual embeddings");
DEFINE_bool(tune_vision, true, "update Vision Encoder parameters");
DEFINE_bool(tune_llm, false, "update LLM parameters");
DEFINE_int32(num_iterations, 1, "number of forward/backward iterations over the alignment sample");
DEFINE_int32(warmup_iterations, 0, "initial training iterations excluded from benchmark aggregates");
DEFINE_double(learning_rate, 0.0, "Adam learning rate; zero disables parameter updates");
DEFINE_double(adam_beta2, 0.95, "Adam beta2 used by the official FM9GV training script");
DEFINE_double(weight_decay, 0.1, "decoupled AdamW weight decay");
DEFINE_double(max_grad_norm, 1.0, "global L2 gradient clipping threshold; zero disables clipping");
DEFINE_double(gradient_scale, 1.0, "additional diagnostic scale applied after gradient clipping");
DEFINE_bool(fp32_master_weights, true, "maintain FP32 master weights and optimizer states for BF16 training");
DEFINE_bool(autocast, true, "enable InfiniTrain autocast around the model forward");
DEFINE_string(metrics_json, "", "optional JSON path for per-step loss, timing, gradient norm and memory metrics");
DEFINE_string(parameter_output_name, "", "optional named parameter to dump after the final iteration");
DEFINE_string(parameter_output_bin, "", "optional path for the selected parameter in FP32");
DEFINE_string(parameter_initial_output_name, "", "optional named parameter to dump before the first iteration");
DEFINE_string(parameter_initial_output_bin, "", "optional path for the initial selected parameter in FP32");
DEFINE_string(parameter_grad_output_name, "", "optional named parameter whose gradient is dumped after backward");
DEFINE_string(parameter_grad_output_bin, "", "optional path for the selected parameter gradient in FP32");

using namespace infini_train;

namespace {
constexpr char kMagic[8] = {'F', 'M', '9', 'M', 'M', '0', '0', '1'};
constexpr char kVpmMagic[8] = {'F', 'M', '9', 'V', 'P', '0', '0', '1'};
constexpr char kVpmInputMagic[8] = {'F', 'M', '9', 'V', 'I', '0', '0', '1'};

struct SampleHeader {
    char magic[8];
    int64_t batch;
    int64_t sequence;
    int64_t visual_tokens;
    int64_t hidden;
};

struct Sample {
    std::shared_ptr<Tensor> input_ids;
    std::shared_ptr<Tensor> labels;
    std::shared_ptr<Tensor> image_indices;
    std::shared_ptr<Tensor> vision_embeddings;
};

struct VpmSample {
    std::shared_ptr<Tensor> features;
    std::shared_ptr<Tensor> patch_sizes;
};

struct VpmInput {
    std::shared_ptr<Tensor> pixels;
    std::shared_ptr<Tensor> patch_sizes;
};

class FP32MasterAdamW final : public Optimizer {
public:
    FP32MasterAdamW(const NamedParameterList &named_params, float learning_rate, float beta1, float beta2, float eps,
                    float weight_decay)
        : Optimizer(named_params, learning_rate), weight_decay_(weight_decay), beta1_(beta1), beta2_(beta2), eps_(eps) {
        master_params_.reserve(named_params.size());
        m_.reserve(named_params.size());
        v_.reserve(named_params.size());
        apply_weight_decay_.reserve(named_params.size());
        for (const auto &[name, parameter] : named_params) {
            auto master = std::make_shared<Tensor>(parameter->To(DataType::kFLOAT32));
            master->set_requires_grad(false);
            master_params_.push_back(master);
            m_.push_back(std::make_shared<Tensor>(parameter->Dims(), DataType::kFLOAT32, parameter->GetDevice()));
            v_.push_back(std::make_shared<Tensor>(parameter->Dims(), DataType::kFLOAT32, parameter->GetDevice()));
            m_.back()->Fill(0.0f);
            v_.back()->Fill(0.0f);
            // Match the standard Trainer AdamW grouping: matrix-like weights
            // decay, while bias and normalization vectors do not.
            apply_weight_decay_.push_back(parameter->Dims().size() > 1);
        }
    }

    void ZeroGrad(bool set_to_none = true) override { Optimizer::ZeroGrad(set_to_none); }

    void SetGradientScale(float scale) { gradient_scale_ = scale; }

    void Step() override {
        ++t_;
        CHECK(!params_.empty());
        const auto device = params_.front()->GetDevice();
        core::DeviceGuard guard(device);
        const auto &kernel = Dispatcher::Instance().GetKernel({device.type(), "AdamWFP32MasterStep"});
        for (size_t i = 0; i < params_.size(); ++i) {
            const auto &grad = params_[i]->grad();
            if (!grad) {
                continue;
            }
            CHECK(params_[i]->GetDevice() == device);
            kernel.Call<void>(grad, params_[i], master_params_[i], m_[i], v_[i], learning_rate_, beta1_, beta2_, eps_,
                              t_, gradient_scale_, apply_weight_decay_[i] ? weight_decay_ : 0.0f);
        }
    }

private:
    float weight_decay_;
    float beta1_;
    float beta2_;
    float eps_;
    float gradient_scale_ = 1.0f;
    int64_t t_ = 0;
    std::vector<std::shared_ptr<Tensor>> master_params_;
    std::vector<std::shared_ptr<Tensor>> m_;
    std::vector<std::shared_ptr<Tensor>> v_;
    std::vector<bool> apply_weight_decay_;
};

double GlobalGradNorm(const NamedParameterList &named_params) {
    std::shared_ptr<Tensor> first_grad = nullptr;
    for (const auto &[name, parameter] : named_params) {
        if (parameter->grad()) {
            first_grad = parameter->grad();
            break;
        }
    }
    CHECK(first_grad != nullptr) << "No gradients available for global norm";
    const auto device = first_grad->GetDevice();
    auto total_square = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32, device);
    total_square->Fill(0.0f);
    const auto &norm_kernel = Dispatcher::Instance().GetKernel({device.type(), "L2NormAccumulate"});
    for (const auto &[name, parameter] : named_params) {
        const auto &grad = parameter->grad();
        if (grad) {
            norm_kernel.Call<void>(grad, total_square);
        }
    }
    auto total_cpu = total_square->To(Device());
    core::GetDeviceGuardImpl(device.type())
        ->SynchronizeStream(core::GetDeviceGuardImpl(device.type())->GetStream(device));
    return std::sqrt(static_cast<const float *>(total_cpu.DataPtr())[0]);
}

void ScaleGradients(const NamedParameterList &named_params, float scale) {
    if (scale == 1.0f) {
        return;
    }
    for (const auto &[name, parameter] : named_params) {
        if (parameter->grad()) {
            parameter->set_grad(parameter->grad()->Mul(scale));
        }
    }
}

void ReadExact(std::ifstream &stream, void *data, size_t size, const char *name) {
    stream.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
    CHECK(stream) << "Failed to read " << name;
}

Sample LoadSample(const std::string &path) {
    std::ifstream stream(path, std::ios::binary);
    CHECK(stream) << "Failed to open sample: " << path;
    SampleHeader header{};
    ReadExact(stream, &header, sizeof(header), "sample header");
    CHECK_EQ(std::memcmp(header.magic, kMagic, sizeof(kMagic)), 0) << "Invalid multimodal sample magic";
    CHECK_GT(header.batch, 0);
    CHECK_GT(header.sequence, 0);
    CHECK_GT(header.visual_tokens, 0);
    CHECK_GT(header.hidden, 0);

    Sample sample;
    sample.input_ids = std::make_shared<Tensor>(std::vector<int64_t>{header.batch, header.sequence}, DataType::kINT64);
    sample.labels = std::make_shared<Tensor>(std::vector<int64_t>{header.batch, header.sequence}, DataType::kINT64);
    sample.image_indices
        = std::make_shared<Tensor>(std::vector<int64_t>{header.batch, header.visual_tokens}, DataType::kINT64);
    sample.vision_embeddings = std::make_shared<Tensor>(
        std::vector<int64_t>{header.batch, header.visual_tokens, header.hidden}, DataType::kFLOAT32);
    ReadExact(stream, sample.input_ids->DataPtr(), sample.input_ids->SizeInBytes(), "input_ids");
    ReadExact(stream, sample.labels->DataPtr(), sample.labels->SizeInBytes(), "labels");
    ReadExact(stream, sample.image_indices->DataPtr(), sample.image_indices->SizeInBytes(), "image_indices");
    ReadExact(stream, sample.vision_embeddings->DataPtr(), sample.vision_embeddings->SizeInBytes(),
              "vision_embeddings");
    CHECK_EQ(stream.peek(), std::ifstream::traits_type::eof()) << "Unexpected trailing data in sample";
    return sample;
}

VpmSample LoadVpmSample(const std::string &path) {
    std::ifstream stream(path, std::ios::binary);
    CHECK(stream) << "Failed to open VPM sample: " << path;
    char magic[8];
    int64_t dims[5];
    ReadExact(stream, magic, sizeof(magic), "VPM magic");
    CHECK_EQ(std::memcmp(magic, kVpmMagic, sizeof(kVpmMagic)), 0);
    ReadExact(stream, dims, sizeof(dims), "VPM dimensions");
    VpmSample sample;
    sample.features = std::make_shared<Tensor>(std::vector<int64_t>{dims[0], dims[1], dims[4]}, DataType::kFLOAT32);
    ReadExact(stream, sample.features->DataPtr(), sample.features->SizeInBytes(), "VPM features");
    sample.patch_sizes = std::make_shared<Tensor>(std::vector<int64_t>{dims[0], 2}, DataType::kINT64);
    auto *patch_sizes = static_cast<int64_t *>(sample.patch_sizes->DataPtr());
    for (int64_t batch = 0; batch < dims[0]; ++batch) {
        patch_sizes[2 * batch] = dims[2];
        patch_sizes[2 * batch + 1] = dims[3];
    }
    CHECK_EQ(stream.peek(), std::ifstream::traits_type::eof());
    return sample;
}

VpmInput LoadVpmInput(const std::string &path) {
    std::ifstream stream(path, std::ios::binary);
    CHECK(stream) << "Failed to open VPM input: " << path;
    char magic[8];
    int64_t dims[6];
    ReadExact(stream, magic, sizeof(magic), "VPM input magic");
    CHECK_EQ(std::memcmp(magic, kVpmInputMagic, sizeof(kVpmInputMagic)), 0);
    ReadExact(stream, dims, sizeof(dims), "VPM input dimensions");
    VpmInput input;
    input.pixels
        = std::make_shared<Tensor>(std::vector<int64_t>{dims[0], dims[1], dims[2], dims[3]}, DataType::kFLOAT32);
    ReadExact(stream, input.pixels->DataPtr(), input.pixels->SizeInBytes(), "VPM pixel values");
    input.patch_sizes = std::make_shared<Tensor>(std::vector<int64_t>{dims[0], 2}, DataType::kINT64);
    auto *sizes = static_cast<int64_t *>(input.patch_sizes->DataPtr());
    for (int64_t batch = 0; batch < dims[0]; ++batch) {
        sizes[2 * batch] = dims[4];
        sizes[2 * batch + 1] = dims[5];
    }
    CHECK_EQ(stream.peek(), std::ifstream::traits_type::eof());
    return input;
}

Device ParseDevice() {
    if (FLAGS_device == "cpu") {
        return Device();
    }
    if (FLAGS_device == "cuda") {
        return Device(Device::DeviceType::kCUDA, 0);
    }
    if (FLAGS_device == "dcu") {
        return Device(Device::DeviceType::kDCU, 0);
    }
    LOG(FATAL) << "Unsupported device: " << FLAGS_device;
}

DataType ParseDtype() {
    if (FLAGS_dtype == "float32") {
        return DataType::kFLOAT32;
    }
    if (FLAGS_dtype == "bfloat16") {
        return DataType::kBFLOAT16;
    }
    LOG(FATAL) << "Unsupported dtype: " << FLAGS_dtype;
}

void WriteTensorFP32(const std::string &path, const std::shared_ptr<Tensor> &tensor) {
    auto fp32
        = tensor->Dtype() == DataType::kFLOAT32 ? tensor : std::make_shared<Tensor>(tensor->To(DataType::kFLOAT32));
    auto cpu
        = fp32->GetDevice().type() == Device::DeviceType::kCPU ? fp32 : std::make_shared<Tensor>(fp32->To(Device()));
    std::ofstream stream(path, std::ios::binary);
    CHECK(stream) << "Failed to open output: " << path;
    stream.write(static_cast<const char *>(cpu->DataPtr()), static_cast<std::streamsize>(cpu->SizeInBytes()));
    CHECK(stream) << "Failed to write output: " << path;
}
} // namespace

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    CHECK(!FLAGS_fm9gv_filepath.empty());
    CHECK(!FLAGS_sample_bin.empty());

    nn::parallel::global::InitAllEnv(1, 1, false, 1, 1);
    const Device device = ParseDevice();
    const DataType dtype = ParseDtype();
    auto sample = LoadSample(FLAGS_sample_bin);
    auto model = fm9gv::LoadFromFM9GBin(FLAGS_fm9gv_filepath);
    CHECK_EQ(sample.vision_embeddings->Dims()[2], model->Config().n_embd);
    for (const auto &parameter : model->Parameters()) { parameter->set_requires_grad(FLAGS_tune_llm); }
    model->To(dtype);
    model->To(device);

    sample.input_ids = std::make_shared<Tensor>(sample.input_ids->To(device));
    sample.labels = std::make_shared<Tensor>(sample.labels->To(device));
    sample.image_indices = std::make_shared<Tensor>(sample.image_indices->To(device));
    sample.vision_embeddings = std::make_shared<Tensor>(sample.vision_embeddings->To(dtype));
    sample.vision_embeddings = std::make_shared<Tensor>(sample.vision_embeddings->To(device))->RequiresGrad();

    std::shared_ptr<fm9gv::Resampler> resampler;
    std::shared_ptr<fm9gv::VisionTransformer> vision_model;
    std::shared_ptr<fm9gv::MultimodalModel> multimodal_model;
    std::shared_ptr<Tensor> vpm_features;
    std::shared_ptr<Tensor> pixel_values;
    std::shared_ptr<Tensor> patch_sizes;
    if (!FLAGS_vpm_input_bin.empty()) {
        CHECK(FLAGS_vpm_sample_bin.empty()) << "Use either --vpm_input_bin or --vpm_sample_bin";
        CHECK(!FLAGS_vision_filepath.empty());
        CHECK(!FLAGS_resampler_filepath.empty());
        auto vpm_input = LoadVpmInput(FLAGS_vpm_input_bin);
        pixel_values = std::make_shared<Tensor>(vpm_input.pixels->To(dtype));
        pixel_values = std::make_shared<Tensor>(pixel_values->To(device))->RequiresGrad();
        patch_sizes = vpm_input.patch_sizes;
        vision_model = std::make_shared<fm9gv::VisionTransformer>();
        vision_model->LoadFromBin(FLAGS_vision_filepath);
        vision_model->To(dtype);
        vision_model->To(device);
        resampler = std::make_shared<fm9gv::Resampler>();
        resampler->LoadFromBin(FLAGS_resampler_filepath);
        resampler->To(dtype);
        resampler->To(device);
        multimodal_model = std::make_shared<fm9gv::MultimodalModel>(model, vision_model, resampler);
        multimodal_model->SetLLMRequiresGrad(FLAGS_tune_llm);
        multimodal_model->SetVisionRequiresGrad(FLAGS_tune_vision);
    } else if (!FLAGS_vpm_sample_bin.empty()) {
        CHECK(!FLAGS_resampler_filepath.empty());
        auto vpm_sample = LoadVpmSample(FLAGS_vpm_sample_bin);
        vpm_features = std::make_shared<Tensor>(vpm_sample.features->To(dtype));
        vpm_features = std::make_shared<Tensor>(vpm_features->To(device))->RequiresGrad();
        patch_sizes = vpm_sample.patch_sizes;
        resampler = std::make_shared<fm9gv::Resampler>();
        resampler->LoadFromBin(FLAGS_resampler_filepath);
        resampler->To(dtype);
        resampler->To(device);
    }

    CHECK_GT(FLAGS_num_iterations, 0);
    CHECK_GE(FLAGS_warmup_iterations, 0);
    CHECK_LT(FLAGS_warmup_iterations, FLAGS_num_iterations);
    CHECK_GE(FLAGS_learning_rate, 0.0);
    CHECK_GE(FLAGS_max_grad_norm, 0.0);
    CHECK_EQ(FLAGS_parameter_initial_output_name.empty(), FLAGS_parameter_initial_output_bin.empty());
    CHECK_EQ(FLAGS_parameter_output_name.empty(), FLAGS_parameter_output_bin.empty());
    CHECK_EQ(FLAGS_parameter_grad_output_name.empty(), FLAGS_parameter_grad_output_bin.empty());
    std::shared_ptr<Optimizer> optimizer;
    std::shared_ptr<FP32MasterAdamW> fp32_master_optimizer;
    NamedParameterList trainable_parameters;
    if (FLAGS_backward && multimodal_model != nullptr) {
        for (const auto &[name, parameter] : multimodal_model->NamedParameters()) {
            if (parameter->requires_grad()) {
                trainable_parameters.emplace_back(name, parameter);
            }
        }
    }
    uint64_t trainable_numel = 0;
    for (const auto &[name, parameter] : trainable_parameters) { trainable_numel += parameter->NumElements(); }
    if (FLAGS_learning_rate > 0.0) {
        CHECK(FLAGS_backward) << "Parameter updates require --backward=true";
        CHECK(multimodal_model != nullptr)
            << "Multimodal parameter updates require raw pixels, Vision Encoder and Resampler checkpoints";
        CHECK(!trainable_parameters.empty());
        if (FLAGS_fp32_master_weights && dtype != DataType::kFLOAT32) {
            fp32_master_optimizer = std::make_shared<FP32MasterAdamW>(
                trainable_parameters, static_cast<float>(FLAGS_learning_rate), 0.9f,
                static_cast<float>(FLAGS_adam_beta2), 1e-8f, static_cast<float>(FLAGS_weight_decay));
            optimizer = fp32_master_optimizer;
        } else {
            optimizer
                = std::make_shared<optimizers::Adam>(trainable_parameters, static_cast<float>(FLAGS_learning_rate),
                                                     0.9f, static_cast<float>(FLAGS_adam_beta2), 1e-8f);
        }
        LOG(INFO) << "Optimizing " << trainable_parameters.size() << " multimodal parameters"
                  << " (" << trainable_numel << " elements)"
                  << " | tune_vision=" << FLAGS_tune_vision << " | tune_llm=" << FLAGS_tune_llm;
    }
    if (!FLAGS_parameter_initial_output_bin.empty()) {
        CHECK(multimodal_model != nullptr);
        bool found = false;
        for (const auto &[name, parameter] : multimodal_model->NamedParameters()) {
            if (name == FLAGS_parameter_initial_output_name) {
                WriteTensorFP32(FLAGS_parameter_initial_output_bin, parameter);
                found = true;
                break;
            }
        }
        CHECK(found) << "Unknown initial parameter: " << FLAGS_parameter_initial_output_name;
    }

    auto impl = core::GetDeviceGuardImpl(device.type());
    std::shared_ptr<Tensor> vision_embeddings;
    std::shared_ptr<Tensor> logits;
    std::shared_ptr<Tensor> loss;
    const bool capture_intermediate
        = !FLAGS_vision_output_bin.empty() || !FLAGS_vision_grad_bin.empty() || !FLAGS_vpm_grad_bin.empty();
    std::vector<double> step_losses;
    std::vector<double> step_elapsed_ms;
    std::vector<double> step_grad_norms;
    std::vector<double> step_clip_scales;
    std::vector<size_t> step_peak_used_mb;
    std::vector<size_t> step_peak_reserved_mb;
    std::shared_ptr<Tensor> selected_parameter_grad;
    for (int32_t step = 0; step < FLAGS_num_iterations; ++step) {
        // InfiniTrain keeps the backward graph reachable from output tensors.
        // Drop the previous iteration before measuring the next one so steady-
        // state memory does not include two overlapping graphs.
        loss.reset();
        logits.reset();
        vision_embeddings.reset();
        if (optimizer) {
            optimizer->ZeroGrad();
        } else {
            for (const auto &[name, parameter] : trainable_parameters) { parameter->ZeroGrad(); }
        }
        if (pixel_values) {
            pixel_values->ZeroGrad();
        }
        if (vpm_features) {
            vpm_features->ZeroGrad();
        }
        sample.vision_embeddings->ZeroGrad();
        impl->SynchronizeDevice(device);
        impl->ResetMemPoolHighWatermarks(device);
        const auto started = std::chrono::steady_clock::now();
        AutocastGuard autocast(device.type(), dtype);
        if (!FLAGS_autocast) {
            autocast.Disable();
        }
        if (multimodal_model && !capture_intermediate) {
            logits = (*multimodal_model)({sample.input_ids, pixel_values, patch_sizes, sample.image_indices})[0];
        } else if (multimodal_model) {
            vision_embeddings = multimodal_model->EncodeVision(pixel_values, patch_sizes);
            logits = multimodal_model->ForwardWithVisionHiddenStates(sample.input_ids, vision_embeddings,
                                                                     sample.image_indices)[0];
        } else {
            vision_embeddings = resampler ? (*resampler)({vpm_features, patch_sizes})[0] : sample.vision_embeddings;
            logits = (*model)({sample.input_ids, vision_embeddings, sample.image_indices})[0];
        }
        autocast.Disable();
        const int64_t vocab_size = logits->Dims().back();
        auto flat_logits = logits->View({static_cast<int64_t>(logits->NumElements()) / vocab_size, vocab_size});
        auto flat_labels = sample.labels->View({static_cast<int64_t>(sample.labels->NumElements())});
        loss = std::make_shared<nn::CrossEntropyLoss>(-100)->Forward({flat_logits, flat_labels})[0];
        if (FLAGS_backward) {
            loss->Backward();
        }
        if (step == FLAGS_num_iterations - 1 && !FLAGS_parameter_grad_output_bin.empty()) {
            CHECK(multimodal_model != nullptr);
            for (const auto &[name, parameter] : multimodal_model->NamedParameters()) {
                if (name == FLAGS_parameter_grad_output_name) {
                    CHECK(parameter->grad() != nullptr) << "Parameter has no gradient: " << name;
                    selected_parameter_grad = parameter->grad();
                    break;
                }
            }
            CHECK(selected_parameter_grad != nullptr) << "Unknown parameter: " << FLAGS_parameter_grad_output_name;
        }
        double grad_norm = 0.0;
        float clip_scale = 1.0f;
        if (optimizer) {
            grad_norm = GlobalGradNorm(trainable_parameters);
            if (FLAGS_max_grad_norm > 0.0) {
                clip_scale = std::min(1.0f, static_cast<float>(FLAGS_max_grad_norm / (grad_norm + 1e-6)));
            }
            const float step_gradient_scale = clip_scale * static_cast<float>(FLAGS_gradient_scale);
            if (fp32_master_optimizer) {
                fp32_master_optimizer->SetGradientScale(step_gradient_scale);
            } else {
                ScaleGradients(trainable_parameters, step_gradient_scale);
            }
            optimizer->Step();
        }
        impl->SynchronizeDevice(device);
        auto loss_cpu = loss->To(Device());
        const auto finished = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(finished - started).count();
        size_t peak_used_mb = 0;
        size_t peak_reserved_mb = 0;
        std::tie(peak_used_mb, peak_reserved_mb) = impl->GetMemPoolPeakMB(device);
        const double loss_value = static_cast<const float *>(loss_cpu.DataPtr())[0];
        step_losses.push_back(loss_value);
        step_elapsed_ms.push_back(elapsed_ms);
        step_grad_norms.push_back(grad_norm);
        step_clip_scales.push_back(clip_scale);
        step_peak_used_mb.push_back(peak_used_mb);
        step_peak_reserved_mb.push_back(peak_reserved_mb);
        LOG(ERROR) << "step " << step + 1 << "/" << FLAGS_num_iterations << " | phase "
                   << (step < FLAGS_warmup_iterations ? "warmup" : "measure") << " | multimodal loss " << loss_value
                   << " | grad norm " << grad_norm << " | clip scale " << clip_scale << " | " << elapsed_ms
                   << " ms | peak used " << peak_used_mb << " MB | peak reserved " << peak_reserved_mb << " MB";
    }
    const auto measured_begin = step_elapsed_ms.begin() + FLAGS_warmup_iterations;
    const double measured_total = std::accumulate(measured_begin, step_elapsed_ms.end(), 0.0);
    const double measured_mean = measured_total / (FLAGS_num_iterations - FLAGS_warmup_iterations);
    const auto peak_begin = step_peak_used_mb.begin() + FLAGS_warmup_iterations;
    const auto reserved_begin = step_peak_reserved_mb.begin() + FLAGS_warmup_iterations;
    const size_t measured_peak_used = *std::max_element(peak_begin, step_peak_used_mb.end());
    const size_t measured_peak_reserved = *std::max_element(reserved_begin, step_peak_reserved_mb.end());
    LOG(ERROR) << "benchmark summary | measured iterations " << FLAGS_num_iterations - FLAGS_warmup_iterations
               << " | mean " << measured_mean << " ms | peak used " << measured_peak_used << " MB | peak reserved "
               << measured_peak_reserved << " MB";
    if (!FLAGS_metrics_json.empty()) {
        std::ofstream metrics(FLAGS_metrics_json);
        CHECK(metrics) << "Failed to open metrics output: " << FLAGS_metrics_json;
        metrics << "{\n  \"dtype\": \"" << FLAGS_dtype << "\",\n"
                << "  \"iterations\": " << FLAGS_num_iterations << ",\n"
                << "  \"warmup_iterations\": " << FLAGS_warmup_iterations << ",\n"
                << "  \"learning_rate\": " << FLAGS_learning_rate << ",\n"
                << "  \"adam_beta2\": " << FLAGS_adam_beta2 << ",\n"
                << "  \"weight_decay\": " << FLAGS_weight_decay << ",\n"
                << "  \"max_grad_norm\": " << FLAGS_max_grad_norm << ",\n"
                << "  \"gradient_scale\": " << FLAGS_gradient_scale << ",\n"
                << "  \"trainable_parameters\": " << trainable_numel << ",\n"
                << "  \"fp32_master_weights\": "
                << ((FLAGS_fp32_master_weights && dtype != DataType::kFLOAT32) ? "true" : "false") << ",\n"
                << "  \"tune_vision\": " << (FLAGS_tune_vision ? "true" : "false") << ",\n"
                << "  \"tune_llm\": " << (FLAGS_tune_llm ? "true" : "false") << ",\n"
                << "  \"mean_elapsed_ms\": " << measured_mean << ",\n"
                << "  \"peak_used_mb\": " << measured_peak_used << ",\n"
                << "  \"peak_reserved_mb\": " << measured_peak_reserved << ",\n";
        const auto write_array = [&](const char *name, const auto &values, bool last) {
            metrics << "  \"" << name << "\": [";
            for (size_t index = 0; index < values.size(); ++index) {
                if (index > 0) {
                    metrics << ", ";
                }
                metrics << values[index];
            }
            metrics << "]" << (last ? "\n" : ",\n");
        };
        write_array("losses", step_losses, false);
        write_array("elapsed_ms", step_elapsed_ms, false);
        write_array("grad_norms", step_grad_norms, false);
        write_array("clip_scales", step_clip_scales, false);
        write_array("peak_used_mb_per_step", step_peak_used_mb, false);
        write_array("peak_reserved_mb_per_step", step_peak_reserved_mb, true);
        metrics << "}\n";
        CHECK(metrics);
    }
    if (!FLAGS_vision_output_bin.empty()) {
        WriteTensorFP32(FLAGS_vision_output_bin, vision_embeddings);
    }
    if (FLAGS_backward && !FLAGS_vision_grad_bin.empty()) {
        CHECK(vision_embeddings->grad() != nullptr);
        WriteTensorFP32(FLAGS_vision_grad_bin, vision_embeddings->grad());
        LOG(INFO) << "Wrote visual embedding gradient to " << FLAGS_vision_grad_bin;
    }
    if (FLAGS_backward && !FLAGS_vpm_grad_bin.empty()) {
        CHECK(vpm_features != nullptr);
        CHECK(vpm_features->grad() != nullptr);
        WriteTensorFP32(FLAGS_vpm_grad_bin, vpm_features->grad());
        LOG(INFO) << "Wrote VPM-output gradient to " << FLAGS_vpm_grad_bin;
    }
    if (FLAGS_backward && !FLAGS_pixel_grad_bin.empty()) {
        CHECK(pixel_values != nullptr);
        CHECK(pixel_values->grad() != nullptr);
        WriteTensorFP32(FLAGS_pixel_grad_bin, pixel_values->grad());
        LOG(INFO) << "Wrote VPM pixel-input gradient to " << FLAGS_pixel_grad_bin;
    }
    if (!FLAGS_parameter_output_bin.empty()) {
        CHECK(multimodal_model != nullptr);
        CHECK(!FLAGS_parameter_output_name.empty());
        bool found = false;
        for (const auto &[name, parameter] : multimodal_model->NamedParameters()) {
            if (name == FLAGS_parameter_output_name) {
                WriteTensorFP32(FLAGS_parameter_output_bin, parameter);
                found = true;
                break;
            }
        }
        CHECK(found) << "Unknown parameter: " << FLAGS_parameter_output_name;
    }
    if (!FLAGS_parameter_grad_output_bin.empty()) {
        CHECK(selected_parameter_grad != nullptr);
        WriteTensorFP32(FLAGS_parameter_grad_output_bin, selected_parameter_grad);
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();
    return 0;
}
