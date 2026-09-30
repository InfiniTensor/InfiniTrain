#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Public training contracts depend only on forward declarations. In particular,
// including them from PP must not pull TransformerModel back into the scheduler.
namespace infini_train {
class Dataset;
class Device;
class Tensor;
namespace nn {
class Module;
class TransformerModel;
struct TransformerConfig;
} // namespace nn
} // namespace infini_train

namespace infini_train::training {

using TensorList = std::vector<std::shared_ptr<Tensor>>;
using LossFunction = std::function<std::shared_ptr<Tensor>(const TensorList &)>;

// Task forward returns activations and a deferred, microbatch-mean loss.
// The training executor owns loss scaling, backward and gradient synchronization.
struct ForwardStepResult {
    TensorList output;
    LossFunction loss_func;
};

// Configured step used by both ordinary training and pipeline execution. Under
// PP, inputs are tokens on the first chunk and activations on later chunks;
// target is supplied only to the last chunk, whose loss callback is evaluated.
using ForwardStepFunction
    = std::function<ForwardStepResult(nn::Module &, const TensorList &, const std::shared_ptr<Tensor> &target)>;

// Parsed once before worker threads start; providers must not mutate CLI flags.
struct TrainingOptions {
    uint32_t batch_size = 4;
    uint32_t sequence_length = 64;
    uint32_t total_batch_size = 256;
    uint32_t num_iteration = 10;
    double learning_rate = 1e-5;
    int32_t zero_stage = 0;
    double min_lr = 0.0;
    std::string lr_decay_style = "constant";
    int64_t lr_warmup_iters = 0;
    double lr_warmup_init = 0.0;
    int64_t lr_decay_iters = 0;
    uint32_t val_loss_every = 0;
    uint32_t sample_every = 0;
    bool overfit_single_batch = true;
    std::string device = "cuda";
    int32_t nthread_per_process = 1;
    uint32_t tensor_parallel = 1;
    bool sequence_parallel = false;
    uint32_t pipeline_parallel = 1;
    uint32_t virtual_pipeline_parallel = 1;
    std::string dtype = "float32";
    std::string precision_check;
    uint32_t save_interval = 0;
    std::string load = "";
    std::string save = "";
    uint32_t max_checkpoint_keep = 3;
    bool load_optimizer_state = true;
    bool save_optimizer_state = true;
    int32_t lora_rank = 0;
    double lora_alpha = 16.0;
    std::string lora_target_modules = "c_attn,c_proj,c_fc,c_fc2";
    std::string lora_save_path = "";
    std::string lora_load_path = "";
    std::string optimizer = "adam";
    std::string profile_name = "pretrain";
    // Some backends require synchronization after model upload and before teardown.
    bool synchronize_device = false;
};

struct DatasetSplits {
    std::shared_ptr<Dataset> train;
    std::shared_ptr<Dataset> valid;
};

// Called once per rank, after TP/PP thread-local state and process groups exist.
// The provider returns the local TransformerModel on CPU, without DDP/PP wrappers.
// TransformerModel constructs all local VPP chunks using the existing global layout.
using ModelProvider = std::function<std::shared_ptr<nn::TransformerModel>()>;
// The argument is the total requested number of global training samples (not tokens).
// A finite dataset may be returned: the trainer cycles it and restores its position.
using DatasetProvider = std::function<DatasetSplits(size_t)>;
// Called for every microbatch (and every local chunk under PP). Only the last
// chunk's loss callback is evaluated. The config is from the loaded model.
using ForwardStep = std::function<ForwardStepResult(nn::Module &, const TensorList &, const std::shared_ptr<Tensor> &,
                                                    const nn::TransformerConfig &)>;
// Optional reporting/sampling hook, called on the last global rank after each step.
using AfterStep = std::function<void(nn::Module &, const Device &, int64_t)>;

// Owns initialization, model wrapping, optimizer/scheduler, data iterators,
// checkpoint resume/save and forward/backward scheduling. Providers own task logic.
void Pretrain(const TrainingOptions &options, const DatasetProvider &dataset_provider,
              const ModelProvider &model_provider, const ForwardStep &forward_step, const AfterStep &after_step = {});

} // namespace infini_train::training
