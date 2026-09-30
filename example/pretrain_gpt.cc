// Out-of-tree providers register before parsing --device.
#ifdef INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_HEADER
#include INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_HEADER
#endif

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "example/common/tiny_shakespeare_dataset.h"
#include "example/common/tokenizer.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/training/arguments.h"
#include "infini_train/include/training/model_provider_registry.h"
#include "infini_train/include/training/training.h"

DEFINE_string(input_bin, "", "input .bin to train on");
DEFINE_string(input_val_bin, "", "input .bin for validation (evaluation is not implemented yet)");
DEFINE_string(tokenizer_bin, "", "input tokenizer .bin");
DEFINE_string(llmc_filepath, "", "LLMC weights to initialize the model; --load resumes training state");
DEFINE_string(model, "", "Registered model name; see --help for available models");
DEFINE_uint32(freq_generate_txt, 10, "text generation frequency; 0 disables generation");
DEFINE_uint32(text_length, 64, "number of tokens to generate");

using namespace infini_train;

namespace {

training::ForwardStepResult ForwardStepGPT(nn::Module &model, const training::TensorList &inputs,
                                           const std::shared_ptr<Tensor> &target, const nn::TransformerConfig &config) {
    auto output = model(inputs);
    if (!target) {
        return {std::move(output), {}};
    }
    std::shared_ptr<nn::Module> loss_fn;
    if (nn::parallel::global::GetTensorParallelSize() > 1) {
        loss_fn = std::make_shared<nn::parallel::VocabParallelCrossEntropyLoss>(config.original_vocab_size);
    } else {
        loss_fn = std::make_shared<nn::CrossEntropyLoss>();
    }
    return {std::move(output), [loss_fn, target](const training::TensorList &logits) {
                return (*loss_fn)({logits.at(0), target}).at(0);
            }};
}

} // namespace

int main(int argc, char *argv[]) {
#ifdef INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_REGISTRAR
    INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_REGISTRAR();
#endif
    const auto &models = training::ModelProviderRegistry::Instance();
    gflags::SetUsageMessage("GPT pretraining. Available models: " + models.Names()
                            + "\nUse --model=<name> or --flagfile=example/<model>/train.flags.");
#ifdef INFINITRAIN_GPT_DEFAULT_FLAGFILE
    training::ParseTrainingFlags(&argc, &argv, INFINITRAIN_GPT_DEFAULT_FLAGFILE);
#else
    training::ParseTrainingFlags(&argc, &argv);
#endif
    google::InitGoogleLogging(argv[0]);

    const std::string model_name = FLAGS_model;
    CHECK(!model_name.empty()) << "--model is required; pass it directly or use --flagfile=example/<model>/train.flags";
    const auto model_provider = models.Resolve(model_name, FLAGS_llmc_filepath);
    CHECK(!FLAGS_input_bin.empty()) << "--input_bin is required";
    auto options = training::TrainingOptionsFromFlags();
#ifdef INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_WORKAROUNDS
    options.synchronize_device = INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_WORKAROUNDS
                              && Device::ParseType(options.device) == Device::DeviceType::kPrivateUse1;
#endif

    const std::string train_path = FLAGS_input_bin;
    const std::string valid_path = FLAGS_input_val_bin;
    const auto dataset_provider = [train_path, valid_path, sequence_length = options.sequence_length](size_t) {
        training::DatasetSplits datasets;
        datasets.train = std::make_shared<TinyShakespeareDataset>(train_path, sequence_length);
        if (!valid_path.empty()) {
            datasets.valid = std::make_shared<TinyShakespeareDataset>(valid_path, sequence_length);
        }
        return datasets;
    };

    std::shared_ptr<Tokenizer> tokenizer;
    if (!FLAGS_tokenizer_bin.empty() && FLAGS_freq_generate_txt > 0) {
        CHECK_EQ(options.pipeline_parallel, 1) << "Text generation does not support PP";
        tokenizer = std::make_shared<Tokenizer>(FLAGS_tokenizer_bin);
    }
    const auto after_step = [tokenizer, frequency = FLAGS_freq_generate_txt, text_length = FLAGS_text_length,
                             &options](nn::Module &model, const Device &device, int64_t step) {
        if (tokenizer && step % frequency == 0) {
            tokenizer->GenerateText(model, options.batch_size, options.sequence_length, text_length, device);
        }
    };

    training::Pretrain(options, dataset_provider, model_provider, ForwardStepGPT, after_step);

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();
#ifdef INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_WORKAROUNDS
    // Preserve the external backend's existing static-destruction workaround.
    if (INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_WORKAROUNDS
        && Device::ParseType(options.device).value() == Device::DeviceType::kPrivateUse1) {
        std::_Exit(0);
    }
#endif
    return 0;
}
