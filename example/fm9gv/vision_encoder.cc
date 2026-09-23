#include "example/fm9gv/vision_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/tensor.h"

namespace fm9gv {

using infini_train::DataType;
using infini_train::Device;
using infini_train::Tensor;
namespace nn = infini_train::nn;

namespace {
constexpr char kMagic[8] = {'F', 'M', '9', 'V', 'E', '0', '0', '1'};

void ReadExact(std::ifstream &stream, void *data, size_t bytes, const std::string &name) {
    stream.read(static_cast<char *>(data), static_cast<std::streamsize>(bytes));
    CHECK(stream) << "Failed to read Vision Transformer tensor " << name;
}

void ReadTensor(std::ifstream &stream, const std::shared_ptr<Tensor> &tensor, const std::string &name) {
    CHECK(tensor->Dtype() == DataType::kFLOAT32);
    ReadExact(stream, tensor->DataPtr(), tensor->SizeInBytes(), name);
}
} // namespace

VisionEmbeddings::VisionEmbeddings(const VisionConfig &config) : CloneableModule(kType), config_(config) {
    modules_["patch_embedding"]
        = std::make_shared<nn::Linear>(config.num_channels * config.patch_size * config.patch_size, config.hidden_size);
    const int64_t positions_per_side = config.image_size / config.patch_size;
    modules_["position_embedding"]
        = std::make_shared<nn::Embedding>(positions_per_side * positions_per_side, config.hidden_size);
}

std::vector<std::shared_ptr<Tensor>>
VisionEmbeddings::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK_EQ(inputs.size(), 2);
    const auto &pixels = inputs[0];
    const auto &target_sizes = inputs[1];
    CHECK_EQ(pixels->Dims().size(), 4);
    CHECK_EQ(pixels->Dims()[1], config_.num_channels);
    CHECK(target_sizes->Dtype() == DataType::kINT64);
    CHECK_EQ(target_sizes->NumElements(), pixels->Dims()[0] * 2);

    const int64_t batch = pixels->Dims()[0];
    const int64_t height = pixels->Dims()[2] / config_.patch_size;
    const int64_t width = pixels->Dims()[3] / config_.patch_size;
    CHECK_EQ(height * config_.patch_size, pixels->Dims()[2]);
    CHECK_EQ(width * config_.patch_size, pixels->Dims()[3]);

    auto patches = pixels->View(
        {batch, config_.num_channels, height, config_.patch_size, width, config_.patch_size});
    patches = patches->Transpose(1, 2)->Transpose(2, 4)->Transpose(3, 4)->Contiguous();
    patches = patches->View(
        {batch, height * width, config_.num_channels * config_.patch_size * config_.patch_size});
    auto embeddings = (*modules_["patch_embedding"])({patches})[0];

    auto sizes_cpu = target_sizes->GetDevice().type() == Device::DeviceType::kCPU
                         ? target_sizes
                         : std::make_shared<Tensor>(target_sizes->To(Device()));
    const auto *sizes = static_cast<const int64_t *>(sizes_cpu->DataPtr());
    auto position_ids = std::make_shared<Tensor>(std::vector<int64_t>{batch, height * width}, DataType::kINT64);
    auto *ids = static_cast<int64_t *>(position_ids->DataPtr());
    std::fill(ids, ids + position_ids->NumElements(), int64_t{0});
    const int64_t positions_per_side = config_.image_size / config_.patch_size;
    for (int64_t b = 0; b < batch; ++b) {
        const int64_t target_height = sizes[2 * b];
        const int64_t target_width = sizes[2 * b + 1];
        CHECK_LE(target_height * target_width, height * width)
            << "Target patch layout exceeds the padded patch sequence";
        for (int64_t row = 0; row < target_height; ++row) {
            const int64_t bucket_row = row * positions_per_side / target_height;
            for (int64_t col = 0; col < target_width; ++col) {
                const int64_t bucket_col = col * positions_per_side / target_width;
                ids[b * height * width + row * target_width + col]
                    = bucket_row * positions_per_side + bucket_col;
            }
        }
    }
    position_ids = std::make_shared<Tensor>(position_ids->To(pixels->GetDevice()));
    auto positions = (*modules_["position_embedding"])({position_ids})[0];
    return {embeddings + positions};
}

VisionAttention::VisionAttention(const VisionConfig &config) : CloneableModule(kType), config_(config) {
    modules_["k_proj"] = std::make_shared<nn::Linear>(config.hidden_size, config.hidden_size);
    modules_["v_proj"] = std::make_shared<nn::Linear>(config.hidden_size, config.hidden_size);
    modules_["q_proj"] = std::make_shared<nn::Linear>(config.hidden_size, config.hidden_size);
    modules_["out_proj"] = std::make_shared<nn::Linear>(config.hidden_size, config.hidden_size);
}

std::vector<std::shared_ptr<Tensor>>
VisionAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK(inputs.size() == 1 || inputs.size() == 2);
    const int64_t batch = inputs[0]->Dims()[0];
    const int64_t sequence = inputs[0]->Dims()[1];
    const int64_t head_dim = config_.hidden_size / config_.num_attention_heads;
    auto query = (*modules_["q_proj"])(inputs)[0]
                     ->View({batch, sequence, config_.num_attention_heads, head_dim})
                     ->Transpose(1, 2);
    auto key = (*modules_["k_proj"])(inputs)[0]
                   ->View({batch, sequence, config_.num_attention_heads, head_dim})
                   ->Transpose(1, 2);
    auto value = (*modules_["v_proj"])(inputs)[0]
                     ->View({batch, sequence, config_.num_attention_heads, head_dim})
                     ->Transpose(1, 2);
    auto attention = query->Matmul(key->Transpose(-2, -1)) * (1.0f / std::sqrt(static_cast<float>(head_dim)));
    if (inputs.size() == 2) { attention = attention + inputs[1]; }
    attention = nn::function::Softmax(attention, -1);
    auto output = attention->Matmul(value)->Transpose(1, 2)->Contiguous()->View({batch, sequence, config_.hidden_size});
    return (*modules_["out_proj"])({output});
}

VisionMLP::VisionMLP(const VisionConfig &config) : CloneableModule(kType) {
    modules_["fc1"] = std::make_shared<nn::Linear>(config.hidden_size, config.intermediate_size);
    modules_["activation_fn"] = std::make_shared<nn::NewGELU>();
    modules_["fc2"] = std::make_shared<nn::Linear>(config.intermediate_size, config.hidden_size);
}

std::vector<std::shared_ptr<Tensor>> VisionMLP::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    auto hidden = (*modules_["fc1"])(inputs)[0];
    hidden = (*modules_["activation_fn"])({hidden})[0];
    return (*modules_["fc2"])({hidden});
}

VisionEncoderLayer::VisionEncoderLayer(const VisionConfig &config) : CloneableModule(kType) {
    modules_["self_attn"] = std::make_shared<VisionAttention>(config);
    modules_["layer_norm1"] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.hidden_size}, config.layer_norm_eps);
    modules_["mlp"] = std::make_shared<VisionMLP>(config);
    modules_["layer_norm2"] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.hidden_size}, config.layer_norm_eps);
}

std::vector<std::shared_ptr<Tensor>>
VisionEncoderLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK(inputs.size() == 1 || inputs.size() == 2);
    auto hidden = inputs[0];
    auto normalized = (*modules_["layer_norm1"])({hidden})[0];
    hidden = hidden + (*modules_["self_attn"])(inputs.size() == 2
                                                    ? std::vector<std::shared_ptr<Tensor>>{normalized, inputs[1]}
                                                    : std::vector<std::shared_ptr<Tensor>>{normalized})[0];
    normalized = (*modules_["layer_norm2"])({hidden})[0];
    return {hidden + (*modules_["mlp"])({normalized})[0]};
}

VisionTransformer::VisionTransformer(const VisionConfig &config) : CloneableModule(kType), config_(config) {
    CHECK_EQ(config.hidden_size % config.num_attention_heads, 0);
    modules_["embeddings"] = std::make_shared<VisionEmbeddings>(config);
    std::vector<std::shared_ptr<nn::Module>> layers;
    layers.reserve(config.num_hidden_layers);
    for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
        layers.push_back(std::make_shared<VisionEncoderLayer>(config));
    }
    modules_["encoder"] = std::make_shared<nn::ModuleList>(std::move(layers));
    modules_["post_layernorm"]
        = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.hidden_size}, config.layer_norm_eps);
}

std::vector<std::shared_ptr<Tensor>>
VisionTransformer::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK_EQ(inputs.size(), 2) << "VisionTransformer expects {pixel_values, target_sizes}";
    auto hidden = (*modules_["embeddings"])(inputs)[0];
    const int64_t batch = hidden->Dims()[0];
    const int64_t sequence = hidden->Dims()[1];
    auto sizes_cpu = inputs[1]->GetDevice().type() == Device::DeviceType::kCPU
                         ? inputs[1]
                         : std::make_shared<Tensor>(inputs[1]->To(Device()));
    const auto *sizes = static_cast<const int64_t *>(sizes_cpu->DataPtr());
    bool has_padding = false;
    for (int64_t b = 0; b < batch; ++b) {
        has_padding |= sizes[2 * b] * sizes[2 * b + 1] < sequence;
    }
    std::shared_ptr<Tensor> attention_mask;
    if (has_padding) {
        attention_mask = std::make_shared<Tensor>(
            std::vector<int64_t>{batch, 1, sequence, sequence}, DataType::kFLOAT32);
        auto *mask = static_cast<float *>(attention_mask->DataPtr());
        for (int64_t b = 0; b < batch; ++b) {
            const int64_t valid = sizes[2 * b] * sizes[2 * b + 1];
            for (int64_t query = 0; query < sequence; ++query) {
                for (int64_t key = 0; key < sequence; ++key) {
                    mask[(b * sequence + query) * sequence + key]
                        = key < valid ? 0.0f : -std::numeric_limits<float>::infinity();
                }
            }
        }
        if (hidden->Dtype() != DataType::kFLOAT32) {
            attention_mask = std::make_shared<Tensor>(attention_mask->To(hidden->Dtype()));
        }
        attention_mask = std::make_shared<Tensor>(attention_mask->To(hidden->GetDevice()));
    }
    auto &layers = static_cast<nn::ModuleList &>(*modules_["encoder"]);
    for (auto &layer : layers) {
        hidden = (*layer)(attention_mask ? std::vector<std::shared_ptr<Tensor>>{hidden, attention_mask}
                                         : std::vector<std::shared_ptr<Tensor>>{hidden})[0];
    }
    return (*modules_["post_layernorm"])({hidden});
}

void VisionTransformer::LoadFromBin(const std::string &path) {
    CHECK(std::filesystem::exists(path)) << "Vision Transformer checkpoint not found: " << path;
    std::ifstream stream(path, std::ios::binary);
    char magic[8];
    ReadExact(stream, magic, sizeof(magic), "header");
    CHECK_EQ(std::memcmp(magic, kMagic, sizeof(kMagic)), 0);
    int64_t dims[7];
    float layer_norm_eps;
    ReadExact(stream, dims, sizeof(dims), "configuration");
    ReadExact(stream, &layer_norm_eps, sizeof(layer_norm_eps), "layer_norm_eps");
    CHECK_EQ(dims[0], config_.hidden_size);
    CHECK_EQ(dims[1], config_.intermediate_size);
    CHECK_EQ(dims[2], config_.num_hidden_layers);
    CHECK_EQ(dims[3], config_.num_attention_heads);
    CHECK_EQ(dims[4], config_.num_channels);
    CHECK_EQ(dims[5], config_.image_size);
    CHECK_EQ(dims[6], config_.patch_size);
    CHECK_EQ(layer_norm_eps, config_.layer_norm_eps);

    auto state = StateDict();
    const auto read = [&](const std::string &name) {
        auto iter = state.find(name);
        CHECK(iter != state.end()) << "Missing InfiniTrain VPM parameter " << name;
        ReadTensor(stream, iter->second, name);
    };
    read("embeddings.patch_embedding.weight");
    read("embeddings.patch_embedding.bias");
    read("embeddings.position_embedding.weight");
    for (int64_t i = 0; i < config_.num_hidden_layers; ++i) {
        const std::string prefix = "encoder." + std::to_string(i) + ".";
        for (const char *projection : {"k_proj", "v_proj", "q_proj", "out_proj"}) {
            read(prefix + "self_attn." + projection + ".weight");
            read(prefix + "self_attn." + projection + ".bias");
        }
        read(prefix + "layer_norm1.weight");
        read(prefix + "layer_norm1.bias");
        read(prefix + "mlp.fc1.weight");
        read(prefix + "mlp.fc1.bias");
        read(prefix + "mlp.fc2.weight");
        read(prefix + "mlp.fc2.bias");
        read(prefix + "layer_norm2.weight");
        read(prefix + "layer_norm2.bias");
    }
    read("post_layernorm.weight");
    read("post_layernorm.bias");
    CHECK_EQ(stream.peek(), std::ifstream::traits_type::eof()) << "Unexpected trailing VPM checkpoint data";
}

} // namespace fm9gv
