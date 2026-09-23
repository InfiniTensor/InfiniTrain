#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace fm9gv {

struct VisionConfig {
    int64_t hidden_size = 1152;
    int64_t intermediate_size = 4304;
    int64_t num_hidden_layers = 27;
    int64_t num_attention_heads = 16;
    int64_t num_channels = 3;
    int64_t image_size = 980;
    int64_t patch_size = 14;
    float layer_norm_eps = 1e-6f;
};

class VisionEmbeddings : public infini_train::nn::CloneableModule<VisionEmbeddings> {
public:
    static constexpr char kType[] = "FM9GVVisionEmbeddings";
    explicit VisionEmbeddings(const VisionConfig &config);
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;

private:
    VisionConfig config_;
};

class VisionAttention : public infini_train::nn::CloneableModule<VisionAttention> {
public:
    static constexpr char kType[] = "FM9GVVisionAttention";
    explicit VisionAttention(const VisionConfig &config);
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;

private:
    VisionConfig config_;
};

class VisionMLP : public infini_train::nn::CloneableModule<VisionMLP> {
public:
    static constexpr char kType[] = "FM9GVVisionMLP";
    explicit VisionMLP(const VisionConfig &config);
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;
};

class VisionEncoderLayer : public infini_train::nn::CloneableModule<VisionEncoderLayer> {
public:
    static constexpr char kType[] = "FM9GVVisionEncoderLayer";
    explicit VisionEncoderLayer(const VisionConfig &config);
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;
};

class VisionTransformer : public infini_train::nn::CloneableModule<VisionTransformer> {
public:
    static constexpr char kType[] = "FM9GVVisionTransformer";
    explicit VisionTransformer(const VisionConfig &config = VisionConfig());

    // Mirrors SiglipVisionTransformer.forward for the FM9GV training path:
    // {pixel_values[B,C,H,W], target_sizes[B,2]} -> last_hidden_state.
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;

    void LoadFromBin(const std::string &path);
    const VisionConfig &Config() const { return config_; }

private:
    VisionConfig config_;
};

} // namespace fm9gv
