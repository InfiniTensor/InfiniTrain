#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "example/common/pipeline_layout_parser.h"
#include "example/gpt2/checkpoint_loader.h"
#include "example/gpt2/config.h"
#include "example/llama3/checkpoint_loader.h"
#include "example/llama3/config.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#ifdef USE_OMP
#include <omp.h>
#endif

namespace infini_train::examples {
namespace {
namespace pp = nn::parallel;

// Independent tiny LLMC fixture: each file element has a distinct, exactly
// representable value. Expected parameter values are recorded as it is written.
class CheckpointFixture {
public:
    explicit CheckpointFixture(bool llama, int num_layers = 4) {
        path = std::filesystem::temp_directory_path()
             / ("infinitrain-loader-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
                + ".bin");
        std::ofstream out(path, std::ios::binary);
        std::array<char, 1024> header{};
        auto put = [&](int offset, auto value) { std::memcpy(header.data() + offset, &value, sizeof(value)); };
        put(0, llama ? 20240803 : 20240326);
        put(4, 3);
        put(8, 4);
        put(12, 16);
        put(16, num_layers);
        put(20, 2);
        if (llama) {
            put(24, 1);
            put(28, 8); // GQA: 2 Q heads, 1 KV head.
            put(32, 1.0f);
            put(36, 8);
            put(40, 1e-5f);
            put(44, 10000.0f);
            put(48, 0);
            put(52, 2);
            put(56, 3);
            put(60, 0);
        } else {
            put(24, 8);
            put(28, 20); // Include padded WTE rows to check file positioning.
        }
        out.write(header.data(), header.size());
        float next = 1;
        auto append = [&](const std::string &name, int count) {
            std::vector<float> values(count);
            for (auto &v : values) { v = next++; }
            out.write(reinterpret_cast<const char *>(values.data()), count * sizeof(float));
            if (!name.empty()) {
                expected[name] = std::move(values);
            }
        };
        append("transformer.wte.weight", 16 * 8);
        if (!llama) {
            expected["lm_head.weight"] = expected.at("transformer.wte.weight");
            append("", 4 * 8);
            append("transformer.wpe.weight", 4 * 8);
        }
        auto layers = [&](const std::string &suffix, int count) {
            for (int layer = 0; layer < num_layers; ++layer) {
                append("transformer.h." + std::to_string(layer) + "." + suffix, count);
            }
        };
        layers("ln_1.weight", 8);
        if (!llama) {
            layers("ln_1.bias", 8);
        }
        layers("attn.c_attn.weight", (llama ? 16 : 24) * 8);
        if (!llama) {
            layers("attn.c_attn.bias", 24);
        }
        layers("attn.c_proj.weight", 8 * 8);
        if (!llama) {
            layers("attn.c_proj.bias", 8);
        }
        layers("ln_2.weight", 8);
        if (!llama) {
            layers("ln_2.bias", 8);
        }
        layers("mlp.c_fc.weight", (llama ? 24 : 32) * 8);
        if (llama) {
            layers("mlp.c_fc2.weight", 24 * 8);
        } else {
            layers("mlp.c_fc.bias", 32);
        }
        layers("mlp.c_proj.weight", 8 * (llama ? 24 : 32));
        if (!llama) {
            layers("mlp.c_proj.bias", 8);
        }
        append("transformer.ln_f.weight", 8);
        if (llama) {
            append("lm_head.weight", 16 * 8);
        } else {
            append("transformer.ln_f.bias", 8);
        }
        if (!out) {
            throw std::runtime_error("cannot write checkpoint fixture");
        }
    }
    ~CheckpointFixture() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
    std::filesystem::path path;
    std::map<std::string, std::vector<float>> expected;
};

nn::TransformerConfig GPT2FixtureConfig() {
    auto config = gpt2::GPT2Config();
    config.block_size = 4;
    config.vocab_size = 16;
    config.original_vocab_size = 16;
    config.n_layer = 4;
    config.n_head = 2;
    config.n_kv_head = 2;
    config.n_embd = 8;
    return config;
}

nn::TransformerConfig LLaMA3FixtureConfig() {
    auto config = llama3::LLaMA3Config();
    config.block_size = 4;
    config.vocab_size = 16;
    config.original_vocab_size = 16;
    config.n_layer = 4;
    config.n_head = 2;
    config.n_kv_head = 1;
    config.n_embd = 8;
    config.ffn_dim_multiplier = 1.0f;
    config.multiple_of = 8;
    config.norm_eps = 1e-5f;
    config.rope_theta = 10000.0f;
    config.use_scaled_rope = false;
    config.max_gen_batch_size = 2;
    return config;
}

std::shared_ptr<nn::TransformerModel> Load(bool llama, const CheckpointFixture &fixture,
                                           const std::optional<PipelineLayoutRequest> &request) {
    const auto config = llama ? LLaMA3FixtureConfig() : GPT2FixtureConfig();
    const auto layout = request ? ResolvePipelineLayout(config.n_layer, pp::global::GetPipelineParallelSize(),
                                                        pp::global::GetVirtualPipelineParallelSize(), *request)
                                : nullptr;
    auto model = llama ? llama3::LoadFromLLMC(fixture.path.string(), config, layout)
                       : gpt2::LoadFromLLMC(fixture.path.string(), config, layout);
    EXPECT_EQ(model->GetPipelineLayout(), layout);
    return model;
}

TEST(PipelineLoaderTest, LoadsExactOwnedValuesAcrossAllRequestSources) {
    const int stages = pp::global::GetPipelineParallelSize();
    const int vpp = pp::global::GetVirtualPipelineParallelSize();
    std::vector<std::optional<PipelineLayoutRequest>> requests{std::nullopt, PipelineLayoutRequest{std::monostate{}}};
    if (stages == 1) {
        requests.emplace_back(ParsePipelineLayoutRequest("4", "", stages, vpp));
    } else if (vpp == 1) {
        requests.emplace_back(ParsePipelineLayoutRequest("1,3", "", stages, vpp));
        requests.emplace_back(ParsePipelineLayoutRequest("", "1:1,0:3", stages, vpp));
    } else {
        requests.emplace_back(ParsePipelineLayoutRequest("1,1,1,1", "", stages, vpp));
        requests.emplace_back(ParsePipelineLayoutRequest("", "1:1,0:1,1:2", stages, vpp));
    }
    for (bool llama : {false, true}) {
        CheckpointFixture fixture(llama);
        for (std::size_t mode = 0; mode < requests.size(); ++mode) {
            std::size_t checked = 0;
            for (int stage = 0; stage < stages; ++stage) {
                SCOPED_TRACE(::testing::Message() << "llama=" << llama << " mode=" << mode << " stage=" << stage);
                pp::pp_rank = stage;
                const auto model = Load(llama, fixture, requests[mode]);
                auto layout = model->GetPipelineLayout();
                EXPECT_EQ(static_cast<bool>(layout), requests[mode].has_value());
                if (!layout) {
                    layout = std::make_shared<const pp::PipelineLayout>(
                        pp::PipelineLayout::BuildUniformLayout(4, stages, vpp));
                }
                EXPECT_EQ(model->Config().n_layer, 4);
                EXPECT_EQ(model->Config().n_embd, 8);
                EXPECT_EQ(model->Config().original_vocab_size, 16);
                if (!llama) {
                    EXPECT_EQ(model->Config().tie_weights, stages == 1);
                }
                const auto &stage_layout = layout->GetStage(stage);
                std::vector<int> globals;
                for (const auto &chunk : stage_layout.chunks) {
                    for (auto layer = chunk.layer_range.begin; layer < chunk.layer_range.end; ++layer) {
                        globals.push_back(layer);
                    }
                }
                const auto state = model->StateDict();
                ASSERT_FALSE(state.empty());
                if (!llama && stage_layout.has_embedding && stage_layout.has_lm_head) {
                    const auto &wte = state.at("transformer.wte.weight");
                    const auto &head = state.at("lm_head.weight");
                    EXPECT_EQ(wte.get() == head.get(), stages == 1);
                }
                for (const auto &[local_name, tensor] : state) {
                    std::string name = local_name;
                    const std::string prefix = "transformer.h.";
                    if (name.starts_with(prefix)) {
                        const auto dot = name.find('.', prefix.size());
                        const int local = std::stoi(name.substr(prefix.size(), dot - prefix.size()));
                        name = prefix + std::to_string(globals.at(local)) + name.substr(dot);
                    }
                    // The causal mask is a constructed buffer, not checkpoint data.
                    if (name.ends_with(".attn.bias")) {
                        ASSERT_EQ(tensor->Dims(), (std::vector<int64_t>{1, 1, 4, 4}));
                        const auto *mask = static_cast<const float *>(tensor->DataPtr());
                        for (int row = 0; row < 4; ++row) {
                            for (int col = 0; col < 4; ++col) {
                                ASSERT_EQ(mask[row * 4 + col], col <= row ? 1.0f : 0.0f);
                            }
                        }
                        continue;
                    }
                    ASSERT_TRUE(fixture.expected.contains(name)) << name;
                    const auto &expected = fixture.expected.at(name);
                    ASSERT_EQ(tensor->NumElements(), expected.size()) << name;
                    const auto *actual = static_cast<const float *>(tensor->DataPtr());
                    for (std::size_t i = 0; i < expected.size(); ++i) {
                        ASSERT_EQ(actual[i], expected[i]) << name << " element=" << i;
                    }
                    ++checked;
                }
            }
            EXPECT_EQ(checked, fixture.expected.size()); // No missing or extra parameters across stages.
        }
    }
    pp::pp_rank = 0;
}

TEST(PipelineLoaderTest, RejectsLayerSumAgainstExternalConfig) {
    pp::pp_rank = 0;
    const int stages = pp::global::GetPipelineParallelSize();
    const int vpp = pp::global::GetVirtualPipelineParallelSize();
    std::vector<pp::LayerIndex> counts(stages * vpp, 1);
    counts.front() += 4; // Syntactically legal, but sum disagrees with the external 4-layer config.
    for (bool llama : {false, true}) {
        CheckpointFixture fixture(llama);
        EXPECT_THROW((void)Load(llama, fixture, PipelineLayoutRequest{counts}), std::invalid_argument);
    }
}
TEST(PipelineLoaderTest, RejectsExternalGPT2ConfigBeforeReadingWeights) {
    pp::pp_rank = 0;
    CheckpointFixture fixture(false);
    // A header-only file proves these errors occur before loading any weights.
    std::filesystem::resize_file(fixture.path, 1024);
    const auto valid = GPT2FixtureConfig();
    std::vector<nn::TransformerConfig> invalid;
    auto add = [&](auto change) {
        auto config = valid;
        change(config);
        invalid.push_back(config);
    };
    add([](auto &c) { c.block_size = 8; });
    add([](auto &c) { c.original_vocab_size = 15; });
    add([](auto &c) { c.vocab_size = 17; });
    add([](auto &c) { c.n_layer = 8; });
    add([](auto &c) { c.n_head = c.n_kv_head = 4; });
    add([](auto &c) { c.n_embd = 16; });
    add([](auto &c) { c.ffn_expansion_ratio = 2.0f; });
    for (const auto &config : invalid) {
        EXPECT_THROW((void)gpt2::LoadFromLLMC(fixture.path.string(), config), std::invalid_argument);
    }
    // The one-argument overload now enforces GPT2Config rather than inferring a tiny model.
    EXPECT_THROW((void)gpt2::LoadFromLLMC(fixture.path.string()), std::invalid_argument);
}

TEST(PipelineLoaderTest, RejectsExternalGPT2LayoutBeforeReadingWeights) {
    pp::pp_rank = 0;
    CheckpointFixture fixture(false);
    std::filesystem::resize_file(fixture.path, 1024);
    const auto config = GPT2FixtureConfig();
    const int stages = pp::global::GetPipelineParallelSize();
    const int vpp = pp::global::GetVirtualPipelineParallelSize();
    const auto wrong_layers
        = std::make_shared<const pp::PipelineLayout>(pp::PipelineLayout::BuildUniformLayout(8, stages, vpp));
    EXPECT_THROW((void)gpt2::LoadFromLLMC(fixture.path.string(), config, wrong_layers), std::invalid_argument);
    const auto wrong_stages = std::make_shared<const pp::PipelineLayout>(
        pp::PipelineLayout::BuildUniformLayout(4, stages == 1 ? 2 : 1, vpp));
    EXPECT_THROW((void)gpt2::LoadFromLLMC(fixture.path.string(), config, wrong_stages), std::invalid_argument);
    const auto wrong_vpp = std::make_shared<const pp::PipelineLayout>(
        pp::PipelineLayout::BuildUniformLayout(4, stages, vpp == 1 ? 2 : 1));
    EXPECT_THROW((void)gpt2::LoadFromLLMC(fixture.path.string(), config, wrong_vpp), std::invalid_argument);
}
TEST(PipelineLoaderTest, RejectsExternalLLaMA3ConfigBeforeReadingWeights) {
    pp::pp_rank = 0;
    CheckpointFixture fixture(true);
    std::filesystem::resize_file(fixture.path, 1024);
    const auto valid = LLaMA3FixtureConfig();
    std::vector<nn::TransformerConfig> invalid;
    auto add = [&](auto change) {
        auto config = valid;
        change(config);
        invalid.push_back(config);
    };
    add([](auto &c) { c.block_size = 8; });
    add([](auto &c) { c.original_vocab_size = 15; });
    add([](auto &c) { c.vocab_size = 17; });
    add([](auto &c) { c.n_layer = 8; });
    add([](auto &c) { c.n_head = 4; });
    add([](auto &c) { c.n_kv_head = 2; });
    add([](auto &c) { c.n_embd = 16; });
    add([](auto &c) { c.ffn_dim_multiplier = 1.5f; });
    add([](auto &c) { c.ffn_dim_multiplier.reset(); });
    add([](auto &c) { c.multiple_of = 16; });
    add([](auto &c) { c.norm_eps = 1e-4f; });
    add([](auto &c) { c.rope_theta = 500000.0f; });
    add([](auto &c) { c.use_scaled_rope = true; });
    add([](auto &c) { c.max_gen_batch_size = 4; });
    add([](auto &c) { c.tie_weights = true; });
    for (const auto &config : invalid) {
        EXPECT_THROW((void)llama3::LoadFromLLMC(fixture.path.string(), config), std::invalid_argument);
    }
    EXPECT_THROW((void)llama3::LoadFromLLMC(fixture.path.string()), std::invalid_argument);
}

TEST(PipelineLoaderTest, RejectsExternalLLaMA3LayoutBeforeReadingWeights) {
    pp::pp_rank = 0;
    CheckpointFixture fixture(true);
    std::filesystem::resize_file(fixture.path, 1024);
    const auto config = LLaMA3FixtureConfig();
    const int stages = pp::global::GetPipelineParallelSize();
    const int vpp = pp::global::GetVirtualPipelineParallelSize();
    const auto wrong_layers
        = std::make_shared<const pp::PipelineLayout>(pp::PipelineLayout::BuildUniformLayout(8, stages, vpp));
    EXPECT_THROW((void)llama3::LoadFromLLMC(fixture.path.string(), config, wrong_layers), std::invalid_argument);
    const auto wrong_stages = std::make_shared<const pp::PipelineLayout>(
        pp::PipelineLayout::BuildUniformLayout(4, stages == 1 ? 2 : 1, vpp));
    EXPECT_THROW((void)llama3::LoadFromLLMC(fixture.path.string(), config, wrong_stages), std::invalid_argument);
    const auto wrong_vpp = std::make_shared<const pp::PipelineLayout>(
        pp::PipelineLayout::BuildUniformLayout(4, stages, vpp == 1 ? 2 : 1));
    EXPECT_THROW((void)llama3::LoadFromLLMC(fixture.path.string(), config, wrong_vpp), std::invalid_argument);
}
TEST(PipelineLoaderTest, RejectsTruncatedCheckpointHeaders) {
    pp::pp_rank = 0;
    for (bool llama : {false, true}) {
        CheckpointFixture fixture(llama);
        std::filesystem::resize_file(fixture.path, 64);
        const auto config = llama ? LLaMA3FixtureConfig() : GPT2FixtureConfig();
        EXPECT_THROW((void)(llama ? llama3::LoadFromLLMC(fixture.path.string(), config)
                                  : gpt2::LoadFromLLMC(fixture.path.string(), config)),
                     std::invalid_argument);
    }
}

TEST(PipelineLoaderTest, RejectsMatchingButInvalidCheckpointMetadata) {
    pp::pp_rank = 0;
    for (bool llama : {false, true}) {
        CheckpointFixture fixture(llama);
        std::filesystem::resize_file(fixture.path, 1024);
        auto config = llama ? LLaMA3FixtureConfig() : GPT2FixtureConfig();
        config.n_layer = 0;
        {
            std::fstream out(fixture.path, std::ios::binary | std::ios::in | std::ios::out);
            const uint32_t zero = 0;
            out.seekp(16);
            out.write(reinterpret_cast<const char *>(&zero), sizeof(zero));
        }
        EXPECT_THROW((void)(llama ? llama3::LoadFromLLMC(fixture.path.string(), config)
                                 : gpt2::LoadFromLLMC(fixture.path.string(), config)), std::invalid_argument);
    }
    CheckpointFixture fixture(true);
    std::filesystem::resize_file(fixture.path, 1024);
    auto config = LLaMA3FixtureConfig();
    config.norm_eps = std::numeric_limits<float>::infinity();
    {
        std::fstream out(fixture.path, std::ios::binary | std::ios::in | std::ios::out);
        out.seekp(40);
        out.write(reinterpret_cast<const char *>(&config.norm_eps), sizeof(config.norm_eps));
    }
    EXPECT_THROW((void)llama3::LoadFromLLMC(fixture.path.string(), config), std::invalid_argument);
}

TEST(PipelineLoaderTest, ReportsExpectedAndCheckpointLayerCounts) {
    pp::pp_rank = 0;
    for (bool llama : {false, true}) {
        CheckpointFixture fixture(llama);
        std::filesystem::resize_file(fixture.path, 1024);
        auto config = llama ? LLaMA3FixtureConfig() : GPT2FixtureConfig();
        config.n_layer = 8;
        try {
            (void)(llama ? llama3::LoadFromLLMC(fixture.path.string(), config)
                         : gpt2::LoadFromLLMC(fixture.path.string(), config));
            FAIL() << "Mismatched checkpoint layer count was accepted";
        } catch (const std::invalid_argument &error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("n_layer"), std::string::npos);
            EXPECT_NE(message.find("expected=8"), std::string::npos);
            EXPECT_NE(message.find("checkpoint=4"), std::string::npos);
        }
    }
}

TEST(PipelineLoaderTest, PreservesUnderfilledDefaultLayoutFallback) {
    pp::pp_rank = 0;
    const int stages = pp::global::GetPipelineParallelSize();
    const int vpp = pp::global::GetVirtualPipelineParallelSize();
    for (bool llama : {false, true}) {
        CheckpointFixture fixture(llama, 1);
        auto config = llama ? LLaMA3FixtureConfig() : GPT2FixtureConfig();
        config.n_layer = 1;
        const auto layout = ResolvePipelineLayout(config.n_layer, stages, vpp, PipelineLayoutRequest{std::monostate{}});
        EXPECT_EQ(static_cast<bool>(layout), stages * vpp == 1);
        const auto model = llama ? llama3::LoadFromLLMC(fixture.path.string(), config, layout)
                                 : gpt2::LoadFromLLMC(fixture.path.string(), config, layout);
        EXPECT_EQ(model->GetPipelineLayout(), layout);
        EXPECT_EQ(model->Config().n_layer, 1);
        EXPECT_FALSE(model->StateDict().empty());
    }
}

} // namespace
} // namespace infini_train::examples

// Each CTest process has a separate GlobalEnv; no communicator or GPU is created.
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
#ifdef USE_OMP
    omp_set_num_threads(1);
#endif
    const int pp = (std::getenv("PIPELINE_LOADER_TEST_PP") ? std::atoi(std::getenv("PIPELINE_LOADER_TEST_PP")) : 1);
    const int vpp = (std::getenv("PIPELINE_LOADER_TEST_VPP") ? std::atoi(std::getenv("PIPELINE_LOADER_TEST_VPP")) : 1);
    infini_train::nn::parallel::global::InitAllEnv(1, 1, false, pp, vpp);
    return RUN_ALL_TESTS();
}
