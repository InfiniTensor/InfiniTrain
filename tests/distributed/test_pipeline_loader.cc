#include "gtest/gtest.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "example/common/parser.h"
#include "example/gpt2/checkpoint_loader.h"
#include "example/llama3/checkpoint_loader.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

namespace infini_train::examples {
namespace {
namespace pp = nn::parallel;

// Independent tiny LLMC fixture: each file element has a distinct, exactly
// representable value. Expected parameter values are recorded as it is written.
class CheckpointFixture {
public:
    explicit CheckpointFixture(bool llama) {
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
        put(16, 4);
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
            for (int layer = 0; layer < 4; ++layer) {
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

std::shared_ptr<nn::TransformerModel> Load(bool llama, const CheckpointFixture &fixture,
                                           const std::optional<PipelineLayoutRequest> &request) {
    if (llama) {
        return request ? llama3::LoadFromLLMC(fixture.path.string(), *request)
                       : llama3::LoadFromLLMC(fixture.path.string());
    }
    return request ? gpt2::LoadFromLLMC(fixture.path.string(), *request) : gpt2::LoadFromLLMC(fixture.path.string());
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

TEST(PipelineLoaderTest, RejectsLayerSumFromCheckpointHeader) {
    pp::pp_rank = 0;
    const int stages = pp::global::GetPipelineParallelSize();
    const int vpp = pp::global::GetVirtualPipelineParallelSize();
    std::vector<pp::LayerIndex> counts(stages * vpp, 1);
    counts.front() += 4; // Syntactically legal, but sum disagrees with the 4-layer file.
    for (bool llama : {false, true}) {
        CheckpointFixture fixture(llama);
        EXPECT_THROW((void)Load(llama, fixture, PipelineLayoutRequest{counts}), std::invalid_argument);
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
