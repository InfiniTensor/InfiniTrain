#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class OptimizerStateTest : public test::InfiniTrainTest {};

// ---------- Adam StateDict ----------
TEST_P(OptimizerStateTest, AdamStateDictKeys) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}, DataType::kFLOAT32, GetDevice());
    param->set_requires_grad(true);
    param->Fill(1.0f);

    auto adam = std::make_shared<optimizers::Adam>(std::vector<std::shared_ptr<Tensor>>{{param}}, 0.001);

    adam->ZeroGrad();
    adam->Step(); // t=1
    adam->Step(); // t=2

    auto state = adam->StateDict();
    EXPECT_GT(state.size(), 0);
    EXPECT_TRUE(state.count("adam.m.0"));
    EXPECT_TRUE(state.count("adam.v.0"));
    EXPECT_TRUE(state.count("adam.t"));

    auto t_cpu = state["adam.t"]->To(Device());
    int64_t t_val = *static_cast<const int64_t *>(t_cpu.DataPtr());
    EXPECT_EQ(t_val, 2);
}

// ---------- Adam LoadStateDict roundtrip ----------
TEST_P(OptimizerStateTest, AdamStateDictRoundTrip) {
    auto param1 = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice());
    param1->set_requires_grad(true);
    param1->Fill(1.0f);

    auto adam1 = std::make_shared<optimizers::Adam>(std::vector<std::shared_ptr<Tensor>>{{param1}}, 0.001);
    adam1->ZeroGrad();
    adam1->Step();
    adam1->Step();
    adam1->Step();

    auto saved = adam1->StateDict();

    auto param2 = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice());
    param2->set_requires_grad(true);
    param2->Fill(1.0f);

    auto adam2 = std::make_shared<optimizers::Adam>(std::vector<std::shared_ptr<Tensor>>{{param2}}, 0.001);
    adam2->LoadStateDict(saved);

    adam2->ZeroGrad();
    adam2->Step();
    auto restored = adam2->StateDict();
    auto t_cpu = restored["adam.t"]->To(Device());
    EXPECT_EQ(*static_cast<const int64_t *>(t_cpu.DataPtr()), 4); // 3 + 1

    auto saved_state = adam1->StateDict();
    for (const auto &[key, tensor] : saved_state) {
        if (key == "adam.t") {
            continue;
        }
        ASSERT_TRUE(restored.count(key)) << "Missing optimizer state key: " << key;
        auto s_cpu = tensor->To(Device());
        auto r_cpu = restored.at(key)->To(Device());
        EXPECT_EQ(s_cpu.Dims(), r_cpu.Dims()) << "Shape mismatch for " << key;
        const float *s = static_cast<const float *>(s_cpu.DataPtr());
        const float *r = static_cast<const float *>(r_cpu.DataPtr());
        for (size_t i = 0; i < s_cpu.NumElements(); ++i) {
            EXPECT_NEAR(s[i], r[i], 1e-6) << "Value mismatch for " << key << " at index " << i;
        }
    }
}

// ---------- SGD ----------
TEST_P(OptimizerStateTest, SGDStateDictEmpty) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice());
    param->set_requires_grad(true);
    auto sgd = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{{param}}, 0.01);
    EXPECT_TRUE(sgd->StateDict().empty());
}

INFINI_TRAIN_REGISTER_TEST(OptimizerStateTest);
