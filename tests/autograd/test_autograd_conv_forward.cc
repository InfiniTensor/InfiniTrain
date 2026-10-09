#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/conv.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {
// Deterministic pseudo-random golden data, generated with torch 2.14 under seed 20260905
// (conv2d reference).
constexpr float kGoldenAbsError = 2e-6f;
} // namespace

class AutogradConvForwardTest : public infini_train::test::InfiniTrainTest {};

// An asymmetric kernel pins the cross-correlation semantics: flipping the kernel would swap
// the roles of the taps and change every output value.
TEST_P(AutogradConvForwardTest, ConvForwardAsymmetricKernel) {
    std::vector<float> input_values;
    for (int idx = 0; idx < 16; ++idx) { input_values.push_back(static_cast<float>(idx)); }
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{1, 1, 4, 4}, DataType::kFLOAT32,
                                          GetDevice());
    const std::vector<float> weight_values{1.0f, 2.0f, 3.0f, 4.0f};
    auto weight = std::make_shared<Tensor>(weight_values.data(), std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32,
                                           GetDevice());
    // out[i][j] = 1*x[i][j] + 2*x[i][j+1] + 3*x[i+1][j] + 4*x[i+1][j+1]
    const std::vector<float> expected{34.0f, 44.0f, 54.0f, 74.0f, 84.0f, 94.0f, 114.0f, 124.0f, 134.0f};

    auto conv_fn = std::make_shared<autograd::Conv2d>();
    auto result = conv_fn->Apply({input, weight});
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{1, 1, 3, 3}));
    test::ExpectTensorFloatEqual(result[0], expected);
}

// Random 4D case checked against PyTorch conv2d (fixed seed, see file-level comment).
TEST_P(AutogradConvForwardTest, ConvForwardTorchGolden) {
    const std::vector<float> input_values{
        2.014464855f,  -0.671311080f, -0.945236862f, -0.096128546f, 0.889558852f,  1.726294398f,  -0.078931913f,
        0.205890238f,  0.094608001f,  0.173616216f,  0.437049866f,  -0.557070732f, 0.455583423f,  -0.736482263f,
        -0.718647420f, 0.926798999f,  1.732108712f,  -0.138735890f, -1.508729339f, 2.127460241f,  0.079739936f,
        0.217489868f,  0.589455068f,  -0.052687794f, -1.659490585f, -0.380473137f, -0.166608080f, 0.350353301f,
        -1.411424637f, 0.626776755f,  1.336345553f,  -0.203511983f, -0.757243156f, 1.381434083f,  -0.417290032f,
        -0.668979943f, -0.569557011f, -1.035447836f, -0.555948615f, -1.279154539f, 0.056448594f,  -0.388204783f,
        2.071142673f,  -0.803012371f, 1.956272721f,  0.610922813f,  -1.668144941f, -0.628582716f, -0.756579638f,
        0.987797141f,  -0.849777400f, -1.654165864f, -0.326211363f, -0.040599853f, 0.281366080f,  0.499955416f,
        -0.518889010f, -0.409607649f, 0.689743340f,  1.129151702f,  0.798748374f,  -1.004696369f, -0.104189672f,
        1.257721663f,
    };
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{2, 2, 4, 4}, DataType::kFLOAT32,
                                          GetDevice());
    const std::vector<float> weight_values{
        -2.448281050f, -0.608355999f, -0.958948731f, 1.009868145f,  0.031469870f,  -1.101765275f, 0.571945250f,
        -1.224039674f, -0.426795214f, -0.612229824f, -0.744571507f, -1.164491534f, -0.726353824f, -1.124407530f,
        -0.498482078f, -0.423681349f, -1.231436968f, -0.689355612f, -0.170030892f, -0.120427191f, 1.032825112f,
        1.369398594f,  1.237134099f,  0.458183825f,  -0.396747291f, 0.548917472f,  0.043999992f,  0.891736746f,
        -1.292207360f, 1.713299513f,  0.982937992f,  0.432634443f,  0.293460459f,  0.318505853f,  -0.146791920f,
        0.983720183f,  -0.162205622f, 0.616937041f,  0.929942310f,  -0.411319345f, 0.301661789f,  -1.444623590f,
        -0.730446517f, 2.391211271f,  0.400584310f,  -0.909495890f, -0.702420592f, 0.372547686f,  0.994188488f,
        0.768918931f,  0.740486383f,  -1.176032424f, 0.868664801f,  0.422459871f,
    };
    auto weight = std::make_shared<Tensor>(weight_values.data(), std::vector<int64_t>{3, 2, 3, 3}, DataType::kFLOAT32,
                                           GetDevice());
    const std::vector<float> bias_values{-0.640517414f, 0.982376575f, 0.729396820f};
    auto bias = std::make_shared<Tensor>(bias_values.data(), std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice());
    const std::vector<float> expected{
        -2.073040247f, 1.047590375f,  -2.980665922f, -6.143202305f, 1.986891150f,  9.790361404f,
        0.198000669f,  0.174701050f,  0.093229778f,  2.889255762f,  -0.725508392f, 0.107455865f,
        1.509838820f,  -5.231677532f, 2.129449368f,  4.556076050f,  -5.035542011f, -1.076976299f,
        0.719715834f,  1.997749925f,  1.977930188f,  7.832613468f,  -5.848879814f, -0.770412445f,
    };

    auto conv_fn = std::make_shared<autograd::Conv2d>();
    auto result = conv_fn->Apply({input, weight, bias});
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3, 2, 2}));
    test::ExpectTensorNear(result[0], expected, kGoldenAbsError);
}

// Mismatched channel counts and a kernel larger than the image are rejected before any kernel
// launch.
TEST_P(AutogradConvForwardTest, ConvForwardRejectsInvalidShapes) {
    EXPECT_DEATH(
        {
            auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 3, 3}, DataType::kFLOAT32, GetDevice());
            auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 3, 2, 2}, DataType::kFLOAT32, GetDevice());
            auto conv_fn = std::make_shared<autograd::Conv2d>();
            auto result = conv_fn->Apply({input, weight});
            (void)result;
        },
        "");

    EXPECT_DEATH(
        {
            auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32, GetDevice());
            auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 3, 3}, DataType::kFLOAT32, GetDevice());
            auto conv_fn = std::make_shared<autograd::Conv2d>();
            auto result = conv_fn->Apply({input, weight});
            (void)result;
        },
        "");
}

INFINI_TRAIN_REGISTER_TEST(AutogradConvForwardTest);
