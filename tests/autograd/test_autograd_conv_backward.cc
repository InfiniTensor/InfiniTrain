#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/conv.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {
// Same golden case as the forward test (torch 2.14, seed 20260905, autograd.grad reference).
constexpr float kGoldenAbsError = 2e-6f;
} // namespace

class AutogradConvBackwardTest : public infini_train::test::InfiniTrainTest {};

// Hand-checkable case: x = arange(9) reshaped (1,1,3,3), w = [[1,2],[3,4]], bias = 0.5,
// grad_output = ones(1,1,2,2). bias does not influence grad_input / grad_weight.
TEST_P(AutogradConvBackwardTest, ConvBackwardGradients) {
    ONLY_CPU();
    std::vector<float> input_values;
    for (int idx = 0; idx < 9; ++idx) { input_values.push_back(static_cast<float>(idx)); }
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{1, 1, 3, 3}, DataType::kFLOAT32,
                                          GetDevice())
                     ->RequiresGrad();
    const std::vector<float> weight_values{1.0f, 2.0f, 3.0f, 4.0f};
    auto weight = std::make_shared<Tensor>(weight_values.data(), std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32,
                                           GetDevice())
                      ->RequiresGrad();
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.5f);

    auto conv_fn = std::make_shared<autograd::Conv2d>();
    auto result = conv_fn->Apply({input, weight, bias});
    auto grad_output
        = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32, GetDevice(), true);
    grad_output->Fill(1.0f);
    auto grad_inputs = conv_fn->Backward({grad_output});
    ASSERT_EQ(grad_inputs.size(), 3);
    ASSERT_NE(grad_inputs[0], nullptr);
    ASSERT_NE(grad_inputs[1], nullptr);
    ASSERT_NE(grad_inputs[2], nullptr);

    // grad_input(x, y) sums the taps of every window covering (x, y).
    const std::vector<float> expected_grad_input{1.0f, 3.0f, 2.0f, 4.0f, 10.0f, 6.0f, 3.0f, 7.0f, 4.0f};
    EXPECT_EQ(grad_inputs[0]->Dims(), (std::vector<int64_t>{1, 1, 3, 3}));
    test::ExpectTensorFloatEqual(grad_inputs[0], expected_grad_input);

    // grad_weight(u, v) sums the input elements seen through tap (u, v) over all windows.
    const std::vector<float> expected_grad_weight{8.0f, 12.0f, 20.0f, 24.0f};
    EXPECT_EQ(grad_inputs[1]->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    test::ExpectTensorFloatEqual(grad_inputs[1], expected_grad_weight);

    test::ExpectTensorFloatEqual(grad_inputs[2], std::vector<float>{4.0f});
}

TEST_P(AutogradConvBackwardTest, ConvBackwardNoBias) {
    ONLY_CPU();
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 2, 4, 4}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{3, 2, 3, 3}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto conv_fn = std::make_shared<autograd::Conv2d>();
    auto result = conv_fn->Apply({input, weight});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 2, 2}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = conv_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);

    // No-bias path returns {grad_input, grad_weight}: both must be produced and correct.
    ASSERT_NE(grad_inputs[0], nullptr);
    EXPECT_EQ(grad_inputs[0]->Dims(), (std::vector<int64_t>{2, 2, 4, 4}));
    ASSERT_NE(grad_inputs[1], nullptr);
    EXPECT_EQ(grad_inputs[1]->Dims(), (std::vector<int64_t>{3, 2, 3, 3}));
    // With all-ones input/weight/grad_output, grad_weight(o,c,u,v) = sum_{n,i,j} 1 = 2*2*2.
    test::ExpectTensorFloatEqual(grad_inputs[1], 8.0f);
}

// H = W = kernel: a single window, so grad_input mirrors the weight and grad_weight the input.
TEST_P(AutogradConvBackwardTest, ConvBackwardKernelEqualsSpatial) {
    ONLY_CPU();
    const std::vector<float> input_values{1.0f, 2.0f, 3.0f, 4.0f};
    auto input = std::make_shared<Tensor>(input_values.data(), std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32,
                                          GetDevice())
                     ->RequiresGrad();
    const std::vector<float> weight_values{5.0f, 6.0f, 7.0f, 8.0f};
    auto weight = std::make_shared<Tensor>(weight_values.data(), std::vector<int64_t>{1, 1, 2, 2}, DataType::kFLOAT32,
                                           GetDevice())
                      ->RequiresGrad();
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.25f);

    auto conv_fn = std::make_shared<autograd::Conv2d>();
    auto result = conv_fn->Apply({input, weight, bias});
    auto grad_output
        = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 1, 1}, DataType::kFLOAT32, GetDevice(), true);
    grad_output->Fill(1.0f);
    auto grad_inputs = conv_fn->Backward({grad_output});
    ASSERT_EQ(grad_inputs.size(), 3);

    EXPECT_EQ(grad_inputs[0]->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    test::ExpectTensorFloatEqual(grad_inputs[0], weight_values);
    EXPECT_EQ(grad_inputs[1]->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    test::ExpectTensorFloatEqual(grad_inputs[1], input_values);
    test::ExpectTensorFloatEqual(grad_inputs[2], std::vector<float>{1.0f});
}

// Training-shaped call: the activation is a leaf without requires_grad, so only the parameter
// gradients must be produced.
TEST_P(AutogradConvBackwardTest, ConvBackwardInputNotRequired) {
    ONLY_CPU();
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 2, 4, 4}, DataType::kFLOAT32, GetDevice());
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{3, 2, 3, 3}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(1.0f);

    auto conv_fn = std::make_shared<autograd::Conv2d>();
    auto result = conv_fn->Apply({input, weight, bias});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 2, 2}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = conv_fn->Backward({grad});
    ASSERT_EQ(grad_inputs.size(), 3);
    EXPECT_EQ(grad_inputs[0], nullptr);
    EXPECT_NE(grad_inputs[1], nullptr);
    EXPECT_NE(grad_inputs[2], nullptr);
    // 2 images * 4 windows * 9 taps of 1.0
    test::ExpectTensorFloatEqual(grad_inputs[1], 8.0f);
    // 2 images * 4 spatial positions
    test::ExpectTensorFloatEqual(grad_inputs[2], 8.0f);
}

// Random 4D case checked against PyTorch autograd (fixed seed, see file-level comment).
TEST_P(AutogradConvBackwardTest, ConvBackwardTorchGolden) {
    ONLY_CPU();
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
                                          GetDevice())
                     ->RequiresGrad();
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
                                           GetDevice())
                      ->RequiresGrad();
    const std::vector<float> bias_values{-0.640517414f, 0.982376575f, 0.729396820f};
    auto bias = std::make_shared<Tensor>(bias_values.data(), std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice())
                    ->RequiresGrad();

    auto conv_fn = std::make_shared<autograd::Conv2d>();
    auto result = conv_fn->Apply({input, weight, bias});
    const std::vector<float> grad_output_values{
        -1.277466655f, -1.339194655f, -1.159473658f, 0.086730979f,  0.438649237f,  0.602657557f,
        -0.738640666f, -0.410562515f, 0.201109841f,  0.716394842f,  0.003985748f,  0.676588118f,
        0.412507862f,  -0.146394536f, -0.880041420f, -1.247093678f, -1.212666154f, -0.393535852f,
        0.564200640f,  0.279093444f,  -2.798053741f, -1.615590930f, -0.518746257f, -0.283052236f,
    };
    auto grad_output = std::make_shared<Tensor>(grad_output_values.data(), std::vector<int64_t>{2, 3, 2, 2},
                                                DataType::kFLOAT32, GetDevice());

    auto grad_inputs = conv_fn->Backward({grad_output});
    ASSERT_EQ(grad_inputs.size(), 3);
    ASSERT_NE(grad_inputs[0], nullptr);
    ASSERT_NE(grad_inputs[1], nullptr);
    ASSERT_NE(grad_inputs[2], nullptr);

    const std::vector<float> expected_grad_input{
        3.020392179f,  3.908452511f,  3.049194813f,  2.572864771f,  2.191555023f,  0.285838872f,  3.004242659f,
        0.838663280f,  -3.235622168f, -0.945002913f, 4.960352898f,  -0.376030058f, -0.373013139f, 0.741603315f,
        1.750292301f,  0.215949476f,  0.990354240f,  0.948824525f,  2.029216766f,  2.858904839f,  1.606565118f,
        4.838562012f,  3.308694601f,  0.822548687f,  0.564552128f,  2.793590069f,  4.189336777f,  2.155962467f,
        0.251298606f,  0.576505065f,  0.615549624f,  -0.177835792f, -0.349884391f, -1.143750548f, -5.110340118f,
        -1.768475175f, 2.049649715f,  0.845555902f,  3.547870159f,  3.535832882f,  2.858142614f,  -6.855783939f,
        -3.052171230f, 1.308768988f,  -0.348264217f, -0.470771074f, 1.195474505f,  0.431147456f,  1.210889816f,
        4.433355331f,  -1.848075271f, -1.105654240f, -2.759691477f, -3.466245174f, -1.316413283f, 0.586114824f,
        3.407652855f,  0.809629261f,  -2.299243927f, -0.474773228f, 1.162620664f,  1.500420690f,  2.191398382f,
        1.014662623f,
    };
    const std::vector<float> expected_grad_weight{
        -1.278161168f, 2.350493193f,  3.455903053f,  -3.191775799f, -4.849991322f, -1.593288898f, -3.348839760f,
        1.063873172f,  4.325201988f,  -2.219501734f, 2.213379383f,  -2.589243412f, 2.336852074f,  -1.198173761f,
        -2.431171656f, 4.716279507f,  0.842501462f,  -3.181046009f, -1.122435212f, -4.357190609f, -0.400157630f,
        2.311078310f,  2.235447884f,  2.117345333f,  1.470544577f,  0.713402152f,  -3.305890560f, 0.861350715f,
        -2.180587769f, 2.186314106f,  1.552511692f,  0.699985623f,  0.372371852f,  0.788659871f,  -1.649568439f,
        -1.754016399f, 1.571256876f,  -3.355353594f, 2.778887510f,  4.880561829f,  3.997585535f,  2.531520605f,
        -1.071573257f, -2.245790720f, -3.141553879f, 1.098210573f,  -2.158659220f, 5.950089931f,  1.271268964f,
        0.028057545f,  -1.955230832f, 1.795806646f,  1.293214321f,  -3.971021652f,
    };
    const std::vector<float> expected_grad_bias{-5.550425529f, -0.870804369f, -3.617365122f};

    EXPECT_EQ(grad_inputs[0]->Dims(), (std::vector<int64_t>{2, 2, 4, 4}));
    test::ExpectTensorNear(grad_inputs[0], expected_grad_input, kGoldenAbsError);
    EXPECT_EQ(grad_inputs[1]->Dims(), (std::vector<int64_t>{3, 2, 3, 3}));
    test::ExpectTensorNear(grad_inputs[1], expected_grad_weight, kGoldenAbsError);
    EXPECT_EQ(grad_inputs[2]->Dims(), (std::vector<int64_t>{3}));
    test::ExpectTensorNear(grad_inputs[2], expected_grad_bias, kGoldenAbsError);
}

INFINI_TRAIN_REGISTER_TEST(AutogradConvBackwardTest);
