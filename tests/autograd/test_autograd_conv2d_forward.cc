#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/conv.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {
// a_input: shape [1, 2, 4, 4]
const std::vector<float> a_input
    = {1.926915f,  1.487284f,  0.900717f,  -2.105521f, 0.678418f,  -1.234545f, -0.043067f, -1.604667f,
       -0.752135f, 1.648723f,  -0.392479f, -1.403607f, -0.727881f, -0.559430f, -0.768839f, 0.762445f,
       1.642317f,  -0.159597f, -0.497398f, 0.439589f,  -0.758131f, 1.078318f,  0.800801f,  1.680621f,
       1.279124f,  1.296423f,  0.610466f,  1.334738f,  -0.231624f, 0.041759f,  -0.251575f, 0.859859f};
// a_weight: shape [3, 2, 3, 3]
const std::vector<float> a_weight
    = {-1.384674f, -0.871236f, -0.223366f, 1.717361f,  0.318880f,  -0.424519f, 0.305721f,  -0.774593f, -1.557572f,
       0.995636f,  -0.879786f, -0.601142f, -1.274151f, 2.122785f,  -1.234653f, -0.487914f, -0.913823f, -0.658137f,
       0.078024f,  0.525809f,  -0.487992f, 1.191369f,  -0.814008f, -0.735993f, -1.403248f, 0.036004f,  -0.063477f,
       0.675615f,  -0.097807f, 1.844594f,  -1.184537f, 1.383549f,  1.445134f,  0.856413f,  2.218076f,  0.523166f,
       0.346647f,  -0.197331f, 1.141202f,  0.051644f,  0.728110f,  -0.710642f, -0.602068f, 0.960449f,  0.404814f,
       -1.354343f, 1.334703f,  0.483539f,  -0.197562f, 1.268311f,  1.224263f,  0.098117f,  1.742253f,  -1.352674f};
// a_bias: shape [3]
const std::vector<float> a_bias = {-0.437065f, 0.077618f, -0.922759f};
// a_output: shape [1, 3, 2, 2]
const std::vector<float> a_output = {-2.577903f, -5.072696f, -1.058225f, 1.205943f,  11.600315f, 5.375094f,
                                     0.386035f,  9.683105f,  2.902302f,  -1.870026f, 5.919574f,  -1.493631f};
// a_grad_output: shape [1, 3, 2, 2]
const std::vector<float> a_grad_output = {-0.728986f, 0.727698f,  -0.007153f, -0.977891f, 1.003980f,  -0.024754f,
                                          0.147247f,  -0.130241f, -0.755130f, -0.636322f, -0.684569f, -1.285823f};
// a_grad_input: shape [1, 2, 4, 4]
const std::vector<float> a_grad_input
    = {0.825979f,  0.081896f,  -1.710306f, -0.876634f, -0.310730f, 0.704758f,  0.081797f,  -1.023905f,
       -1.049271f, -2.005676f, -1.170570f, 0.035285f,  0.203346f,  0.011301f,  -0.757518f, 1.010884f,
       0.975205f,  1.104877f,  0.437927f,  -0.790798f, 0.908273f,  -2.130304f, 1.231674f,  -1.987390f,
       1.111350f,  2.113331f,  -4.305401f, -0.186185f, 0.062427f,  -0.620124f, -0.627753f, 2.314747f};
// a_grad_weight: shape [3, 2, 3, 3]
const std::vector<float> a_grad_weight
    = {0.879997f,  -0.377813f, -0.619297f, -2.999825f, 1.240634f,  0.239064f,  2.300337f,  -0.731660f, -1.475380f,
       -2.362419f, -1.036420f, -0.966707f, 0.060448f,  -0.809580f, -0.670382f, -0.028238f, -0.255124f, -0.312783f,
       2.158452f,  1.294732f,  1.159075f,  0.386198f,  -0.944505f, 0.121499f,  -0.830260f, 1.682760f,  -0.571807f,
       1.400730f,  -0.093438f, -0.611229f, -0.768341f, 1.174173f,  0.678437f,  1.212578f,  1.325385f,  0.430823f,
       -1.278481f, -0.795731f, 2.752429f,  -1.331804f, 0.335641f,  3.127074f,  0.736455f,  0.376301f,  0.735471f,
       -2.006141f, -1.330848f, -2.613303f, -2.656287f, -2.996279f, -3.808266f, -1.685980f, -1.072527f, -2.243709f};
// a_grad_bias: shape [3]
const std::vector<float> a_grad_bias = {-0.986331f, 0.996232f, -3.361843f};
// b_input: shape [2, 1, 5, 5]
const std::vector<float> b_input
    = {0.654741f,  0.576005f,  -0.360910f, -0.060590f, 0.073255f,  0.818653f,  1.480474f,  0.344929f,  -0.686651f,
       0.636814f,  0.217553f,  -0.046655f, -1.433521f, -0.566527f, -0.425283f, 0.262519f,  -0.732803f, 0.104298f,
       1.041401f,  -0.399731f, -2.293334f, 0.497563f,  -0.425723f, -1.337147f, -1.195454f, 0.812337f,  -0.306278f,
       -0.330158f, -0.980803f, 0.194734f,  -1.653521f, 0.681419f,  0.174820f,  -1.093929f, 0.716537f,  1.533467f,
       -1.450979f, -0.786135f, -0.956316f, -1.247602f, -0.749942f, -0.592190f, -1.532617f, -0.725131f, 0.466404f,
       0.666725f,  -0.043871f, 0.236808f,  -0.706068f, -0.716906f};
// b_weight: shape [2, 1, 2, 2]
const std::vector<float> b_weight
    = {0.494319f, -0.642960f, 0.711319f, 0.399978f, -1.203922f, -0.419752f, -1.192891f, -0.935063f};
// b_bias: shape [2]
const std::vector<float> b_bias = {0.213803f, -1.284212f};
// b_output: shape [2, 2, 3, 3]
const std::vector<float> b_output
    = {0.475685f,  0.479170f,  0.200004f,  -0.225542f, 0.117290f,  -1.108152f, -0.872269f, -0.031849f, -0.443695f,
       -1.896435f, -1.633849f, -1.280432f, -1.831269f, -1.815284f, 0.348629f,  0.750008f,  -0.641214f, 0.342706f,
       0.538720f,  -0.136115f, -0.405972f, 1.890304f,  -0.908306f, -1.966910f, 0.962661f,  0.969996f,  -1.233508f,
       -2.043798f, -0.610136f, -0.296308f, -2.024031f, 0.287977f,  2.039392f,  -1.592851f, -0.097037f, 0.905630f};
// b_grad_output: shape [2, 2, 3, 3]
const std::vector<float> b_grad_output
    = {0.361835f, 1.999347f,  0.663007f,  0.704733f,  -0.004505f, 1.666792f,  0.153920f,  -1.060253f, 0.507098f,
       0.082078f, 0.443975f,  -0.724034f, -0.071988f, -0.906094f, -2.048712f, -1.081056f, -0.982707f, 0.301771f,
       0.178692f, -0.129309f, -0.685482f, 0.563559f,  -1.507175f, -1.610666f, -1.479047f, 0.432274f,  -0.125025f,
       0.782118f, -1.598768f, -0.109130f, 0.715199f,  0.039139f,  1.305860f,  0.246593f,  -1.977591f, 0.017896f};
// b_grad_input: shape [2, 1, 5, 5]
const std::vector<float> b_grad_input
    = {0.067978f,  0.892559f,  0.384550f,  1.335303f,  0.942206f,  -0.422898f, 1.088640f,  0.383232f,  3.290416f,
       -0.211729f, 0.349190f,  1.077667f,  0.845453f,  3.629510f,  2.582355f,  0.354811f,  0.658999f,  1.094194f,
       -0.112641f, -0.452713f, 1.072420f,  0.418085f,  0.494815f,  0.000728f,  -0.079347f, -0.659857f, 1.815175f,
       1.443228f,  -0.357416f, -0.172134f, -0.662553f, -0.792145f, 0.952625f,  -2.368335f, 0.487456f,  -0.443345f,
       -1.118770f, -0.639434f, -2.703445f, -1.865292f, 0.847460f,  2.594546f,  0.552163f,  -0.083348f, 0.072874f,
       -0.822166f, 2.666535f,  2.022072f,  -0.110281f, -0.066741f};
// b_grad_weight: shape [2, 1, 2, 2]
const std::vector<float> b_grad_weight
    = {0.723482f, -0.597131f, 3.470205f, 1.825737f, 0.855981f, 0.431211f, -0.024724f, 4.793805f};
// b_grad_bias: shape [2]
const std::vector<float> b_grad_bias = {0.629796f, -5.565448f};

std::shared_ptr<Tensor> MakeTensor(const std::vector<int64_t> &dims, const std::vector<float> &values,
                                   const Device &device, bool requires_grad = false) {
    auto cpu_tensor = std::make_shared<Tensor>(dims, DataType::kFLOAT32, Device());
    std::copy(values.begin(), values.end(), static_cast<float *>(cpu_tensor->DataPtr()));
    auto tensor = std::make_shared<Tensor>(cpu_tensor->To(device));
    return requires_grad ? tensor->RequiresGrad() : tensor;
}
} // namespace

class AutogradConv2dForwardTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradConv2dForwardTest, Conv2dForwardStride1NoPadding) {
    const std::vector<int64_t> input_dims = {1, 2, 4, 4};
    const std::vector<int64_t> output_dims = {1, 3, 2, 2};
    auto input = MakeTensor(input_dims, a_input, GetDevice());
    auto weight = MakeTensor({3, 2, 3, 3}, a_weight, GetDevice());
    auto bias = MakeTensor({3}, a_bias, GetDevice());

    auto conv_fn = std::make_shared<autograd::Conv2d>(/*stride=*/1, /*padding=*/0);
    auto result = conv_fn->Apply({input, weight, bias});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), output_dims);
    test::ExpectTensorNear(result[0], a_output, 1e-4f);
}

TEST_P(AutogradConv2dForwardTest, Conv2dForwardStride2Padding1Batch) {
    const std::vector<int64_t> input_dims = {2, 1, 5, 5};
    const std::vector<int64_t> output_dims = {2, 2, 3, 3};
    auto input = MakeTensor(input_dims, b_input, GetDevice());
    auto weight = MakeTensor({2, 1, 2, 2}, b_weight, GetDevice());
    auto bias = MakeTensor({2}, b_bias, GetDevice());

    auto conv_fn = std::make_shared<autograd::Conv2d>(/*stride=*/2, /*padding=*/1);
    auto result = conv_fn->Apply({input, weight, bias});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), output_dims);
    test::ExpectTensorNear(result[0], b_output, 1e-4f);
}

TEST_P(AutogradConv2dForwardTest, Conv2dForwardNoBias) {
    const std::vector<int64_t> input_dims = {1, 2, 4, 4};
    auto input = MakeTensor(input_dims, a_input, GetDevice());
    auto weight = MakeTensor({3, 2, 3, 3}, a_weight, GetDevice());

    auto conv_fn = std::make_shared<autograd::Conv2d>(/*stride=*/1, /*padding=*/0);
    auto result = conv_fn->Apply({input, weight});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{1, 3, 2, 2}));
    // Bias-free output is the biased reference minus the bias per channel; layout is (C, H_out, W_out).
    std::vector<float> expected(a_output);
    constexpr int64_t kP = 2 * 2;
    for (int64_t c = 0; c < 3; ++c) {
        for (int64_t i = 0; i < kP; ++i) { expected[c * kP + i] -= a_bias[c]; }
    }
    test::ExpectTensorNear(result[0], expected, 1e-4f);
}

INFINI_TRAIN_REGISTER_TEST(AutogradConv2dForwardTest);
