// Golden-value parity for Conv2d forward/backward vs PyTorch (CPU).
//
// Config: N=1, Cin=2, Cout=2, H=4, W=5, K=3, stride=2, padding=1, seed=0.
// Goldens generated ONCE from torch (float32, torch 2.14+cpu) via
// /tmp/opencode/conv2d_golden.py (NOT committed; see OSpec P5 notes).
// Threshold: abs err <= 1e-5 (fp32 vs-PyTorch).
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/conv.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradConvParityTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradConvParityTest, Conv2dGoldenParity) {
    ONLY_CPU();
    const Device device = GetDevice();

    // torch.manual_seed(0) randn inputs.
    const std::vector<float> input_vals = {
        -1.1258398294f, -1.1523602009f, -0.2505785823f, -0.4338788092f, 0.8487103581f,  0.6920091510f,
        -0.3160127699f, -2.1152193546f, 0.3222749233f,  -1.2633347511f, 0.3499831855f,  0.3081339300f,
        0.1198415086f,  1.2376579046f,  1.1167771816f,  -0.2472781539f, -1.3526537418f, -1.6959311962f,
        0.5666506290f,  0.7935083508f,  0.5988394618f,  -1.5550950766f, -0.3413603902f, 1.8530061245f,
        -0.2158632576f, -0.7425481677f, 0.5627213717f,  0.2596274018f,  -0.1739609987f, -0.6787462234f,
        0.9382607341f,  0.4888698161f,  1.2032237053f,  0.0845346972f,  -1.2001394033f, -0.0047857380f,
        -0.5180748105f, -0.3067041934f, -1.5809938908f, 1.7066433430f,
    };
    const std::vector<float> weight_vals = {
        0.2055256665f,  -0.4503297508f, -0.5730770826f, -0.5553584099f, 0.5943230391f,  1.5419425964f,
        0.5073344111f,  -0.5910331607f, -1.3253259659f, 0.1885535717f,  -0.0690726861f, -0.4949253500f,
        -1.4959149361f, -0.1938371211f, 0.4455121756f,  1.3252748251f,  1.5091218948f,  2.0819554329f,
        1.7067116499f,  2.3803675175f,  1.9414620399f,  0.7914980650f,  -0.0202518273f, -0.4371695518f,
        1.6458669901f,  -1.3601689339f, 0.3445654213f,  0.5198677182f,  -0.3656187952f, -1.3024404049f,
        0.0994034633f,  0.4418220222f,  0.2469263971f,  0.0768870041f,  0.3380058110f,  0.4544017613f,
    };
    const std::vector<float> bias_vals = {0.1752833277f, -0.9315211177f};
    const std::vector<float> grad_out_vals = {
        -1.5054897070f, -0.6609825492f, 1.3232016563f,  0.0371143036f, -0.2849093080f, -0.1334417462f,
        1.8929104805f,  3.1110441685f,  -0.4583958089f, -0.3359880745f, -1.5699861050f, 1.2315003872f,
    };

    // PyTorch goldens (F.conv2d + autograd, 10 decimals).
    const std::vector<float> expected_forward = {
        -3.0188088417f, 4.6534223557f,  -2.1541304588f, 1.3896170855f, -2.9418153763f, 1.2058488131f,
        -1.5697779655f, 1.0232539177f,  0.8026736379f,  -0.3325381279f, -5.7407088280f, -2.4872004986f,
    };
    const std::vector<float> expected_dinput = {
        -0.9330821037f, -0.3194339275f, -0.4558414817f, -3.4769215584f, 0.7956926227f,  -2.5013723373f,
        4.0208745003f,  -7.4497237206f, 1.0544195175f,  2.8329558372f,  0.0288622584f,  -0.8803023696f,
        -0.1375330836f, 1.2958744764f,  -0.1042476371f, 0.4350647628f,  -2.8934910297f, 2.3038370609f,
        1.7958209515f,  -1.5961800814f, 1.1281492710f,  1.0947177410f,  1.5026507378f,  -1.5512402058f,
        -0.4590149522f, -1.5118727684f, -3.3616752625f, 0.6477436423f,  4.5567674637f,  1.4008895159f,
        -0.1556410640f, 0.2037085742f,  -0.6384283900f, -0.1925686896f, 0.5699699521f,  -0.0575559139f,
        -0.5736979246f, -0.9606273174f, -1.3887335062f, 0.2148744166f,
    };
    const std::vector<float> expected_dweight = {
        0.0470300466f,  0.7969107032f,  -0.1035477147f, -0.0653646588f, 2.8134040833f,  1.6804685593f,
        0.9450824261f,  -0.9472144842f, 0.0510890186f,  -0.1371109039f, -0.0109563395f, 0.0704481155f,
        3.3292274475f,  -1.1093821526f, 1.1104340553f,  -0.2435595840f, -0.0923609436f, -0.3009741902f,
        0.8930173516f,  1.5325608253f,  -0.3997906148f, -2.3457450867f, -2.2301483154f, -5.5777654648f,
        1.6906189919f,  -0.9686450958f, -0.0307304859f, -1.0976977348f, -0.9940003157f, 0.0840486735f,
        -6.3507938385f, -3.5117480755f, 2.5241556168f,  0.6967696548f,  2.2981307507f,  3.1801860332f,
    };
    const std::vector<float> expected_dbias = {-1.2245074511f, 3.8710851669f};
    constexpr float kAbsTol = 1e-5f;

    auto input = std::make_shared<Tensor>(input_vals.data(), std::vector<int64_t>{1, 2, 4, 5},
                                          DataType::kFLOAT32, device)
                     ->RequiresGrad();
    auto weight = std::make_shared<Tensor>(weight_vals.data(), std::vector<int64_t>{2, 2, 3, 3},
                                           DataType::kFLOAT32, device)
                      ->RequiresGrad();
    auto bias = std::make_shared<Tensor>(bias_vals.data(), std::vector<int64_t>{2}, DataType::kFLOAT32,
                                         device)
                    ->RequiresGrad();

    auto fn = std::make_shared<autograd::Conv2d>(2, 1);
    auto out = fn->Apply({input, weight, bias});
    ASSERT_EQ(out.size(), 1);
    EXPECT_EQ(out[0]->Dims(), (std::vector<int64_t>{1, 2, 2, 3}));
    test::ExpectTensorNear(out[0], expected_forward, kAbsTol);

    auto grad_out = std::make_shared<Tensor>(grad_out_vals.data(), std::vector<int64_t>{1, 2, 2, 3},
                                             DataType::kFLOAT32, device);
    auto grads = fn->Backward({grad_out});
    ASSERT_EQ(grads.size(), 3);
    ASSERT_NE(grads[0], nullptr);
    ASSERT_NE(grads[1], nullptr);
    ASSERT_NE(grads[2], nullptr);
    EXPECT_EQ(grads[0]->Dims(), (std::vector<int64_t>{1, 2, 4, 5}));
    EXPECT_EQ(grads[1]->Dims(), (std::vector<int64_t>{2, 2, 3, 3}));
    EXPECT_EQ(grads[2]->Dims(), (std::vector<int64_t>{2}));
    // Forward logits + full backward grads (dInput spot-check superset, dWeight, dBias).
    test::ExpectTensorNear(grads[0], expected_dinput, kAbsTol);
    test::ExpectTensorNear(grads[1], expected_dweight, kAbsTol);
    test::ExpectTensorNear(grads[2], expected_dbias, kAbsTol);
}

INFINI_TRAIN_REGISTER_TEST(AutogradConvParityTest);
