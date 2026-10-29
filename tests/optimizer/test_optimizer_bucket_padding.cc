#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel_config.h"
#include "infini_train/include/nn/parallel/ddp/param_and_grad_buffer.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class OptimizerBucketPaddingTest : public test::InfiniTrainTest {};

TEST_P(OptimizerBucketPaddingTest,
       ParameterRangesRetainAlignmentForBothZeROStages) {
  for (int stage : {1, 2}) {
    auto first = std::make_shared<Tensor>(std::vector<int64_t>{32},
                                          DataType::kFLOAT32, GetDevice());
    auto second = std::make_shared<Tensor>(std::vector<int64_t>{32},
                                           DataType::kFLOAT32, GetDevice());
    first->Fill(1.0f);
    second->Fill(2.0f);
    auto dtype = DataType::kFLOAT32;
    nn::parallel::DistributedDataParallelConfig config;
    config.zero_stage = stage;
    nn::parallel::ParamAndGradBuffer buffer({first, second}, dtype, dtype,
                                            nullptr, 1.0f, config);
    ASSERT_EQ(buffer.buckets().size(), 1);
    const auto bucket = buffer.buckets()[0];
    size_t begin = 0, end = 0;
    ASSERT_TRUE(bucket->GetTensorLocInBucket(second, begin, end));
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 32);
    ASSERT_TRUE(bucket->GetTensorLocInBucket(first, begin, end));
    EXPECT_EQ(begin,
              64);  // 32 padding elements follow the preceding parameter.
    EXPECT_EQ(end, 96);
    test::ExpectTensorFloatEqual(first, 1.0f);
    test::ExpectTensorFloatEqual(second, 2.0f);
  }
}

TEST_P(OptimizerBucketPaddingTest,
       TwoOptimizerShardsUpdateOnlyTheirActualParameters) {
  auto first = std::make_shared<Tensor>(std::vector<int64_t>{32},
                                        DataType::kFLOAT32, GetDevice());
  auto second = std::make_shared<Tensor>(std::vector<int64_t>{32},
                                         DataType::kFLOAT32, GetDevice());
  first->Fill(1.0f);
  second->Fill(2.0f);
  auto dtype = DataType::kFLOAT32;
  nn::parallel::DistributedDataParallelConfig config;
  config.zero_stage = 1;
  nn::parallel::ParamAndGradBuffer buffer({first, second}, dtype, dtype,
                                          nullptr, 1.0f, config);
  const auto bucket = buffer.buckets()[0];
  buffer.grad_buffer()->Fill(0.0f);
  first->grad()->Fill(0.25f);
  second->grad()->Fill(-0.5f);

  // Emulate two optimizer ranks after reduce-scatter, using the same public
  // bucket metadata consumed by DistributedOptimizer. No communication is
  // needed to expose incorrect offsets, so this also runs on CPU.
  const auto shard_size = bucket->param_data()->NumElements() / 2;
  for (size_t rank = 0; rank < 2; ++rank) {
    std::vector<std::shared_ptr<Tensor>> pieces;
    for (const auto &parameter : {first, second}) {
      size_t begin = 0, end = 0;
      ASSERT_TRUE(bucket->GetTensorLocInBucket(parameter, begin, end));
      begin = std::max(begin, rank * shard_size);
      end = std::min(end, (rank + 1) * shard_size);
      if (begin >= end) {
        continue;
      }
      std::vector<int64_t> shape{static_cast<int64_t>(end - begin)};
      auto piece = std::make_shared<Tensor>(*bucket->param_data(),
                                            begin * sizeof(float), shape);
      piece->set_grad(std::make_shared<Tensor>(*bucket->grad_data(),
                                               begin * sizeof(float), shape));
      pieces.push_back(piece);
    }
    optimizers::SGD optimizer(pieces, 0.1f);
    optimizer.Step();
  }
  test::ExpectTensorFloatEqual(first, 0.975f);
  test::ExpectTensorFloatEqual(second, 2.05f);
}

TEST_P(OptimizerBucketPaddingTest,
       SecondBucketParameterRangesAreRelativeAndAligned) {
  for (int stage : {1, 2}) {
    SCOPED_TRACE(stage);
    std::vector<std::shared_ptr<Tensor>> params;
    for (int64_t numel : {32, 32, 64, 32}) {
      auto param = std::make_shared<Tensor>(std::vector<int64_t>{numel},
                                            DataType::kFLOAT32, GetDevice());
      param->Fill(static_cast<float>(params.size() + 1));
      params.push_back(param);
    }
    auto dtype = DataType::kFLOAT32;
    nn::parallel::DistributedDataParallelConfig config;
    config.zero_stage = stage;
    config.bucket_size_in_elements = 128;
    nn::parallel::ParamAndGradBuffer buffer(params, dtype, dtype, nullptr, 1.0f,
                                            config);
    const auto buckets = buffer.buckets();
    ASSERT_EQ(buckets.size(), 2);
    // Packing is reversed: params[3] and params[2] fill the first bucket;
    // params[1] and params[0] occupy the second, starting at global offset 128.
    for (size_t bucket_idx = 0; bucket_idx < buckets.size(); ++bucket_idx) {
      SCOPED_TRACE(bucket_idx);
      const auto &bucket = buckets[bucket_idx];
      EXPECT_EQ(bucket->offset(), bucket_idx * 128);
      ASSERT_EQ(bucket->params().size(), 2);
      ASSERT_EQ(bucket->param_data()->NumElements(), 128);
      for (size_t slot = 0; slot < 2; ++slot) {
        const size_t param_idx = 3 - bucket_idx * 2 - slot;
        const auto &param = params[param_idx];
        size_t begin = 0, end = 0;
        ASSERT_TRUE(bucket->GetTensorLocInBucket(param, begin, end));
        EXPECT_EQ(begin, slot * 64);
        EXPECT_EQ(end, slot * 64 + param->NumElements());
        auto view = std::make_shared<Tensor>(
            *bucket->param_data(), slot * 64 * sizeof(float), param->Dims());
        test::ExpectTensorFloatEqual(view, static_cast<float>(param_idx + 1));
        test::ExpectTensorFloatEqual(param, static_cast<float>(param_idx + 1));
      }
    }
  }
}

TEST_P(OptimizerBucketPaddingTest,
       ZeRO2AccumulatesGradientsAndUpdatesAlignedParameters) {
  auto first = std::make_shared<Tensor>(std::vector<int64_t>{32},
                                        DataType::kFLOAT32, GetDevice());
  auto second = std::make_shared<Tensor>(std::vector<int64_t>{32},
                                         DataType::kFLOAT32, GetDevice());
  first->Fill(1.0f);
  second->Fill(2.0f);
  auto dtype = DataType::kFLOAT32;
  nn::parallel::DistributedDataParallelConfig config;
  config.zero_stage = 2;
  config.overlap_grad_reduce = false;
  nn::parallel::ParamAndGradBuffer buffer({first, second}, dtype, dtype,
                                          nullptr, 1.0f, config);
  ASSERT_EQ(buffer.buckets().size(), 1);
  const auto bucket = buffer.buckets()[0];
  ASSERT_EQ(bucket->param_data()->NumElements(), 128);
  ASSERT_EQ(bucket->grad_data(), nullptr);
  // Exercise the ZeRO-2 accumulation and synchronization path on a single
  // rank, so the test also runs on CPU without a communication backend.
  nn::parallel::ParamAndGradBucketGroup group(buffer.buckets(), nullptr, 1,
                                              config);
  auto grad =
      std::make_shared<Tensor>(std::vector<int64_t>{32}, dtype, GetDevice());
  grad->Fill(0.25f);
  group.AccumulateParamGrad(first, grad, true, 1.0f);
  grad->Fill(-0.5f);
  group.AccumulateParamGrad(second, grad, true, 1.0f);
  grad->Fill(0.5f);
  group.AccumulateParamGrad(first, grad, false, 1.0f);
  grad->Fill(-0.25f);
  group.AccumulateParamGrad(second, grad, false, 1.0f);
  group.FinishGradSync();

  const auto local_grad = group.GetLocalGradShardBuffer(0);
  ASSERT_NE(local_grad, nullptr);
  // Check actual storage independently of GetTensorLocInBucket: otherwise
  // gradient writes and optimizer reads could share the same wrong offsets.
  std::vector<float> expected_grad(128, 0.0f);
  std::fill_n(expected_grad.begin(), 32, -0.75f);
  std::fill_n(expected_grad.begin() + 64, 32, 0.75f);
  test::ExpectTensorFloatEqual(local_grad, expected_grad);

  std::vector<std::shared_ptr<Tensor>> pieces;
  for (const auto &param : {first, second}) {
    size_t begin = 0, end = 0;
    ASSERT_TRUE(bucket->GetTensorLocInBucket(param, begin, end));
    const std::vector<int64_t> shape{static_cast<int64_t>(end - begin)};
    auto piece = std::make_shared<Tensor>(*bucket->param_data(),
                                          begin * sizeof(float), shape);
    piece->set_grad(
        std::make_shared<Tensor>(*local_grad, begin * sizeof(float), shape));
    pieces.push_back(piece);
  }
  optimizers::SGD optimizer(pieces, 0.125f);
  optimizer.Step();
  test::ExpectTensorFloatEqual(first, 0.90625f);
  test::ExpectTensorFloatEqual(second, 2.09375f);
}

INFINI_TRAIN_REGISTER_TEST(OptimizerBucketPaddingTest);
