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

INFINI_TRAIN_REGISTER_TEST(OptimizerBucketPaddingTest);
