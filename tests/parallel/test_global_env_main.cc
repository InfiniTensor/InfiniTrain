#include <cstdlib>
#include <optional>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/model_parallel_config.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  setenv("WORLD_SIZE", "16", 1);
  setenv("LOCAL_WORLD_SIZE", "1", 1);
  setenv("RANK", "0", 1);
  setenv("LOCAL_RANK", "0", 1);
  infini_train::nn::parallel::ModelParallelConfig model_parallel_config;
  model_parallel_config.tensor_model_parallel_size = 4;
  model_parallel_config.pipeline_model_parallel_size = 2;
  model_parallel_config.expert_model_parallel_size = 2;
  model_parallel_config.expert_tensor_parallel_size = 1;
  infini_train::nn::parallel::global::GlobalEnv::Instance().Init(
      /*nthread_per_process=*/1, model_parallel_config);
  return RUN_ALL_TESTS();
}
