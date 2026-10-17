#include <cstdlib>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/global.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    setenv("WORLD_SIZE", "16", 1);
    setenv("LOCAL_WORLD_SIZE", "1", 1);
    setenv("RANK", "0", 1);
    setenv("LOCAL_RANK", "0", 1);
    infini_train::nn::parallel::global::GlobalEnv::Instance().Init(
        /*nthread_per_process=*/1,
        /*tensor_parallel_size=*/4,
        /*sequence_parallel_enabled=*/false,
        /*pipeline_parallel_size=*/2,
        /*virtual_pipeline_parallel_size=*/1,
        /*expert_parallel_size=*/2,
        /*expert_tensor_parallel_size=*/1);
    return RUN_ALL_TESTS();
}
