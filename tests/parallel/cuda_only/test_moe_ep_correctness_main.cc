#include <cstdlib>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/global.h"

#ifndef INFINI_TRAIN_TEST_EP
#error "INFINI_TRAIN_TEST_EP must be defined"
#endif

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    setenv("NNODES", "1", 1);
    setenv("NPROC_PER_NODE", "1", 1);
    setenv("WORLD_SIZE", "1", 1);
    setenv("GLOBAL_PROC_RANK", "0", 1);
    setenv("LOCAL_PROC_RANK", "0", 1);
    infini_train::nn::parallel::global::InitAllEnv(
        /*nthread_per_process=*/INFINI_TRAIN_TEST_EP,
        /*tensor_parallel_size=*/1,
        /*sequence_parallel_enabled=*/false,
        /*pipeline_parallel_size=*/1,
        /*virtual_pipeline_parallel=*/1,
        /*expert_parallel_size=*/INFINI_TRAIN_TEST_EP,
        /*expert_tensor_parallel_size=*/1);
    return RUN_ALL_TESTS();
}
