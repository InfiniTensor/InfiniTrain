#include <optional>

#include "gtest/gtest.h"

#if defined(INFINI_TRAIN_TEST_BACKEND_HEADER) && !defined(INFINI_TRAIN_TEST_BACKEND_REGISTRAR)
#error "INFINI_TRAIN_TEST_BACKEND_REGISTRAR must be defined with INFINI_TRAIN_TEST_BACKEND_HEADER"
#elif !defined(INFINI_TRAIN_TEST_BACKEND_HEADER) && defined(INFINI_TRAIN_TEST_BACKEND_REGISTRAR)
#error "INFINI_TRAIN_TEST_BACKEND_HEADER must be defined with INFINI_TRAIN_TEST_BACKEND_REGISTRAR"
#endif

#if defined(INFINI_TRAIN_TEST_BACKEND_HEADER)
#include INFINI_TRAIN_TEST_BACKEND_HEADER
#endif

#include "infini_train/include/nn/parallel/global.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
#if defined(INFINI_TRAIN_TEST_BACKEND_REGISTRAR)
    INFINI_TRAIN_TEST_BACKEND_REGISTRAR();
#endif
    infini_train::nn::parallel::global::GlobalEnv::Instance().Init(
        /*nthread_per_process=*/1,
        /*tensor_parallel_size=*/1,
        /*sequence_parallel_enabled=*/false,
        /*pipeline_parallel_size=*/1,
        /*virtual_pipeline_parallel_size=*/1,
        /*expert_parallel_size=*/1,
        /*expert_tensor_parallel_size=*/1);
    return RUN_ALL_TESTS();
}
