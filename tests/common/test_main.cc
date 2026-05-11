#include "infini_train/include/nn/parallel/global.h"
#include "gtest/gtest.h"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    infini_train::nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1);
    return RUN_ALL_TESTS();
}
