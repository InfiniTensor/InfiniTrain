#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "test_utils.h"

using namespace infini_train;

TEST(SlowTest, Cpu) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_TRUE(true);
}

TEST(SlowTest, Cuda) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto count = infini_train::test::GetCudaDeviceCount();
    EXPECT_GT(count, 0);
#endif
}

TEST(SlowTest, Distributed) {
    REQUIRE_DISTRIBUTED();
#if defined(USE_CUDA) && defined(USE_NCCL)
    EXPECT_GE(infini_train::test::GetCudaDeviceCount(), 2);
#endif
}
