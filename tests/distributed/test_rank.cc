#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/rank.h"

namespace infini_train::nn::parallel {
namespace {

TEST(RankTest, DetectsParallelismFromGlobalWorldSize) {
    const Rank rank(global::GetGlobalProcRank(), 0, global::GetNprocPerNode(), global::GetNthreadPerProc());

    EXPECT_EQ(rank.IsParallel(), global::GetWorldSize() > 1);
}

} // namespace
} // namespace infini_train::nn::parallel
