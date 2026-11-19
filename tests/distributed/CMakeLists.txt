# ==========================================================================
# Distributed tests
# ==========================================================================

infini_train_add_test(test_rank
  SOURCES test_rank.cc
  LABELS cpu
)

add_test(
  NAME RankTest.MultiNodeSingleProcessIsParallel
  COMMAND ${CMAKE_COMMAND} -E env
    "WORLD_SIZE=2"
    "LOCAL_WORLD_SIZE=1"
    "RANK=0"
    "LOCAL_RANK=0"
    $<TARGET_FILE:test_rank>
    --gtest_filter=RankTest.DetectsParallelismFromGlobalWorldSize
)

set_tests_properties(RankTest.MultiNodeSingleProcessIsParallel
  PROPERTIES
    LABELS cpu
    TIMEOUT 10
)

add_test(
  NAME MultinodeSshLauncherTest
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/test_launch_multinode_ssh.sh
)

set_tests_properties(MultinodeSshLauncherTest
  PROPERTIES
    LABELS cpu
    TIMEOUT 60
)
