# ==========================================================================
# Distributed tests
# ==========================================================================

infini_train_add_test(test_rank
  SOURCES test_rank.cc
  LABELS cpu
)

infini_train_add_test(test_pipeline_layout
  SOURCES test_pipeline_layout.cc
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

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME PipelineLayoutSuggestionTest
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_pipeline_layout_suggestion.py
  )
  set_tests_properties(PipelineLayoutSuggestionTest PROPERTIES LABELS cpu TIMEOUT 10)
endif()
