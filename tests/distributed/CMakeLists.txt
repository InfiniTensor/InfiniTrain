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

infini_train_add_test(test_pipeline_layout
  SOURCES test_pipeline_layout.cc
  LABELS cpu
)

infini_train_add_test(test_pipeline_parallel
  SOURCES test_pipeline_parallel.cc
  LABELS cpu
)

infini_train_add_test(test_pp_layout_parser
  SOURCES test_pp_layout_parser.cc ${CMAKE_SOURCE_DIR}/example/common/parser.cc
  LABELS cpu
)

# Loader fixtures run in separate CPU processes because GlobalEnv is initialized once.
add_executable(test_pipeline_loader
  test_pipeline_loader.cc
  ${CMAKE_SOURCE_DIR}/example/common/parser.cc
  ${CMAKE_SOURCE_DIR}/example/common/utils.cc
  ${CMAKE_SOURCE_DIR}/example/gpt2/checkpoint_loader.cc
  ${CMAKE_SOURCE_DIR}/example/llama3/checkpoint_loader.cc
)
target_link_libraries(test_pipeline_loader PRIVATE GTest::gtest)
link_infini_train_exe(test_pipeline_loader)
foreach(topology IN ITEMS "1:1" "2:1" "2:2")
  string(REPLACE ":" ";" dimensions "${topology}")
  list(GET dimensions 0 pp)
  list(GET dimensions 1 vpp)
  add_test(NAME PipelineLoader.PP${pp}VPP${vpp}
    COMMAND ${CMAKE_COMMAND} -E env WORLD_SIZE=${pp} LOCAL_WORLD_SIZE=${pp} RANK=0 LOCAL_RANK=0
      PIPELINE_LOADER_TEST_PP=${pp} PIPELINE_LOADER_TEST_VPP=${vpp}
      $<TARGET_FILE:test_pipeline_loader>)
  set_tests_properties(PipelineLoader.PP${pp}VPP${vpp} PROPERTIES LABELS cpu TIMEOUT 30)
endforeach()

infini_train_add_test(test_pipeline_schedule
  SOURCES test_pipeline_schedule.cc
  LABELS cpu
)
