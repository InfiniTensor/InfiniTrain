# ==========================================================================
# LoRA tests (single binary)
# ==========================================================================

set(LORA_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LORA_TEST_SOURCES
  ${LORA_TEST_DIR}/test_lora.cc
)

add_executable(test_lora ${LORA_TEST_SOURCES})
target_compile_options(test_lora PRIVATE -Wno-error)
link_infini_train_exe(test_lora)
target_link_libraries(test_lora PRIVATE GTest::gtest GTest::gtest_main)
target_include_directories(test_lora PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${glog_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(test_lora
  EXTRA_ARGS --gtest_output=xml:%T.xml
  PROPERTIES LABELS "cpu"
)
