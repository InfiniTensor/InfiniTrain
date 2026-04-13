# ============================================================================
# InfiniTrain Test Macros
# ============================================================================
# Unified test configuration interface to reduce boilerplate.
#
# Usage:
#   1. Include this file in tests/CMakeLists.txt
#   2. Use infini_train_add_test macro to register tests
#
# Examples:
#   infini_train_add_test(
#     test_tensor_create
#     SOURCES test_tensor_create.cc
#     LABELS cpu cuda
#   )
# ============================================================================

include_guard(GLOBAL)

# Path to this file's directory (tests/common/)
set(TEST_MACROS_DIR "${CMAKE_CURRENT_LIST_DIR}")

# -----------------------------------------------------------------------------
# Load GoogleTest module (provides gtest_discover_tests)
# -----------------------------------------------------------------------------
include(GoogleTest)

# -----------------------------------------------------------------------------
# infini_train_add_test - Test registration macro
# -----------------------------------------------------------------------------
# Features:
#   1. Create executable target
#   2. Configure compile options, link libraries, and include paths
#   3. Use gtest_discover_tests to auto-discover test cases
#   4. Set test labels
#
# Arguments:
#   SOURCES:    Source file list (required)
#   LABELS:     Test labels, e.g. "cpu" "cuda" "distributed" (optional, default "cpu")
#   TEST_FILTER: gtest test filter pattern (optional)
#
# Examples:
#   # Single-label test (one liner)
#   infini_train_add_test(test_example SOURCES test_example.cc LABELS cpu)
#
#   # Filter same binary by label suffix (one call per label)
#   infini_train_add_test(test_example SOURCES test_example.cc LABELS cpu TEST_FILTER "-*CUDA*")
#   infini_train_add_test(test_example_cuda SOURCES test_example.cc LABELS cuda TEST_FILTER "*CUDA*")
# -----------------------------------------------------------------------------
macro(infini_train_add_test)
  cmake_parse_arguments(ARG "" "TEST_NAME;TEST_FILTER" "SOURCES;LABELS" ${ARGN})

  if(NOT ARG_TEST_NAME)
    set(ARG_TEST_NAME ${ARG_UNPARSED_ARGUMENTS})
  endif()

  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "infini_train_add_test: TEST_NAME and SOURCES are required")
  endif()

  # 1. Create executable target
  add_executable(${ARG_TEST_NAME} ${ARG_SOURCES})

  # 2. Disable -Werror so tests can run under relaxed warning levels
  target_compile_options(${ARG_TEST_NAME} PRIVATE -Wno-error)

  # 3. Link Google Test
  target_link_libraries(${ARG_TEST_NAME} PRIVATE
    GTest::gtest
    GTest::gtest_main
  )

  # 4. Add include paths
  target_include_directories(${ARG_TEST_NAME} PRIVATE
    ${TEST_MACROS_DIR}
    ${glog_SOURCE_DIR}/src
  )

  # 5. Link project library (reuses framework linking strategy)
  link_infini_train_exe(${ARG_TEST_NAME})

  # 6. Auto-discover gtest cases and register as ctest tests
  set(labels "cpu")
  if(ARG_LABELS)
    set(labels "${ARG_LABELS}")
  endif()

  if(ARG_TEST_FILTER)
    gtest_discover_tests(${ARG_TEST_NAME}
      EXTRA_ARGS --gtest_output=xml:%T.xml
      TEST_FILTER "${ARG_TEST_FILTER}"
      PROPERTIES LABELS "${labels}"
    )
  else()
    gtest_discover_tests(${ARG_TEST_NAME}
      EXTRA_ARGS --gtest_output=xml:%T.xml
      PROPERTIES LABELS "${labels}"
    )
  endif()
endmacro()

# -----------------------------------------------------------------------------
# infini_train_add_test_suite - Register cpu/cuda/distributed targets in one call
# -----------------------------------------------------------------------------
# Calls infini_train_add_test three times (or fewer) with the correct
# TEST_FILTER and LABELS derived from the label list.
#
# Arguments:
#   <name>   Base name; each target is named <name>_<label>
#   SOURCES  Source file list (required)
#   LABELS   Subset of {cpu cuda distributed} (optional, default: all three)
#
# Examples:
#   infini_train_add_test_suite(test_tensor SOURCES ${TENSOR_TEST_SOURCES})
#   infini_train_add_test_suite(test_lora   SOURCES test_lora.cc LABELS cpu)
# -----------------------------------------------------------------------------
macro(infini_train_add_test_suite)
  cmake_parse_arguments(SUITE "" "" "SOURCES;LABELS" ${ARGN})
  set(_suite_name ${SUITE_UNPARSED_ARGUMENTS})

  if(NOT SUITE_LABELS)
    set(SUITE_LABELS cpu cuda distributed)
  endif()

  foreach(_label IN LISTS SUITE_LABELS)
    if(_label STREQUAL "cpu")
      set(_filter "CPU/*")
    elseif(_label STREQUAL "cuda")
      set(_filter "CUDA/*")
    elseif(_label STREQUAL "distributed")
      set(_filter "Distributed/*")
    else()
      message(FATAL_ERROR "infini_train_add_test_suite: unknown label '${_label}'")
    endif()
    infini_train_add_test(${_suite_name}_${_label}
      SOURCES ${SUITE_SOURCES}
      LABELS ${_label}
      TEST_FILTER "${_filter}"
    )
  endforeach()
endmacro()
