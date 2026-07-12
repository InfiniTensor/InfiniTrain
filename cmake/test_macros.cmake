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
#   3. Use gtest_discover_tests to auto-discover CPU test cases
#   4. Register CUDA tests at binary granularity with CTest GPU resources
#   5. Set test labels
#
# Arguments:
#   SOURCES:    Source file list (required)
#   LABELS:     Test labels, e.g. "cpu" "cuda" "distributed" (optional, default "cpu")
#   TEST_FILTER:  gtest test filter pattern (optional)
#   TEST_TIMEOUT: ctest timeout in seconds (optional, default 10)
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
  cmake_parse_arguments(ARG "" "TEST_NAME;TEST_FILTER;TEST_TIMEOUT" "SOURCES;LABELS" ${ARGN})

  if(NOT ARG_TEST_NAME)
    set(ARG_TEST_NAME ${ARG_UNPARSED_ARGUMENTS})
  endif()

  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "infini_train_add_test: TEST_NAME and SOURCES are required")
  endif()

  # 1. Create executable target
  add_executable(${ARG_TEST_NAME} ${ARG_SOURCES} $<TARGET_OBJECTS:test_main>)

  # 2. Disable -Werror so tests can run under relaxed warning levels
  target_compile_options(${ARG_TEST_NAME} PRIVATE -Wno-error)

  # 3. Link Google Test (uses custom main from test_main that initializes GlobalEnv)
  target_link_libraries(${ARG_TEST_NAME} PRIVATE GTest::gtest)

  # 4. Add include paths
  target_include_directories(${ARG_TEST_NAME} PRIVATE
    ${glog_SOURCE_DIR}/src
  )

  # 5. Link project library (reuses framework linking strategy)
  link_infini_train_exe(${ARG_TEST_NAME})

  # 6. Register tests
  set(labels "cpu")
  if(ARG_LABELS)
    set(labels "${ARG_LABELS}")
  endif()

  set(test_timeout 10)
  if(ARG_TEST_TIMEOUT)
    set(test_timeout ${ARG_TEST_TIMEOUT})
  endif()

  list(FIND labels cuda _has_cuda_label)
  if(NOT _has_cuda_label EQUAL -1)
    set(_cuda_test_args)
    if(ARG_TEST_FILTER)
      list(APPEND _cuda_test_args --gtest_filter=${ARG_TEST_FILTER})
    endif()

    add_test(
      NAME ${ARG_TEST_NAME}
      COMMAND $<TARGET_FILE:${ARG_TEST_NAME}> ${_cuda_test_args}
    )
    set_tests_properties(${ARG_TEST_NAME}
      PROPERTIES
        LABELS "${labels}"
        TIMEOUT ${test_timeout}
    )
  elseif(ARG_TEST_FILTER)
    gtest_discover_tests(${ARG_TEST_NAME}
      TEST_FILTER "${ARG_TEST_FILTER}"
      DISCOVERY_TIMEOUT 10
      PROPERTIES LABELS "${labels}" TIMEOUT ${test_timeout}
    )
  else()
    gtest_discover_tests(${ARG_TEST_NAME}
      PROPERTIES LABELS "${labels}" TIMEOUT ${test_timeout}
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
#   LABELS   Subset of {cpu cuda} (optional, default: both)
#   TEST_TIMEOUT ctest timeout in seconds (optional, default 10)
#
# Examples:
#   infini_train_add_test_suite(test_tensor SOURCES ${TENSOR_TEST_SOURCES})
#   infini_train_add_test_suite(test_lora   SOURCES test_lora.cc LABELS cpu)
# -----------------------------------------------------------------------------
macro(infini_train_add_test_suite)
  cmake_parse_arguments(SUITE "" "TEST_TIMEOUT" "SOURCES;LABELS" ${ARGN})
  set(_suite_name ${SUITE_UNPARSED_ARGUMENTS})

  if(NOT SUITE_LABELS)
    set(SUITE_LABELS cpu cuda)
  endif()

  set(suite_test_timeout 10)
  if(SUITE_TEST_TIMEOUT)
    set(suite_test_timeout ${SUITE_TEST_TIMEOUT})
  endif()

  foreach(_label IN LISTS SUITE_LABELS)
    string(TOUPPER ${_label} _label_upper)
    set(_filter "${_label_upper}/*")
    infini_train_add_test(${_suite_name}_${_label}
      SOURCES ${SUITE_SOURCES}
      LABELS ${_label}
      TEST_FILTER "${_filter}"
      TEST_TIMEOUT ${suite_test_timeout}
    )
  endforeach()
endmacro()
