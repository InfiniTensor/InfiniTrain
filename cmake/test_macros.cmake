# ============================================================================
# InfiniTrain Test Macros
# ============================================================================
# Unified test configuration interface to reduce boilerplate.
#
# Usage:
#   1. Include this file in tests/CMakeLists.txt
#   2. Use infini_train_add_test to register tests
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
# infini_train_add_test - Test registration function
# -----------------------------------------------------------------------------
# Features:
#   1. Create executable target
#   2. Configure compile options, link libraries, and include paths
#   3. Use gtest_discover_tests to auto-discover host test cases
#   4. Register accelerator tests at binary granularity
#   5. Set test labels and execution properties
#
# Arguments:
#   SOURCES:    Source file list (required)
#   LABELS:     Test labels, e.g. "cpu" "cuda" "distributed" (optional, default "cpu")
#   TEST_FILTER:       gtest test filter pattern (optional)
#   TEST_TIMEOUT:      ctest timeout in seconds (optional, default 10)
#   TEST_MAIN_SOURCE:  per-target main source (optional)
#   TEST_MAIN_TARGET:  object-library main target (optional, defaults to test_main)
#   RUNTIME_OUTPUT_DIRECTORY: executable output directory (optional)
#   LINK_LIBRARIES:    additional link libraries (optional)
#   COMPILE_DEFINITIONS: target compile definitions (optional)
#   BINARY_REGISTRATION: register the binary as one CTest test (optional)
#   SKIP_DEFAULT_FRAMEWORK_LINK: LINK_LIBRARIES supplies the full link closure
#   RUN_SERIAL: do not run this CTest target concurrently (optional)
#
# Examples:
#   # Single-label test (one liner)
#   infini_train_add_test(test_example SOURCES test_example.cc LABELS cpu)
#
#   # Filter same binary by label suffix (one call per label)
#   infini_train_add_test(test_example SOURCES test_example.cc LABELS cpu TEST_FILTER "-*CUDA*")
#   infini_train_add_test(test_example_cuda SOURCES test_example.cc LABELS cuda TEST_FILTER "*CUDA*")
# -----------------------------------------------------------------------------
function(infini_train_add_test)
  cmake_parse_arguments(ARG
    "BINARY_REGISTRATION;RUN_SERIAL;SKIP_DEFAULT_FRAMEWORK_LINK"
    "TEST_NAME;TEST_FILTER;TEST_TIMEOUT;TEST_MAIN_SOURCE;TEST_MAIN_TARGET;RUNTIME_OUTPUT_DIRECTORY"
    "SOURCES;LABELS;LINK_LIBRARIES;COMPILE_DEFINITIONS"
    ${ARGN}
  )

  if(ARG_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "infini_train_add_test: missing values for ${ARG_KEYWORDS_MISSING_VALUES}")
  endif()

  if(ARG_TEST_NAME AND ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "infini_train_add_test: unexpected arguments: ${ARG_UNPARSED_ARGUMENTS}")
  elseif(NOT ARG_TEST_NAME)
    set(ARG_TEST_NAME ${ARG_UNPARSED_ARGUMENTS})
  endif()

  list(LENGTH ARG_TEST_NAME _test_name_count)
  if(NOT _test_name_count EQUAL 1 OR NOT ARG_SOURCES)
    message(FATAL_ERROR "infini_train_add_test: TEST_NAME and SOURCES are required")
  endif()
  if(ARG_TEST_MAIN_SOURCE AND ARG_TEST_MAIN_TARGET)
    message(FATAL_ERROR
      "infini_train_add_test: TEST_MAIN_SOURCE and TEST_MAIN_TARGET are mutually exclusive")
  endif()

  # 1. Create executable target.
  if(ARG_TEST_MAIN_SOURCE)
    add_executable(${ARG_TEST_NAME} ${ARG_SOURCES} ${ARG_TEST_MAIN_SOURCE})
  elseif(ARG_TEST_MAIN_TARGET)
    if(NOT TARGET ${ARG_TEST_MAIN_TARGET})
      message(FATAL_ERROR
        "infini_train_add_test: TEST_MAIN_TARGET '${ARG_TEST_MAIN_TARGET}' does not exist")
    endif()
    add_executable(${ARG_TEST_NAME} ${ARG_SOURCES} $<TARGET_OBJECTS:${ARG_TEST_MAIN_TARGET}>)
  else()
    if(NOT TARGET test_main)
      message(FATAL_ERROR
        "infini_train_add_test: default test_main target does not exist")
    endif()
    add_executable(${ARG_TEST_NAME} ${ARG_SOURCES} $<TARGET_OBJECTS:test_main>)
  endif()
  if(ARG_RUNTIME_OUTPUT_DIRECTORY)
    set_target_properties(${ARG_TEST_NAME} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${ARG_RUNTIME_OUTPUT_DIRECTORY}")
  endif()

  # 2. Disable -Werror so tests can run under relaxed warning levels
  target_compile_options(${ARG_TEST_NAME} PRIVATE -Wno-error)

  # 3. Link Google Test (uses custom main from test_main that initializes GlobalEnv)
  target_link_libraries(${ARG_TEST_NAME} PRIVATE GTest::gtest)

  # 4. Link the framework. External backends may supply a complete executable
  # link target that owns static archive retention and link ordering.
  if(NOT ARG_SKIP_DEFAULT_FRAMEWORK_LINK)
    link_infini_train_exe(${ARG_TEST_NAME})
  endif()
  if(ARG_LINK_LIBRARIES)
    target_link_libraries(${ARG_TEST_NAME} PRIVATE ${ARG_LINK_LIBRARIES})
  endif()
  if(ARG_COMPILE_DEFINITIONS)
    target_compile_definitions(${ARG_TEST_NAME} PRIVATE ${ARG_COMPILE_DEFINITIONS})
  endif()

  # 5. Register tests
  set(labels "cpu")
  if(ARG_LABELS)
    set(labels "${ARG_LABELS}")
  endif()

  set(test_timeout 10)
  if(DEFINED ARG_TEST_TIMEOUT AND NOT ARG_TEST_TIMEOUT STREQUAL "")
    set(test_timeout ${ARG_TEST_TIMEOUT})
  endif()

  # Hardware discovery would execute accelerator binaries during the build.
  # Register them as one CTest entry and defer execution to ctest.
  if(ARG_BINARY_REGISTRATION)
    set(_binary_test_args)
    if(ARG_TEST_FILTER)
      list(APPEND _binary_test_args --gtest_filter=${ARG_TEST_FILTER})
    endif()

    add_test(
      NAME ${ARG_TEST_NAME}
      COMMAND $<TARGET_FILE:${ARG_TEST_NAME}> ${_binary_test_args}
    )
    set_tests_properties(${ARG_TEST_NAME}
      PROPERTIES
        LABELS "${labels}"
        TIMEOUT ${test_timeout}
    )
    if(ARG_RUN_SERIAL)
      set_tests_properties(${ARG_TEST_NAME} PROPERTIES RUN_SERIAL TRUE)
    endif()
  elseif(ARG_TEST_FILTER)
    set(_discovered_test_properties LABELS "${labels}" TIMEOUT ${test_timeout})
    if(ARG_RUN_SERIAL)
      list(APPEND _discovered_test_properties RUN_SERIAL TRUE)
    endif()
    gtest_discover_tests(${ARG_TEST_NAME}
      TEST_FILTER "${ARG_TEST_FILTER}"
      DISCOVERY_TIMEOUT 10
      PROPERTIES ${_discovered_test_properties}
    )
  else()
    set(_discovered_test_properties LABELS "${labels}" TIMEOUT ${test_timeout})
    if(ARG_RUN_SERIAL)
      list(APPEND _discovered_test_properties RUN_SERIAL TRUE)
    endif()
    gtest_discover_tests(${ARG_TEST_NAME}
      PROPERTIES ${_discovered_test_properties}
    )
  endif()
endfunction()

# -----------------------------------------------------------------------------
# infini_train_add_test_suite - Declare a shared device-parameterized suite
# -----------------------------------------------------------------------------
# Declaration is intentionally separate from device instantiation. This lets a
# top-level InfiniTrain build create CPU/CUDA targets while an embedding
# provider creates only its own PrivateUse1 targets from the same source list.
#
# Arguments:
#   <name>   Base name; each instantiated target is named <name>_<suffix>
#   SOURCES  Source file list (required)
#   LABELS   Built-in devices from {cpu cuda} (optional, default: both)
#   TEST_TIMEOUT ctest timeout in seconds (optional, default 10)
#   EXCLUDE_PRIVATEUSE1 do not expose this suite to PrivateUse1 providers
#
# Examples:
#   infini_train_add_test_suite(test_tensor SOURCES ${TENSOR_TEST_SOURCES})
#   infini_train_add_test_suite(test_lora   SOURCES test_lora.cc LABELS cpu)
# -----------------------------------------------------------------------------
function(infini_train_add_test_suite suite_name)
  cmake_parse_arguments(SUITE "EXCLUDE_PRIVATEUSE1" "TEST_TIMEOUT" "SOURCES;LABELS" ${ARGN})
  if(SUITE_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "infini_train_add_test_suite: missing values for ${SUITE_KEYWORDS_MISSING_VALUES}")
  endif()
  if(SUITE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "infini_train_add_test_suite: unexpected arguments: ${SUITE_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT suite_name OR NOT SUITE_SOURCES)
    message(FATAL_ERROR
      "infini_train_add_test_suite: a suite name and SOURCES are required")
  endif()

  # Resolve paths at the declaration site because a provider may instantiate
  # the suite from another directory. Generator expressions stay deferred.
  set(_suite_sources)
  foreach(_source IN LISTS SUITE_SOURCES)
    if(IS_ABSOLUTE "${_source}" OR _source MATCHES "^\\$<")
      list(APPEND _suite_sources "${_source}")
    else()
      list(APPEND _suite_sources "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
    endif()
  endforeach()

  if(NOT SUITE_LABELS)
    set(SUITE_LABELS cpu cuda)
  endif()
  list(REMOVE_DUPLICATES SUITE_LABELS)

  set(_suite_timeout 10)
  if(DEFINED SUITE_TEST_TIMEOUT AND NOT SUITE_TEST_TIMEOUT STREQUAL "")
    set(_suite_timeout ${SUITE_TEST_TIMEOUT})
  endif()

  # Global properties carry suite metadata across directory scopes and make it
  # visible to an embedding PrivateUse1 provider.
  get_property(_registered_suites GLOBAL PROPERTY INFINI_TRAIN_TEST_SUITES)
  if(suite_name IN_LIST _registered_suites)
    message(FATAL_ERROR
      "infini_train_add_test_suite: duplicate shared suite '${suite_name}'")
  endif()
  set_property(GLOBAL APPEND PROPERTY INFINI_TRAIN_TEST_SUITES "${suite_name}")
  set_property(GLOBAL PROPERTY
    "INFINI_TRAIN_TEST_SUITE_${suite_name}_SOURCES" "${_suite_sources}")
  set_property(GLOBAL PROPERTY
    "INFINI_TRAIN_TEST_SUITE_${suite_name}_TIMEOUT" "${_suite_timeout}")
  set_property(GLOBAL PROPERTY
    "INFINI_TRAIN_TEST_SUITE_${suite_name}_BINARY_DIR" "${CMAKE_CURRENT_BINARY_DIR}")

  foreach(_label IN LISTS SUITE_LABELS)
    if(_label STREQUAL "cpu")
      set_property(GLOBAL APPEND PROPERTY INFINI_TRAIN_CPU_TEST_SUITES "${suite_name}")
    elseif(_label STREQUAL "cuda")
      set_property(GLOBAL APPEND PROPERTY INFINI_TRAIN_CUDA_TEST_SUITES "${suite_name}")
    else()
      message(FATAL_ERROR
        "infini_train_add_test_suite: unsupported built-in label '${_label}'")
    endif()
  endforeach()

  if(NOT SUITE_EXCLUDE_PRIVATEUSE1)
    set_property(GLOBAL APPEND PROPERTY INFINI_TRAIN_PRIVATEUSE1_TEST_SUITES "${suite_name}")
  endif()
endfunction()

# -----------------------------------------------------------------------------
# _infini_train_instantiate_test_suites - Instantiate one device target per suite
# -----------------------------------------------------------------------------
function(_infini_train_instantiate_test_suites)
  cmake_parse_arguments(INSTANCE
    "ACCELERATOR;RUN_SERIAL;SKIP_DEFAULT_FRAMEWORK_LINK"
    "SUITE_REGISTRY;TARGET_SUFFIX;GTEST_PREFIX;DEVICE_TYPE;DEVICE_INDEX;TEST_TIMEOUT;TEST_MAIN_TARGET"
    "LABELS;LINK_LIBRARIES"
    ${ARGN}
  )

  if(INSTANCE_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "_infini_train_instantiate_test_suites: missing values for ${INSTANCE_KEYWORDS_MISSING_VALUES}")
  endif()
  if(INSTANCE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "_infini_train_instantiate_test_suites: unexpected arguments: ${INSTANCE_UNPARSED_ARGUMENTS}")
  endif()
  foreach(_required SUITE_REGISTRY TARGET_SUFFIX GTEST_PREFIX DEVICE_TYPE)
    if(NOT DEFINED INSTANCE_${_required} OR INSTANCE_${_required} STREQUAL "")
      message(FATAL_ERROR
        "_infini_train_instantiate_test_suites: ${_required} is required")
    endif()
  endforeach()
  if(NOT INSTANCE_LABELS)
    message(FATAL_ERROR
      "_infini_train_instantiate_test_suites: LABELS are required")
  endif()

  if(NOT DEFINED INSTANCE_DEVICE_INDEX OR INSTANCE_DEVICE_INDEX STREQUAL "")
    set(INSTANCE_DEVICE_INDEX 0)
  endif()

  get_property(_registered_suites GLOBAL PROPERTY "${INSTANCE_SUITE_REGISTRY}")
  if(NOT _registered_suites)
    return()
  endif()

  # Artifact suffix, GTest prefix, and dispatcher device type are independent
  # identities; for example, MACA uses maca, PRIVATEUSE1, and kPrivateUse1.
  set(_device_compile_definitions
    "INFINI_TRAIN_TEST_DEVICE_TYPE=::infini_train::Device::DeviceType::${INSTANCE_DEVICE_TYPE}"
    "INFINI_TRAIN_TEST_DEVICE_INDEX=${INSTANCE_DEVICE_INDEX}"
    "INFINI_TRAIN_TEST_DEVICE_PREFIX=${INSTANCE_GTEST_PREFIX}"
  )

  set(_device_test_options)
  if(INSTANCE_ACCELERATOR)
    list(APPEND _device_test_options BINARY_REGISTRATION)
  endif()
  if(INSTANCE_RUN_SERIAL)
    list(APPEND _device_test_options RUN_SERIAL)
  endif()
  if(INSTANCE_SKIP_DEFAULT_FRAMEWORK_LINK)
    list(APPEND _device_test_options SKIP_DEFAULT_FRAMEWORK_LINK)
  endif()

  set(_device_link_args)
  if(INSTANCE_LINK_LIBRARIES)
    list(APPEND _device_link_args LINK_LIBRARIES ${INSTANCE_LINK_LIBRARIES})
  endif()
  set(_device_main_args)
  if(INSTANCE_TEST_MAIN_TARGET)
    list(APPEND _device_main_args TEST_MAIN_TARGET ${INSTANCE_TEST_MAIN_TARGET})
  endif()

  foreach(_suite_name IN LISTS _registered_suites)
    get_property(_suite_sources GLOBAL PROPERTY
      "INFINI_TRAIN_TEST_SUITE_${_suite_name}_SOURCES")
    get_property(_suite_timeout GLOBAL PROPERTY
      "INFINI_TRAIN_TEST_SUITE_${_suite_name}_TIMEOUT")
    get_property(_suite_binary_dir GLOBAL PROPERTY
      "INFINI_TRAIN_TEST_SUITE_${_suite_name}_BINARY_DIR")
    if(NOT _suite_sources OR NOT _suite_binary_dir)
      message(FATAL_ERROR
        "Shared test suite '${_suite_name}' has incomplete registration metadata")
    endif()
    if(DEFINED INSTANCE_TEST_TIMEOUT AND NOT INSTANCE_TEST_TIMEOUT STREQUAL "")
      set(_suite_timeout ${INSTANCE_TEST_TIMEOUT})
    endif()

    set(_target_name "${_suite_name}_${INSTANCE_TARGET_SUFFIX}")
    if(TARGET ${_target_name})
      message(FATAL_ERROR "Test target '${_target_name}' already exists")
    endif()

    # Preserve the declaring suite's output layout even when an external
    # provider creates the target from its own CMake directory.
    infini_train_add_test(${_target_name}
      SOURCES ${_suite_sources}
      LABELS ${INSTANCE_LABELS}
      TEST_TIMEOUT ${_suite_timeout}
      RUNTIME_OUTPUT_DIRECTORY "${_suite_binary_dir}"
      COMPILE_DEFINITIONS ${_device_compile_definitions}
      ${_device_main_args}
      ${_device_link_args}
      ${_device_test_options}
    )
  endforeach()
endfunction()

# -----------------------------------------------------------------------------
# infini_train_add_privateuse1_test_suites - Instantiate a provider's suites
# -----------------------------------------------------------------------------
# A PrivateUse1 provider calls this after creating its final executable link
# target. The provider name is used as the target/binary suffix and CTest label;
# the GTest prefix and device type remain on the PrivateUse1 contract.
#
# Example:
#   infini_train_add_privateuse1_test_suites(
#     BACKEND_NAME ${INFINITRAIN_BACKEND}
#     LINK_LIBRARIES PrivateUse1Backend::Executable
#     BACKEND_HEADER "privateuse1_backend/backend.h"
#     BACKEND_REGISTRAR privateuse1_backend::RegisterBackend
#     RUN_SERIAL
#   )
# -----------------------------------------------------------------------------
function(infini_train_add_privateuse1_test_suites)
  cmake_parse_arguments(PRIVATEUSE1
    "RUN_SERIAL"
    "BACKEND_NAME;DEVICE_INDEX;TEST_TIMEOUT;BACKEND_HEADER;BACKEND_REGISTRAR"
    "LINK_LIBRARIES"
    ${ARGN}
  )

  if(PRIVATEUSE1_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "infini_train_add_privateuse1_test_suites: missing values for ${PRIVATEUSE1_KEYWORDS_MISSING_VALUES}")
  endif()
  if(PRIVATEUSE1_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "infini_train_add_privateuse1_test_suites: unexpected arguments: ${PRIVATEUSE1_UNPARSED_ARGUMENTS}")
  endif()

  if(USE_CUDA)
    message(FATAL_ERROR
      "infini_train_add_privateuse1_test_suites requires USE_CUDA=OFF")
  endif()

  foreach(_required BACKEND_NAME BACKEND_HEADER BACKEND_REGISTRAR)
    if(NOT PRIVATEUSE1_${_required})
      message(FATAL_ERROR
        "infini_train_add_privateuse1_test_suites: ${_required} is required")
    endif()
  endforeach()
  if(PRIVATEUSE1_BACKEND_NAME STREQUAL "cpu" OR
     PRIVATEUSE1_BACKEND_NAME STREQUAL "cuda" OR
     NOT PRIVATEUSE1_BACKEND_NAME MATCHES "^[a-z0-9_]+$")
    message(FATAL_ERROR
      "infini_train_add_privateuse1_test_suites: BACKEND_NAME must be non-reserved and contain only lowercase ASCII letters, digits, and underscores")
  endif()
  if(NOT PRIVATEUSE1_LINK_LIBRARIES)
    message(FATAL_ERROR
      "infini_train_add_privateuse1_test_suites: LINK_LIBRARIES must provide the complete executable link closure")
  endif()
  if(NOT DEFINED PRIVATEUSE1_DEVICE_INDEX OR PRIVATEUSE1_DEVICE_INDEX STREQUAL "")
    set(PRIVATEUSE1_DEVICE_INDEX 0)
  endif()

  get_property(_registered_suites GLOBAL PROPERTY INFINI_TRAIN_PRIVATEUSE1_TEST_SUITES)
  if(NOT _registered_suites)
    message(FATAL_ERROR
      "No shared PrivateUse1 test suites are registered. Configure InfiniTrain with BUILD_TEST=ON first.")
  endif()

  # All providers occupy the single kPrivateUse1 slot, so instantiate at most
  # one provider in a build tree.
  get_property(_privateuse1_registered GLOBAL PROPERTY INFINI_TRAIN_PRIVATEUSE1_TEST_PROVIDER_REGISTERED)
  if(_privateuse1_registered)
    message(FATAL_ERROR
      "PrivateUse1 test suites have already been instantiated in this build")
  endif()

  set(_privateuse1_options SKIP_DEFAULT_FRAMEWORK_LINK ACCELERATOR)
  if(PRIVATEUSE1_RUN_SERIAL)
    list(APPEND _privateuse1_options RUN_SERIAL)
  endif()

  set(_privateuse1_timeout_args)
  if(DEFINED PRIVATEUSE1_TEST_TIMEOUT AND NOT PRIVATEUSE1_TEST_TIMEOUT STREQUAL "")
    list(APPEND _privateuse1_timeout_args TEST_TIMEOUT ${PRIVATEUSE1_TEST_TIMEOUT})
  endif()

  # Only the process entry point needs provider initialization. Compile it once
  # and share the object across all suites for this provider.
  set(_privateuse1_main_target
    "infini_train_test_main_${PRIVATEUSE1_BACKEND_NAME}")
  if(TARGET ${_privateuse1_main_target})
    message(FATAL_ERROR
      "Test main target '${_privateuse1_main_target}' already exists")
  endif()
  add_library(${_privateuse1_main_target} OBJECT
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tests/common/test_main.cc")
  target_include_directories(${_privateuse1_main_target} PRIVATE
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
  target_compile_definitions(${_privateuse1_main_target} PRIVATE
    "INFINI_TRAIN_TEST_BACKEND_HEADER=\"${PRIVATEUSE1_BACKEND_HEADER}\""
    "INFINI_TRAIN_TEST_BACKEND_REGISTRAR=${PRIVATEUSE1_BACKEND_REGISTRAR}"
  )
  target_link_libraries(${_privateuse1_main_target} PRIVATE
    GTest::gtest
    ${PRIVATEUSE1_LINK_LIBRARIES}
  )

  _infini_train_instantiate_test_suites(
    SUITE_REGISTRY INFINI_TRAIN_PRIVATEUSE1_TEST_SUITES
    TARGET_SUFFIX ${PRIVATEUSE1_BACKEND_NAME}
    GTEST_PREFIX PRIVATEUSE1
    DEVICE_TYPE kPrivateUse1
    DEVICE_INDEX ${PRIVATEUSE1_DEVICE_INDEX}
    LABELS ${PRIVATEUSE1_BACKEND_NAME} accelerator hardware
    TEST_MAIN_TARGET ${_privateuse1_main_target}
    LINK_LIBRARIES ${PRIVATEUSE1_LINK_LIBRARIES}
    ${_privateuse1_timeout_args}
    ${_privateuse1_options}
  )
  set_property(GLOBAL PROPERTY INFINI_TRAIN_PRIVATEUSE1_TEST_PROVIDER_REGISTERED TRUE)
endfunction()
