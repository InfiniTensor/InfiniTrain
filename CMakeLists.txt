cmake_minimum_required(VERSION 3.28)

option(USE_CUDA "Support NVIDIA CUDA" OFF)
option(PROFILE_MODE "ENABLE PROFILE MODE" OFF)
option(USE_OMP "Use OpenMP as backend for Eigen" ON)
option(USE_NCCL "Build project for distributed running on CUDA using NCCL" ON)
option(BUILD_TEST "Build InfiniTrain tests" OFF)

project(infini_train VERSION 0.6.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wunused -Wunused-function)

# ------------------------------------------------------------------------------
# GoogleTest (submodule)
# ------------------------------------------------------------------------------
if(BUILD_TEST)
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt)
    message(FATAL_ERROR "googletest submodule not found at third_party/googletest. "
      "Run: git submodule update --init third_party/googletest")
  endif()
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory(third_party/googletest)
  enable_testing()
endif()

# ------------------------------------------------------------------------------
# Third-party deps
# ------------------------------------------------------------------------------

# gflags
add_subdirectory(third_party/gflags)
include_directories(${gflags_SOURCE_DIR}/include)

# glog
set(WITH_GFLAGS OFF CACHE BOOL "Disable glog finding system gflags" FORCE)
set(WITH_GTEST OFF CACHE BOOL "Disable glog finding system gtest" FORCE)
get_property(_infinitrain_build_testing_was_cached CACHE BUILD_TESTING PROPERTY TYPE SET)
block()
  set(BUILD_TESTING OFF)
  add_subdirectory(third_party/glog)
endblock()
if(NOT _infinitrain_build_testing_was_cached)
  unset(BUILD_TESTING CACHE)
endif()
unset(_infinitrain_build_testing_was_cached)
include_directories(${glog_SOURCE_DIR}/src)

# eigen
if(USE_OMP)
  find_package(OpenMP REQUIRED)
endif()
add_subdirectory(third_party/eigen)
include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen)

include_directories(${PROJECT_SOURCE_DIR})

if(PROFILE_MODE)
  add_compile_definitions(PROFILE_MODE=1)
endif()

# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

# Framework core sources (*.cc), excluding cpu kernels (they are built separately)
file(GLOB_RECURSE SRC ${PROJECT_SOURCE_DIR}/infini_train/src/*.cc)
list(FILTER SRC EXCLUDE REGEX ".*kernels/cpu/.*")

# Exclude backend-specific runtime/ccl translation units when the corresponding
# backend is disabled. This keeps each build self-contained and avoids pulling
# in headers (e.g. <cuda_runtime.h> / <mcr/mc_runtime.h>) that are not on the
# include path.
if(NOT USE_CUDA)
  list(FILTER SRC EXCLUDE REGEX ".*/(ccl|runtime)/cuda/.*")
elseif(NOT USE_NCCL)
  list(FILTER SRC EXCLUDE REGEX ".*/ccl/cuda/.*")
endif()

# CPU kernels (*.cc)
file(GLOB_RECURSE CPU_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/kernels/cpu/*.cc)

# ------------------------------------------------------------------------------
# CPU kernels library
# ------------------------------------------------------------------------------

add_library(infini_train_cpu_kernels STATIC ${CPU_KERNELS})
target_link_libraries(infini_train_cpu_kernels PUBLIC glog Eigen3::Eigen)

if(USE_OMP)
  add_compile_definitions(USE_OMP=1)
  target_link_libraries(infini_train_cpu_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()

# ------------------------------------------------------------------------------
# CUDA kernels library (optional)
# ------------------------------------------------------------------------------

if(USE_CUDA)
  add_compile_definitions(USE_CUDA=1)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})

  # CUDA compilation options
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

  # Only compile CUDA kernels / cuda sources here (your original used src/*.cu)
  file(GLOB_RECURSE CUDA_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/*.cu)

  add_library(infini_train_cuda_kernels STATIC ${CUDA_KERNELS})
  set_target_properties(infini_train_cuda_kernels PROPERTIES CUDA_ARCHITECTURES "75;80;90")

  target_link_libraries(infini_train_cuda_kernels
    PUBLIC
      glog
      CUDA::cudart
      CUDA::cublas
      CUDA::cuda_driver
  )

  if(USE_NCCL)
    message(STATUS "Add USE_NCCL, use NCCL with CUDA")
    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
    find_package(NCCL REQUIRED)
    add_compile_definitions(USE_NCCL=1)
    target_link_libraries(infini_train_cuda_kernels PUBLIC nccl)
  endif()
endif()

# ------------------------------------------------------------------------------
# Main framework library
# ------------------------------------------------------------------------------

add_library(infini_train STATIC ${SRC})
add_library(InfiniTrain::infini_train ALIAS infini_train)
add_library(InfiniTrain::cpu_kernels ALIAS infini_train_cpu_kernels)
target_compile_features(infini_train PUBLIC cxx_std_20)
target_include_directories(infini_train PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
target_link_libraries(infini_train
  PUBLIC
    glog
    gflags
    Eigen3::Eigen
)
if(USE_OMP)
  target_link_libraries(infini_train PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_CUDA)
  # infini_train contains cuda runtime wrappers (*.cc) like cuda_blas_handle.cc/cuda_guard.cc
  # Those may need CUDA runtime/driver/cublas symbols at final link, so attach them here too.
  target_link_libraries(infini_train
    PUBLIC
      CUDA::cudart
      CUDA::cublas
      CUDA::cuda_driver
  )

  if(USE_NCCL)
    # If your core library code also directly references NCCL symbols (not only kernels),
    # keep this. Otherwise it's harmless.
    target_link_libraries(infini_train PUBLIC nccl)
  endif()
endif()

# ------------------------------------------------------------------------------
# Final executable link interface
# ------------------------------------------------------------------------------

set(_infini_train_executable_archives
  infini_train
  infini_train_cpu_kernels
)
if(USE_CUDA)
  list(APPEND _infini_train_executable_archives infini_train_cuda_kernels)
endif()
set_property(GLOBAL PROPERTY INFINI_TRAIN_EXECUTABLE_ARCHIVES
  ${_infini_train_executable_archives})

function(link_infini_train_exe target_name)
  cmake_parse_arguments(LINK "" "" "EXTRA_ARCHIVES" ${ARGN})
  if(LINK_KEYWORDS_MISSING_VALUES)
    message(FATAL_ERROR
      "link_infini_train_exe: missing values for ${LINK_KEYWORDS_MISSING_VALUES}")
  endif()
  if(LINK_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "link_infini_train_exe: unexpected arguments: ${LINK_UNPARSED_ARGUMENTS}")
  endif()

  get_target_property(_target_type ${target_name} TYPE)
  if(_target_type STREQUAL "INTERFACE_LIBRARY")
    set(_link_scope INTERFACE)
  elseif(_target_type STREQUAL "EXECUTABLE")
    set(_link_scope PRIVATE)
  else()
    message(FATAL_ERROR
      "link_infini_train_exe requires an executable or interface library target")
  endif()

  get_property(_framework_archives GLOBAL PROPERTY INFINI_TRAIN_EXECUTABLE_ARCHIVES)
  target_link_libraries(${target_name} ${_link_scope}
    "-Wl,--start-group"
    "-Wl,--whole-archive"
    ${_framework_archives}
    ${LINK_EXTRA_ARCHIVES}
    "-Wl,--no-whole-archive"
    "-Wl,--end-group"
  )
endfunction()

# Libraries and providers link InfiniTrain::infini_train for the core API. Final
# executables must link InfiniTrain::infini_train_executable or a provider-specific
# executable interface; linking only InfiniTrain::infini_train may discard static
# registration objects.
# EXTRA_ARCHIVES is reserved for archives that still rely on static initialization.
add_library(infini_train_executable INTERFACE)
link_infini_train_exe(infini_train_executable)
add_library(InfiniTrain::infini_train_executable ALIAS infini_train_executable)
unset(_infini_train_executable_archives)

# ------------------------------------------------------------------------------
# Examples
# ------------------------------------------------------------------------------

if(PROJECT_IS_TOP_LEVEL)
add_executable(mnist
  example/mnist/main.cc
  example/mnist/dataset.cc
  example/mnist/net.cc
)
link_infini_train_exe(mnist)

add_executable(gpt2
  example/gpt2/main.cc
  example/common/tiny_shakespeare_dataset.cc
  example/common/utils.cc
  example/common/tokenizer.cc
  example/gpt2/checkpoint_loader.cc
)
link_infini_train_exe(gpt2)

add_executable(mixtral
  example/mixtral/main.cc
  example/common/tiny_shakespeare_dataset.cc
  example/common/utils.cc
  example/mixtral/checkpoint_loader.cc
)
link_infini_train_exe(mixtral)

add_executable(llama3
  example/llama3/main.cc
  example/common/tiny_shakespeare_dataset.cc
  example/common/utils.cc
  example/common/tokenizer.cc
  example/llama3/checkpoint_loader.cc
)
link_infini_train_exe(llama3)
endif()

# Tools
if(PROJECT_IS_TOP_LEVEL)
add_subdirectory(tools/infini_run)
set_target_properties(infini_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

# Tests
if(BUILD_TEST)
  add_subdirectory(tests)
endif()

# Negative compile test: missing dtype registration must fail at compile time.
set(DTYPE_DISPATCH_COMPILE_FAIL_SOURCE
  ${PROJECT_SOURCE_DIR}/tests/dtype/test_dtype_dispatch_compile_fail.cc)

add_executable(test_dtype_dispatch_compile_fail EXCLUDE_FROM_ALL
  ${DTYPE_DISPATCH_COMPILE_FAIL_SOURCE}
)

target_include_directories(test_dtype_dispatch_compile_fail PRIVATE
  ${PROJECT_SOURCE_DIR}
)

link_infini_train_exe(test_dtype_dispatch_compile_fail)
