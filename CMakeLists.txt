cmake_minimum_required(VERSION 3.28)

option(USE_CUDA "Support NVIDIA CUDA" OFF)
option(PROFILE_MODE "ENABLE PROFILE MODE" OFF)
option(USE_OMP "Use OpenMP as backend for Eigen" ON)
option(USE_NCCL "Build project for distributed running" ON)
option(BUILD_TEST "Build InfiniTrain tests" OFF)
option(USE_FLASH_ATTENTION "Enable FlashAttention 2 CUDA backend" OFF)

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
add_subdirectory(third_party/glog)
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
if(NOT USE_CUDA)
  list(FILTER SRC EXCLUDE REGEX ".*runtime/cuda/.*")
  list(FILTER SRC EXCLUDE REGEX ".*ccl/cuda/.*")
endif()
if(NOT USE_NCCL)
  list(FILTER SRC EXCLUDE REGEX ".*infini_train/src/core/ccl/cuda/.*")
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

  if(USE_FLASH_ATTENTION)
    set(FLASH_ATTN_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/flash-attention" CACHE PATH
      "Path to vendored FlashAttention source tree")
    if(NOT EXISTS "${FLASH_ATTN_SOURCE_DIR}/csrc/flash_attn/src/flash.h")
      message(FATAL_ERROR "FlashAttention submodule not found at ${FLASH_ATTN_SOURCE_DIR}. "
        "Run: git submodule update --init --recursive third_party/flash-attention")
    endif()
    if(NOT EXISTS "${FLASH_ATTN_SOURCE_DIR}/csrc/cutlass/include/cutlass/cutlass.h")
      message(FATAL_ERROR "FlashAttention CUTLASS dependency not found. "
        "Run: git -C third_party/flash-attention submodule update --init csrc/cutlass")
    endif()

    # Minimal training subset: causal FP16/BF16 kernels for GPT-2/LLaMA head dims.
    # TODO: add other head dims and architectures when InfiniTrain models require them.
    set(FLASH_ATTN_CUDA_SOURCES
      "${FLASH_ATTN_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm80.cu"
      "${FLASH_ATTN_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_hdim64_bf16_causal_sm80.cu"
      "${FLASH_ATTN_SOURCE_DIR}/csrc/flash_attn/src/flash_bwd_hdim64_fp16_causal_sm80.cu"
      "${FLASH_ATTN_SOURCE_DIR}/csrc/flash_attn/src/flash_bwd_hdim64_bf16_causal_sm80.cu"
      "${FLASH_ATTN_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu"
      "${FLASH_ATTN_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_hdim128_bf16_causal_sm80.cu"
      "${FLASH_ATTN_SOURCE_DIR}/csrc/flash_attn/src/flash_bwd_hdim128_fp16_causal_sm80.cu"
      "${FLASH_ATTN_SOURCE_DIR}/csrc/flash_attn/src/flash_bwd_hdim128_bf16_causal_sm80.cu"
    )
    add_library(flash_attn_native STATIC ${FLASH_ATTN_CUDA_SOURCES})
    set_target_properties(flash_attn_native PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "80"
      POSITION_INDEPENDENT_CODE ON
    )
    target_compile_definitions(flash_attn_native PRIVATE
      FLASHATTENTION_DISABLE_DROPOUT
      FLASHATTENTION_DISABLE_ALIBI
      FLASHATTENTION_DISABLE_SOFTCAP
      FLASHATTENTION_DISABLE_LOCAL
    )
    target_include_directories(flash_attn_native PRIVATE
      "${PROJECT_SOURCE_DIR}/infini_train/src/kernels/cuda/flash_attention_compat"
      "${FLASH_ATTN_SOURCE_DIR}/csrc/flash_attn"
      "${FLASH_ATTN_SOURCE_DIR}/csrc/flash_attn/src"
      "${FLASH_ATTN_SOURCE_DIR}/csrc/cutlass/include"
    )
    target_compile_options(flash_attn_native PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-O3>
      $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_OPERATORS__>
      $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF_CONVERSIONS__>
      $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_HALF2_OPERATORS__>
      $<$<COMPILE_LANGUAGE:CUDA>:-U__CUDA_NO_BFLOAT16_CONVERSIONS__>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
    target_link_libraries(flash_attn_native PUBLIC CUDA::cudart)
  endif()

  # CUDA compilation options
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

  # Only compile CUDA kernels / cuda sources here (your original used src/*.cu)
  file(GLOB_RECURSE CUDA_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/*.cu)
  if(NOT USE_FLASH_ATTENTION)
    list(FILTER CUDA_KERNELS EXCLUDE REGEX ".*/kernels/cuda/flash_attention\\.cu$")
  endif()

  add_library(infini_train_cuda_kernels STATIC ${CUDA_KERNELS})
  set_target_properties(infini_train_cuda_kernels PROPERTIES CUDA_ARCHITECTURES "75;80;90")

  target_link_libraries(infini_train_cuda_kernels
    PUBLIC
      glog
      CUDA::cudart
      CUDA::cublas
      CUDA::cuda_driver
  )

  if(USE_FLASH_ATTENTION)
    target_compile_definitions(infini_train_cuda_kernels PRIVATE
      FLASHATTENTION_DISABLE_DROPOUT
      FLASHATTENTION_DISABLE_ALIBI
      FLASHATTENTION_DISABLE_SOFTCAP
      FLASHATTENTION_DISABLE_LOCAL
    )
    target_include_directories(infini_train_cuda_kernels PRIVATE
      "${PROJECT_SOURCE_DIR}/infini_train/src/kernels/cuda/flash_attention_compat"
      "${FLASH_ATTN_SOURCE_DIR}/csrc/flash_attn/src"
      "${FLASH_ATTN_SOURCE_DIR}/csrc/cutlass/include"
    )
    target_link_libraries(infini_train_cuda_kernels PUBLIC flash_attn_native)
  endif()

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
target_link_libraries(infini_train
  PUBLIC
    glog
    gflags
    infini_train_cpu_kernels
)

if(USE_CUDA)
  # infini_train contains cuda runtime wrappers (*.cc) like cuda_blas_handle.cc/cuda_guard.cc
  # Those may need CUDA runtime/driver/cublas symbols at final link, so attach them here too.
  target_link_libraries(infini_train
    PUBLIC
      infini_train_cuda_kernels
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
# Helper: link libraries in a group to fix static lib one-pass resolution
# (THIS is what fixes "undefined reference" from cuda_kernels -> core symbols)
# ------------------------------------------------------------------------------
function(link_infini_train_exe target_name)
  if(USE_CUDA)
    target_link_libraries(${target_name} PRIVATE
      "-Wl,--start-group"
      "-Wl,--whole-archive"
        infini_train
        infini_train_cpu_kernels
        infini_train_cuda_kernels
      "-Wl,--no-whole-archive"
      "-Wl,--end-group"
    )
  else()
    target_link_libraries(${target_name} PRIVATE
      "-Wl,--start-group"
      "-Wl,--whole-archive"
        infini_train
        infini_train_cpu_kernels
      "-Wl,--no-whole-archive"
      "-Wl,--end-group"
    )
  endif()
endfunction()


# ------------------------------------------------------------------------------
# Examples
# ------------------------------------------------------------------------------

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

# Tools
add_subdirectory(tools/infini_run)
set_target_properties(infini_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

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
