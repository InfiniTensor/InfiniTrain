cmake_minimum_required(VERSION 3.28)

option(USE_CUDA "Support NVIDIA CUDA" OFF)
option(PROFILE_MODE "ENABLE PROFILE MODE" OFF)
option(USE_OMP "Use OpenMP as backend for Eigen" ON)
option(USE_NCCL "Build project for distributed running" ON)

project(infini_train VERSION 0.5.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

  # CUDA compilation options
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

  # FlashAttention-2 support (optional)
  option(USE_FLASH_ATTN "Enable FlashAttention-2 support" OFF)

  # Only compile CUDA kernels / cuda sources here (your original used src/*.cu)
  file(GLOB_RECURSE CUDA_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/*.cu)

  # When FlashAttention is disabled, exclude flash_attention.cu from framework kernels
  if(NOT USE_FLASH_ATTN)
    list(FILTER CUDA_KERNELS EXCLUDE REGEX ".*flash_attention\\.cu$")
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

  # Build FlashAttention-2 as a separate static library when enabled
  if(USE_FLASH_ATTN)
    add_compile_definitions(USE_FLASH_ATTN=1)
    message(STATUS "FlashAttention-2 support enabled")

    # FlashAttention-2 source files
    file(GLOB FLASH_ATTN_SRCS
      ${PROJECT_SOURCE_DIR}/third_party/flash-attention/csrc/flash_attn/src/*.cu)

    add_library(flash_attn STATIC ${FLASH_ATTN_SRCS})
    set_target_properties(flash_attn PROPERTIES CUDA_ARCHITECTURES "80;90")

    target_include_directories(flash_attn PUBLIC
      ${PROJECT_SOURCE_DIR}/third_party/flash-attention/csrc/flash_attn
      ${PROJECT_SOURCE_DIR}/third_party/flash-attention/csrc/flash_attn/src
      ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    )

    target_compile_options(flash_attn PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda -O2>)

    # Let the framework kernel find flash_attn headers
    target_include_directories(infini_train_cuda_kernels PUBLIC
      ${PROJECT_SOURCE_DIR}/third_party/flash-attention/csrc/flash_attn
      ${PROJECT_SOURCE_DIR}/third_party/flash-attention/csrc/flash_attn/src
      ${PROJECT_SOURCE_DIR}/third_party/cutlass/include
    )

    target_link_libraries(infini_train_cuda_kernels PUBLIC flash_attn)
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
    if(USE_FLASH_ATTN)
      target_link_libraries(${target_name} PRIVATE
        "-Wl,--start-group"
        "-Wl,--whole-archive"
          infini_train
          infini_train_cpu_kernels
          infini_train_cuda_kernels
          flash_attn
        "-Wl,--no-whole-archive"
        "-Wl,--end-group"
      )
    else()
      target_link_libraries(${target_name} PRIVATE
        "-Wl,--start-group"
        "-Wl,--whole-archive"
          infini_train
          infini_train_cpu_kernels
          infini_train_cuda_kernels
        "-Wl,--no-whole-archive"
        "-Wl,--end-group"
      )
    endif()
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
  example/gpt2/net.cc
  example/common/tokenizer.cc
)
link_infini_train_exe(gpt2)

add_executable(llama3
  example/llama3/main.cc
  example/common/tiny_shakespeare_dataset.cc
  example/common/utils.cc
  example/llama3/net.cc
  example/common/tokenizer.cc
)
link_infini_train_exe(llama3)

# Tools
add_subdirectory(tools/infini_run)
set_target_properties(infini_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Tests
add_executable(test_hook test/hook/test_hook.cc)
target_link_libraries(test_hook infini_train)

add_executable(test_precision_check test/hook/test_precision_check.cc)
target_link_libraries(test_precision_check infini_train)
