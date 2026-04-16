cmake_minimum_required(VERSION 3.28)

# Platforms
option(USE_CUDA "Support NVIDIA CUDA" OFF)
option(USE_MACA "Support MetaX MACA" OFF)

option(PROFILE_MODE "ENABLE PROFILE MODE" OFF)
option(USE_OMP "Use OpenMP as backend for Eigen" ON)
option(USE_NCCL "Build project for distributed running on CUDA using NCCL" ON)
option(USE_MCCL "Build project for distributed running on MACA using MCCL" ON)

project(infini_train VERSION 0.5.0 LANGUAGES CXX)

# Switch to mxcc after project() so that third-party libs (glog, gflags) are
# configured with the host compiler and their feature-detection checks pass.
if(USE_MACA)
  set(MACA_PATH $ENV{MACA_PATH})
  if(NOT MACA_PATH)
    message(FATAL_ERROR "USE_MACA=ON but environment variable MACA_PATH is not set.")
  endif()
  set(CMAKE_C_COMPILER   "${MACA_PATH}/mxgpu_llvm/bin/mxcc")
  set(CMAKE_CXX_COMPILER "${MACA_PATH}/mxgpu_llvm/bin/mxcc")
endif()

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
set(BUILD_TESTING OFF CACHE BOOL "Disable glog unit tests" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build glog as static library" FORCE)

add_subdirectory(third_party/glog)
include_directories(${glog_SOURCE_DIR}/src)

# eigen
if(USE_OMP)
  find_package(OpenMP REQUIRED)

  set(INFINI_OMP_LIBS OpenMP::OpenMP_CXX)

  # Under MACA/mxcc, the host compiler is LLVM-based; link mxomp (iomp5) instead
  # of libgomp to stay ABI-compatible with the MACA toolchain.
  if(USE_MACA)
    find_library(INFINI_MACA_OMP_LIB
      NAMES omp iomp5
      HINTS
        "${MACA_PATH}/lib"
        "${MACA_PATH}/mxgpu_llvm/lib"
        "${MACA_PATH}/mxgpu_llvm/lib64"
      REQUIRED
    )
    set(INFINI_OMP_LIBS OpenMP::OpenMP_CXX ${INFINI_MACA_OMP_LIB})
  endif()
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
# in headers (e.g. <cuda_runtime.h> / <mcr/mc_runtime.h>) that aren't on the
# include path.
if(NOT USE_CUDA)
  list(FILTER SRC EXCLUDE REGEX ".*infini_train/src/core/runtime/cuda/.*")
  list(FILTER SRC EXCLUDE REGEX ".*infini_train/src/core/ccl/cuda/.*")
endif()
if(NOT USE_MACA)
  list(FILTER SRC EXCLUDE REGEX ".*infini_train/src/core/runtime/maca/.*")
  list(FILTER SRC EXCLUDE REGEX ".*infini_train/src/core/ccl/maca/.*")
endif()
if(NOT USE_NCCL)
  list(FILTER SRC EXCLUDE REGEX ".*infini_train/src/core/ccl/cuda/.*")
endif()
if(NOT USE_MCCL)
  list(FILTER SRC EXCLUDE REGEX ".*infini_train/src/core/ccl/maca/.*")
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
  target_link_libraries(infini_train_cpu_kernels PUBLIC ${INFINI_OMP_LIBS})
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
# MACA kernels library (optional, MetaX backend)
# ------------------------------------------------------------------------------

if(USE_MACA)
  add_compile_definitions(USE_MACA=1)

  # ---- MACA SDK include / link paths ----
  include_directories("${MACA_PATH}/include")
  link_directories("${MACA_PATH}/lib")

  # ---- MACA runtime / blas / (optional) mccl libraries ----
  find_library(MACA_RUNTIME_LIB NAMES mcruntime HINTS "${MACA_PATH}/lib" REQUIRED)
  find_library(MACA_DNN_LIB     NAMES mcdnn     HINTS "${MACA_PATH}/lib" REQUIRED)
  find_library(MACA_BLAS_LIB    NAMES mcblas    HINTS "${MACA_PATH}/lib" REQUIRED)

  # ---- Collect .maca kernel sources and build as a CXX static lib with -x maca ----
  file(GLOB_RECURSE MACA_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/kernels/maca/*.maca)
  set_source_files_properties(${MACA_KERNELS} PROPERTIES
    LANGUAGE CXX
    COMPILE_OPTIONS "-x;maca"
  )

  add_library(infini_train_maca_kernels STATIC ${MACA_KERNELS})
  target_link_libraries(infini_train_maca_kernels
    PUBLIC
      glog
      ${MACA_RUNTIME_LIB}
      ${MACA_DNN_LIB}
      ${MACA_BLAS_LIB}
  )

  if(USE_MCCL)
    message(STATUS "Add USE_MCCL, use MCCL with MACA")
    find_library(MACA_COMM_LIB NAMES mccl HINTS "${MACA_PATH}/lib" REQUIRED)
    add_compile_definitions(USE_MCCL=1)
    target_link_libraries(infini_train_maca_kernels PUBLIC ${MACA_COMM_LIB})
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

if(USE_MACA)
  # infini_train contains MACA runtime wrappers (maca_guard_impl.cc / maca_runtime_common.cc /
  # mccl_impl.cc) which reference mcruntime / mcblas / mccl symbols directly at final link.
  target_link_libraries(infini_train
    PUBLIC
      infini_train_maca_kernels
      ${MACA_RUNTIME_LIB}
      ${MACA_DNN_LIB}
      ${MACA_BLAS_LIB}
  )

  if(USE_MCCL)
    target_link_libraries(infini_train PUBLIC ${MACA_COMM_LIB})
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
  elseif(USE_MACA)
    target_link_libraries(${target_name} PRIVATE
      "-Wl,--start-group"
      "-Wl,--whole-archive"
        infini_train
        infini_train_cpu_kernels
        infini_train_maca_kernels
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
  example/gpt2/checkpoint_loader.cc
  example/common/tokenizer.cc
)
link_infini_train_exe(gpt2)

add_executable(llama3
  example/llama3/main.cc
  example/common/tiny_shakespeare_dataset.cc
  example/common/utils.cc
  example/llama3/checkpoint_loader.cc
  example/common/tokenizer.cc
)
link_infini_train_exe(llama3)

# Tools
add_subdirectory(tools/infini_run)
set_target_properties(infini_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Tests
add_executable(test_hook test/hook/test_hook.cc)
link_infini_train_exe(test_hook)

add_executable(test_precision_check test/hook/test_precision_check.cc)
link_infini_train_exe(test_precision_check)

add_executable(test_lora test/lora/test_lora.cc)
link_infini_train_exe(test_lora)

add_executable(test_transformer_architecture test/transformer/test_transformer_architecture.cc)
link_infini_train_exe(test_transformer_architecture)

add_executable(test_scalar test/dtype/test_scalar.cc)
link_infini_train_exe(test_scalar)

add_executable(test_dtype_dispatch test/dtype/test_dtype_dispatch.cc)
link_infini_train_exe(test_dtype_dispatch)

# Negative compile test: missing dtype registration must fail at compile time.
set(DTYPE_DISPATCH_COMPILE_FAIL_SOURCE
  ${PROJECT_SOURCE_DIR}/test/dtype/test_dtype_dispatch_compile_fail.cc)

try_compile(DTYPE_DISPATCH_COMPILE_UNEXPECTEDLY_SUCCEEDED
  ${CMAKE_BINARY_DIR}/CMakeFiles/try_compile_dtype_dispatch_missing_map
  SOURCES ${DTYPE_DISPATCH_COMPILE_FAIL_SOURCE}
  CMAKE_FLAGS
    "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
    "-DCMAKE_CXX_STANDARD_REQUIRED=ON"
    "-DCMAKE_CXX_EXTENSIONS=OFF"
    "-DCMAKE_CXX_FLAGS=-I${PROJECT_SOURCE_DIR}"
  OUTPUT_VARIABLE DTYPE_DISPATCH_TRY_COMPILE_OUTPUT
)

if(DTYPE_DISPATCH_COMPILE_UNEXPECTEDLY_SUCCEEDED)
  message(FATAL_ERROR
    "dtype dispatch compile-fail test unexpectedly succeeded.\n"
    "Source: ${DTYPE_DISPATCH_COMPILE_FAIL_SOURCE}\n"
    "Output:\n${DTYPE_DISPATCH_TRY_COMPILE_OUTPUT}")
endif()

add_custom_target(test_dtype_dispatch_compile_fail
  COMMAND ${CMAKE_COMMAND} -E echo
    "dtype dispatch compile-fail check passed (missing dtype registration correctly fails to compile)."
  VERBATIM
)

add_dependencies(test_dtype_dispatch test_dtype_dispatch_compile_fail)
