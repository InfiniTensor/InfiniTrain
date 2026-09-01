cmake_minimum_required(VERSION 3.28)

option(USE_CUDA "Support NVIDIA CUDA" OFF)
option(USE_DCU "Support Hygon DCU through DTK/HIP" OFF)
option(PROFILE_MODE "ENABLE PROFILE MODE" OFF)
option(USE_OMP "Use OpenMP as backend for Eigen" ON)
option(USE_NCCL "Build project for distributed running" ON)
option(USE_RCCL "Build project for distributed running on DCU using RCCL" OFF)
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
if(NOT USE_DCU)
  list(FILTER SRC EXCLUDE REGEX ".*/(ccl|runtime)/dcu/.*")
elseif(NOT USE_RCCL)
  list(FILTER SRC EXCLUDE REGEX ".*/ccl/dcu/.*")
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
# DCU kernels library (optional)
# ------------------------------------------------------------------------------

if(USE_DCU)
  add_compile_definitions(USE_DCU=1)

  set(DCU_PATH "$ENV{DTK_PATH}" CACHE PATH "Hygon DTK installation root")
  set(DCU_ARCH "" CACHE STRING "Optional HIP offload architecture reported by rocminfo")
  if(NOT DCU_PATH)
    set(DCU_PATH /opt/dtk)
  endif()

  find_program(HIPCC_EXECUTABLE hipcc
    HINTS "${DCU_PATH}/bin" "${DCU_PATH}/llvm/bin" /opt/rocm/bin
    REQUIRED)
  if(NOT CMAKE_CXX_COMPILER MATCHES "hipcc")
    message(WARNING
      "DCU kernels must be compiled by hipcc. Reconfigure with "
      "-DCMAKE_CXX_COMPILER=${HIPCC_EXECUTABLE}")
  endif()

  include_directories("${DCU_PATH}/include")
  link_directories("${DCU_PATH}/lib" "${DCU_PATH}/lib64")

  find_library(DCU_RUNTIME_LIB NAMES amdhip64 hip_hcc
    HINTS "${DCU_PATH}/lib" "${DCU_PATH}/lib64" /opt/rocm/lib /opt/rocm/lib64
    REQUIRED)
  find_library(DCU_BLAS_LIB NAMES hipblas
    HINTS "${DCU_PATH}/lib" "${DCU_PATH}/lib64" /opt/rocm/lib /opt/rocm/lib64
    REQUIRED)

  file(GLOB_RECURSE DCU_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/kernels/dcu/*.hip)
  set_source_files_properties(${DCU_KERNELS} PROPERTIES
    LANGUAGE CXX
    COMPILE_OPTIONS "-x;hip"
  )
  add_library(infini_train_dcu_kernels STATIC ${DCU_KERNELS})
  if(DCU_ARCH)
    target_compile_options(infini_train_dcu_kernels PRIVATE "--offload-arch=${DCU_ARCH}")
  endif()
  target_link_libraries(infini_train_dcu_kernels PUBLIC glog ${DCU_RUNTIME_LIB} ${DCU_BLAS_LIB})
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

if(USE_DCU)
  target_link_libraries(infini_train PUBLIC infini_train_dcu_kernels ${DCU_RUNTIME_LIB} ${DCU_BLAS_LIB})

  if(USE_RCCL)
    message(STATUS "Add USE_RCCL, use RCCL with DCU")
    find_library(DCU_COMM_LIB NAMES rccl nccl
      HINTS
        "${DCU_PATH}/lib"
        "${DCU_PATH}/lib64"
        "${DCU_PATH}/rccl/lib"
        /opt/rocm/lib
        /opt/rocm/lib64
      REQUIRED)
    add_compile_definitions(USE_RCCL=1)
    target_link_libraries(infini_train PUBLIC ${DCU_COMM_LIB})
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
  elseif(USE_DCU)
    target_link_libraries(${target_name} PRIVATE
      "-Wl,--start-group"
      "-Wl,--whole-archive"
        infini_train
        infini_train_cpu_kernels
        infini_train_dcu_kernels
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
  example/fm9gv/checkpoint_loader.cc
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
