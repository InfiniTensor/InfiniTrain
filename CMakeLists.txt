option(USE_CUDA "Support NVIDIA CUDA" OFF)
option(PROFILE_MODE "ENABLE PROFILE MODE" OFF)
option(USE_OMP "Use OpenMP as backend for Eigen" ON)
option(USE_NCCL "Build project for distributed running" OFF)
cmake_minimum_required(VERSION 3.28)

project(infini_train VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add gflags
add_subdirectory(third_party/gflags)
include_directories(${gflags_SOURCE_DIR}/include)

set(WITH_GFLAGS OFF CACHE BOOL "Disable glog finding system gflags" FORCE)
set(WITH_GTEST OFF CACHE BOOL "Disable glog finding system gtest" FORCE)

# Add glog
add_subdirectory(third_party/glog)
include_directories(${glog_SOURCE_DIR}/src)

# Add eigen
if(USE_OMP)
    find_package(OpenMP REQUIRED)
endif()
# find_package(OpenBLAS REQUIRED)
# include_directories(${OpenBLAS_INCLUDE_DIR})
add_subdirectory(third_party/eigen)
include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen)
# add_definitions(-DEIGEN_USE_BLAS)

include_directories(${PROJECT_SOURCE_DIR})
file(GLOB_RECURSE SRC ${PROJECT_SOURCE_DIR}/infini_train/src/*.cc)
list(FILTER SRC EXCLUDE REGEX ".*kernels/cpu/.*")

if(PROFILE_MODE)
    add_compile_definitions(PROFILE_MODE=1)
endif()

file (GLOB_RECURSE CPU_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/kernels/cpu/*.cc)
add_library(infini_train_cpu_kernels STATIC ${CPU_KERNELS})
target_link_libraries(infini_train_cpu_kernels glog Eigen3::Eigen)
if(USE_OMP)
    add_compile_definitions(USE_OMP=1)
    target_link_libraries(infini_train_cpu_kernels OpenMP::OpenMP_CXX)
endif()

if(USE_CUDA)
    add_compile_definitions(USE_CUDA=1)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})

    # enable CUDA-related compilation options
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
    file(GLOB_RECURSE CUDA_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/*.cu)
    add_library(infini_train_cuda_kernels STATIC ${CUDA_KERNELS})
    set_target_properties(infini_train_cuda_kernels PROPERTIES CUDA_ARCHITECTURES "75;80;90")
    target_link_libraries(infini_train_cuda_kernels glog CUDA::cudart CUDA::cublas CUDA::cuda_driver)

    add_library(infini_train STATIC ${SRC})
    target_link_libraries(infini_train glog gflags "-Wl,--whole-archive" infini_train_cpu_kernels infini_train_cuda_kernels "-Wl,--no-whole-archive")

    if (USE_NCCL)
        message(STATUS "Add USE_NCCL, use NCCL with CUDA")
        list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
        find_package(NCCL REQUIRED)
        add_compile_definitions(USE_NCCL=1)
        target_link_libraries(infini_train nccl)
    endif()
else()
    add_library(infini_train STATIC ${SRC})
    target_link_libraries(infini_train glog gflags "-Wl,--whole-archive" infini_train_cpu_kernels "-Wl,--no-whole-archive")
endif()

# Examples
add_executable(mnist example/mnist/main.cc example/mnist/dataset.cc example/mnist/net.cc)
target_link_libraries(mnist infini_train)

add_executable(gpt2 example/gpt2/main.cc example/common/tiny_shakespeare_dataset.cc example/common/utils.cc example/gpt2/net.cc example/common/tokenizer.cc)
target_link_libraries(gpt2 infini_train)

add_executable(llama3 example/llama3/main.cc example/common/tiny_shakespeare_dataset.cc example/common/utils.cc example/llama3/net.cc example/common/tokenizer.cc)
target_link_libraries(llama3 infini_train)

add_executable(gpt2_tp example/gpt2_tp/main.cc example/common/tiny_shakespeare_dataset.cc example/common/utils.cc example/gpt2_tp/net.cc example/common/tokenizer.cc)
target_link_libraries(gpt2_tp infini_train)

add_executable(llama3_tp example/llama3_tp/main.cc example/common/tiny_shakespeare_dataset.cc example/common/utils.cc example/llama3_tp/net.cc example/common/tokenizer.cc)
target_link_libraries(llama3_tp infini_train)
