option(USE_CUDA "Support NVIDIA CUDA" OFF)
option(PROFILE_MODE "ENABLE PROFILE MODE" OFF)
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
find_package(OpenMP REQUIRED)
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
target_link_libraries(infini_train_cpu_kernels glog Eigen3::Eigen OpenMP::OpenMP_CXX)

if(USE_CUDA)
    add_compile_definitions(USE_CUDA=1)
    enable_language(CUDA)
    include(FindCUDAToolkit)

    # enable CUDA-related compilation options
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
    file(GLOB_RECURSE CUDA_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/*.cu)
    add_library(infini_train_cuda_kernels STATIC ${CUDA_KERNELS})
    set_target_properties(infini_train_cuda_kernels PROPERTIES CUDA_ARCHITECTURES "75;80")
    target_link_libraries(infini_train_cuda_kernels glog CUDA::cudart CUDA::cublas)

    add_library(infini_train STATIC ${SRC})
    target_link_libraries(infini_train glog gflags "-Wl,--whole-archive" infini_train_cpu_kernels infini_train_cuda_kernels "-Wl,--no-whole-archive")
else()
    add_library(infini_train STATIC ${SRC})
    target_link_libraries(infini_train glog gflags "-Wl,--whole-archive" infini_train_cpu_kernels "-Wl,--no-whole-archive")
endif()

# Examples
add_executable(mnist example/mnist/main.cc example/mnist/dataset.cc example/mnist/net.cc)
target_link_libraries(mnist infini_train)

add_executable(gpt2 example/gpt2/main.cc example/common/tiny_shakespeare_dataset.cc example/gpt2/net.cc example/common/tokenizer.cc)
target_link_libraries(gpt2 infini_train)

add_executable(llama3 example/llama3/main.cc example/common/tiny_shakespeare_dataset.cc example/llama3/net.cc example/common/tokenizer.cc)
target_link_libraries(llama3 infini_train)
