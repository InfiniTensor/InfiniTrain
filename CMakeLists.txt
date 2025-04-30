option(USE_CUDA "Support NVIDIA CUDA" OFF)
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

if(USE_CUDA)
    add_compile_definitions(USE_CUDA=1)
    enable_language(CUDA)
    include(FindCUDAToolkit)

    # enable CUDA-related compilation options
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

    file(GLOB_RECURSE CUDA_SRC ${PROJECT_SOURCE_DIR}/infini_train/src/*.cu)
    list(APPEND SRC ${CUDA_SRC})

    add_library(infini_train STATIC ${SRC})
    set_target_properties(infini_train PROPERTIES CUDA_ARCHITECTURES "70;80")
    target_link_libraries(infini_train glog gflags CUDA::cudart CUDA::cublas Eigen3::Eigen)
else()
    add_library(infini_train STATIC ${SRC})
    target_link_libraries(infini_train glog gflags Eigen3::Eigen)
endif()

# Examples
add_executable(mnist example/mnist/main.cc example/mnist/dataset.cc example/mnist/net.cc)
target_link_libraries(mnist glog gflags infini_train Eigen3::Eigen)

add_executable(gpt2 example/gpt2/main.cc example/gpt2/dataset.cc example/gpt2/net.cc)
target_link_libraries(gpt2 glog gflags infini_train Eigen3::Eigen)

# OpenBLAS
# target_link_libraries(infini_train ${OpenBLAS_LIBRARIES})
# target_link_libraries(mnist ${OpenBLAS_LIBRARIES})
# target_link_libraries(gpt2 ${OpenBLAS_LIBRARIES})

# OpenMP
target_link_libraries(infini_train OpenMP::OpenMP_CXX)
target_link_libraries(mnist OpenMP::OpenMP_CXX)
target_link_libraries(gpt2 OpenMP::OpenMP_CXX)

