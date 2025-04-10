option(USE_CUDA "Support NVIDIA CUDA" OFF)
cmake_minimum_required(VERSION 3.28)

project(infini_train VERSION 0.2.0 LANGUAGES CXX)

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

if(USE_CUDA)
    add_compile_definitions(USE_CUDA=1)
    enable_language(CUDA)
    include(FindCUDAToolkit)

    # enable CUDA-related compilation options
    # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CMAKE_INCLUDE_PATH} -Xcompiler -fPIC --expt-relaxed-constexpr")

    include_directories(${PROJECT_SOURCE_DIR})
    file(GLOB_RECURSE SRC ${PROJECT_SOURCE_DIR}/infini_train/src/*.cc)
    file(GLOB_RECURSE CUDA_SRC ${PROJECT_SOURCE_DIR}/infini_train/src/*.cu)
    list(APPEND SRC ${CUDA_SRC})

    add_library(infini_train STATIC ${SRC})
    set_target_properties(infini_train PROPERTIES CUDA_ARCHITECTURES "70;80")
    target_link_libraries(infini_train glog gflags CUDA::cudart CUDA::cublas)

    # Examples
    add_executable(mnist example/mnist/main.cc example/mnist/dataset.cc example/mnist/net.cc)
    target_link_libraries(mnist glog gflags infini_train)

    add_executable(gpt2 example/gpt2/main.cc example/gpt2/dataset.cc example/gpt2/net.cc)
    target_link_libraries(gpt2 glog gflags infini_train)
else()
    include_directories(${PROJECT_SOURCE_DIR})
    file(GLOB_RECURSE SRC ${PROJECT_SOURCE_DIR}/infini_train/src/*.cc)

    add_library(infini_train STATIC ${SRC})
    target_link_libraries(infini_train glog gflags)

    # Examples
    add_executable(mnist example/mnist/main.cc example/mnist/dataset.cc example/mnist/net.cc)
    target_link_libraries(mnist glog gflags infini_train)

    add_executable(gpt2 example/gpt2/main.cc example/gpt2/dataset.cc example/gpt2/net.cc)
    target_link_libraries(gpt2 glog gflags infini_train)
endif()
