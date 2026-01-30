# Platforms
option(USE_CUDA "Support NVIDIA CUDA" OFF)
option(USE_MACA "Support MetaX MACA" OFF)

option(PROFILE_MODE "ENABLE PROFILE MODE" OFF)
option(USE_OMP "Use OpenMP as backend for Eigen" ON)
option(USE_NCCL "Build project for distributed running on CUDA using NCCL" ON)
option(USE_MCCL "Build project for distributed running on MACA using MCCL" ON)
option(USE_MPI "Enable MPI for inter-node CPU communication" ON)
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

    set(INFINI_OMP_LIBS OpenMP::OpenMP_CXX)

    # Under MACA/mxcc, use mxomp instead of original libgomp
    if(USE_MACA)
        set(MACA_PATH $ENV{MACA_PATH})
        find_library(OMP_RUNTIME_LIB
            NAMES omp iomp5
            HINTS
                "${MACA_PATH}/lib"
                "${MACA_PATH}/mxgpu_llvm/lib"
                "${MACA_PATH}/mxgpu_llvm/lib64"
            REQUIRED
        )

        set(INFINI_OMP_LIBS OpenMP::OpenMP_CXX ${OMP_RUNTIME_LIB})
    endif()
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
    target_link_libraries(infini_train_cpu_kernels ${INFINI_OMP_LIBS})
endif()

# =========================
# MPI (optional)
# =========================
if (USE_MPI)
    add_compile_definitions(USE_MPI=1)
    set(OPENMPI_ROOT /opt/openmpi-4.1.6 CACHE PATH "OpenMPI root directory")

    # ---- MPI include & lib (explicit OpenMPI path) ----
    set(MPI_INCLUDE_DIR ${OPENMPI_ROOT}/include)
    set(MPI_LIB_DIR     ${OPENMPI_ROOT}/lib)

    include_directories(${MPI_INCLUDE_DIR})
    link_directories(${MPI_LIB_DIR})

    # OpenMPI core libs (C++ bindings are deprecated; MPI is C ABI)
    set(MPI_LIBS mpi)

    # mxcc 不支持 -pthread，用 Threads::Threads（-lpthread）
    if (USE_MACA)
        set(THREADS_PREFER_PTHREAD_FLAG OFF)
        find_package(Threads REQUIRED)
    endif()
endif()

# =========================
# CUDA backend
# =========================
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

    if (USE_MPI)
        target_link_libraries(infini_train ${MPI_LIBS})
    endif()

# =========================
# MACA backend (MetaX)
# =========================
elseif(USE_MACA)
    add_compile_definitions(USE_MACA=1)

    # ---- configure MACA SDK paths ----
    # Typical: /opt/maca (can be overridden by -DMACA_PATH=...)
    set(MACA_PATH $ENV{MACA_PATH})
    set(CMAKE_C_COMPILER ${MACA_PATH}/mxgpu_llvm/bin/mxcc)
    set(CMAKE_CXX_COMPILER ${MACA_PATH}/mxgpu_llvm/bin/mxcc)

    include_directories("${MACA_PATH}/include")
    link_directories("${MACA_PATH}/lib")

    # Libraries: mcruntime / mcdnn / mcblas
    find_library(MACA_RUNTIME_LIB NAMES mcruntime HINTS "${MACA_PATH}/lib" REQUIRED)
    find_library(MACA_DNN_LIB     NAMES mcdnn     HINTS "${MACA_PATH}/lib" REQUIRED)
    find_library(MACA_BLAS_LIB    NAMES mcblas    HINTS "${MACA_PATH}/lib" REQUIRED)

    file(GLOB_RECURSE MACA_KERNELS ${PROJECT_SOURCE_DIR}/infini_train/src/kernels/maca/*.maca)
    set_source_files_properties(${MACA_KERNELS} PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS "-x;maca"
    )
    add_library(infini_train_maca_kernels STATIC ${MACA_KERNELS})
    target_link_libraries(infini_train_maca_kernels glog ${MACA_RUNTIME_LIB} ${MACA_DNN_LIB} ${MACA_BLAS_LIB})

    add_library(infini_train STATIC ${SRC})
    target_link_libraries(infini_train glog gflags "-Wl,--whole-archive" infini_train_cpu_kernels infini_train_maca_kernels "-Wl,--no-whole-archive")

    if (USE_MCCL)
        message(STATUS "Add USE_MCCL under MACA backend, use MCCL (mccl)")
        find_library(MACA_COMM_LIB NAMES mccl HINTS "${MACA_PATH}/lib" REQUIRED)
        add_compile_definitions(USE_MCCL=1)
        target_link_libraries(infini_train ${MACA_COMM_LIB})
    endif()

    if (USE_MPI)
        target_link_libraries(infini_train ${MPI_LIBS} Threads::Threads)

        # 有些 MPI 还需要额外 link flags（比如 -Wl,...），也一并带上
        if (MPI_CXX_LINK_FLAGS)
            set_target_properties(infini_train PROPERTIES
                LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
            )
        endif()
    endif()

# =========================
# CPU-only backend
# =========================
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

add_subdirectory(tools/infini_run)

set_target_properties(infini_run PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)