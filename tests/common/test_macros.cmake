# ============================================================================
# InfiniTrain 测试宏
# ============================================================================
# 提供统一的测试配置接口，降低接入成本
#
# 使用方法：
#   1. 在 tests/CMakeLists.txt 中 include 此文件
#   2. 使用 infini_train_add_test 宏注册测试
#
# 示例：
#   infini_train_add_test(
#     test_tensor_create
#     SOURCES test_tensor_create.cc
#     LABELS cpu cuda
#   )
# ============================================================================

include_guard(GLOBAL)

# 获取 test_macros.cmake 所在目录（tests/common/）
set(TEST_MACROS_DIR "${CMAKE_CURRENT_LIST_DIR}")

# -----------------------------------------------------------------------------
# 加载 GoogleTest 模块（提供 gtest_discover_tests）
# -----------------------------------------------------------------------------
include(GoogleTest)

# -----------------------------------------------------------------------------
# infini_train_add_test - 测试注册宏
# -----------------------------------------------------------------------------
# 功能：
#   1. 创建可执行文件
#   2. 配置编译选项、链接库和头文件路径
#   3. 使用 gtest_discover_tests 自动发现测试用例
#   4. 设置测试标签
#
# 参数：
#   SOURCES: 源文件列表（必填）
#   LABELS: 测试标签，如 "cpu" "cuda" "distributed"（可选，默认 "cpu"）
#
# 示例：
#   # 简单测试（1行）
#   infini_train_add_test(test_example SOURCES test_example.cc LABELS cpu)
#
#   # 多标签测试
#   infini_train_add_test(test_cuda_example SOURCES test_cuda.cc LABELS cuda distributed)
# -----------------------------------------------------------------------------
macro(infini_train_add_test)
  cmake_parse_arguments(ARG "" "TEST_NAME" "SOURCES;LABELS" ${ARGN})
  
  if(NOT ARG_TEST_NAME)
    set(ARG_TEST_NAME ${ARG_UNPARSED_ARGUMENTS})
  endif()
  
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "infini_train_add_test: TEST_NAME and SOURCES are required")
  endif()
  
  # 1. 创建可执行文件
  add_executable(${ARG_TEST_NAME} ${ARG_SOURCES})
  
  # 2. 配置编译选项（禁用警告转错误，以便在宽松编译环境下运行）
  target_compile_options(${ARG_TEST_NAME} PRIVATE -Wno-error)
  
  # 3. 链接 Google Test
  target_link_libraries(${ARG_TEST_NAME} PRIVATE
    GTest::gtest
    GTest::gtest_main
  )
  
  # 4. 添加头文件路径
  target_include_directories(${ARG_TEST_NAME} PRIVATE 
    ${TEST_MACROS_DIR}
    ${glog_SOURCE_DIR}/src
  )
  
  # 5. 链接项目库（复用框架链接策略，包含 CUDA/静态库依赖处理）
  link_infini_train_exe(${ARG_TEST_NAME})
  
  # 6. 使用 gtest_discover_tests 自动发现测试用例
  #    这会自动为每个 TEST_F() 创建一个 ctest 测试
  set(labels "cpu")
  if(ARG_LABELS)
    set(labels "${ARG_LABELS}")
  endif()
  
  gtest_discover_tests(${ARG_TEST_NAME}
    # 自动将测试输出重定向到 XML（便于 CI 集成）
    EXTRA_ARGS --gtest_output=xml:%T.xml
    PROPERTIES LABELS "${labels}"
  )
endmacro()
