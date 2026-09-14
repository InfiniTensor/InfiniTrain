# CMake generated Testfile for 
# Source directory: /home/fabu/桌面/planning_tmp/InfiniTrain/tests/dtype
# Build directory: /home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/dtype
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/dtype/test_dtype_cpu[1]_include.cmake")
add_test([=[test_dtype_cuda]=] "/home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/dtype/test_dtype_cuda" "--gtest_filter=CUDA/*")
set_tests_properties([=[test_dtype_cuda]=] PROPERTIES  LABELS "cuda" TIMEOUT "40" _BACKTRACE_TRIPLES "/home/fabu/桌面/planning_tmp/InfiniTrain/cmake/test_macros.cmake;95;add_test;/home/fabu/桌面/planning_tmp/InfiniTrain/cmake/test_macros.cmake;149;infini_train_add_test;/home/fabu/桌面/planning_tmp/InfiniTrain/tests/dtype/CMakeLists.txt;8;infini_train_add_test_suite;/home/fabu/桌面/planning_tmp/InfiniTrain/tests/dtype/CMakeLists.txt;0;")
