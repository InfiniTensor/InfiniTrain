# CMake generated Testfile for 
# Source directory: /home/fabu/桌面/planning_tmp/InfiniTrain/tests/tensor
# Build directory: /home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/tensor
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/tensor/test_tensor_cpu[1]_include.cmake")
include("/home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/tensor/test_tensor_cpu_only[1]_include.cmake")
add_test([=[test_tensor_cuda]=] "/home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/tensor/test_tensor_cuda" "--gtest_filter=CUDA/*")
set_tests_properties([=[test_tensor_cuda]=] PROPERTIES  LABELS "cuda" TIMEOUT "10" _BACKTRACE_TRIPLES "/home/fabu/桌面/planning_tmp/InfiniTrain/cmake/test_macros.cmake;95;add_test;/home/fabu/桌面/planning_tmp/InfiniTrain/cmake/test_macros.cmake;149;infini_train_add_test;/home/fabu/桌面/planning_tmp/InfiniTrain/tests/tensor/CMakeLists.txt;12;infini_train_add_test_suite;/home/fabu/桌面/planning_tmp/InfiniTrain/tests/tensor/CMakeLists.txt;0;")
