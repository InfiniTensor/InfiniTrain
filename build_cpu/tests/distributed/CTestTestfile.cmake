# CMake generated Testfile for 
# Source directory: /home/fabu/桌面/planning_tmp/InfiniTrain/tests/distributed
# Build directory: /home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/distributed
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/distributed/test_rank[1]_include.cmake")
add_test([=[RankTest.MultiNodeSingleProcessIsParallel]=] "/usr/bin/cmake" "-E" "env" "WORLD_SIZE=2" "LOCAL_WORLD_SIZE=1" "RANK=0" "LOCAL_RANK=0" "/home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/distributed/test_rank" "--gtest_filter=RankTest.DetectsParallelismFromGlobalWorldSize")
set_tests_properties([=[RankTest.MultiNodeSingleProcessIsParallel]=] PROPERTIES  LABELS "cpu" TIMEOUT "10" _BACKTRACE_TRIPLES "/home/fabu/桌面/planning_tmp/InfiniTrain/tests/distributed/CMakeLists.txt;10;add_test;/home/fabu/桌面/planning_tmp/InfiniTrain/tests/distributed/CMakeLists.txt;0;")
