add_test([=[RankTest.DetectsParallelismFromGlobalWorldSize]=]  [==[/home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/distributed/test_rank]==] [==[--gtest_filter=RankTest.DetectsParallelismFromGlobalWorldSize]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[RankTest.DetectsParallelismFromGlobalWorldSize]=]  PROPERTIES WORKING_DIRECTORY [==[/home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/distributed]==] SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==] LABELS cpu TIMEOUT 10)
set(  test_rank_TESTS RankTest.DetectsParallelismFromGlobalWorldSize)
