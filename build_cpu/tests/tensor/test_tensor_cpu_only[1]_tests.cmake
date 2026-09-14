add_test([=[TensorCopyCpuTest.CopiesCPUToCPU]=]  [==[/home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/tensor/test_tensor_cpu_only]==] [==[--gtest_filter=TensorCopyCpuTest.CopiesCPUToCPU]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[TensorCopyCpuTest.CopiesCPUToCPU]=]  PROPERTIES WORKING_DIRECTORY [==[/home/fabu/桌面/planning_tmp/InfiniTrain/build_cpu/tests/tensor]==] SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==] LABELS cpu TIMEOUT 10)
set(  test_tensor_cpu_only_TESTS TensorCopyCpuTest.CopiesCPUToCPU)
