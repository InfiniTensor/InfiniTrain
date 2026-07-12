add_executable(generator_benchmark generator/generator_benchmark.cc)
link_infini_train_exe(generator_benchmark)

set_target_properties(generator_benchmark PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
)
