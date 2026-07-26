# Generator tests are kept separate from Tensor tests. Shared behavior is
# compiled for both backends, while CUDA-specific coverage stays isolated.
file(GLOB GENERATOR_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)
list(REMOVE_ITEM GENERATOR_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_generator_ddp.cc)
infini_train_add_test(test_generator_cpu
  SOURCES ${GENERATOR_TEST_SOURCES}
  LABELS cpu
  TEST_FILTER "CPU/*"
)

if(USE_CUDA)
  file(GLOB GENERATOR_CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_only/test_*.cc)
  infini_train_add_test(test_generator_cuda
    SOURCES ${GENERATOR_TEST_SOURCES} ${GENERATOR_CUDA_SOURCES}
    LABELS cuda
    TEST_FILTER "CUDA/*"
  )

  if(USE_NCCL)
    add_executable(test_generator_ddp test_generator_ddp.cc)
    target_compile_options(test_generator_ddp PRIVATE -Wno-error)
    target_link_libraries(test_generator_ddp PRIVATE GTest::gtest)
    target_include_directories(test_generator_ddp PRIVATE ${glog_SOURCE_DIR}/src)
    link_infini_train_exe(test_generator_ddp)
    add_test(NAME test_generator_ddp COMMAND $<TARGET_FILE:test_generator_ddp>)
    set_tests_properties(test_generator_ddp PROPERTIES LABELS "cuda;distributed" TIMEOUT 30 RUN_SERIAL TRUE)
  endif()
endif()
