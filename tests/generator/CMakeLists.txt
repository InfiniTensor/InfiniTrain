# Shared CPU/CUDA tests; DDP is built separately below.
file(GLOB GENERATOR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)
set(GENERATOR_DDP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/test_generator_ddp.cc)
list(REMOVE_ITEM GENERATOR_SOURCES ${GENERATOR_DDP_SOURCE})

infini_train_add_test_suite(test_generator
  SOURCES ${GENERATOR_SOURCES}
  TEST_TIMEOUT 40
)

file(GLOB GENERATOR_CPU_ONLY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cpu_only/test_*.cc)
if(GENERATOR_CPU_ONLY_SOURCES)
  infini_train_add_test(test_generator_cpu_only
    SOURCES ${GENERATOR_CPU_ONLY_SOURCES}
    LABELS cpu
    TEST_TIMEOUT 40
  )
endif()

if(USE_CUDA)
  file(GLOB GENERATOR_CUDA_ONLY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_only/test_*.cc)
  if(GENERATOR_CUDA_ONLY_SOURCES)
    infini_train_add_test(test_generator_cuda_only
      SOURCES ${GENERATOR_CUDA_ONLY_SOURCES}
      LABELS cuda
      TEST_TIMEOUT 40
    )
  endif()

  # DDP uses a separate executable with its own main.
  add_executable(test_generator_ddp ${GENERATOR_DDP_SOURCE})
  target_compile_options(test_generator_ddp PRIVATE -Wno-error)
  target_link_libraries(test_generator_ddp PRIVATE GTest::gtest)
  target_include_directories(test_generator_ddp PRIVATE ${glog_SOURCE_DIR}/src)
  link_infini_train_exe(test_generator_ddp)

  add_test(NAME test_generator_ddp COMMAND $<TARGET_FILE:test_generator_ddp>)
  set_tests_properties(test_generator_ddp PROPERTIES LABELS "cuda;distributed" TIMEOUT 40)
endif()
