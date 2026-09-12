# Shared CPU/CUDA tests.
# TODO: Add backend-independent DDP coverage when the test harness supports multi-node execution.
file(GLOB GENERATOR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

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
endif()
