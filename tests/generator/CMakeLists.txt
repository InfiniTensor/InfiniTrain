# ==========================================================================
# Generator tests
# ==========================================================================
file(GLOB GENERATOR_SHARED_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)
infini_train_add_test_suite(test_generator SOURCES ${GENERATOR_SHARED_SOURCES})

file(GLOB GENERATOR_CPU_ONLY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cpu_only/test_*.cc)
if(GENERATOR_CPU_ONLY_SOURCES)
  infini_train_add_test(test_generator_cpu_only
    SOURCES ${GENERATOR_CPU_ONLY_SOURCES}
    LABELS cpu
  )
endif()
