# ==========================================================================
# Generator tests
# ==========================================================================

file(GLOB GENERATOR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

infini_train_add_test_suite(test_generator
  SOURCES ${GENERATOR_SOURCES}
  TEST_TIMEOUT 40
)
