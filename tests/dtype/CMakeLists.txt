# ==========================================================================
# Dtype dispatch tests
# ==========================================================================

file(GLOB DTYPE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)
list(FILTER DTYPE_SOURCES EXCLUDE REGEX ".*compile_fail.*")

infini_train_add_test_suite(test_dtype
  SOURCES ${DTYPE_SOURCES}
)
