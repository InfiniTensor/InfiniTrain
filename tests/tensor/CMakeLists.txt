# ==========================================================================
# Tensor tests
# ==========================================================================

file(GLOB TENSOR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

infini_train_add_test_suite(test_tensor
  SOURCES ${TENSOR_SOURCES}
)
