# ==========================================================================
# Tensor tests
# ==========================================================================

set(TENSOR_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TENSOR_TEST_SOURCES
  ${TENSOR_TEST_DIR}/test_tensor.cc
  ${TENSOR_TEST_DIR}/test_tensor_create.cc
  ${TENSOR_TEST_DIR}/test_tensor_copy.cc
  ${TENSOR_TEST_DIR}/test_tensor_delete.cc
)

infini_train_add_test_suite(test_tensor
  SOURCES ${TENSOR_TEST_SOURCES}
)
