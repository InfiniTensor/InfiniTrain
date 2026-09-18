# ==========================================================================
# Example tests (MNIST dataset, ...)
# ==========================================================================

infini_train_add_test(test_mnist_dataset
  SOURCES test_mnist_dataset.cc ${CMAKE_SOURCE_DIR}/example/mnist/dataset.cc
  LABELS cpu
  TEST_TIMEOUT 60
)

# MnistCnn <-> torch parity fixture (OSpec P12): dumps a deterministic 10-step
# SGD trajectory to $MNIST_PARITY_OUT_DIR (or ./mnist_parity_dump).
infini_train_add_test(test_mnist_parity
  SOURCES mnist_parity.cc ${CMAKE_SOURCE_DIR}/example/mnist/net.cc
  LABELS cpu
  TEST_TIMEOUT 300
)
