infini_train_add_test(test_mnist_dataset
  SOURCES
    test_mnist_dataset.cc
    ${CMAKE_SOURCE_DIR}/example/mnist/dataset.cc
  LABELS cpu
)

target_include_directories(test_mnist_dataset PRIVATE ${CMAKE_SOURCE_DIR})

# External fixture producer/comparator is delivered separately from the framework PR.
add_executable(mnist_parity mnist_parity.cc ${CMAKE_SOURCE_DIR}/example/mnist/net.cc)
target_include_directories(mnist_parity PRIVATE ${CMAKE_SOURCE_DIR})
link_infini_train_exe(mnist_parity)
