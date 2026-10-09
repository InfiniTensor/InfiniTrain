infini_train_add_test_suite(test_mnist
  SOURCES
    test_mnist.cc
    ${PROJECT_SOURCE_DIR}/example/mnist/net.cc
    ${PROJECT_SOURCE_DIR}/example/mnist/dataset.cc
  TEST_TIMEOUT 60
)

if(USE_CUDA AND USE_NCCL)
  add_executable(test_mnist_ddp test_mnist_ddp.cc ${PROJECT_SOURCE_DIR}/example/mnist/net.cc)
  link_infini_train_exe(test_mnist_ddp)
  option(BUILD_MNIST_DDP_TESTS "Register MNIST integration tests requiring two visible CUDA GPUs" OFF)
  if(BUILD_MNIST_DDP_TESTS)
   foreach(buckets IN ITEMS true false)
    add_test(NAME mnist_ddp_${buckets}
      COMMAND $<TARGET_FILE:infini_run> --nproc_per_node=2 $<TARGET_FILE:test_mnist_ddp> --ddp_buckets=${buckets})
    set_tests_properties(mnist_ddp_${buckets} PROPERTIES TIMEOUT 120 PROCESSORS 2
      RESOURCE_LOCK mnist_gpu LABELS "mnist;ddp;cuda;requires_2_gpus")
   endforeach()
  endif()
endif()
