# Shared Generator behavior tests. The existing suite helper builds separate
# CPU and CUDA executables and filters the parameterized instances by label.
infini_train_add_test_suite(test_generator
  SOURCES test_generator.cc
)

# Device-independent handle and validation tests are not parameterized, so
# keep them in a separate executable that is not subject to CPU/* filters.
infini_train_add_test(test_generator_interface
  SOURCES test_generator_interface.cc
  LABELS cpu
)

if(USE_CUDA)
  infini_train_add_test(test_generator_cuda_only
    SOURCES cuda_only/test_cuda_generator.cc
    LABELS cuda
  )
endif()
