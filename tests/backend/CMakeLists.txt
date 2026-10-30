infini_train_add_test(test_privateuse1_backend
  SOURCES test_privateuse1_backend.cc
  LABELS cpu
)

infini_train_add_test(test_privateuse1_backend_validation
  SOURCES test_privateuse1_backend_validation.cc
  LABELS cpu
)

if(USE_CUDA)
  infini_train_add_test(test_stream_guard_cuda
    SOURCES test_stream_guard.cc
    LABELS cuda
    BINARY_REGISTRATION
    RUN_SERIAL
    TEST_TIMEOUT 60
  )
endif()
