# ============================================================================
# Autograd tests
# ============================================================================

file(GLOB AUTOGRAD_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)
list(FILTER AUTOGRAD_SOURCES EXCLUDE REGEX ".*/test_autograd_scaled_dot_product_attention\\.cc$")

infini_train_add_test_suite(test_autograd
  SOURCES ${AUTOGRAD_SOURCES}
)

if(USE_FLASH_ATTENTION)
  infini_train_add_test_suite(test_flash_attention
    SOURCES test_autograd_scaled_dot_product_attention.cc
    LABELS cuda
    EXCLUDE_PRIVATEUSE1
    TEST_TIMEOUT 60
  )
endif()
