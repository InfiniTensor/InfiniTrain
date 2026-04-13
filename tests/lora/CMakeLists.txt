# ==========================================================================
# LoRA tests
# ==========================================================================

set(LORA_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

infini_train_add_test_suite(test_lora
  SOURCES ${LORA_TEST_DIR}/test_lora.cc
)
