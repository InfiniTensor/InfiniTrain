# ==========================================================================
# LoRA tests
# ==========================================================================

file(GLOB LORA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

infini_train_add_test_suite(test_lora
  SOURCES ${LORA_SOURCES}
)
