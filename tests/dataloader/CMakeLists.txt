# ==========================================================================
# DataLoader tests
# ==========================================================================

infini_train_add_test(test_dataloader
  SOURCES test_dataloader.cc
  LABELS cpu
)

infini_train_add_test(test_dataloader_stack
  SOURCES test_dataloader_stack.cc
  LABELS cpu
)
