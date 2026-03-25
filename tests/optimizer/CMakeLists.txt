# ============================================================================
# Optimizer tests
# ============================================================================
# 重构版本：使用 infini_train_add_test 宏简化配置
#
# 新增测试只需 1 行：
#   infini_train_add_test(test_name SOURCES test_name.cc LABELS cpu)
# ============================================================================

# -----------------------------------------------------------------------------
# Optimizer tests
# -----------------------------------------------------------------------------
infini_train_add_test(test_optimizer_creation SOURCES test_optimizer_creation.cc LABELS cpu)
infini_train_add_test(test_optimizer_step SOURCES test_optimizer_step.cc LABELS cpu)
infini_train_add_test(test_optimizer_cuda SOURCES test_optimizer_cuda.cc LABELS cuda)
infini_train_add_test(test_optimizer_distributed SOURCES test_optimizer_distributed.cc LABELS cuda distributed)

# -----------------------------------------------------------------------------
# Legacy combined tests
# 注意：使用 gtest_discover_tests，所有 TEST_F 都会被自动发现
# -----------------------------------------------------------------------------
infini_train_add_test(test_optimizer_legacy SOURCES test_optimizer.cc LABELS cpu cuda distributed)
