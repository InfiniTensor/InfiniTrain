# ============================================================================
# Hook tests
# ============================================================================
# 重构版本：使用 infini_train_add_test 宏简化配置
#
# 新增测试只需 1 行：
#   infini_train_add_test(test_name SOURCES test_name.cc LABELS cpu)
# ============================================================================

# -----------------------------------------------------------------------------
# Hook tests
# -----------------------------------------------------------------------------
infini_train_add_test(test_hook SOURCES test_hook.cc LABELS cpu cuda distributed)
infini_train_add_test(test_precision_check SOURCES test_precision_check.cc LABELS cpu)
