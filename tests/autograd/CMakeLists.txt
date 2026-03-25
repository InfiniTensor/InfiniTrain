# ============================================================================
# Autograd tests
# ============================================================================
# 重构版本：使用 infini_train_add_test 宏简化配置
#
# 新增测试只需 1 行：
#   infini_train_add_test(test_name SOURCES test_name.cc LABELS cpu)
# ============================================================================

# -----------------------------------------------------------------------------
# Elementwise tests
# -----------------------------------------------------------------------------
infini_train_add_test(test_autograd_elementwise_forward SOURCES test_autograd_elementwise_forward.cc LABELS cpu)
infini_train_add_test(test_autograd_elementwise_backward SOURCES test_autograd_elementwise_backward.cc LABELS cpu)

# -----------------------------------------------------------------------------
# Matmul tests
# -----------------------------------------------------------------------------
infini_train_add_test(test_autograd_matmul_forward SOURCES test_autograd_matmul_forward.cc LABELS cpu)
infini_train_add_test(test_autograd_matmul_backward SOURCES test_autograd_matmul_backward.cc LABELS cpu)

# -----------------------------------------------------------------------------
# Reduction tests
# -----------------------------------------------------------------------------
infini_train_add_test(test_autograd_reduction_forward SOURCES test_autograd_reduction_forward.cc LABELS cpu)
infini_train_add_test(test_autograd_reduction_backward SOURCES test_autograd_reduction_backward.cc LABELS cpu)

# -----------------------------------------------------------------------------
# Linear tests
# -----------------------------------------------------------------------------
infini_train_add_test(test_autograd_linear_forward SOURCES test_autograd_linear_forward.cc LABELS cpu)
infini_train_add_test(test_autograd_linear_backward SOURCES test_autograd_linear_backward.cc LABELS cpu)

# -----------------------------------------------------------------------------
# Softmax tests
# -----------------------------------------------------------------------------
infini_train_add_test(test_autograd_softmax_forward SOURCES test_autograd_softmax_forward.cc LABELS cpu)
infini_train_add_test(test_autograd_softmax_backward SOURCES test_autograd_softmax_backward.cc LABELS cpu)

# -----------------------------------------------------------------------------
# Transform tests
# -----------------------------------------------------------------------------
infini_train_add_test(test_autograd_transform_forward SOURCES test_autograd_transform_forward.cc LABELS cpu)
infini_train_add_test(test_autograd_transform_backward SOURCES test_autograd_transform_backward.cc LABELS cpu)

# -----------------------------------------------------------------------------
# Normalization tests
# -----------------------------------------------------------------------------
infini_train_add_test(test_autograd_normalization_forward SOURCES test_autograd_normalization_forward.cc LABELS cpu)
infini_train_add_test(test_autograd_normalization_backward SOURCES test_autograd_normalization_backward.cc LABELS cpu)

# -----------------------------------------------------------------------------
# Legacy combined tests
# 注意：使用 gtest_discover_tests，所有 TEST_F 都会被自动发现
# -----------------------------------------------------------------------------
infini_train_add_test(test_autograd_legacy SOURCES test_autograd.cc LABELS cpu cuda distributed)
