#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace {

constexpr float kEps = 1e-6f;

std::shared_ptr<infini_train::Optimizer> MakeDummyOptimizer(float lr) {
    std::vector<std::shared_ptr<infini_train::Tensor>> empty_params;
    return std::make_shared<infini_train::optimizers::SGD>(empty_params, lr);
}

bool FloatNear(float a, float b, float eps = kEps) {
    return std::fabs(a - b) < eps;
}

int g_fail_count = 0;

void Check(bool cond, const char *expr, int line) {
    if (!cond) {
        std::cerr << "FAIL [line " << line << "]: " << expr << std::endl;
        ++g_fail_count;
    }
}

#define ASSERT_TRUE(cond) Check((cond), #cond, __LINE__)
#define ASSERT_FLOAT_EQ(a, b) \
    Check(FloatNear((a), (b)), #a " == " #b, __LINE__)
#define ASSERT_FLOAT_NEAR(a, b, eps) \
    Check(FloatNear((a), (b), (eps)), #a " ≈ " #b, __LINE__)

}  // namespace