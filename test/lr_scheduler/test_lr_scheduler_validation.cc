#include <functional>
#include <iostream>
#include <memory>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "infini_train/include/lr_scheduler.h"
#include "test/lr_scheduler/test_helpers.h"

using namespace infini_train;
using namespace infini_train::lr_schedulers;

namespace {

bool ExpectDeath(const std::function<void()> &fn) {
    pid_t pid = fork();
    if (pid == -1) {
        return false;
    }

    if (pid == 0) {
        fn();
        _exit(0);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) == -1) {
        return false;
    }

    return !WIFEXITED(status) || WEXITSTATUS(status) != 0;
}

void TestStepLRRejectsNonPositiveStepSize() {
    ASSERT_TRUE(ExpectDeath([] {
        auto opt = MakeDummyOptimizer(0.1f);
        auto sched = CreateLRScheduler(opt, {
                                                .type = "step",
                                                .step_size = 0,
                                                .step_gamma = 0.1f,
                                            });
        (void)sched;
    }));
}

void TestLinearLRRejectsNonPositiveTotalIters() {
    ASSERT_TRUE(ExpectDeath([] {
        auto opt = MakeDummyOptimizer(0.1f);
        auto sched = CreateLRScheduler(opt, {
                                                .type = "linear",
                                                .linear_start_factor = 0.5f,
                                                .linear_end_factor = 1.0f,
                                                .linear_total_iters = 0,
                                            });
        (void)sched;
    }));
}

void TestLambdaLRRejectsNullLambda() {
    ASSERT_TRUE(ExpectDeath([] {
        auto opt = MakeDummyOptimizer(0.1f);
        auto sched = CreateLRScheduler(opt, {
                                                .type = "lambda",
                                            });
        (void)sched;
    }));
}

void TestSequentialLRRejectsMismatchedOptimizer() {
    ASSERT_TRUE(ExpectDeath([] {
        auto opt1 = MakeDummyOptimizer(0.1f);
        auto opt2 = MakeDummyOptimizer(0.1f);

        auto s1 = CreateLRScheduler(opt1, {
                                              .type = "linear",
                                              .linear_start_factor = 0.5f,
                                              .linear_end_factor = 1.0f,
                                              .linear_total_iters = 2,
                                          });
        auto s2 = CreateLRScheduler(opt2, {
                                              .type = "step",
                                              .step_size = 2,
                                              .step_gamma = 0.5f,
                                          });

        auto sched = LRScheduler::Create<SequentialLR>(
            opt1, std::vector<std::shared_ptr<LRScheduler>>{s1, s2}, std::vector<int64_t>{1});
        (void)sched;
    }));
}

void TestSequentialLRRejectsNullChild() {
    ASSERT_TRUE(ExpectDeath([] {
        auto opt = MakeDummyOptimizer(0.1f);
        auto sched = LRScheduler::Create<SequentialLR>(opt, std::vector<std::shared_ptr<LRScheduler>>{nullptr},
                                                       std::vector<int64_t>{});
        (void)sched;
    }));
}

void TestChainedSchedulerRejectsEmptyChildren() {
    ASSERT_TRUE(ExpectDeath([] {
        auto opt = MakeDummyOptimizer(0.1f);
        auto sched = LRScheduler::Create<ChainedScheduler>(opt, std::vector<std::shared_ptr<LRScheduler>>{});
        (void)sched;
    }));
}

void TestChainedSchedulerRejectsMismatchedOptimizer() {
    ASSERT_TRUE(ExpectDeath([] {
        auto opt1 = MakeDummyOptimizer(0.1f);
        auto opt2 = MakeDummyOptimizer(0.1f);

        auto s1 = CreateLRScheduler(opt1, {
                                              .type = "step",
                                              .step_size = 2,
                                              .step_gamma = 0.5f,
                                          });
        auto s2 = CreateLRScheduler(opt2, {
                                              .type = "constant",
                                              .constant_factor = 0.5f,
                                              .constant_total_iters = 2,
                                          });

        auto sched = LRScheduler::Create<ChainedScheduler>(opt1, std::vector<std::shared_ptr<LRScheduler>>{s1, s2});
        (void)sched;
    }));
}

void TestChainedSchedulerRejectsNullChild() {
    ASSERT_TRUE(ExpectDeath([] {
        auto opt = MakeDummyOptimizer(0.1f);
        auto sched = LRScheduler::Create<ChainedScheduler>(opt, std::vector<std::shared_ptr<LRScheduler>>{nullptr});
        (void)sched;
    }));
}

} // namespace

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    std::cout << "=== LR Scheduler Validation Tests ===" << std::endl;
    TestStepLRRejectsNonPositiveStepSize();
    TestLinearLRRejectsNonPositiveTotalIters();
    TestLambdaLRRejectsNullLambda();
    TestSequentialLRRejectsMismatchedOptimizer();
    TestSequentialLRRejectsNullChild();
    TestChainedSchedulerRejectsEmptyChildren();
    TestChainedSchedulerRejectsMismatchedOptimizer();
    TestChainedSchedulerRejectsNullChild();

    if (g_fail_count == 0) {
        std::cout << "All Tests PASSED" << std::endl;
    } else {
        std::cout << g_fail_count << " test(s) FAILED" << std::endl;
    }

    return g_fail_count > 0 ? 1 : 0;
}
