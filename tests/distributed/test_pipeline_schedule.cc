#include "gtest/gtest.h"
#include <algorithm>
#include <deque>
#include <sstream>

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_layout.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
namespace {
using Scheduler = PipelineParallelScheduler;

// Model ProcessGroup's stream-ordered, unbuffered P2P operations. Send/Recv
// enqueue a compute-stream wait, so a later operation cannot unblock an earlier
// unmatched operation on that same rank. This is stronger than tensor DAG order.
std::string PendingRendezvous(const std::vector<Scheduler::Task> &tasks, int stages, const std::vector<int> &owners) {
    struct Event {
        bool send;
        int peer;
        int mb;
        int edge;
        bool forward;
    };
    std::vector<std::deque<Event>> queues(stages);
    const int chunks = static_cast<int>(owners.size());
    for (const auto &task : tasks) {
        const int gid = task.global_chunk_id, rank = owners.at(gid), mb = task.microbatch_id;
        auto &events = queues.at(rank);
        if (task.is_forward) {
            if (gid > 0 && owners[gid - 1] != rank) {
                events.push_back({false, owners[gid - 1], mb, gid - 1, true});
            }
            if (gid + 1 < chunks && owners[gid + 1] != rank) {
                events.push_back({true, owners[gid + 1], mb, gid, true});
            }
        } else {
            if (gid + 1 < chunks && owners[gid + 1] == rank) {
                continue;
            }
            if (gid + 1 < chunks) {
                events.push_back({false, owners[gid + 1], mb, gid, false});
            }
            int first = gid;
            while (first > 0 && owners[first - 1] == rank) { --first; }
            if (first > 0) {
                events.push_back({true, owners[first - 1], mb, first - 1, false});
            }
        }
    }
    for (;;) {
        bool progress = false;
        for (int rank = 0; rank < stages; ++rank) {
            if (queues[rank].empty()) {
                continue;
            }
            const auto a = queues[rank].front();
            if (queues[a.peer].empty()) {
                continue;
            }
            const auto b = queues[a.peer].front();
            if (a.send != b.send && b.peer == rank && a.mb == b.mb && a.edge == b.edge && a.forward == b.forward) {
                queues[rank].pop_front();
                queues[a.peer].pop_front();
                progress = true;
            }
        }
        if (progress) {
            continue;
        }
        std::ostringstream pending;
        for (int rank = 0; rank < stages; ++rank) {
            if (queues[rank].empty()) {
                continue;
            }
            const auto &e = queues[rank].front();
            pending << "rank=" << rank << (e.send ? " send to=" : " recv from=") << e.peer << " mb=" << e.mb
                    << " edge=" << e.edge << (e.forward ? " F; " : " B; ");
        }
        return pending.str();
    }
}

TEST(PipelineScheduleTest, CommunicationOrderMakesProgressAcrossAllOwners) {
    // Exhaust two-stage owners through six chunks and three-stage owners through five.
    for (int stages : {2, 3}) {
        for (int count = stages; count <= (stages == 2 ? 6 : 5); ++count) {
            int combinations = 1;
            for (int i = 0; i < count; ++i) { combinations *= stages; }
            for (int code = 0; code < combinations; ++code) {
                std::vector<int> owners, present(stages, 0);
                std::vector<PipelineChunkSpec> specs;
                int digits = code;
                for (int gid = 0; gid < count; ++gid) {
                    const int owner = digits % stages;
                    digits /= stages;
                    owners.push_back(owner);
                    ++present[owner];
                    specs.push_back({owner, 1});
                }
                if (std::find(present.begin(), present.end(), 0) != present.end()) {
                    continue;
                }
                const auto layout = PipelineLayout::BuildChunkLayout(count, stages, specs);
                for (int n = 1; n <= 4; ++n) {
                    SCOPED_TRACE(::testing::Message() << "stages=" << stages << " chunks=" << count
                                                      << " owners=" << code << " microbatches=" << n);
                    const auto tasks = Scheduler::GenerateGPipeSchedule(n, stages, layout.GetMaxLocalChunks(), &layout);
                    ASSERT_EQ(PendingRendezvous(tasks, stages, owners), "");
                }
            }
        }
    }
}

TEST(PipelineScheduleTest, DetectsLegacyCyclicCommunicationDeadlock) {
    // Fixed old PP2/vPP2/n2 forward sequence; do not derive it from the new generator.
    const std::vector<std::array<int, 3>> rows{{0, 0, 0}, {1, 0, 1}, {1, 1, 0}, {2, 1, 1},
                                               {2, 0, 2}, {3, 0, 3}, {3, 1, 2}, {4, 1, 3}};
    std::vector<Scheduler::Task> tasks;
    for (const auto &row : rows) { tasks.push_back(Scheduler::CreateTask(row[0], row[1], row[2], 2, 4, true)); }
    EXPECT_EQ(PendingRendezvous(tasks, 2, {0, 1, 0, 1}),
              "rank=0 send to=1 mb=1 edge=0 F; rank=1 send to=0 mb=0 edge=1 F; ");
}

TEST(PipelineScheduleTest, CyclicGPipeUsesProgressSafeOrderAcrossAllRequestSources) {
    const auto uniform = PipelineLayout::BuildUniformLayout(12, 2, 2);
    const std::vector<LayerIndex> counts{1, 2, 3, 6};
    const auto custom = PipelineLayout::BuildCustomLayout(12, 2, counts, 2);
    const std::vector<PipelineChunkSpec> specs{{0, 1}, {1, 2}, {0, 3}, {1, 6}};
    const auto explicit_layout = PipelineLayout::BuildChunkLayout(12, 2, specs);
    const std::vector<std::array<int, 3>> expected{
        {0, 0, 0}, {1, 0, 1}, {2, 0, 2},  {3, 0, 3},  {4, 1, 0},  {5, 1, 1},  {6, 1, 2},  {7, 1, 3},
        {8, 1, 3}, {9, 1, 2}, {10, 1, 1}, {11, 1, 0}, {12, 0, 3}, {13, 0, 2}, {14, 0, 1}, {15, 0, 0}};
    for (const auto *layout : {static_cast<const PipelineLayout *>(nullptr), &uniform, &custom, &explicit_layout}) {
        const auto tasks = Scheduler::GenerateGPipeSchedule(2, 2, 2, layout);
        ASSERT_EQ(tasks.size(), expected.size());
        for (size_t i = 0; i < tasks.size(); ++i) {
            const auto &task = tasks[i];
            EXPECT_EQ((std::array<int, 3>{task.step, task.microbatch_id, task.global_chunk_id}), expected[i]);
            EXPECT_EQ(task.is_forward, i < 8);
            EXPECT_EQ(task.stage_id, expected[i][2] % 2);
            EXPECT_EQ(task.local_chunk_idx, expected[i][2] / 2);
        }
        EXPECT_EQ(PendingRendezvous(tasks, 2, {0, 1, 0, 1}), "");
    }
}

TEST(PipelineScheduleTest, PreservesPinnedLegacyTaskOrder) {
    // Captured from f604940's generators, not from the modified implementation.
    // Each row is (step, microbatch, global chunk), with forward flags listed separately.
    struct Case {
        int vpp;
        bool one_f_one_b;
        std::vector<std::array<int, 3>> rows;
        const char *directions;
    };
    const std::vector<Case> cases{
        {1,
         false,
         {{0, 0, 0}, {1, 0, 1}, {1, 1, 0}, {2, 1, 1}, {3, 1, 1}, {4, 0, 1}, {4, 1, 0}, {5, 0, 0}},
         "FFFFBBBB"},
        {1, true, {{0, 0, 0}, {1, 0, 1}, {1, 0, 1}, {1, 1, 0}, {2, 0, 0}, {2, 1, 1}, {2, 1, 1}, {3, 1, 0}}, "FFBFBFBB"},
        {2,
         true,
         {{0, 0, 0},
          {1, 0, 1},
          {1, 1, 0},
          {2, 0, 2},
          {2, 1, 1},
          {3, 0, 3},
          {3, 0, 3},
          {3, 1, 2},
          {4, 0, 2},
          {4, 1, 3},
          {4, 1, 3},
          {5, 0, 1},
          {5, 1, 2},
          {6, 0, 0},
          {6, 1, 1},
          {7, 1, 0}},
         "FFFFFFBFBFBBBBBB"},
    };
    for (const auto &c : cases) {
        const auto uniform = PipelineLayout::BuildUniformLayout(12, 2, c.vpp);
        const std::vector<LayerIndex> counts
            = c.vpp == 1 ? std::vector<LayerIndex>{3, 9} : std::vector<LayerIndex>{1, 2, 3, 6};
        const auto custom = PipelineLayout::BuildCustomLayout(12, 2, counts, c.vpp);
        std::vector<PipelineChunkSpec> specs;
        for (size_t gid = 0; gid < counts.size(); ++gid) { specs.push_back({static_cast<int>(gid % 2), counts[gid]}); }
        const auto explicit_interleaved = PipelineLayout::BuildChunkLayout(12, 2, specs);
        for (const auto *layout :
             {static_cast<const PipelineLayout *>(nullptr), &uniform, &custom, &explicit_interleaved}) {
            SCOPED_TRACE(::testing::Message() << "vpp=" << c.vpp << " 1F1B=" << c.one_f_one_b << " layout=" << layout);
            const auto tasks = c.one_f_one_b ? Scheduler::GenerateInterleaved1F1BSchedule(2, 2, c.vpp, layout)
                                             : Scheduler::GenerateGPipeSchedule(2, 2, c.vpp, layout);
            ASSERT_EQ(tasks.size(), c.rows.size());
            for (size_t i = 0; i < tasks.size(); ++i) {
                const auto &t = tasks[i];
                EXPECT_EQ((std::array<int, 3>{t.step, t.microbatch_id, t.global_chunk_id}), c.rows[i]);
                EXPECT_EQ(t.is_forward, c.directions[i] == 'F');
                EXPECT_EQ(t.stage_id, c.rows[i][2] % 2);
                EXPECT_EQ(t.local_chunk_idx, c.rows[i][2] / 2);
                EXPECT_EQ(t.is_first_chunk, c.rows[i][2] == 0);
                EXPECT_EQ(t.is_last_chunk, c.rows[i][2] == 2 * c.vpp - 1);
            }
        }
    }
}

TEST(PipelineScheduleTest, ArbitraryMappingsPreserveDependenciesAndOwnership) {
    // Local chains, revisited stages, moved endpoints and unequal local chunk counts.
    const std::vector<std::vector<int>> mappings{{0, 1, 1, 0}, {1, 0, 1, 1}, {2, 0, 0, 1, 2}};
    for (const auto &owners : mappings) {
        const int stages = owners == mappings.back() ? 3 : 2;
        const int count = static_cast<int>(owners.size());
        std::vector<PipelineChunkSpec> specs;
        std::vector<int> locals, next_local(stages, 0);
        for (int owner : owners) {
            specs.push_back({owner, 1});
            locals.push_back(next_local[owner]++);
        }
        const auto layout = PipelineLayout::BuildChunkLayout(count, stages, specs);
        for (int n : {1, 3}) {
            const auto tasks = Scheduler::GenerateGPipeSchedule(n, stages, layout.GetMaxLocalChunks(), &layout);
            ASSERT_EQ(tasks.size(), 2 * n * count);
            std::vector<std::vector<int>> forwards(n, std::vector<int>(count, -1));
            auto backwards = forwards;
            bool backward_started = false;
            for (size_t i = 0; i < tasks.size(); ++i) {
                const auto &t = tasks[i];
                ASSERT_GE(t.microbatch_id, 0);
                ASSERT_LT(t.microbatch_id, n);
                ASSERT_GE(t.global_chunk_id, 0);
                ASSERT_LT(t.global_chunk_id, count);
                const int gid = t.global_chunk_id, mb = t.microbatch_id;
                EXPECT_EQ(t.stage_id, owners[gid]);
                EXPECT_EQ(t.local_chunk_idx, locals[gid]);
                EXPECT_EQ(t.is_first_chunk, gid == 0);
                EXPECT_EQ(t.is_last_chunk, gid == count - 1);
                auto &seen = t.is_forward ? forwards : backwards;
                EXPECT_EQ(seen[mb][gid], -1);
                seen[mb][gid] = static_cast<int>(i);
                if (t.is_forward) {
                    EXPECT_FALSE(backward_started);
                    if (gid > 0) {
                        EXPECT_GE(forwards[mb][gid - 1], 0);
                    }
                } else {
                    backward_started = true;
                    EXPECT_GE(forwards[mb][gid], 0);
                    if (gid + 1 < count) {
                        EXPECT_GE(backwards[mb][gid + 1], 0);
                    }
                }
            }
        }
    }
}

TEST(PipelineScheduleTest, RejectsMismatchedDimensionsAndOverflowBeforeAllocation) {
    const auto layout = PipelineLayout::BuildUniformLayout(8, 2, 2);
    EXPECT_THROW(Scheduler::GenerateGPipeSchedule(1, 3, 2, &layout), std::invalid_argument);
    EXPECT_THROW(Scheduler::GenerateGPipeSchedule(1, 2, 1, &layout), std::invalid_argument);
    EXPECT_THROW(Scheduler::GenerateGPipeSchedule(std::numeric_limits<int>::max(), 2, 2), std::invalid_argument);
    EXPECT_THROW(Scheduler::GenerateGPipeSchedule(std::numeric_limits<int>::max() / 8 + 1, 2, 2),
                 std::invalid_argument);
    EXPECT_TRUE(Scheduler::GenerateGPipeSchedule(0, 2, 2).empty());
    EXPECT_TRUE(Scheduler::GenerateInterleaved1F1BSchedule(0, 2, 2).empty());
}

struct SyncState {
    bool suppressed = false;
    int entries = 0;
    int exits = 0;
    std::vector<bool> suppressed_at_accumulation;
    std::vector<bool> suppressed_at_backward;
};

class ObserveAccumulation : public autograd::PostAccumulateGradHook {
public:
    explicit ObserveAccumulation(std::shared_ptr<SyncState> state) : state_(std::move(state)) {}
    void operator()(const std::shared_ptr<Tensor> &) override {
        state_->suppressed_at_accumulation.push_back(state_->suppressed);
    }

private:
    std::shared_ptr<SyncState> state_;
};

// CPU numerical test of the real executor. No mocked communication or copied executor.
class Scale : public Module {
public:
    explicit Scale(float value) {
        parameters_["weight"] = std::make_shared<Tensor>(&value, std::vector<int64_t>{1}, DataType::kFLOAT32, Device());
        parameters_["weight"]->set_requires_grad(true);
        parameters_["weight"]->RegisterPostAccumulateGradHook(std::make_shared<ObserveAccumulation>(sync));
        backward_hook_ = RegisterBackwardPreHook(
            [state = sync](Module *, const auto &) { state->suppressed_at_backward.push_back(state->suppressed); });
    }
    std::unique_ptr<NoSyncGuard> no_sync() override {
        sync->suppressed = true;
        ++sync->entries;
        return std::make_unique<NoSyncGuard>([state = sync] {
            state->suppressed = false;
            ++state->exits;
        });
    }
    std::shared_ptr<SyncState> sync = std::make_shared<SyncState>();
    std::shared_ptr<HookHandle> backward_hook_;
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &x) override {
        return {x[0] * parameters_["weight"]};
    }
};
class SquaredLoss : public Module {
public:
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &x) override {
        auto error = x[0] - x[1];
        return {(error * error)->Mean(0)};
    }
};
TEST(PipelineScheduleTest, LocalChainAccumulatesGradientsAndUpdatesOncePerStep) {
    auto first = std::make_shared<Scale>(2.0f), second = std::make_shared<Scale>(3.0f);
    std::vector<std::shared_ptr<Module>> chunks{first, second};
    auto stage
        = std::make_shared<PipelineStage>(0, 1, std::vector<std::vector<int64_t>>{{1}}, Device(), std::move(chunks));
    const std::vector<PipelineChunkSpec> specs{{0, 1}, {0, 1}};
    auto layout = std::make_shared<const PipelineLayout>(PipelineLayout::BuildChunkLayout(2, 1, specs));
    PipelineSchedule schedule(stage, 1, 2, layout, true);
    auto p = first->parameter("weight"), q = second->parameter("weight");
    auto optimizer = std::make_shared<optimizers::SGD>(std::vector<std::shared_ptr<Tensor>>{p, q}, 0.01f);
    auto loss_fn = std::make_shared<SquaredLoss>();
    const float xs[]{1, 2}, ys[]{0, 0};
    auto x = std::make_shared<Tensor>(xs, std::vector<int64_t>{2}, DataType::kFLOAT32, Device());
    auto y = std::make_shared<Tensor>(ys, std::vector<int64_t>{2}, DataType::kFLOAT32, Device());
    const auto scalar = [](const auto &t) { return static_cast<const float *>(t->DataPtr())[0]; };
    for (int step = 0; step < 2; ++step) {
        const float a = scalar(p), b = scalar(q);
        // mean((a*b*[1,2])^2) = 2.5*a^2*b^2.
        const float loss = schedule.Step(x, y, optimizer, loss_fn, DataType::kFLOAT32);
        EXPECT_NEAR(loss, 2.5f * a * a * b * b, 1e-4f);
        ASSERT_NE(p->grad(), nullptr);
        ASSERT_NE(q->grad(), nullptr);
        EXPECT_NEAR(scalar(p->grad()), 5 * a * b * b, 1e-4f);
        EXPECT_NEAR(scalar(q->grad()), 5 * a * a * b, 1e-4f);
        EXPECT_NEAR(scalar(p), a - 0.01f * 5 * a * b * b, 1e-5f);
        EXPECT_NEAR(scalar(q), b - 0.01f * 5 * a * a * b, 1e-5f);
    }
    for (const auto &chunk : {first, second}) {
        EXPECT_EQ(chunk->sync->entries, 2);
        EXPECT_EQ(chunk->sync->exits, 2);
        // Every chunk in the chain must leave no_sync before its final backward.
        EXPECT_EQ(chunk->sync->suppressed_at_backward, (std::vector<bool>{true, false, true, false}));
        // GPipe builds all forward graphs first; the shared accumulator fires once per step.
        EXPECT_EQ(chunk->sync->suppressed_at_accumulation, (std::vector<bool>{false, false}));
    }
}
TEST(PipelineScheduleTest, ExplicitStepRejectsMalformedBatchesBeforeOptimizerOrCommunication) {
    auto chunk = std::make_shared<Scale>(2.0f);
    auto stage = std::make_shared<PipelineStage>(0, 1, std::vector<std::vector<int64_t>>{{1}}, Device(),
                                                 std::vector<std::shared_ptr<Module>>{chunk});
    const std::vector<PipelineChunkSpec> specs{{0, 1}};
    auto layout = std::make_shared<const PipelineLayout>(PipelineLayout::BuildChunkLayout(1, 1, specs));
    PipelineSchedule schedule(stage, 1, 2, layout, true);
    const float values[]{1, 2, 3};
    auto valid = std::make_shared<Tensor>(values, std::vector<int64_t>{2}, DataType::kFLOAT32, Device());
    auto uneven = std::make_shared<Tensor>(values, std::vector<int64_t>{3}, DataType::kFLOAT32, Device());
    auto scalar = std::make_shared<Tensor>(values, std::vector<int64_t>{}, DataType::kFLOAT32, Device());
    // Null optimizer is deliberate: each validation must happen before ZeroGrad.
    for (const auto &invalid : {std::shared_ptr<Tensor>{}, scalar, uneven}) {
        EXPECT_THROW(schedule.Step(invalid, valid, nullptr, nullptr, DataType::kFLOAT32), std::invalid_argument);
        EXPECT_THROW(schedule.Step(valid, invalid, nullptr, nullptr, DataType::kFLOAT32), std::invalid_argument);
    }
    EXPECT_EQ(chunk->sync->entries, 0);
    EXPECT_EQ(chunk->parameter("weight")->grad(), nullptr);
}

} // namespace
} // namespace infini_train::nn::parallel
