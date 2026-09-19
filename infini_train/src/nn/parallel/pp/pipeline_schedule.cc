// pipeline_schedule.cc
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pp/pipeline_layout.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"
#include "infini_train/include/nn/parallel/pp/send_recv.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

PipelineSchedule::PipelineSchedule(std::shared_ptr<PipelineStage> stage, int num_stages, int num_micro_batches,
                                   std::shared_ptr<const PipelineLayout> layout, bool explicit_chunk_mode)
    : num_micro_batches_(num_micro_batches), stage_(std::move(stage)), layout_(std::move(layout)),
      explicit_chunk_mode_(explicit_chunk_mode) {
    if (!stage_ || num_micro_batches_ <= 0 || num_stages != stage_->num_stages()) {
        throw std::invalid_argument("invalid pipeline stage or microbatch count");
    }
    if (explicit_chunk_mode_ && !layout_) {
        throw std::invalid_argument("explicit execution requires a layout");
    }
    if (layout_
        && (layout_->GetNumStages() != num_stages
            || stage_->chunks().size() != layout_->GetStage(stage_->stage_index()).chunks.size())) {
        throw std::invalid_argument("stage modules disagree with pipeline layout");
    }
    if (layout_ && !explicit_chunk_mode_) {
        const int count = layout_->GetNumChunks();
        if (count <= 0 || count % num_stages != 0) {
            throw std::invalid_argument("ordinary execution requires a full interleaved layout");
        }
        for (int gid = 0; gid < count; ++gid) {
            const auto &chunk = layout_->GetChunk(gid);
            if (chunk.stage_id != gid % num_stages || chunk.local_chunk_idx != gid / num_stages) {
                throw std::invalid_argument("explicit ownership requires explicit execution mode");
            }
        }
    }
}

void PrintScheduleTable(const std::vector<PipelineParallelScheduler::Task> &schedule, int n, int total_chunks) {
    LOG(INFO) << "Pipeline schedule: microbatches=" << n << ", chunks=" << total_chunks;
    LOG(INFO) << "Step | Type | Microbatch | Global Chunk | Local Chunk | Stage";
    for (const auto &task : schedule) {
        LOG(INFO) << task.step << " | " << (task.is_forward ? "Forward" : "Backward") << " | " << task.microbatch_id
                  << " | " << task.global_chunk_id << " | " << task.local_chunk_idx << " | " << task.stage_id;
    }
}

std::vector<std::shared_ptr<Tensor>> PipelineSchedule::ReceiveFromPrev(int peer_rank) {
    std::vector<std::shared_ptr<Tensor>> recv_tensors;
    auto &shapes = stage_->recv_shape();
    for (size_t i = 0; i < shapes.size(); ++i) {
        // FIXME(jym): The data type between stages is not float32, which will cause a crash
        auto tensor = std::make_shared<Tensor>(shapes[i], DataType::kFLOAT32, stage_->device());
        tensor->set_requires_grad(true);

        // Mark as non-leaf to prevent the autograd engine from creating an AccumulateGrad
        // for this tensor. Otherwise, IRecv's next_functions_ would hold AccumulateGrad,
        // which holds a shared_ptr back to this tensor, forming a reference cycle:
        //   tensor -> grad_fn_(IRecv) -> next_functions_ -> AccumulateGrad -> tensor
        // This cycle prevents the autograd graph from being released after backward.
        tensor->set_is_leaf(false);
        recv_tensors.push_back(tensor);
    }

    return IRecv(recv_tensors, stage_->device(), peer_rank);
}

std::vector<std::shared_ptr<Tensor>> PipelineSchedule::SendToNext(const std::vector<std::shared_ptr<Tensor>> &tensors,
                                                                  int peer_rank) {
    return ISend(tensors, stage_->device(), peer_rank, stage_->recv_shape());
}

PipelineParallelScheduler::Task PipelineParallelScheduler::CreateTask(int step, int mb, int global_chunk,
                                                                      int num_stages, int total_chunks, bool is_forward,
                                                                      const PipelineLayout *layout) {
    PipelineParallelScheduler::Task task;
    task.step = step;
    task.microbatch_id = mb;
    task.global_chunk_id = global_chunk;
    task.local_chunk_idx = layout ? layout->GetChunk(global_chunk).local_chunk_idx : global_chunk / num_stages;
    task.is_forward = is_forward;
    task.stage_id = layout ? layout->GetChunk(global_chunk).stage_id : global_chunk % num_stages;
    task.is_last_chunk = (global_chunk == total_chunks - 1);
    task.is_first_chunk = (global_chunk == 0);
    return task;
}

std::vector<PipelineParallelScheduler::Task>
PipelineParallelScheduler::GenerateGPipeSchedule(int n, int num_stages, int vpp_size, const PipelineLayout *layout) {
    std::vector<Task> schedule;
    if (n <= 0) {
        return schedule;
    }
    if (num_stages <= 0 || vpp_size <= 0
        || static_cast<int64_t>(num_stages) * vpp_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("invalid schedule dimensions");
    }
    int total_global_chunks = layout ? layout->GetNumChunks() : num_stages * vpp_size;
    if (total_global_chunks <= 0 || total_global_chunks > std::numeric_limits<int>::max() / 2
        || n > std::numeric_limits<int>::max() / 2 - total_global_chunks) {
        throw std::invalid_argument("schedule step count overflow");
    }

    bool interleaved = true;
    if (layout) {
        if (layout->GetNumStages() != num_stages || layout->GetMaxLocalChunks() != vpp_size) {
            throw std::invalid_argument("schedule dimensions disagree with layout");
        }
        interleaved = (total_global_chunks == static_cast<int64_t>(num_stages) * vpp_size);
        for (int gid = 0; interleaved && gid < total_global_chunks; ++gid) {
            const auto &chunk = layout->GetChunk(gid);
            interleaved = (chunk.stage_id == gid % num_stages) && (chunk.local_chunk_idx == gid / num_stages);
        }
    }
    // A rank revisited after a cross-rank edge cannot safely pipeline the old
    // diagonal task order: Send/Recv put a completion wait on its compute stream.
    // Example: owners 0,1,0 can make rank 0 send mb1 while rank 1 sends mb0 back.
    std::vector<bool> visited(num_stages, false);
    int previous_owner = -1;
    bool revisits_stage = false;
    for (int gid = 0; gid < total_global_chunks; ++gid) {
        const int owner = layout ? layout->GetChunk(gid).stage_id : gid % num_stages;
        if (owner != previous_owner) {
            revisits_stage = revisits_stage || visited[owner];
            visited[owner] = true;
            previous_owner = owner;
        }
    }
    if (revisits_stage) {
        // Here step numbers tasks, not diagonal waves; protect 2*n*chunks.
        if (n > std::numeric_limits<int>::max() / 2 / total_global_chunks) {
            throw std::invalid_argument("schedule task count overflow");
        }
        int step = 0;
        for (int mb = 0; mb < n; ++mb) {
            for (int gid = 0; gid < total_global_chunks; ++gid) {
                schedule.push_back(CreateTask(step++, mb, gid, num_stages, total_global_chunks, true, layout));
            }
        }
        for (int mb = n - 1; mb >= 0; --mb) {
            for (int gid = total_global_chunks - 1; gid >= 0; --gid) {
                schedule.push_back(CreateTask(step++, mb, gid, num_stages, total_global_chunks, false, layout));
            }
        }
        return schedule;
    }
    if (!interleaved) {
        const int waves = n + total_global_chunks - 1;
        for (int direction = 0; direction < 2; ++direction) {
            for (int wave = 0; wave < waves; ++wave) {
                for (int mb = 0; mb < n; ++mb) {
                    const int offset = wave - mb;
                    if (offset < 0 || offset >= total_global_chunks) {
                        continue;
                    }
                    const int gid = direction == 0 ? offset : total_global_chunks - 1 - offset;
                    schedule.push_back(CreateTask(wave + direction * waves, mb, gid, num_stages, total_global_chunks,
                                                  direction == 0, layout));
                }
            }
        }
        return schedule;
    }

    // Keep the existing GPipe wave order for layouts without rank revisits
    int total_steps = n + total_global_chunks - 1;

    // ======== Forward Pass ========
    for (int step = 0; step < total_steps; ++step) {
        for (int mb = 0; mb < n; ++mb) {
            int global_chunk_id = step - mb;
            if (global_chunk_id >= 0 && global_chunk_id < total_global_chunks) {
                auto is_forward = true;
                auto task = CreateTask(step, mb, global_chunk_id, num_stages, total_global_chunks, is_forward, layout);
                schedule.push_back(task);
            }
        }
    }

    // ======== Backward Pass ========
    for (int step = 0; step < total_steps; ++step) {
        for (int mb = 0; mb < n; ++mb) {
            int global_chunk_id = (total_steps - 1 - step) - mb;
            if (global_chunk_id >= 0 && global_chunk_id < total_global_chunks) {
                auto is_forward = false;
                auto task = CreateTask(step + total_steps, mb, global_chunk_id, num_stages, total_global_chunks,
                                       is_forward, layout);
                schedule.push_back(task);
            }
        }
    }

    // sorted according to step, local_chunk_idx
    std::sort(schedule.begin(), schedule.end(), [](const Task &a, const Task &b) {
        if (a.step != b.step) {
            return a.step < b.step;
        }

        return a.local_chunk_idx < b.local_chunk_idx;
    });

    return schedule;
}

std::vector<PipelineParallelScheduler::Task>
PipelineParallelScheduler::GenerateInterleaved1F1BSchedule(int n, int num_stages, int vpp_size,
                                                           const PipelineLayout *layout) {
    std::vector<Task> schedule;
    // Preserve legacy empty-schedule behavior for non-positive dimensions.
    if (n <= 0 || num_stages <= 0 || vpp_size <= 0) {
        return schedule;
    }
    if (n <= 0 || num_stages <= 0 || vpp_size <= 0
        || static_cast<int64_t>(num_stages) * vpp_size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("invalid schedule dimension");
    }

    int total_global_chunks = layout ? layout->GetNumChunks() : num_stages * vpp_size;
    if (total_global_chunks <= 0 || total_global_chunks > std::numeric_limits<int>::max() / 2
        || n > std::numeric_limits<int>::max() / 2 - total_global_chunks) {
        throw std::invalid_argument("schedule step count overflow");
    }

    int warmup_steps = total_global_chunks - 1;
    int total_steps = 2 * warmup_steps + n;

    // ================ Warm-up ================
    for (int step = 0; step < warmup_steps; ++step) {
        for (int mb = 0; mb < n; ++mb) {
            int forward_global_chunk = step - mb;
            if (forward_global_chunk >= 0 && forward_global_chunk < total_global_chunks) {
                auto is_forward = true;
                auto task
                    = CreateTask(step, mb, forward_global_chunk, num_stages, total_global_chunks, is_forward, layout);
                schedule.push_back(task);
            }
        }
    }

    // ================ Steady ================
    for (int step = warmup_steps; step < warmup_steps + n; ++step) {
        int stable_step = step - warmup_steps;

        for (int mb = 0; mb < n; ++mb) {
            // Forward
            int forward_global_chunk = step - mb;
            if (forward_global_chunk >= 0 && forward_global_chunk < total_global_chunks) {
                auto is_forward = true;
                auto task
                    = CreateTask(step, mb, forward_global_chunk, num_stages, total_global_chunks, is_forward, layout);
                schedule.push_back(task);
            }

            // Backward
            int backward_global_chunk = (total_global_chunks - 1) - (stable_step - mb);

            if (backward_global_chunk >= 0 && backward_global_chunk < total_global_chunks) {
                auto is_forward = false;
                auto task
                    = CreateTask(step, mb, backward_global_chunk, num_stages, total_global_chunks, is_forward, layout);
                schedule.push_back(task);
            }
        }
    }

    // ================ Cool-down ================
    for (int step = warmup_steps + n; step < total_steps; ++step) {
        for (int mb = 0; mb < n; ++mb) {
            int backward_step = step - (warmup_steps);
            int backward_global_chunk = (total_global_chunks - 1) - (backward_step - mb);
            if (backward_global_chunk >= 0 && backward_global_chunk < total_global_chunks) {
                auto is_forward = false;
                auto task
                    = CreateTask(step, mb, backward_global_chunk, num_stages, total_global_chunks, is_forward, layout);
                schedule.push_back(task);
            }
        }
    }

    return schedule;
}

float PipelineSchedule::StepMicroBatches(const std::vector<std::shared_ptr<Tensor>> &microbatch_inputs,
                                         const std::vector<std::shared_ptr<Tensor>> &microbatch_targets,
                                         const std::shared_ptr<Module> &loss_fn, DataType dtype) {

    const int n = num_micro_batches_;
    int num_stages = stage_->num_stages();
    int stage_idx = stage_->stage_index();
    int vpp_size = layout_ ? layout_->GetMaxLocalChunks() : global::GetVirtualPipelineParallelSize();
    const auto local_count = stage_->chunks().size();
    if (microbatch_inputs.size() != static_cast<std::size_t>(n)
        || microbatch_targets.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("microbatch containers disagree with microbatch count");
    }

    auto schedule = PipelineParallelScheduler::GenerateGPipeSchedule(n, num_stages, vpp_size, layout_.get());
    if (!has_printed_schedule_ && stage_idx == (layout_ ? layout_->GetInputStage() : 0)) {
        PrintScheduleTable(schedule, n, layout_ ? layout_->GetNumChunks() : num_stages * vpp_size);
        has_printed_schedule_ = true;
    }

    using TensorList = std::vector<std::shared_ptr<Tensor>>;
    std::vector<std::vector<TensorList>> activations(local_count, std::vector<TensorList>(n));
    std::vector<std::unique_ptr<nn::NoSyncGuard>> no_sync_guards;
    for (const auto &chunk : stage_->chunks()) { no_sync_guards.push_back(chunk->no_sync()); }
    std::vector<int> backward_counts(local_count, 0);
    float total_loss = 0.0f;

    for (const auto &task : schedule) {
        if (task.stage_id != stage_idx) {
            continue;
        }
        const int gid = task.global_chunk_id;
        const int local = task.local_chunk_idx;
        const int mb = task.microbatch_id;
        if (task.is_forward) {
            infini_train::AutocastGuard autocast_guard(stage_->device().type(), dtype);
            TensorList inputs;
            if (task.is_first_chunk) {
                if (!microbatch_inputs.at(mb)) {
                    throw std::invalid_argument("missing input on logical input stage");
                }
                inputs = {microbatch_inputs.at(mb)};
            } else if (layout_) {
                const auto &previous = layout_->GetChunk(gid - 1);
                inputs = previous.stage_id == stage_idx ? activations.at(previous.local_chunk_idx).at(mb)
                                                        : ReceiveFromPrev(previous.stage_id);
            } else {
                inputs = ReceiveFromPrev(stage_->IsFirstStage() ? num_stages - 1 : stage_->prev_rank());
            }
            auto &output = activations.at(local).at(mb);
            output = stage_->ForwardOneChunk(inputs, local);
            if (!task.is_last_chunk) {
                const int next_owner
                    = layout_ ? layout_->GetChunk(gid + 1).stage_id : (stage_->IsLastStage() ? 0 : stage_->next_rank());
                if (!layout_ || next_owner != stage_idx) {
                    output = SendToNext(output, next_owner);
                }
            }
        } else {
            // A successor on the same stage already traversed this local graph.
            if (layout_ && !task.is_last_chunk && layout_->GetChunk(gid + 1).stage_id == stage_idx) {
                continue;
            }
            if (layout_) {
                for (int chain_gid = gid; chain_gid >= 0; --chain_gid) {
                    const auto &chunk = layout_->GetChunk(chain_gid);
                    if (chunk.stage_id != stage_idx) {
                        break;
                    }
                    if (++backward_counts.at(chunk.local_chunk_idx) == n) {
                        no_sync_guards.at(chunk.local_chunk_idx).reset();
                    }
                }
            } else if (++backward_counts.at(local) == n) {
                no_sync_guards.at(local).reset();
            }
            const auto &output = activations.at(local).at(mb);
            if (output.empty() || !output[0]) {
                throw std::invalid_argument("missing activation ");
            }
            if (task.is_last_chunk) {
                if (!microbatch_targets.at(mb) || !loss_fn) {
                    throw std::invalid_argument("missing target or loss function");
                }
                std::shared_ptr<Tensor> loss;
                {
                    infini_train::AutocastGuard autocast_guard(stage_->device().type(), dtype);
                    auto target = microbatch_targets.at(mb)->To(output[0]->GetDevice());
                    loss = (*loss_fn)({output[0], std::make_shared<Tensor>(target)})[0] / n;
                }
                loss->Backward();
                total_loss += static_cast<const float *>(loss->To(Device()).DataPtr())[0];
            } else {
                auto dummy = std::make_shared<Tensor>(output[0]->Dims(), output[0]->Dtype(), output[0]->GetDevice());
                dummy->Fill(0.0f);
                output[0]->Backward(dummy);
            }
        }
    }
    for (int completed : backward_counts) {
        if (completed != n) {
            throw std::runtime_error("incomplete local chunk backward traversal");
        }
    }
    return total_loss;
}

float PipelineSchedule::Step(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
                             const std::shared_ptr<Optimizer> &optimizer, const std::shared_ptr<Module> &loss_fn,
                             DataType dtype) {
    const int stage_idx = stage_->stage_index();
    const int n = num_micro_batches_;
    const int input_owner = layout_ ? layout_->GetInputStage() : 0;
    const int output_owner = layout_ ? layout_->GetOutputStage() : stage_->num_stages() - 1;
    auto split = [this, n](const std::shared_ptr<Tensor> &tensor) {
        if (explicit_chunk_mode_
            && (!tensor || tensor->Dims().empty() || tensor->Dims()[0] <= 0 || tensor->Dims()[0] % n != 0)) {
            throw std::invalid_argument("invalid input/target microbatch dimension");
        }
        return tensor->Split(tensor->Dims()[0] / n);
    };
    std::vector<std::shared_ptr<Tensor>> micro_batches(n), target_mbs(n);

    if (stage_idx == input_owner) {
        micro_batches = split(input);
    }
    if (stage_idx == output_owner) {
        target_mbs = split(target);
    }

    optimizer->ZeroGrad();

    float lossf = StepMicroBatches(micro_batches, target_mbs, loss_fn, dtype);

    optimizer->Step();

    return lossf;
}

} // namespace infini_train::nn::parallel
