#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train {

// Bookkeeping shared by the sparse embedding backward and the optimizer, so that a vocab-sized
// gradient buffer can be kept alive across steps and only the rows actually hit by the batch need
// to be scattered into, cleared and updated.
//
// Invariant: grad_buffer is all-zero except for the rows listed in row_list[0..*count), which were
// claimed (deduplicated against stamp) since the last clear. generation is bumped at every clear
// so the stamp array can tag a row as "already claimed this cycle" without ever being reset.
struct SparseRowGradState {
    const Tensor *weight = nullptr;
    std::weak_ptr<Tensor> weight_guard; // detects a recycled pointer after the weight was freed
    int64_t vocab = 0;
    int64_t dim = 0;
    int64_t capacity = 0; // row_list capacity; == vocab, so dedup alone guarantees no overflow
    int32_t generation = 1;  // stamp value identifying the current accumulation cycle
    bool initialized = false; // one-time device zero-fill of grad_buffer/stamp/count is done
    bool poisoned = false; // a dense (non-aliased) accumulation landed in a foreign storage
    std::shared_ptr<Tensor> grad_buffer; // dense [vocab, dim], persistent storage for the grad
    std::shared_ptr<Tensor> stamp; // kINT32 [vocab], per-row claim tags
    std::shared_ptr<Tensor> row_list; // kINT32 [capacity], deduplicated rows since the last clear
    std::shared_ptr<Tensor> count; // kINT32 [1], number of valid entries in row_list
};

class SparseRowGradRegistry {
public:
    static SparseRowGradRegistry &Instance() {
        static thread_local SparseRowGradRegistry instance;
        return instance;
    }

    SparseRowGradState *Lookup(const Tensor *weight) {
        auto it = states_.find(weight);
        if (it == states_.end()) {
            return nullptr;
        }
        if (it->second->weight_guard.lock().get() != weight) {
            states_.erase(it); // the address was recycled by a different tensor
            return nullptr;
        }
        return it->second.get();
    }

    SparseRowGradState *GetOrCreate(const std::shared_ptr<Tensor> &weight) {
        if (auto *state = Lookup(weight.get())) {
            return state;
        }
        auto state = std::make_unique<SparseRowGradState>();
        state->weight = weight.get();
        state->weight_guard = weight;
        state->vocab = weight->Dims()[0];
        state->dim = weight->Dims()[1];
        state->capacity = state->vocab;
        const auto device = weight->GetDevice();
        state->grad_buffer = std::make_shared<Tensor>(weight->Dims(), weight->Dtype(), device);
        state->stamp = std::make_shared<Tensor>(std::vector<int64_t>{state->vocab}, DataType::kINT32, device);
        state->row_list = std::make_shared<Tensor>(std::vector<int64_t>{state->capacity}, DataType::kINT32, device);
        state->count = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kINT32, device);
        auto *raw = state.get();
        states_[weight.get()] = std::move(state);
        return raw;
    }

    void Erase(const Tensor *weight) { states_.erase(weight); }

private:
    SparseRowGradRegistry() = default;
    std::unordered_map<const Tensor *, std::unique_ptr<SparseRowGradState>> states_;
};

// Free-function shim so call sites do not have to spell out the singleton.
inline SparseRowGradState *GetSparseRowGradState(const Tensor *weight) {
    return SparseRowGradRegistry::Instance().Lookup(weight);
}

// Zero the rows made dirty since the last clear, reset the row count and start a new claim
// generation. The caller must hold the device guard for the state's device. After this call the
// buffer is (again) all-zero, so nothing else needs to be touched to "zero the grad".
inline void ClearSparseRowGradRows(SparseRowGradState *state) {
    const auto device = state->grad_buffer->GetDevice();
    auto clear_kernel = Dispatcher::Instance().GetKernel({device.type(), "SparseRowClearRows"});
    clear_kernel.Call<void>(state->grad_buffer, state->row_list, state->count);
    auto reset_kernel = Dispatcher::Instance().GetKernel({device.type(), "SparseRowResetCount"});
    reset_kernel.Call<void>(state->count);
    ++state->generation;
}

} // namespace infini_train
