#pragma once

#include <vector>

namespace infini_train {

class HookHandle {
public:
    virtual ~HookHandle() = default;
    virtual void Remove() = 0;
};

template <typename HookType> class HookHandleImpl : public HookHandle {
public:
    HookHandleImpl(std::vector<HookType> *hooks, size_t id) : hooks_(hooks), id_(id) {}

    void Remove() override {
        if (!removed_ && hooks_ && id_ < hooks_->size()) {
            (*hooks_)[id_] = nullptr;
            removed_ = true;
        }
    }

private:
    std::vector<HookType> *hooks_;
    size_t id_;
    bool removed_ = false;
};

} // namespace infini_train
