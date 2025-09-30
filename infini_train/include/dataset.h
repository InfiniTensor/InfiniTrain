#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "infini_train/include/tensor.h"

namespace infini_train {
class Dataset {
public:
    virtual ~Dataset() = default;

    virtual std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> operator[](size_t idx) const = 0;
    virtual size_t Size() const = 0;
};
} // namespace infini_train
