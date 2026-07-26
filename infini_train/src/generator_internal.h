#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "infini_train/include/device.h"

namespace infini_train {

class Generator;

class GeneratorImpl {
public:
    virtual ~GeneratorImpl() = default;

    virtual void ManualSeed(uint64_t seed) = 0;
    virtual uint64_t Seed() = 0;
    virtual uint64_t InitialSeed() const = 0;
    virtual std::vector<uint8_t> GetState() const = 0;
    virtual void SetState(const std::vector<uint8_t> &state) = 0;
    virtual Device GetDevice() const = 0;
};

namespace detail {

class GeneratorAccessor {
public:
    static std::pair<uint64_t, uint64_t> ReserveCUDARandomOffset(const std::shared_ptr<Generator> &generator,
                                                                 uint64_t increment);
    static void FillCPUUniform(const std::shared_ptr<Generator> &generator, float *data, size_t num_elements,
                               float from, float to);
    static void FillCPUNormal(const std::shared_ptr<Generator> &generator, float *data, size_t num_elements, float mean,
                              float stddev);
};

} // namespace detail
} // namespace infini_train
