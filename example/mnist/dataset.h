#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "infini_train/include/dataset.h"
#include "infini_train/include/tensor.h"

class MNISTDataset : public infini_train::Dataset {
public:
    enum class SN3PascalVincentType : int {
        kUINT8,
        kINT8,
        kINT16,
        kINT32,
        kFLOAT32,
        kFLOAT64,
        kINVALID,
    };

    struct SN3PascalVincentFile {
        SN3PascalVincentType type = SN3PascalVincentType::kINVALID;
        std::vector<int64_t> dims;
        infini_train::Tensor tensor;
    };

    MNISTDataset(const std::string &prefix, bool train);

    std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
    operator[](size_t idx) const override;

    size_t Size() const override;

private:
    SN3PascalVincentFile image_file_;
    SN3PascalVincentFile label_file_;
    std::vector<int64_t> image_dims_;
    std::vector<int64_t> label_dims_;
    const size_t image_size_in_bytes_ = 0;
    const size_t label_size_in_bytes_ = 0;
};
