#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "infini_train/include/dataset.h"
#include "infini_train/include/tensor.h"

class TinyShakespeareDataset : public infini_train::Dataset {
    /*
        Dataset bin file is downloaded and processed using the script at
        https://github.com/karpathy/llm.c/blob/master/dev/data/tinyshakespeare.py
    */
public:
    enum class TinyShakespeareType : int {
        kUINT16, // For GPT-2
        kUINT32, // For LLaMA 3
        kINVALID,
    };

    struct TinyShakespeareFile {
        TinyShakespeareType type = TinyShakespeareType::kINVALID;
        std::vector<int64_t> dims;
        infini_train::Tensor tensor;
    };

    TinyShakespeareDataset(const std::string &filepath, size_t sequence_length);

    std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
    operator[](size_t idx) const override;

    size_t Size() const override;

private:
    TinyShakespeareFile text_file_;
    const size_t sequence_length_ = 0;
    const size_t sequence_size_in_bytes_ = 0;
    const size_t num_samples_ = 0;
};
