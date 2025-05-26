#include <cctype>
#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"

#include "example/gpt2/dataset.h"
#include "example/gpt2/net.h"

namespace infini_train {

class Tokenizer {
public:
    enum class TokenizerType : int {
        kUINT16,
        kUINT32,
        kINVALID,
    };

    enum class Version : uint32_t {
        kV1 = 1,
        kV2 = 2,
    };

    Tokenizer(const std::string &filepath);

    std::string Decode(uint32_t token_id) const;

    void GenerateText(GPT2 &model, uint32_t batch_size, uint32_t sequence_length, uint32_t text_length,
                      Device device) const;

    uint32_t GetEndToken() const { return eot_token_; };

private:
    uint32_t vocab_size_ = 0;
    std::vector<std::string> token_table_;
    uint32_t eot_token_ = 0;
};
} // namespace infini_train
