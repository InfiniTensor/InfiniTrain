#include <cctype>
#include <cstdint>
#include <vector>

#include "infini_train/include/device.h"

namespace infini_train {
namespace nn {
class Module;
}
class Tokenizer {
public:
    enum class Version : uint32_t {
        kV1 = 1,
        kV2 = 2,
    };

    Tokenizer(const std::string &filepath);

    std::string Decode(uint32_t token_id) const;

    void GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                      uint32_t text_length, Device device) const;

    uint32_t GetEndToken() const { return eot_token_; };

private:
    uint32_t magic_number_ = 0;
    uint32_t vocab_size_ = 0;
    std::vector<std::string> token_table_;
    uint32_t eot_token_ = 0;
};
} // namespace infini_train
