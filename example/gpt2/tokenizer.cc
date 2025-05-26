#include "example/gpt2/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using TokenizerType = Tokenizer::TokenizerType;
using Version = Tokenizer::Version;

const std::unordered_map<int, TokenizerType> kTypeMap = {
    {20240328, TokenizerType::kUINT32}, // GPT-2
};

const std::unordered_map<TokenizerType, size_t> kTypeToSize = {
    {TokenizerType::kUINT16, 2},
    {TokenizerType::kUINT32, 4},
};

const std::unordered_map<TokenizerType, DataType> kTypeToDataType = {
    {TokenizerType::kUINT16, DataType::kUINT16},
    {TokenizerType::kUINT32, DataType::kINT32},
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File not found: " << filepath;
    }

    std::ifstream ifs(filepath, std::ios::binary);
    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);

    const uint32_t magic = BytesToType<uint32_t>(header, 0);
    const uint32_t version_num = BytesToType<uint32_t>(header, 4);
    vocab_size_ = BytesToType<uint32_t>(header, 8);
    if (kTypeMap.find(magic) == kTypeMap.end()) {
        LOG(FATAL) << "Unsupported tokenizer magic: " << magic;
    }

    Version version = static_cast<Version>(version_num);
    if (version == Version::kV1) {
        eot_token_ = kGpt2Eot;
    } else if (version == Version::kV2) {
        const uint32_t eot_token_2 = BytesToType<uint32_t>(header, 12);
        eot_token_ = eot_token_2;
    } else {
        LOG(FATAL) << "Unsupported tokenizer version: " << version_num;
        return;
    }

    token_table_.resize(vocab_size_);
    for (uint32_t i = 0; i < vocab_size_; ++i) {
        uint8_t length;
        ifs.read(reinterpret_cast<char *>(&length), sizeof(length));

        std::vector<char> buffer(length);
        ifs.read(buffer.data(), length);

        token_table_[i] = std::string(buffer.data(), length);
    }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    if (token_id >= vocab_size_) {
        return "[INVALID_TOKEN]";
    }
    return token_table_[token_id];
}

void Tokenizer::GenerateText(GPT2 &model, uint32_t batch_size, uint32_t sequence_length, uint32_t text_length,
                             Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = kGpt2Eot; }

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t kRngState = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = 1; t < text_length; t++) {
        x = std::make_shared<infini_train::Tensor>(x->To(device)); // CPU->calc device
        // TODO(jym): use no_grad forward later
        auto logits = model.Forward({x})[0];
        auto logits_orignal = nn::function::Softmax(logits, -1);
        auto logits_cpu = logits_orignal->To(Device());
        auto data = logits_cpu.DataPtr();
        auto vocab_size = logits->Dims()[2];
        float *probs = static_cast<float *>(data) + (t - 1) * vocab_size;
        float coin = RandomF32(kRngState);
        int next_token = SampleMult(probs, vocab_size, coin);

        x = std::make_shared<infini_train::Tensor>(x->To(Device())); // calc device->CPU
        auto data_temp = static_cast<int64_t *>(x->DataPtr());
        data_temp[t] = next_token;

        std::cout << Decode(next_token);
    }
    std::cout << std::endl;
}
} // namespace infini_train
