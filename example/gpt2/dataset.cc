#include "example/gpt2/dataset.h"

#include <cstddef>
#include <cstdlib>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
};

constexpr char kTrainPostfix[] = "train";
constexpr char kTestPostfix[] = "val";

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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, const size_t &sequence_length) {
    /*
      HEADER (1024 bytes)                         | DATA (tokens) |
      magic    | version  | num_toks | reserved   |
      4 bytes  | 4 bytes  | 4 bytes  | 1012 bytes | # bytes       |
    */
    TinyShakespeareFile text_file;
    std::ifstream ifs(path, std::ios::binary);
    const auto header = ReadSeveralBytesFromIfstream(1024, &ifs);
    const int magic = BytesToType<int32_t>(header, 0);
    const int version = BytesToType<int32_t>(header, 4);
    const int num_tokens = BytesToType<int32_t>(header, 8);
    text_file.type = kTypeMap.at(magic);

    const int num_sequences = num_tokens / sequence_length;
    text_file.dims.assign({num_sequences, sequence_length});

    const int data_size_in_bytes
        = kTypeToSize.at(text_file.type)
        * std::accumulate(text_file.dims.begin(), text_file.dims.end(), 1, std::multiplies<int>());
    text_file.tensor = infini_train::Tensor(text_file.dims, kTypeToDataType.at(text_file.type));
    ifs.read(reinterpret_cast<char *>(text_file.tensor.DataPtr()), data_size_in_bytes);
    return text_file;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &dataset, const size_t &sequence_length, bool train)
    : text_file_(ReadTinyShakespeareFile(
        std::format("{}/tiny_shakespeare_{}.bin", dataset, train ? kTrainPostfix : kTestPostfix), sequence_length)),
      sequence_length_(sequence_length), sequence_size_in_bytes_(sequence_length * kTypeToSize.at(text_file_.type)),
      num_samples_(text_file_.dims[0] - 1) {
    CHECK_LE(sequence_length, 1024); // GPT-2: max_seq_length = 1024
    CHECK_EQ(text_file_.dims[1], sequence_length_);
    CHECK_EQ(static_cast<int>(text_file_.tensor.Dtype()), static_cast<int>(DataType::kUINT16));
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, std::move(dims)),
            std::make_shared<infini_train::Tensor>(
                text_file_.tensor, idx * sequence_size_in_bytes_ + kTypeToSize.at(text_file_.type), std::move(dims))};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
