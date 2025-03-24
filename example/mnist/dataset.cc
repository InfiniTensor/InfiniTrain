#include "example/mnist/dataset.h"

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
using SN3PascalVincentType = MNISTDataset::SN3PascalVincentType;
using SN3PascalVincentFile = MNISTDataset::SN3PascalVincentFile;

const std::unordered_map<int, SN3PascalVincentType> kTypeMap = {
    {8, SN3PascalVincentType::kUINT8},  {9, SN3PascalVincentType::kINT8},     {11, SN3PascalVincentType::kINT16},
    {12, SN3PascalVincentType::kINT32}, {13, SN3PascalVincentType::kFLOAT32}, {14, SN3PascalVincentType::kFLOAT64},
};

const std::unordered_map<SN3PascalVincentType, size_t> kSN3TypeToSize = {
    {SN3PascalVincentType::kUINT8, 1}, {SN3PascalVincentType::kINT8, 1},    {SN3PascalVincentType::kINT16, 2},
    {SN3PascalVincentType::kINT32, 4}, {SN3PascalVincentType::kFLOAT32, 4}, {SN3PascalVincentType::kFLOAT64, 8},
};

const std::unordered_map<SN3PascalVincentType, DataType> kSN3TypeToDataType = {
    {SN3PascalVincentType::kUINT8, DataType::kUINT8},     {SN3PascalVincentType::kINT8, DataType::kINT8},
    {SN3PascalVincentType::kINT16, DataType::kINT16},     {SN3PascalVincentType::kINT32, DataType::kINT32},
    {SN3PascalVincentType::kFLOAT32, DataType::kFLOAT32}, {SN3PascalVincentType::kFLOAT64, DataType::kFLOAT64},
};

constexpr char kTrainPrefix[] = "train";
constexpr char kTestPrefix[] = "t10k";

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

SN3PascalVincentFile ReadSN3PascalVincentFile(const std::string &path) {
    /*
      magic                                     | dims               | data    |
      reserved | reserved | type_int | num_dims |
      1 byte   | 1 byte   | 1 byte   | 1 byte   | 4*{num_dims} bytes | # bytes |
    */
    SN3PascalVincentFile sn3_file;
    std::ifstream ifs(path, std::ios::binary);
    const auto magic = ReadSeveralBytesFromIfstream(4, &ifs);
    const int num_dims = magic[3];
    const int type_int = magic[2];
    sn3_file.type = kTypeMap.at(type_int);

    auto &dims = sn3_file.dims;
    dims.resize(num_dims, 0);
    for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx) {
        for (const auto &v : ReadSeveralBytesFromIfstream(4, &ifs)) {
            dims[dim_idx] <<= 8;
            dims[dim_idx] += v;
        }
    }
    const int data_size_in_bytes
        = kSN3TypeToSize.at(sn3_file.type) * std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    sn3_file.tensor = infini_train::Tensor(dims, kSN3TypeToDataType.at(sn3_file.type));
    ifs.read(reinterpret_cast<char *>(sn3_file.tensor.DataPtr()), data_size_in_bytes);
    return sn3_file;
}
} // namespace

MNISTDataset::MNISTDataset(const std::string &dataset, bool train)
    : image_file_(
        ReadSN3PascalVincentFile(std::format("{}/{}-images-idx3-ubyte", dataset, train ? kTrainPrefix : kTestPrefix))),
      label_file_(ReadSN3PascalVincentFile(
          std::format("{}/{}-labels-idx1-ubyte", dataset, train ? kTrainPrefix : kTestPrefix))),
      image_dims_(image_file_.dims.begin() + 1, image_file_.dims.end()),
      label_dims_(label_file_.dims.begin() + 1, label_file_.dims.end()),
      image_size_in_bytes_(kSN3TypeToSize.at(image_file_.type)
                           * std::accumulate(image_dims_.begin(), image_dims_.end(), 1, std::multiplies<int>())),
      label_size_in_bytes_(kSN3TypeToSize.at(label_file_.type)
                           * std::accumulate(label_dims_.begin(), label_dims_.end(), 1, std::multiplies<int>())) {
    CHECK_EQ(image_file_.dims[0], label_file_.dims[0]);
    CHECK_EQ(static_cast<int>(image_file_.tensor.Dtype()), static_cast<int>(DataType::kUINT8));
    const auto &image_dims = image_file_.tensor.Dims();
    CHECK_EQ(image_dims.size(), 3);
    CHECK_EQ(image_dims[1], 28);
    CHECK_EQ(image_dims[2], 28);

    const auto bs = image_dims[0];
    infini_train::Tensor transposed_tensor(image_dims, DataType::kFLOAT32);
    for (int idx = 0; idx < bs; ++idx) {
        const auto *image_data = reinterpret_cast<uint8_t *>(image_file_.tensor.DataPtr()) + idx * 28 * 28;
        auto *transposed_data = reinterpret_cast<float *>(transposed_tensor.DataPtr()) + idx * 28 * 28;
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) { transposed_data[i * 28 + j] = image_data[i * 28 + j] / 255.0f; }
        }
    }
    image_file_.tensor = std::move(transposed_tensor);
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
MNISTDataset::operator[](size_t idx) const {
    CHECK_LT(idx, image_file_.dims[0]);
    return {std::make_shared<infini_train::Tensor>(image_file_.tensor, idx * image_size_in_bytes_, image_dims_),
            std::make_shared<infini_train::Tensor>(label_file_.tensor, idx * label_size_in_bytes_, label_dims_)};
}

size_t MNISTDataset::Size() const { return image_file_.dims[0]; }
