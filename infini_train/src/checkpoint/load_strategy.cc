#include "infini_train/include/checkpoint/load_strategy.h"

#include <fstream>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/utils/string_utils.h"

namespace infini_train::checkpoint {
namespace {

using FileCache = std::unordered_map<std::string, std::unique_ptr<std::ifstream>>;

std::ifstream &GetFile(FileCache &cache, const std::filesystem::path &checkpoint_dir, const std::string &filename) {
    auto it = cache.find(filename);
    if (it == cache.end()) {
        auto stream = std::make_unique<std::ifstream>(checkpoint_dir / filename, std::ios::binary);
        CHECK(stream->is_open()) << "Failed to open checkpoint file: " << checkpoint_dir / filename;
        it = cache.emplace(filename, std::move(stream)).first;
    }
    return *it->second;
}

void ReadAt(std::ifstream &stream, uint64_t offset, void *destination, uint64_t byte_size, const std::string &key,
            const std::string &filename) {
    stream.clear();
    stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    CHECK(stream.good()) << "Failed to seek tensor " << key << " in " << filename;
    stream.read(static_cast<char *>(destination), static_cast<std::streamsize>(byte_size));
    CHECK_EQ(static_cast<uint64_t>(stream.gcount()), byte_size) << "Truncated tensor " << key << " in " << filename;
}

std::shared_ptr<Tensor> ReadTensor(std::ifstream &stream, const ReadItem &read) {
    auto tensor = std::make_shared<Tensor>(read.source_shape, read.dtype, Device());
    CHECK_EQ(read.byte_size, tensor->SizeInBytes())
        << "Tensor byte size mismatch for " << read.key << " in " << read.filename;
    ReadAt(stream, read.data_offset, tensor->DataPtr(), read.byte_size, read.key, read.filename);
    return tensor;
}

std::shared_ptr<Tensor> ReadTensorRegion(std::ifstream &stream, const ReadItem &read) {
    CHECK_GE(read.shard_dim, 0);
    CHECK_LT(read.shard_dim, static_cast<int>(read.source_shape.size()));
    CHECK_GE(read.source_offset, 0);
    CHECK_GT(read.length, 0);
    CHECK_LE(read.source_offset + read.length, read.source_shape[read.shard_dim]);

    uint64_t source_numel = 1;
    for (const auto size : read.source_shape) {
        CHECK_GT(size, 0);
        source_numel *= static_cast<uint64_t>(size);
    }
    CHECK_EQ(read.byte_size % source_numel, 0u)
        << "Invalid tensor byte size for " << read.key << " in " << read.filename;
    const uint64_t element_size = read.byte_size / source_numel;

    uint64_t outer = 1;
    for (int dim = 0; dim < read.shard_dim; ++dim) { outer *= static_cast<uint64_t>(read.source_shape[dim]); }
    uint64_t inner = 1;
    for (size_t dim = static_cast<size_t>(read.shard_dim + 1); dim < read.source_shape.size(); ++dim) {
        inner *= static_cast<uint64_t>(read.source_shape[dim]);
    }

    auto region_shape = read.source_shape;
    region_shape[read.shard_dim] = read.length;
    auto tensor = std::make_shared<Tensor>(region_shape, read.dtype, Device());
    const uint64_t block_bytes = static_cast<uint64_t>(read.length) * inner * element_size;
    const uint64_t source_stride_bytes
        = static_cast<uint64_t>(read.source_shape[read.shard_dim]) * inner * element_size;
    const uint64_t first_block_offset
        = read.data_offset + static_cast<uint64_t>(read.source_offset) * inner * element_size;
    CHECK_EQ(outer * block_bytes, tensor->SizeInBytes());

    auto *destination = static_cast<char *>(tensor->DataPtr());
    for (uint64_t block = 0; block < outer; ++block) {
        ReadAt(stream, first_block_offset + block * source_stride_bytes, destination + block * block_bytes, block_bytes,
               read.key, read.filename);
    }
    return tensor;
}

} // namespace

LoadedStateDict IndexedRegionLoadStrategy::Execute(const std::filesystem::path &checkpoint_dir, const LoadPlan &plan) {
    FileCache file_cache;
    LoadedStateDict result;

    for (const auto &[key, tensor_plan] : plan.tensors) {
        CHECK(!tensor_plan.reads.empty()
              || (tensor_plan.shard_dim >= 0
                  && tensor_plan.trailing_zero_fill == tensor_plan.target_shape[tensor_plan.shard_dim]))
            << "No reads or padding planned for target tensor: " << key;
        std::vector<std::shared_ptr<Tensor>> pieces;
        pieces.reserve(tensor_plan.reads.size() + (tensor_plan.trailing_zero_fill > 0 ? 1 : 0));

        for (const auto &read : tensor_plan.reads) {
            CHECK_GT(read.data_offset, 0) << "Checkpoint metadata lacks a valid tensor data offset for " << key
                                          << "; regenerate the checkpoint with the current format";
            auto &stream = GetFile(file_cache, checkpoint_dir, read.filename);
            pieces.push_back(read.shard_dim < 0 ? ReadTensor(stream, read) : ReadTensorRegion(stream, read));
        }

        if (tensor_plan.trailing_zero_fill > 0) {
            CHECK_GE(tensor_plan.shard_dim, 0);
            auto padding_shape = tensor_plan.target_shape;
            padding_shape[tensor_plan.shard_dim] = tensor_plan.trailing_zero_fill;
            auto padding = std::make_shared<Tensor>(padding_shape, tensor_plan.dtype, Device());
            padding->Fill(0.0f);
            pieces.push_back(std::move(padding));
        }

        auto target = pieces.front();
        if (pieces.size() > 1) {
            CHECK_GE(tensor_plan.shard_dim, 0);
            target = nn::function::Concat(pieces, tensor_plan.shard_dim)->Contiguous();
        }
        CHECK(target->Dims() == tensor_plan.target_shape)
            << "Target shard shape mismatch for " << key
            << ": expected=" << utils::DimsToString(tensor_plan.target_shape)
            << ", got=" << utils::DimsToString(target->Dims());
        result.emplace(key, std::move(target));
    }
    return result;
}

} // namespace infini_train::checkpoint
