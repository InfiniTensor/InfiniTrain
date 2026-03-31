#include "infini_train/include/core/models/decode_only_transformer/model.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "glog/logging.h"

using namespace infini_train;

namespace {
constexpr int kRandomSeed = 42;

// TODO(dcj): make this rng generator compatible with torch later
static std::mt19937 gen{kRandomSeed};
} // namespace

std::shared_ptr<DecoderOnlyTransformer> DecoderOnlyTransformer::FromPretrained(ModelType model_type) {
    // TODO(dcj): implement this later
    LOG(FATAL) << "Not implemented yet";
    return nullptr;
}

int DecoderOnlyTransformer::GetChunkSize() const { return stage_info_.layer_ranges_per_chunk.size(); }
