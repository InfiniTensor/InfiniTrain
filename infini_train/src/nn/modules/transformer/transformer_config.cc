#include "infini_train/include/nn/modules/transformer/transformer_config.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

namespace infini_train::nn {
bool TransformerConfig::UseGQA() const { return n_kv_head < n_head; }

int TransformerConfig::GetChunkSize() const {
    auto stage_info = parallel::PipelineParallel::GetStageInfo(n_layer, parallel::global::GetPipelineParallelSize(),
                                                               parallel::pp_rank,
                                                               parallel::global::GetVirtualPipelineParallelSize());
    return stage_info.layer_ranges_per_chunk.size();
}
} // namespace infini_train::nn
