#include "infini_train/include/nn/modules/transformer/transformer_config.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

namespace infini_train::nn {
bool TransformerConfig::UseGQA() const { return n_kv_head < n_head; }

int TransformerConfig::GetChunkSize() const {
    return parallel::global::GetPipelineLayout().stage(parallel::pp_rank).global_chunk_ids.size();
}
} // namespace infini_train::nn
