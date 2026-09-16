#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train::nn {
bool TransformerConfig::UseGQA() const { return n_kv_head < n_head; }
} // namespace infini_train::nn
