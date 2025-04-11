#include "infini_train/include/kernels/cpu/fused_embedding.h"

#include <memory>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> FusedEmbeddingForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &wte,
                                         const std::shared_ptr<Tensor> &wpe) {
    /*
        x: [bs, seq_len]
        -> FusedEmbedding (wte: [vocab_size, embed_dim], wpe: [max_position, embed_dim])
        -> o: [bs, seq_len, embed_dim]
    */
    CHECK_EQ(input->Dims().size(), 2);
    CHECK_EQ(wte->Dims().size(), 2);
    CHECK_EQ(wpe->Dims().size(), 2);
    CHECK_LE(input->Dims()[1], wpe->Dims()[0]);
    CHECK_EQ(wte->Dims()[1], wpe->Dims()[1]);

    const int batch_size = input->Dims()[0];
    const int max_seqlen = wpe->Dims()[0];
    const int embed_dim = wpe->Dims()[1];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, max_seqlen, embed_dim}, DataType::kFLOAT32);

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < max_seqlen; t++) {
            int ix = static_cast<int>(reinterpret_cast<const uint16_t *>(input->DataPtr())[b * max_seqlen + t]);
            for (int i = 0; i < embed_dim; i++) {
                reinterpret_cast<float *>(output->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i]
                    = reinterpret_cast<float *>(wte->DataPtr())[ix * embed_dim + i]
                    + reinterpret_cast<float *>(wpe->DataPtr())[t * embed_dim + i];
            }
        }
    }

    return {output};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
FusedEmbeddingBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &wte,
                  const std::shared_ptr<Tensor> &wpe, const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(input->Dims().size(), 2);
    CHECK_EQ(wte->Dims().size(), 2);
    CHECK_EQ(wpe->Dims().size(), 2);
    CHECK_LE(input->Dims()[1], wpe->Dims()[0]);
    CHECK_EQ(wte->Dims()[1], wpe->Dims()[1]);

    const int batch_size = input->Dims()[0];
    const int max_seqlen = wpe->Dims()[0];
    const int embed_dim = wpe->Dims()[1];

    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    auto grad_wte = std::make_shared<Tensor>(wte->Dims(), DataType::kFLOAT32);
    auto grad_wpe = std::make_shared<Tensor>(wpe->Dims(), DataType::kFLOAT32);
    grad_wte->Fill<float>(0.0f);
    grad_wpe->Fill<float>(0.0f);

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < max_seqlen; t++) {
            int ix = static_cast<int>(reinterpret_cast<const uint16_t *>(input->DataPtr())[b * max_seqlen + t]);
            for (int i = 0; i < embed_dim; i++) {
                float d
                    = reinterpret_cast<float *>(grad_output->DataPtr())[b * max_seqlen * embed_dim + t * embed_dim + i];
                reinterpret_cast<float *>(grad_wte->DataPtr())[ix * embed_dim + i] += d;
                reinterpret_cast<float *>(grad_wpe->DataPtr())[t * embed_dim + i] += d;
            }
        }
    }
    return {grad_input, grad_wte, grad_wpe};
}
} // namespace infini_train::kernels::cpu
