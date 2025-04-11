#include "infini_train/include/kernels/cpu/embedding.h"

#include <memory>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> EmbeddingForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight) {
    /*
        x: [bs, seq_len]
        -> Embedding (weight: [num_embeddings, embedding_dim])
        -> o: [bs, seq_len, embed_dim]
    */
    CHECK_EQ(input->Dims().size(), 2);
    CHECK_EQ(weight->Dims().size(), 2);

    const int batch_size = input->Dims()[0];
    const int seq_len = input->Dims()[1];
    const int embed_dim = weight->Dims()[1];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, seq_len, embed_dim}, DataType::kFLOAT32);

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int ix = static_cast<int>(reinterpret_cast<const uint16_t *>(input->DataPtr())[b * seq_len + t]);
            for (int i = 0; i < embed_dim; i++) {
                reinterpret_cast<float *>(output->DataPtr())[b * seq_len * embed_dim + t * embed_dim + i]
                    = reinterpret_cast<float *>(weight->DataPtr())[ix * embed_dim + i];
            }
        }
    }

    return {output};
}

std::shared_ptr<Tensor> EmbeddingBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                          const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(input->Dims().size(), 2);
    CHECK_EQ(weight->Dims().size(), 2);

    const int batch_size = input->Dims()[0];
    const int seq_len = input->Dims()[0];
    const int embed_dim = weight->Dims()[1];

    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), DataType::kFLOAT32);
    grad_weight->Fill<float>(0.0f);

    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int ix = static_cast<int>(reinterpret_cast<const uint16_t *>(input->DataPtr())[b * seq_len + t]);
            for (int i = 0; i < embed_dim; i++) {
                reinterpret_cast<float *>(grad_weight->DataPtr())[ix * embed_dim + i]
                    += reinterpret_cast<float *>(grad_output->DataPtr())[b * seq_len * embed_dim + t * embed_dim + i];
            }
        }
    }
    return {grad_weight};
}
} // namespace infini_train::kernels::cpu
