#include "infini_train/include/kernels/cpu/embedding.h"

#include <memory>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> EmbeddingForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight) {
    /*
        x: [*]
        -> Embedding (weight: [num_embeddings, embedding_dim])
        -> o: [*, embedding_dim]
    */
    const auto &input_dims = input->Dims();
    CHECK_EQ(weight->Dims().size(), 2);
    const int embedding_dim = weight->Dims()[1];
    auto output_dims = input_dims;
    output_dims.push_back(embedding_dim);
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    for (int i = 0; i < input->NumElements(); i++) {
        int idx = static_cast<int>(reinterpret_cast<const uint16_t *>(input->DataPtr())[i]);
        for (int j = 0; j < embedding_dim; j++) {
            reinterpret_cast<float *>(output->DataPtr())[i * embedding_dim + j]
                = reinterpret_cast<float *>(weight->DataPtr())[idx * embedding_dim + j];
        }
    }

    return output;
}

std::shared_ptr<Tensor> EmbeddingBackward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &weight_dims,
                                          const std::shared_ptr<Tensor> &grad_output) {
    CHECK_EQ(weight_dims.size(), 2);
    const int embedding_dim = weight_dims[1];
    CHECK_EQ(input->Dims().size() + 1, grad_output->Dims().size());
    for (int idx = 0; idx < input->Dims().size(); ++idx) { CHECK_EQ(input->Dims()[idx], grad_output->Dims()[idx]); }
    CHECK_EQ(*grad_output->Dims().rbegin(), embedding_dim);

    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    grad_weight->Fill<float>(0.0f);

    for (int i = 0; i < input->NumElements(); i++) {
        int idx = static_cast<int>(reinterpret_cast<const uint16_t *>(input->DataPtr())[i]);
        for (int j = 0; j < embedding_dim; j++) {
            reinterpret_cast<float *>(grad_weight->DataPtr())[idx * embedding_dim + i]
                += reinterpret_cast<float *>(grad_output->DataPtr())[i * embedding_dim + j];
        }
    }
    return grad_weight;
}
} // namespace infini_train::kernels::cpu
