#include "infini_train/src/nn/parallel/pp/send_recv.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::pipeline {
void Isend(const std::vector<std::shared_ptr<Tensor>> &inputs, int destinations) {
    CHECK_GT(inputs.size(), 0);
    auto device = inputs[0]->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "CommNcclSend"});
    kernel.Call<void, std::vector<std::shared_ptr<Tensor>>>(inputs, destinations);
}

void Irecv(const std::vector<std::shared_ptr<Tensor>> &inputs, int source) {
    CHECK_GT(inputs.size(), 0);
    auto device = inputs[0]->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "CommNcclRecv"});
    kernel.Call<void, std::vector<std::shared_ptr<Tensor>>>(inputs, source);
}
}