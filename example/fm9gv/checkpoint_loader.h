#pragma once

#include <memory>
#include <string>

namespace infini_train::nn {
class TransformerModel;
} // namespace infini_train::nn

namespace fm9gv {
std::shared_ptr<infini_train::nn::TransformerModel> LoadFromFM9GBin(const std::string &filepath);
} // namespace fm9gv

