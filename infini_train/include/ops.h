#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::ops {
class Op {
public:
    virtual ~Op() = default;

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors);
    // TODO(dcj): Check the validity of the output_tensor
    void Backward(const Tensor *output_tensor);

    virtual std::vector<std::shared_ptr<Tensor>> ForwardImpl() = 0;
    virtual void BackwardImpl() = 0;

protected:
    std::vector<std::shared_ptr<Tensor>> input_tensors_;
    std::vector<std::shared_ptr<Tensor>> output_tensors_;
    int64_t backward_reached_ = 0; // record the number of tensors this op produced call to backward
};

/*
  x: [bs, in_dim]
  -> Linear (w: [in_dim, out_dim], b: [out_dim])
  -> o: [bs, out_dim]
*/
class Linear : public Op {
public:
    Linear(Tensor *weight, Tensor *bias);

    std::vector<std::shared_ptr<Tensor>> ForwardImpl() override;
    void BackwardImpl() override;

protected:
    int64_t in_dim_ = 0;
    int64_t out_dim_ = 0;
    Tensor *w_ = nullptr; // not owned
    Tensor *b_ = nullptr; // not owned
};

class CUDALinear : public Linear {
public:
    CUDALinear(Tensor *weight, Tensor *bias);

    std::vector<std::shared_ptr<Tensor>> ForwardImpl() override;
    void BackwardImpl() override;
};

/*
  input: [bs, out_dim]
  -> Sigmoid ()
  -> output: [bs, out_dim]
 */
class Sigmoid : public Op {
public:
    Sigmoid() = default;

    std::vector<std::shared_ptr<Tensor>> ForwardImpl() override;
    void BackwardImpl() override;
};

class CUDASigmoid : public Sigmoid {
public:
    std::vector<std::shared_ptr<Tensor>> ForwardImpl() override;
    void BackwardImpl() override;
};

/*
  -> input: [bs, out_dim]
  -> CrossEntropy ()
  -> target: [bs,]
*/
class CrossEntropy : public Op {
public:
    CrossEntropy() = default;

    std::vector<std::shared_ptr<Tensor>> ForwardImpl() override;
    void BackwardImpl() override;
};

class CUDACrossEntropy : public CrossEntropy {
public:
    std::vector<std::shared_ptr<Tensor>> ForwardImpl() override;
    void BackwardImpl() override;
};
} // namespace infini_train::ops
