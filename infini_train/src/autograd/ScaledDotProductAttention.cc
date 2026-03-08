#include "infini_train/include/autograd/ScaledDotProductAttention.h"

#include <cmath>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/kernels/cuda/flash_attention.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
	CHECK_EQ(input_tensors.size(), 3);

	const auto &query = input_tensors[0];
	const auto &key = input_tensors[1];
	const auto &value = input_tensors[2];

	CHECK_GE(query->Dims().size(), 2);
	CHECK_GE(key->Dims().size(), 2);
	CHECK_GE(value->Dims().size(), 2);

	const int64_t query_last_dim = query->Dims().back();
	const float scale = static_cast<float>(
		scale_.has_value() ? *scale_ : 1.0 / std::sqrt(static_cast<double>(query_last_dim)));

	// Use FlashAttention kernel for CUDA device
	auto device = query->GetDevice().type();
	if (device == Device::Device::DeviceType::kCUDA) {
		// Note: is_causal and enable_gqa are supported in FlashAttention kernel
		// Dropout is now supported using seed-based approach for reproducibility
		flash_output_ = Dispatcher::Instance().Call<kernels::cuda::FlashAttentionForwardOutput>(
			{device, "FlashAttentionForward"}, query, key, value, attn_mask_, scale, is_causal_, dropout_p_, enable_gqa_);
		
		// Return output tensor only; saved_tensors_ will be set in SetupContext
		return {flash_output_.output};
	}

	// Fallback to traditional implementation for CPU or other devices
	CHECK_EQ(dropout_p_, 0) << "ScaledDotProductAttention dropout is not implemented yet";
	CHECK(!is_causal_) << "ScaledDotProductAttention causal mask is not implemented yet";
	CHECK(!enable_gqa_) << "ScaledDotProductAttention GQA mode is not implemented yet";

	const int64_t key_rank = static_cast<int64_t>(key->Dims().size());
	auto key_t = key->Transpose(key_rank - 2, key_rank - 1);

	auto attn_scores = query->Matmul(key_t);
	if (scale != 1.0f) {
		attn_scores = attn_scores * scale;
	}
	if (attn_mask_) {
		attn_scores = attn_scores + attn_mask_;
	}

	auto attn_prob = nn::function::Softmax(attn_scores, -1);
	auto output = attn_prob->Matmul(value);

	// Return output tensor only; saved_tensors_ will be set in SetupContext
	return {output};
}

void ScaledDotProductAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                            const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
	const auto &query = input_tensors[0];
	const auto &key = input_tensors[1];
	const auto &value = input_tensors[2];
	const auto &output = output_tensors[0];

	// Use FlashAttention kernel for CUDA device
	auto device = query->GetDevice().type();
	if (device == Device::Device::DeviceType::kCUDA) {
		// Save tensors for backward pass:
		// - query, key, value: input tensors
		// - output: forward output
		// - logsumexp: logsumexp tensor for backward pass (from flash_output_)
		// - dropout_seed: dropout seed tensor for backward pass (from flash_output_)
		saved_tensors_ = {query, key, value, output, flash_output_.logsumexp, flash_output_.dropout_seed};
	} else {
		// Fallback to traditional implementation for CPU or other devices
		// Recompute attn_prob from output and value
		const int64_t value_rank = static_cast<int64_t>(value->Dims().size());
		auto value_t = value->Transpose(value_rank - 2, value_rank - 1);
		auto attn_prob = output->Matmul(value_t);
		saved_tensors_ = {query, key, value, attn_prob};
	}
}

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
	CHECK_GE(saved_tensors_.size(), 4);
	CHECK_EQ(grad_outputs.size(), 1);

	const auto &query = saved_tensors_[0];
	const auto &key = saved_tensors_[1];
	const auto &value = saved_tensors_[2];
	const auto &output = saved_tensors_[3];
	const auto &grad_output = grad_outputs[0];

	// Use FlashAttention backward kernel for CUDA device
	auto device = query->GetDevice().type();
	if (device == Device::Device::DeviceType::kCUDA) {
		CHECK_EQ(saved_tensors_.size(), 6) << "FlashAttention backward expects 6 saved tensors";
		const auto &logsumexp = saved_tensors_[4];
		const auto &dropout_seed = saved_tensors_[5];
		
		const float scale = static_cast<float>(
			scale_.has_value() ? *scale_ : 1.0 / std::sqrt(static_cast<double>(query->Dims().back())));
		
		auto grad_tensors = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
			{device, "FlashAttentionBackward"}, query, key, value, output, grad_output, logsumexp, dropout_seed, attn_mask_, 
			scale, is_causal_, dropout_p_, enable_gqa_);
		
		return grad_tensors;
	}

	// Fallback to traditional implementation for CPU or other devices
	CHECK_EQ(saved_tensors_.size(), 4) << "Traditional backward expects 4 saved tensors";
	const auto &attn_prob = output;
	const int64_t value_rank = static_cast<int64_t>(value->Dims().size());
	const int64_t key_rank = static_cast<int64_t>(key->Dims().size());
	const int64_t attn_rank = static_cast<int64_t>(attn_prob->Dims().size());

	auto value_t = value->Transpose(value_rank - 2, value_rank - 1);
	auto grad_attn_prob = grad_output->Matmul(value_t);

	// softmax backward: grad = (g - sum(g * y, dim=-1, keepdim=true)) * y
	auto grad_dot = (grad_attn_prob * attn_prob)->Sum(-1, true);
	auto grad_attn_scores = (grad_attn_prob - grad_dot) * attn_prob;

	const float scale = static_cast<float>(
		scale_.has_value() ? *scale_ : 1.0 / std::sqrt(static_cast<double>(query->Dims().back())));
	if (scale != 1.0f) {
		grad_attn_scores = grad_attn_scores * scale;
	}

	auto grad_query = grad_attn_scores->Matmul(key);

	auto grad_attn_scores_t = grad_attn_scores->Transpose(attn_rank - 2, attn_rank - 1);
	auto grad_key = grad_attn_scores_t->Matmul(query);

	auto attn_prob_t = attn_prob->Transpose(attn_rank - 2, attn_rank - 1);
	auto grad_value = attn_prob_t->Matmul(grad_output);

	return {grad_query, grad_key, grad_value};
}
} // namespace infini_train::autograd
