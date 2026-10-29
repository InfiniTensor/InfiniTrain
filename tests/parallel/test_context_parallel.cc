#include <stddef.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/context_parallel.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

#if defined(USE_CUDA) && defined(USE_NCCL)
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <thread>

#include <cuda_runtime_api.h>

#include "infini_train/include/autocast.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/utils.h"
#endif

namespace infini_train::test {
namespace {
namespace parallel = nn::parallel;
namespace global = parallel::global;

std::vector<int64_t> LocalPositions(int rank, int64_t sequence_length) {
  const int cp_size = global::GetContextParallelSize();
  const int64_t local = sequence_length / cp_size;
  std::vector<int64_t> positions(local);
  for (int64_t i = 0; i < local; ++i) {
    positions[i] = rank * local + i;
    if (cp_size > 1 && global::GetAttentionCPLayout() == "zigzag") {
      positions[i] =
          i < local / 2
              ? rank * (local / 2) + i
              : sequence_length - (rank + 1) * (local / 2) + i - local / 2;
    }
  }
  return positions;
}

TEST(ContextParallelLayoutTest, SlicesAndGradientsFollowGlobalPositions) {
  const int cp_size = global::GetContextParallelSize();
  const int64_t sequence = 8 * cp_size;
  std::vector<int> coverage(sequence, 0);
  std::vector<float> values(2 * sequence * 3);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i);
  }
  for (int rank = 0; rank < cp_size; ++rank) {
    parallel::cp_rank = rank;
    auto input = std::make_shared<Tensor>(values.data(),
                                          std::vector<int64_t>{2, sequence, 3},
                                          DataType::kFLOAT32, Device());
    input->set_requires_grad(true);
    auto output = parallel::SliceAlongCPRegionFunc(input, -2);
    if (cp_size == 1) {
      EXPECT_EQ(output.get(), input.get());
    }
    auto positions = LocalPositions(rank, sequence);
    EXPECT_EQ(output->Dims(), (std::vector<int64_t>{2, 8, 3}));
    const auto *actual = static_cast<const float *>(output->DataPtr());
    std::vector<float> expected_grad(values.size(), 0.0f);
    std::vector<float> upstream(output->NumElements());
    for (int64_t b = 0; b < 2; ++b) {
      for (int64_t i = 0; i < 8; ++i) {
        for (int64_t d = 0; d < 3; ++d) {
          const auto local_index = (b * 8 + i) * 3 + d;
          const auto global_index = (b * sequence + positions[i]) * 3 + d;
          EXPECT_FLOAT_EQ(actual[local_index], values[global_index]);
          upstream[local_index] = static_cast<float>(local_index + 1);
          expected_grad[global_index] = upstream[local_index];
        }
      }
    }
    for (const auto pos : positions) {
      ++coverage[pos];
    }
    (output * 1.0f)
        ->Backward(std::make_shared<Tensor>(upstream.data(), output->Dims(),
                                            DataType::kFLOAT32, Device()));
    ASSERT_NE(input->grad(), nullptr);
    const auto *grad = static_cast<const float *>(input->grad()->DataPtr());
    for (size_t i = 0; i < expected_grad.size(); ++i) {
      EXPECT_FLOAT_EQ(grad[i], expected_grad[i]);
    }
  }
  parallel::cp_rank = 0;
  for (const auto count : coverage) {
    EXPECT_EQ(count, 1);
  }
}

TEST(ContextParallelLayoutTest, ZigzagRequiresTwoEqualChunksPerRank) {
  const auto cp_size = global::GetContextParallelSize();
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{3 * cp_size},
                                        DataType::kFLOAT32, Device());
  parallel::cp_rank = 0;
  if (cp_size > 1 && global::GetAttentionCPLayout() == "zigzag") {
    EXPECT_DEATH(parallel::SliceAlongCPRegionFunc(input, 0), "2 \\* CP");
  } else {
    EXPECT_EQ(parallel::SliceAlongCPRegionFunc(input, 0)->Dims()[0], 3);
  }
}

#if defined(USE_CUDA) && defined(USE_NCCL)
struct AttentionReference {
  std::vector<double> output;
  std::vector<double> dq;
  std::vector<double> dk;
  std::vector<double> dv;
};

// Independent dense causal reference. All reductions and the softmax derivative
// are computed in double precision; no CP layout or framework attention is
// used.
AttentionReference DenseReference(const std::vector<float> &q,
                                  const std::vector<float> &k,
                                  const std::vector<float> &v,
                                  const std::vector<float> &dout, int batch,
                                  int heads, int kv_heads, int sequence,
                                  int dim) {
  AttentionReference ref{
      std::vector<double>(q.size()), std::vector<double>(q.size()),
      std::vector<double>(k.size()), std::vector<double>(v.size())};
  const double scale = 1.0 / std::sqrt(static_cast<double>(dim));
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < heads; ++h) {
      const int kh = h / (heads / kv_heads);
      for (int row = 0; row < sequence; ++row) {
        const int qi = ((b * heads + h) * sequence + row) * dim;
        std::vector<double> probs(row + 1), dp(row + 1);
        double max_score = -std::numeric_limits<double>::infinity();
        for (int col = 0; col <= row; ++col) {
          const int ki = ((b * kv_heads + kh) * sequence + col) * dim;
          for (int d = 0; d < dim; ++d) {
            probs[col] += static_cast<double>(q[qi + d]) * k[ki + d] * scale;
            dp[col] += static_cast<double>(dout[qi + d]) * v[ki + d];
          }
          max_score = std::max(max_score, probs[col]);
        }
        double sum = 0.0;
        for (auto &p : probs) {
          p = std::exp(p - max_score);
          sum += p;
        }
        double delta = 0.0;
        for (int col = 0; col <= row; ++col) {
          probs[col] /= sum;
          delta += probs[col] * dp[col];
        }
        for (int col = 0; col <= row; ++col) {
          const int ki = ((b * kv_heads + kh) * sequence + col) * dim;
          const double ds = probs[col] * (dp[col] - delta) * scale;
          for (int d = 0; d < dim; ++d) {
            ref.output[qi + d] += probs[col] * v[ki + d];
            ref.dq[qi + d] += ds * k[ki + d];
            ref.dk[ki + d] += ds * q[qi + d];
            ref.dv[ki + d] += probs[col] * dout[qi + d];
          }
        }
      }
    }
  }
  return ref;
}

void CheckAttentionAgainstDoubleReference(bool bf16, bool mixed_inputs = false,
                                          bool single_key = false) {
  const int world = global::GetWorldSize();
  const int cp_size = global::GetContextParallelSize();
  int gpu_count = 0;
  if (cp_size == 1 || cudaGetDeviceCount(&gpu_count) != cudaSuccess ||
      gpu_count < world) {
    GTEST_SKIP() << "Requires CP > 1 and " << world << " CUDA GPUs";
  }
  auto *factory =
      parallel::ProcessGroupFactory::Instance(Device::DeviceType::kCUDA);
  for (int rank = 0; rank < world; ++rank) {
    factory->GetOrCreate(parallel::GetContextParallelProcessGroupName(rank),
                         parallel::GetContextParallelGroupRanks(rank));
    if (global::GetContextParallelCommType() == "a2a+p2p") {
      for (int level = 0; level < 2; ++level) {
        factory->GetOrCreate(
            parallel::GetHierarchicalContextParallelProcessGroupName(rank,
                                                                     level),
            parallel::GetHierarchicalContextParallelGroupRanks(rank, level));
      }
    }
  }
  constexpr int batch = 2, heads = 16;
  const int dim = mixed_inputs ? 64 : 8;
  for (const int kv_heads : {16, 8}) {
    for (int seed = 0; seed < (single_key ? 1 : 2); ++seed) {
      const int sequence = single_key ? 128 : (seed == 0 ? 8 : 6) * cp_size;
      std::vector<float> q(batch * heads * sequence * dim), dout(q.size());
      std::vector<float> k(batch * kv_heads * sequence * dim), v(k.size());
      for (size_t i = 0; i < q.size(); ++i) {
        q[i] = static_cast<float>(0.4 * std::sin(0.17 * i + seed));
        dout[i] = static_cast<float>(std::cos(0.31 * i + seed));
        if (single_key && (i / dim) % sequence != 0) {
          dout[i] = 0.0f;
        }
      }
      for (size_t i = 0; i < k.size(); ++i) {
        k[i] = static_cast<float>(0.3 * std::cos(0.23 * i + seed));
        v[i] = static_cast<float>(std::sin(0.13 * i + seed));
      }
      auto ref_q = q, ref_k = k, ref_v = v, ref_dout = dout;
      if (bf16) {
        // Reference the quantized GEMM operands, including dO. Keep the test
        // inputs unrounded to exercise autocast and FP32 upstream gradients.
        for (auto *values : {&ref_q, &ref_k, &ref_v, &ref_dout}) {
          for (auto &value : *values) {
            auto bits = std::bit_cast<uint32_t>(value);
            bits += 0x7fff + ((bits >> 16) & 1);
            value = std::bit_cast<float>(bits & 0xffff0000U);
          }
        }
      }
      auto ref = DenseReference(ref_q, ref_k, ref_v, ref_dout, batch, heads,
                                kv_heads, sequence, dim);
      std::vector<float> mask(sequence * sequence);
      for (int row = 0; row < sequence; ++row) {
        for (int col = 0; col < sequence; ++col) {
          mask[row * sequence + col] = col > row ? 1.0f : 0.0f;
        }
      }
      std::vector<std::thread> workers;
      for (int rank = 0; rank < world; ++rank) {
        workers.emplace_back([&, rank] {
          global::thread_global_rank = rank;
          const auto *pg =
              factory->Get(parallel::GetContextParallelProcessGroupName(rank));
          parallel::cp_rank = pg->GetGroupRank(rank);
          const Device device(Device::DeviceType::kCUDA, rank);
          core::DeviceGuard guard(device);
          auto make_tensor = [&](const std::vector<float> &data, int nheads,
                                 bool grad, bool shard) {
            auto full = std::make_shared<Tensor>(
                data.data(), std::vector<int64_t>{batch, nheads, sequence, dim},
                DataType::kFLOAT32, device);
            auto tensor =
                shard ? parallel::SliceAlongCPRegionFunc(full, 2) : full;
            tensor->set_requires_grad(grad);
            return tensor;
          };
          auto q_local = make_tensor(q, heads, true, true);
          auto k_local = make_tensor(k, kv_heads, true, true);
          auto v_local = make_tensor(v, kv_heads, true, true);
          auto full_mask = std::make_shared<Tensor>(
              mask.data(), std::vector<int64_t>{1, 1, sequence, sequence},
              DataType::kFLOAT32, device);
          std::unique_ptr<AutocastGuard> autocast;
          std::shared_ptr<Tensor> identity;
          auto query = q_local, key = k_local, value = v_local;
          if (bf16) {
            autocast = std::make_unique<AutocastGuard>(
                Device::DeviceType::kCUDA, DataType::kBFLOAT16);
            // Identity projections model AMP's BF16 activations while leaving
            // Q/K/V leaf gradients in FP32, as in the model projection path.
            std::vector<float> eye(dim * dim, 0.0f);
            for (int d = 0; d < dim; ++d) {
              eye[d * dim + d] = 1.0f;
            }
            identity = std::make_shared<Tensor>(eye.data(),
                                                std::vector<int64_t>{dim, dim},
                                                DataType::kFLOAT32, device);
            if (!mixed_inputs) {
              query = q_local->View({batch * heads * (sequence / cp_size), dim})
                          ->Matmul(identity)
                          ->View(q_local->Dims());
              key =
                  k_local->View({batch * kv_heads * (sequence / cp_size), dim})
                      ->Matmul(identity)
                      ->View(k_local->Dims());
            }
            value =
                v_local->View({batch * kv_heads * (sequence / cp_size), dim})
                    ->Matmul(identity)
                    ->View(v_local->Dims());
          }
          auto output = parallel::AttnForwardFuncWithCP(
              query, key, value,
              parallel::SliceAlongCPRegionFunc(full_mask, 2));
          // Linear backward produces FP32 gradients even for BF16 outputs.
          // Also exercise restoring the forward autocast context in backward.
          autocast.reset();
          auto zero = std::make_shared<Tensor>(std::vector<int64_t>{1},
                                               DataType::kFLOAT32, device);
          zero->Fill(0.0f);
          (output + zero)->Backward(make_tensor(dout, heads, false, true));
          const auto positions = LocalPositions(parallel::cp_rank, sequence);
          auto compare = [&](const char *name,
                             const std::shared_ptr<Tensor> &actual,
                             const std::vector<double> &expected, int nheads) {
            ASSERT_NE(actual, nullptr);
            auto fp32 = actual->To(DataType::kFLOAT32);
            auto cpu = fp32.To(Device());
            const auto *data = static_cast<const float *>(cpu.DataPtr());
            double worst = 0.0, error_norm = 0.0, reference_norm = 0.0;
            double max_abs = 0.0;
            for (int b = 0; b < batch; ++b) {
              for (int h = 0; h < nheads; ++h) {
                for (size_t i = 0; i < positions.size(); ++i) {
                  for (int d = 0; d < dim; ++d) {
                    const auto local_index =
                        ((b * nheads + h) * positions.size() + i) * dim + d;
                    const auto global_index =
                        ((b * nheads + h) * sequence + positions[i]) * dim + d;
                    ASSERT_TRUE(std::isfinite(data[local_index]));
                    const double error =
                        data[local_index] - expected[global_index];
                    error_norm += error * error;
                    max_abs = std::max(max_abs, std::abs(error));
                    reference_norm +=
                        expected[global_index] * expected[global_index];
                    worst = std::max(
                        worst,
                        std::abs(data[local_index] - expected[global_index]) /
                            (1e-6 + 1e-6 * std::abs(expected[global_index])));
                  }
                }
              }
            }
            if (single_key &&
                (std::string(name) == "dq" || std::string(name) == "dk")) {
              EXPECT_LE(max_abs, 1e-6) << "Single visible key: " << name;
            }
            if (bf16) {
              // GQA contributions can nearly cancel. A pure relative norm
              // rejects even dense/contiguous BF16 roundoff near zero, so use
              // an absolute RMS floor in addition to the relative budget.
              const double count = static_cast<double>(cpu.NumElements());
              EXPECT_LE(std::sqrt(error_norm / count),
                        1e-3 + 0.02 * std::sqrt(reference_norm / count))
                  << name << ", rank=" << rank << ", kv_heads=" << kv_heads
                  << ", seed=" << seed;
            } else {
              EXPECT_LE(worst, 1.0)
                  << name << ", rank=" << rank << ", kv_heads=" << kv_heads
                  << ", seed=" << seed;
            }
          };
          compare("output", output, ref.output, heads);
          compare("dq", q_local->grad(), ref.dq, heads);
          compare("dk", k_local->grad(), ref.dk, kv_heads);
          compare("dv", v_local->grad(), ref.dv, kv_heads);

          if (mixed_inputs) {
            // Rounding Q/K before AMP cannot change the forward GEMM inputs,
            // so it must not change backward either. This catches using raw
            // FP32 RoPE operands in the custom backward even when its aggregate
            // error happens to fit the BF16 reference tolerance.
            auto q_rounded = make_tensor(ref_q, heads, true, true);
            auto k_rounded = make_tensor(ref_k, kv_heads, true, true);
            auto v_rounded = make_tensor(ref_v, kv_heads, true, true);
            autocast = std::make_unique<AutocastGuard>(
                Device::DeviceType::kCUDA, DataType::kBFLOAT16);
            auto rounded_value =
                v_rounded->View({batch * kv_heads * (sequence / cp_size), dim})
                    ->Matmul(identity)
                    ->View(v_rounded->Dims());
            auto rounded_output = parallel::AttnForwardFuncWithCP(
                q_rounded, k_rounded, rounded_value,
                parallel::SliceAlongCPRegionFunc(full_mask, 2));
            autocast.reset();
            (rounded_output + zero)
                ->Backward(make_tensor(dout, heads, false, true));
            auto compare_compute =
                [&](const std::shared_ptr<Tensor> &actual,
                    const std::shared_ptr<Tensor> &expected) {
                  auto a = actual->To(DataType::kFLOAT32).To(Device());
                  auto b = expected->To(DataType::kFLOAT32).To(Device());
                  const auto *a_data = static_cast<const float *>(a.DataPtr());
                  const auto *b_data = static_cast<const float *>(b.DataPtr());
                  double worst = 0.0;
                  for (size_t i = 0; i < a.NumElements(); ++i) {
                    worst = std::max(
                        worst,
                        std::abs(static_cast<double>(a_data[i]) - b_data[i]) /
                            (1e-7 + 1e-6 * std::abs(b_data[i])));
                  }
                  EXPECT_LE(worst, 1.0)
                      << "Forward-equivalent compute inputs, rank=" << rank;
                };
            compare_compute(output, rounded_output);
            compare_compute(q_local->grad(), q_rounded->grad());
            compare_compute(k_local->grad(), k_rounded->grad());
            compare_compute(v_local->grad(), v_rounded->grad());

            // Full-sequence attention on this one GPU, with the same FP32 Q/K
            // (as after RoPE), BF16 V and FP32 dO. No CP helpers are used here.
            auto q_dense = make_tensor(q, heads, true, false);
            auto k_dense = make_tensor(k, kv_heads, true, false);
            auto v_dense = make_tensor(v, kv_heads, true, false);
            autocast = std::make_unique<AutocastGuard>(
                Device::DeviceType::kCUDA, DataType::kBFLOAT16);
            auto dense_value = v_dense->View({batch * kv_heads * sequence, dim})
                                   ->Matmul(identity)
                                   ->View(v_dense->Dims());
            auto dense_key = k_dense->RepeatInterleave(heads / kv_heads, 1);
            dense_value = dense_value->RepeatInterleave(heads / kv_heads, 1);
            auto scores =
                q_dense->Matmul(dense_key->Transpose(-2, -1)) *
                static_cast<float>(1.0 / std::sqrt(static_cast<double>(dim)));
            auto probs = nn::function::Softmax(
                scores->MaskedFill(full_mask,
                                   std::numeric_limits<float>::lowest()),
                -1);
            auto dense_output = probs->Matmul(dense_value);
            autocast.reset();
            (dense_output + zero)
                ->Backward(make_tensor(dout, heads, false, false));
            auto compare_dense =
                [&](const char *name, const std::shared_ptr<Tensor> &actual,
                    const std::shared_ptr<Tensor> &expected, int nheads) {
                  auto fp32 = expected->To(DataType::kFLOAT32);
                  auto cpu = fp32.To(Device());
                  const auto *data = static_cast<const float *>(cpu.DataPtr());
                  compare(name, actual,
                          std::vector<double>(data, data + cpu.NumElements()),
                          nheads);
                };
            compare_dense("output", output, dense_output, heads);
            compare_dense("dq", q_local->grad(), q_dense->grad(), heads);
            compare_dense("dk", k_local->grad(), k_dense->grad(), kv_heads);
            compare_dense("dv", v_local->grad(), v_dense->grad(), kv_heads);
          }
        });
      }
      for (auto &worker : workers) {
        worker.join();
      }
    }
  }
}
TEST(ContextParallelAttentionTest, ForwardBackwardMatchesDoubleReference) {
  CheckAttentionAgainstDoubleReference(false);
}

TEST(ContextParallelAttentionTest,
     Bfloat16ForwardBackwardMatchesDoubleReference) {
  CheckAttentionAgainstDoubleReference(true);
}

TEST(ContextParallelAttentionTest, MixedPrecisionMatchesSingleDeviceAttention) {
  CheckAttentionAgainstDoubleReference(true, true);
}

TEST(ContextParallelAttentionTest,
     SingleVisibleKeyHasZeroQueryAndKeyGradients) {
  CheckAttentionAgainstDoubleReference(true, true, true);
}

#endif

}  // namespace
}  // namespace infini_train::test
