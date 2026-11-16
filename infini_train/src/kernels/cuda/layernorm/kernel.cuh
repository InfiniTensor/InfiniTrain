#pragma once

#include <cub/block/block_reduce.cuh>

namespace infini_train::kernels::cuda {

template <int N>
__global__ void StableLNForward32(const float* x, const float* w,
                                  const float* b, int width, float eps,
                                  float* mean, float* rstd, float* y) {
  using R = cub::BlockReduce<float, N>;
  __shared__ typename R::TempStorage smean;
  __shared__ typename R::TempStorage svar;
  __shared__ float mu, inv;
  const int base = blockIdx.x * width;
  const float anchor = x[base];
  float sum = 0;
  for (int i = threadIdx.x; i < width; i += N) sum += x[base + i] - anchor;
  float total = R(smean).Sum(sum);
  if (threadIdx.x == 0) {
    mu = total / width;
    mean[blockIdx.x] = anchor + mu;
  }
  __syncthreads();
  float sq = 0;
  for (int i = threadIdx.x; i < width; i += N) {
    float delta = (x[base + i] - anchor) - mu;
    sq += delta * delta;
  }
  float variance_sum = R(svar).Sum(sq);
  if (threadIdx.x == 0) {
    inv = 1.0f / sqrtf(variance_sum / width + eps);
    rstd[blockIdx.x] = inv;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < width; i += N)
    y[base + i] = ((x[base + i] - anchor) - mu) * inv * w[i] + b[i];
}

template <int N>
__global__ void StableLNInputGrad32(const float* x, const float* dy,
                                    const float* w, const float* rstd,
                                    int width, float* normalized, float* dx) {
  using R = cub::BlockReduce<float, N>;
  __shared__ typename R::TempStorage smean, sdy, sdynorm;
  __shared__ float mu, mean_dy, mean_dynorm;
  const int base = blockIdx.x * width;
  const float anchor = x[base];
  float sum = 0;
  for (int i = threadIdx.x; i < width; i += N) sum += x[base + i] - anchor;
  float total = R(smean).Sum(sum);
  if (threadIdx.x == 0) mu = total / width;
  __syncthreads();
  const float inv = rstd[blockIdx.x];
  float sum_d = 0, sum_dn = 0;
  for (int i = threadIdx.x; i < width; i += N) {
    float n = ((x[base + i] - anchor) - mu) * inv, d = w[i] * dy[base + i];
    normalized[base + i] = float(n);
    sum_d += d;
    sum_dn += d * n;
  }
  float d = R(sdy).Sum(sum_d), dn = R(sdynorm).Sum(sum_dn);
  if (threadIdx.x == 0) {
    mean_dy = d / width;
    mean_dynorm = dn / width;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < width; i += N) {
    float n = ((x[base + i] - anchor) - mu) * inv;
    dx[base + i] = (w[i] * dy[base + i] - mean_dy - n * mean_dynorm) * inv;
  }
}

template <int N>
__global__ void StableLNParameterGrad32(const float* normalized,
                                        const float* dy, int tokens, int width,
                                        float* dw, float* db) {
  using R = cub::BlockReduce<float, N>;
  __shared__ typename R::TempStorage sw, sb;
  const int feature = blockIdx.x;
  float w = 0, b = 0;
  for (int t = threadIdx.x; t < tokens; t += N) {
    int i = t * width + feature;
    w += dy[i] * normalized[i];
    b += dy[i];
  }
  float ww = R(sw).Sum(w), bb = R(sb).Sum(b);
  if (threadIdx.x == 0) {
    dw[feature] = float(ww);
    db[feature] = float(bb);
  }
}

}  // namespace infini_train::kernels::cuda
