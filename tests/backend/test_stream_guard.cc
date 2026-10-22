#include <atomic>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>

#include <cuda_runtime_api.h>

#include "gtest/gtest.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/core/runtime/stream_guard.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::test {
namespace {

static_assert(!std::is_default_constructible_v<core::StreamGuard>);
static_assert(!std::is_copy_constructible_v<core::StreamGuard>);
static_assert(!std::is_copy_assignable_v<core::StreamGuard>);
static_assert(!std::is_move_constructible_v<core::StreamGuard>);
static_assert(!std::is_move_assignable_v<core::StreamGuard>);

class StreamGuardTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (cudaGetDeviceCount(&device_count_) != cudaSuccess ||
        device_count_ == 0) {
      GTEST_SKIP() << "Requires CUDA";
    }
    impl_ = core::GetDeviceGuardImpl(Device::DeviceType::kCUDA);
    original_device_ = impl_->GetDevice();
    impl_->SetDevice(device0_);
    original_stream0_ = impl_->GetStream(device0_);
    if (device_count_ > 1) {
      original_stream1_ = impl_->GetStream(device1_);
    }
  }

  void TearDown() override {
    if (!impl_) {
      return;
    }
    impl_->ExchangeStream(device0_, original_stream0_);
    if (device_count_ > 1) {
      impl_->ExchangeStream(device1_, original_stream1_);
    }
    impl_->SetDevice(original_device_);
  }

  const Device device0_{Device::DeviceType::kCUDA, 0};
  const Device device1_{Device::DeviceType::kCUDA, 1};
  int device_count_ = 0;
  core::DeviceGuardImpl *impl_ = nullptr;
  Device original_device_;
  core::Stream *original_stream0_ = nullptr;
  core::Stream *original_stream1_ = nullptr;
};

TEST_F(StreamGuardTest, NestedSameDeviceRestoresPreviousStream) {
  std::unique_ptr<core::Stream> first(impl_->CreateStream(device0_));
  std::unique_ptr<core::Stream> second(impl_->CreateStream(device0_));
  {
    core::StreamGuard outer(device0_, first.get());
    EXPECT_EQ(impl_->GetDevice(), device0_);
    EXPECT_EQ(impl_->GetStream(device0_), first.get());
    {
      core::StreamGuard inner(device0_, second.get());
      EXPECT_EQ(impl_->GetStream(device0_), second.get());
    }
    EXPECT_EQ(impl_->GetStream(device0_), first.get());
  }
  EXPECT_EQ(impl_->GetStream(device0_), original_stream0_);
  EXPECT_EQ(impl_->GetDevice(), device0_);
}

TEST_F(StreamGuardTest, CrossDeviceRestoresTargetStreamBeforeOriginalDevice) {
  if (device_count_ < 2) {
    GTEST_SKIP() << "Requires two GPUs";
  }
  std::unique_ptr<core::Stream> first(impl_->CreateStream(device0_));
  std::unique_ptr<core::Stream> second(impl_->CreateStream(device1_));
  std::unique_ptr<core::Stream> third(impl_->CreateStream(device0_));
  {
    core::StreamGuard outer(device0_, first.get());
    {
      core::StreamGuard inner(device1_, second.get());
      EXPECT_EQ(impl_->GetDevice(), device1_);
      EXPECT_EQ(impl_->GetStream(device0_), first.get());
      EXPECT_EQ(impl_->GetStream(device1_), second.get());
      {
        core::StreamGuard nested(device0_, third.get());
        EXPECT_EQ(impl_->GetDevice(), device0_);
        EXPECT_EQ(impl_->GetStream(device0_), third.get());
      }
      EXPECT_EQ(impl_->GetDevice(), device1_);
      EXPECT_EQ(impl_->GetStream(device0_), first.get());
    }
    EXPECT_EQ(impl_->GetDevice(), device0_);
    EXPECT_EQ(impl_->GetStream(device0_), first.get());
    EXPECT_EQ(impl_->GetStream(device1_), original_stream1_);
  }
  EXPECT_EQ(impl_->GetStream(device0_), original_stream0_);
}

TEST_F(StreamGuardTest, RestoresAfterException) {
  if (device_count_ < 2) {
    GTEST_SKIP() << "Requires two GPUs";
  }
  std::unique_ptr<core::Stream> stream(impl_->CreateStream(device1_));
  EXPECT_THROW(
      {
        core::StreamGuard guard(device1_, stream.get());
        EXPECT_EQ(impl_->GetDevice(), device1_);
        throw std::runtime_error("leave guarded scope");
      },
      std::runtime_error);
  EXPECT_EQ(impl_->GetDevice(), device0_);
  EXPECT_EQ(impl_->GetStream(device1_), original_stream1_);
}

TEST_F(StreamGuardTest, RestoresEvenIfCurrentDeviceChangesInsideScope) {
  if (device_count_ < 2) {
    GTEST_SKIP() << "Requires two GPUs";
  }
  std::unique_ptr<core::Stream> stream(impl_->CreateStream(device1_));
  {
    core::StreamGuard guard(device1_, stream.get());
    impl_->SetDevice(device0_);
  }
  EXPECT_EQ(impl_->GetStream(device1_), original_stream1_);
  EXPECT_EQ(impl_->GetStream(device0_), original_stream0_);
  EXPECT_EQ(impl_->GetDevice(), device0_);
}

TEST_F(StreamGuardTest, ExchangeStreamAloneDoesNotSwitchDevices) {
  if (device_count_ < 2) {
    GTEST_SKIP() << "Requires two GPUs";
  }
  std::unique_ptr<core::Stream> stream(impl_->CreateStream(device1_));
  auto *previous = impl_->ExchangeStream(device1_, stream.get());
  EXPECT_EQ(previous, original_stream1_);
  EXPECT_EQ(impl_->GetDevice(), device0_);
  EXPECT_EQ(impl_->GetStream(device1_), stream.get());
  impl_->ExchangeStream(device1_, previous);
}

TEST_F(StreamGuardTest, CurrentStreamSelectionIsThreadLocal) {
  std::unique_ptr<core::Stream> stream(impl_->CreateStream(device0_));
  core::StreamGuard guard(device0_, stream.get());
  std::atomic<bool> independent{false};
  std::thread worker([&] {
    core::DeviceGuard device_guard(device0_);
    independent = impl_->GetStream(device0_) == original_stream0_;
  });
  worker.join();
  EXPECT_TRUE(independent.load());
  EXPECT_EQ(impl_->GetStream(device0_), stream.get());
}

TEST_F(StreamGuardTest, BlasHandleBindingIsThreadLocal) {
  std::unique_ptr<core::Stream> first(impl_->CreateStream(device0_));
  std::unique_ptr<core::Stream> second(impl_->CreateStream(device0_));
  core::StreamGuard guard(device0_, first.get());
  auto *handle = dynamic_cast<core::cuda::CudaBlasHandle *>(
      impl_->GetBlasHandle(device0_));
  ASSERT_NE(handle, nullptr);

  cublasHandle_t worker_handle = nullptr;
  std::thread worker([&] {
    core::StreamGuard worker_guard(device0_, second.get());
    auto *worker_blas = dynamic_cast<core::cuda::CudaBlasHandle *>(
        impl_->GetBlasHandle(device0_));
    ASSERT_NE(worker_blas, nullptr);
    worker_handle = worker_blas->cublas_handle();
    EXPECT_EQ(impl_->GetBlasHandle(device0_), worker_blas);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cublasGetStream(worker_handle, &stream), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        stream,
        dynamic_cast<core::cuda::CudaStream *>(second.get())->cuda_stream());
  });
  worker.join();

  EXPECT_NE(handle->cublas_handle(), worker_handle);
  // Do not refresh the main thread's handle: another thread must not change its
  // binding.
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cublasGetStream(handle->cublas_handle(), &stream),
            CUBLAS_STATUS_SUCCESS);
  EXPECT_EQ(stream,
            dynamic_cast<core::cuda::CudaStream *>(first.get())->cuda_stream());
}

TEST_F(StreamGuardTest, BlasHandleIsReusedAfterThreadExit) {
  std::unique_ptr<core::Stream> first(impl_->CreateStream(device0_));
  std::unique_ptr<core::Stream> second(impl_->CreateStream(device0_));
  auto get_handle = [&](core::Stream *selected_stream, cublasHandle_t &result) {
    core::StreamGuard guard(device0_, selected_stream);
    auto *handle = dynamic_cast<core::cuda::CudaBlasHandle *>(
        impl_->GetBlasHandle(device0_));
    ASSERT_NE(handle, nullptr);
    result = handle->cublas_handle();
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cublasGetStream(result, &stream), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        stream,
        dynamic_cast<core::cuda::CudaStream *>(selected_stream)->cuda_stream());
  };
  cublasHandle_t first_handle = nullptr;
  std::thread first_worker([&] { get_handle(first.get(), first_handle); });
  first_worker.join();
  cublasHandle_t second_handle = nullptr;
  std::thread second_worker([&] { get_handle(second.get(), second_handle); });
  second_worker.join();
  ASSERT_NE(first_handle, nullptr);
  EXPECT_EQ(first_handle, second_handle);
}

TEST_F(StreamGuardTest, BlasHandleFollowsNestedStreamGuards) {
  std::unique_ptr<core::Stream> first(impl_->CreateStream(device0_));
  std::unique_ptr<core::Stream> second(impl_->CreateStream(device0_));
  auto expect_blas_stream = [&](core::Stream *expected_stream) {
    auto *handle = dynamic_cast<core::cuda::CudaBlasHandle *>(
        impl_->GetBlasHandle(device0_));
    ASSERT_NE(handle, nullptr);
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cublasGetStream(handle->cublas_handle(), &stream),
              CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        stream,
        dynamic_cast<core::cuda::CudaStream *>(expected_stream)->cuda_stream());
  };
  {
    core::StreamGuard outer(device0_, first.get());
    expect_blas_stream(first.get());
    {
      core::StreamGuard inner(device0_, second.get());
      expect_blas_stream(second.get());
    }
    expect_blas_stream(first.get());
  }
  expect_blas_stream(original_stream0_);
}

TEST_F(StreamGuardTest, AllocationAndGemmFollowGuardedStream) {
  if (device_count_ < 2) {
    GTEST_SKIP() << "Requires two GPUs";
  }
  std::unique_ptr<core::Stream> stream(impl_->CreateStream(device1_));
  std::shared_ptr<Tensor> output;
  {
    core::StreamGuard guard(device1_, stream.get());
    EXPECT_EQ(impl_->GetDevice(), device1_);
    auto *handle = dynamic_cast<core::cuda::CudaBlasHandle *>(
        impl_->GetBlasHandle(device1_));
    ASSERT_NE(handle, nullptr);
    cudaStream_t blas_stream = nullptr;
    ASSERT_EQ(cublasGetStream(handle->cublas_handle(), &blas_stream),
              CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(
        blas_stream,
        dynamic_cast<core::cuda::CudaStream *>(stream.get())->cuda_stream());
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 2},
                                          DataType::kFLOAT32, device1_);
    input->Fill(2.0f);
    output = input->Matmul(input);
    // The guard does not synchronize: join explicitly before consuming output
    // on the original stream.
    impl_->SynchronizeStream(stream.get());
  }
  EXPECT_EQ(impl_->GetDevice(), device0_);
  EXPECT_EQ(impl_->GetStream(device1_), original_stream1_);
  auto cpu = output->To(Device());
  const auto *values = static_cast<const float *>(cpu.DataPtr());
  for (size_t i = 0; i < cpu.NumElements(); ++i) {
    EXPECT_FLOAT_EQ(values[i], 8.0f);
  }
}

}  // namespace
}  // namespace infini_train::test
