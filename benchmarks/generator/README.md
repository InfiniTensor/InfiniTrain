# Generator benchmark

This benchmark measures Generator state-management latency and end-to-end
`Uniform` / `Normal` tensor fill throughput. It is intentionally separate from
the GoogleTest correctness suite: benchmark results are informative and do not
make tests flaky.

## Build

From the repository root:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DUSE_NCCL=OFF \
  -DBUILD_TEST=ON \
  -DBUILD_BENCHMARK=ON
cmake --build build -j
```

For a CPU-only build, use `-DUSE_CUDA=OFF`.

## Correctness tests

```bash
ctest --test-dir build -L cpu --output-on-failure
ctest --test-dir build -L cuda --output-on-failure
```

To run only this feature's tests:

```bash
ctest --test-dir build -R Generator --output-on-failure
```

## Single benchmark

```bash
./build/benchmarks/generator_benchmark \
  --device cuda \
  --device-index 0 \
  --op all \
  --generator explicit \
  --elements 1048576 \
  --warmup 10 \
  --iterations 100 \
  --seed 42
```

The output is CSV with average latency, generated samples per second, and
effective output bandwidth. CUDA timing synchronizes the device before and
after the measured loop, so it includes kernel execution rather than only
asynchronous launch overhead.

## Standard matrix

```bash
bash benchmarks/generator/run_generator_benchmark.sh \
  ./build/benchmarks/generator_benchmark cpu > generator_cpu.csv

bash benchmarks/generator/run_generator_benchmark.sh \
  ./build/benchmarks/generator_benchmark cuda > generator_cuda.csv
```

For comparable reports, keep the compiler, build type, CPU/GPU model, CUDA
version, warmup count, iteration count, and thread-related environment
variables fixed. Compare before/after InfiniTrain commits rather than requiring
element-wise or performance parity with PyTorch.
