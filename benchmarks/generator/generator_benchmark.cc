#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"
#if defined(USE_CUDA)
#include "infini_train/src/core/runtime/cuda/cuda_generator_impl.h"
#endif

namespace {

using Clock = std::chrono::steady_clock;
using infini_train::DataType;
using infini_train::Device;
using infini_train::Generator;
using infini_train::Tensor;

struct Options {
    std::string device = "cpu";
    std::string operation = "all";
    std::string generator = "explicit";
    int device_index = 0;
    int64_t elements = 1 << 20;
    int warmup = 10;
    int iterations = 100;
    uint64_t seed = 42;
};

void PrintUsage(const char *program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --device cpu|cuda\n"
        << "  --device-index N\n"
        << "  --op uniform|normal|state|all\n"
        << "  --generator explicit|default\n"
        << "  --elements N\n"
        << "  --warmup N\n"
        << "  --iterations N\n"
        << "  --seed N\n";
}

std::string RequireValue(int argc, char **argv, int &index) {
    if (++index >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index - 1]);
    }
    return argv[index];
}

Options ParseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--help" || argument == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (argument == "--device") {
            options.device = RequireValue(argc, argv, i);
        } else if (argument == "--device-index") {
            options.device_index = std::stoi(RequireValue(argc, argv, i));
        } else if (argument == "--op") {
            options.operation = RequireValue(argc, argv, i);
        } else if (argument == "--generator") {
            options.generator = RequireValue(argc, argv, i);
        } else if (argument == "--elements") {
            options.elements = std::stoll(RequireValue(argc, argv, i));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(RequireValue(argc, argv, i));
        } else if (argument == "--iterations") {
            options.iterations = std::stoi(RequireValue(argc, argv, i));
        } else if (argument == "--seed") {
            options.seed = std::stoull(RequireValue(argc, argv, i));
        } else {
            throw std::invalid_argument(std::string("unknown option: ") + argv[i]);
        }
    }

    if (options.device != "cpu" && options.device != "cuda") {
        throw std::invalid_argument("--device must be cpu or cuda");
    }
    if (options.operation != "uniform" && options.operation != "normal"
        && options.operation != "state" && options.operation != "all") {
        throw std::invalid_argument("--op must be uniform, normal, state, or all");
    }
    if (options.generator != "explicit" && options.generator != "default") {
        throw std::invalid_argument("--generator must be explicit or default");
    }
    if (options.elements <= 0 || options.warmup < 0 || options.iterations <= 0) {
        throw std::invalid_argument("elements and iterations must be positive; warmup must be non-negative");
    }
    return options;
}

Device MakeDevice(const Options &options) {
    if (options.device == "cpu") {
        return Device(Device::DeviceType::kCPU, 0);
    }
#if defined(USE_CUDA)
    return Device(Device::DeviceType::kCUDA, options.device_index);
#else
    throw std::invalid_argument("CUDA benchmark requested, but InfiniTrain was built without USE_CUDA");
#endif
}

Generator MakeGenerator(Device device, uint64_t seed) {
    if (device.IsCPU()) {
        return infini_train::core::cpu::createCPUGenerator(seed);
    }
#if defined(USE_CUDA)
    return infini_train::core::cuda::createCUDAGenerator(device.index(), seed);
#else
    (void)seed;
    throw std::invalid_argument("CUDA support is disabled");
#endif
}

void Synchronize(Device device) {
    infini_train::core::GetDeviceGuardImpl(device.type())->SynchronizeDevice(device);
}

template <typename Function>
double MeasureMicroseconds(Device device, int warmup, int iterations, Function &&function) {
    for (int i = 0; i < warmup; ++i) {
        function();
    }
    Synchronize(device);
    const auto start = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        function();
    }
    Synchronize(device);
    const auto end = Clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

void PrintHeader() {
    std::cout << "device,device_index,operation,generator,elements,iterations,latency_us,gsamples_s,bandwidth_gbps\n";
}

void PrintResult(const Options &options, std::string_view operation, double latency_us) {
    const double seconds = latency_us * 1e-6;
    const double samples_per_second = static_cast<double>(options.elements) / seconds;
    const double gsamples_per_second = samples_per_second / 1e9;
    const double bandwidth_gbps = samples_per_second * sizeof(float) / 1e9;
    std::cout << options.device << ',' << options.device_index << ',' << operation << ','
              << options.generator << ',' << options.elements << ',' << options.iterations << ','
              << std::fixed << std::setprecision(3) << latency_us << ',' << gsamples_per_second << ','
              << bandwidth_gbps << '\n';
}

void RunDistribution(const Options &options, Device device, std::string_view operation) {
    Generator explicit_generator = MakeGenerator(device, options.seed);
    std::optional<Generator> generator = options.generator == "explicit"
        ? std::optional<Generator>(explicit_generator)
        : std::nullopt;
    if (!generator) {
        infini_train::manual_seed(options.seed);
    }
    auto tensor = std::make_shared<Tensor>(
        std::vector<int64_t>{options.elements}, DataType::kFLOAT32, device);

    const double latency_us = MeasureMicroseconds(device, options.warmup, options.iterations, [&] {
        if (operation == "uniform") {
            infini_train::nn::init::Uniform(tensor, 0.0f, 1.0f, generator);
        } else {
            infini_train::nn::init::Normal(tensor, 0.0f, 1.0f, generator);
        }
    });
    PrintResult(options, operation, latency_us);
}

void RunStateBenchmark(const Options &options, Device device) {
    Generator generator = MakeGenerator(device, options.seed);
    auto state = generator.get_state();
    const double get_state_us = MeasureMicroseconds(device, options.warmup, options.iterations, [&] {
        state = generator.get_state();
    });
    const double set_state_us = MeasureMicroseconds(device, options.warmup, options.iterations, [&] {
        generator.set_state(*state);
    });
    uint64_t seed = options.seed;
    const double manual_seed_us = MeasureMicroseconds(device, options.warmup, options.iterations, [&] {
        generator.set_current_seed(seed++);
    });

    Options state_options = options;
    // State-management operations do not generate tensor elements. Reporting
    // zero avoids presenting meaningless throughput numbers for these rows.
    state_options.elements = 0;
    PrintResult(state_options, "get_state", get_state_us);
    PrintResult(state_options, "set_state", set_state_us);
    PrintResult(state_options, "manual_seed", manual_seed_us);
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        const Device device = MakeDevice(options);
        PrintHeader();
        if (options.operation == "uniform" || options.operation == "all") {
            RunDistribution(options, device, "uniform");
        }
        if (options.operation == "normal" || options.operation == "all") {
            RunDistribution(options, device, "normal");
        }
        if (options.operation == "state" || options.operation == "all") {
            RunStateBenchmark(options, device);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "generator_benchmark: " << error.what() << '\n';
        PrintUsage(argv[0]);
        return 2;
    }
}
