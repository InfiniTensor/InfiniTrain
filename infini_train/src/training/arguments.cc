#include "infini_train/include/training/arguments.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gflags/gflags.h"

DECLARE_string(flagfile);

namespace {
// TrainingOptions owns the framework defaults for both C++ and CLI callers.
const infini_train::training::TrainingOptions kTrainingDefaults;
} // namespace

DEFINE_uint32(batch_size, kTrainingDefaults.batch_size, "batch size, in units of #batch dimensions");
DEFINE_uint32(sequence_length, kTrainingDefaults.sequence_length, "sequence length");
DEFINE_uint32(total_batch_size, kTrainingDefaults.total_batch_size, "total desired batch size, in units of #tokens");
DEFINE_uint32(num_iteration, kTrainingDefaults.num_iteration, "number of iterations to run");
DEFINE_double(learning_rate, kTrainingDefaults.learning_rate, "Peak learning rate.");
DEFINE_int32(zero_stage, kTrainingDefaults.zero_stage, "ZeRO stage (0/1/2/3); 0 disables DistributedOptimizer");
DEFINE_double(min_lr, kTrainingDefaults.min_lr, "Minimum learning rate.");
DEFINE_string(lr_decay_style, kTrainingDefaults.lr_decay_style,
              "LR decay style: none|constant|linear|cosine|inverse-square-root");
DEFINE_int64(lr_warmup_iters, kTrainingDefaults.lr_warmup_iters, "Number of linear warmup iterations.");
DEFINE_double(lr_warmup_init, kTrainingDefaults.lr_warmup_init, "Initial learning rate at the start of warmup.");
DEFINE_int64(lr_decay_iters, kTrainingDefaults.lr_decay_iters,
             "Number of iterations to decay LR over (0 = num_iteration).");
DEFINE_uint32(val_loss_every, kTrainingDefaults.val_loss_every, "every how many steps to evaluate val loss?");
DEFINE_uint32(sample_every, kTrainingDefaults.sample_every, "how often to sample from the model?");
DEFINE_bool(overfit_single_batch, kTrainingDefaults.overfit_single_batch, "overfit just one batch of data");
DEFINE_string(device, kTrainingDefaults.device,
              "device type (cpu/cuda/privateuse1/<registered privateuse1 backend name>), useless if using parallel "
              "training mode");
DEFINE_int32(nthread_per_process, kTrainingDefaults.nthread_per_process,
             "Number of threads to use for each process. "
             "When set > 1, enables data parallelism on the specified accelerator devices.");
DEFINE_uint32(tensor_parallel, kTrainingDefaults.tensor_parallel, "Tensor Parallel world size");
DEFINE_bool(sequence_parallel, kTrainingDefaults.sequence_parallel, "Whether to enable Sequence Parallel");
DEFINE_uint32(pipeline_parallel, kTrainingDefaults.pipeline_parallel,
              "Pipeline Parallel world size, specified the number of PP stages.");
DEFINE_uint32(virtual_pipeline_parallel, kTrainingDefaults.virtual_pipeline_parallel, "Number of chunks in PP stage.");
DEFINE_string(dtype, kTrainingDefaults.dtype, "precision used in training (float32/bfloat16)");
DEFINE_uint32(save_interval, kTrainingDefaults.save_interval, "save checkpoint every N steps; 0 disables saving");
DEFINE_string(load, kTrainingDefaults.load, "checkpoint directory to resume from");
DEFINE_string(save, kTrainingDefaults.save, "root directory used to store checkpoints");
DEFINE_uint32(max_checkpoint_keep, kTrainingDefaults.max_checkpoint_keep, "max number of checkpoint steps to keep");
DEFINE_bool(load_optimizer_state, kTrainingDefaults.load_optimizer_state,
            "whether optimizer state is restored from checkpoints");
DEFINE_bool(save_optimizer_state, kTrainingDefaults.save_optimizer_state,
            "whether optimizer state is persisted in checkpoints");
DEFINE_int32(lora_rank, kTrainingDefaults.lora_rank, "LoRA rank (0 = disabled)");
DEFINE_double(lora_alpha, kTrainingDefaults.lora_alpha, "LoRA alpha scaling factor");
DEFINE_string(lora_target_modules, kTrainingDefaults.lora_target_modules, "LoRA target modules (comma-separated)");
DEFINE_string(lora_save_path, kTrainingDefaults.lora_save_path, "Path to save LoRA weights after training");
DEFINE_string(lora_load_path, kTrainingDefaults.lora_load_path, "Path to load LoRA weights from");
DEFINE_string(optimizer, kTrainingDefaults.optimizer, "Optimizer: sgd|adam");
DEFINE_string(profile_name, kTrainingDefaults.profile_name, "Prefix for profiler output files");
DEFINE_string(precision_check, kTrainingDefaults.precision_check,
              "precision check: level=N,format=simple|table,output_md5=true|false,output_path=PATH,baseline=PATH");

namespace infini_train::training {
namespace {

namespace fs = std::filesystem;

std::string Trim(const std::string &text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    return begin == std::string::npos ? "" : text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1);
}

std::vector<std::string> FileList(const std::string &text) {
    std::vector<std::string> files;
    size_t begin = 0;
    while (begin < text.size()) {
        const auto end = text.find(',', begin);
        auto file = Trim(text.substr(begin, end == std::string::npos ? end : end - begin));
        if (file.empty()) {
            throw std::runtime_error("Empty path in --flagfile list");
        }
        files.push_back(std::move(file));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
        if (begin == text.size()) {
            throw std::runtime_error("Empty path in --flagfile list");
        }
    }
    return files;
}

// gflags runs validators while holding its registry mutex. Snapshot flag types
// before parsing: querying gflags from inside the validator would deadlock.
std::map<std::string, std::string> &FlagTypes() {
    static std::map<std::string, std::string> types;
    return types;
}

std::string FlagType(const std::string &name) {
    const auto &types = FlagTypes();
    if (auto it = types.find(name); it != types.end()) {
        return it->second;
    }
    if (name.starts_with("no")) {
        auto it = types.find(name.substr(2));
        if (it != types.end() && it->second == "bool") {
            return "bool";
        }
    }
    return "";
}

class FlagfileValidator {
public:
    void Validate(const std::string &files) {
        for (const auto &file : FileList(files)) { ValidateFile(file); }
    }

    void ValidateFile(const fs::path &path) {
        if (!fs::is_regular_file(path)) {
            throw std::runtime_error("Cannot read flag file: " + path.string());
        }
        const auto canonical = fs::canonical(path);
        if (active_.size() >= 64 || !active_.insert(canonical).second) {
            throw std::runtime_error("Cyclic or excessively nested flag file: " + path.string());
        }
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("Cannot read flag file: " + path.string());
        }
        std::string line;
        size_t number = 0;
        while (std::getline(input, line)) {
            ++number;
            line = Trim(line);
            if (line.empty() || line.starts_with('#')) {
                continue;
            }
            const auto location = path.string() + ":" + std::to_string(number) + ": ";
            if (!line.starts_with("--")) {
                throw std::runtime_error(location + "Expected --name=value (filename sections are not supported)");
            }
            const auto equal = line.find('=');
            const auto name = line.substr(2, equal == std::string::npos ? equal : equal - 2);
            const auto type = FlagType(name);
            if (type.empty()) {
                throw std::runtime_error(location + "Unknown flag: --" + name);
            }
            if (equal == std::string::npos && type != "bool") {
                throw std::runtime_error(location + "Missing value for --" + name + "; use --name=value");
            }
            if (name == "flagfile") {
                Validate(line.substr(equal + 1));
            } else if (name == "fromenv" || name == "tryfromenv") {
                for (const auto &flag : FileList(line.substr(equal + 1))) {
                    if (flag == "flagfile") {
                        throw std::runtime_error(location + "Use an explicit --flagfile instead of loading it fromenv");
                    }
                }
            }
        }
        if (input.bad()) {
            throw std::runtime_error("Failed reading flag file: " + path.string());
        }
        active_.erase(canonical);
    }

private:
    std::set<fs::path> active_;
};

bool ValidateFlagFiles(const char *, const std::string &files) {
    try {
        FlagfileValidator().Validate(files);
        return true;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return false;
    }
}

bool HasExplicitFlagfile(int argc, char *const argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string argument = argv[i];
        if (argument == "--") {
            break;
        }
        if (!argument.starts_with('-')) {
            continue;
        }
        argument.erase(0, argument.starts_with("--") ? 2 : 1);
        const auto equal = argument.find('=');
        const auto name = argument.substr(0, equal);
        if (name == "flagfile") {
            return true;
        }
        // A string argument's value may itself look like --flagfile; do not
        // mistake that value for a separate command-line option.
        const auto type = FlagType(name);
        if (equal == std::string::npos && !type.empty() && type != "bool") {
            ++i;
        }
    }
    return false;
}

fs::path ExecutableDirectory(const char *argv0) {
#ifdef __linux__
    std::error_code error;
    const auto executable = fs::read_symlink("/proc/self/exe", error);
    if (!error) {
        return executable.parent_path();
    }
#endif
    fs::path executable_path(argv0);
    if (executable_path.has_parent_path()) {
        return fs::canonical(executable_path).parent_path();
    }
    const char *search_path = std::getenv("PATH");
    if (search_path) {
        const std::string paths(search_path);
        size_t begin = 0;
        do {
            const auto end = paths.find(':', begin);
            const auto dir = paths.substr(begin, end == std::string::npos ? end : end - begin);
            const auto candidate = fs::path(dir.empty() ? "." : dir) / executable_path;
            if (fs::is_regular_file(candidate)) {
                return fs::canonical(candidate).parent_path();
            }
            if (end == std::string::npos) {
                break;
            }
            begin = end + 1;
        } while (begin <= paths.size());
    }
    throw std::runtime_error("Cannot locate executable: " + executable_path.string());
}

} // namespace

TrainingOptions TrainingOptionsFromFlags() {
    TrainingOptions options;
    options.profile_name = FLAGS_profile_name;
    options.precision_check = FLAGS_precision_check;
    options.batch_size = FLAGS_batch_size;
    options.sequence_length = FLAGS_sequence_length;
    options.total_batch_size = FLAGS_total_batch_size;
    options.num_iteration = FLAGS_num_iteration;
    options.learning_rate = FLAGS_learning_rate;
    options.zero_stage = FLAGS_zero_stage;
    options.min_lr = FLAGS_min_lr;
    options.lr_decay_style = FLAGS_lr_decay_style;
    options.lr_warmup_iters = FLAGS_lr_warmup_iters;
    options.lr_warmup_init = FLAGS_lr_warmup_init;
    options.lr_decay_iters = FLAGS_lr_decay_iters;
    options.val_loss_every = FLAGS_val_loss_every;
    options.sample_every = FLAGS_sample_every;
    options.overfit_single_batch = FLAGS_overfit_single_batch;
    options.device = FLAGS_device;
    options.nthread_per_process = FLAGS_nthread_per_process;
    options.tensor_parallel = FLAGS_tensor_parallel;
    options.sequence_parallel = FLAGS_sequence_parallel;
    options.pipeline_parallel = FLAGS_pipeline_parallel;
    options.virtual_pipeline_parallel = FLAGS_virtual_pipeline_parallel;
    options.dtype = FLAGS_dtype;
    options.save_interval = FLAGS_save_interval;
    options.load = FLAGS_load;
    options.save = FLAGS_save;
    options.max_checkpoint_keep = FLAGS_max_checkpoint_keep;
    options.load_optimizer_state = FLAGS_load_optimizer_state;
    options.save_optimizer_state = FLAGS_save_optimizer_state;
    options.lora_rank = FLAGS_lora_rank;
    options.lora_alpha = FLAGS_lora_alpha;
    options.lora_target_modules = FLAGS_lora_target_modules;
    options.lora_save_path = FLAGS_lora_save_path;
    options.lora_load_path = FLAGS_lora_load_path;
    options.optimizer = FLAGS_optimizer;
    return options;
}

void ParseTrainingFlags(int *argc, char ***argv, const fs::path &default_flagfile) {
    try {
        std::vector<gflags::CommandLineFlagInfo> flags;
        gflags::GetAllFlags(&flags);
        auto &types = FlagTypes();
        types.clear();
        for (const auto &flag : flags) { types.emplace(flag.name, flag.type); }
        if (!gflags::RegisterFlagValidator(&FLAGS_flagfile, &ValidateFlagFiles)) {
            throw std::runtime_error("Another validator is already registered for --flagfile");
        }
        if (!default_flagfile.empty()) {
            const auto path = default_flagfile.is_absolute() ? default_flagfile
                                                             : ExecutableDirectory((*argv)[0]) / default_flagfile;
            if (fs::exists(path)) {
                // This is one resolved filesystem path, not a comma-separated
                // --flagfile list; executable directories can contain commas.
                FlagfileValidator().ValidateFile(path);
                if (!gflags::ReadFromFlagsFile(path.string(), (*argv)[0], false)) {
                    std::exit(EXIT_FAILURE);
                }
                // Nested includes have already been applied in file order.
                // Do not let ParseCommandLineFlags replay the final include.
                FLAGS_flagfile.clear();
            } else if (!HasExplicitFlagfile(*argc, *argv)) {
                throw std::runtime_error("Default training flag file not found: " + path.string()
                                         + ". Copy configs/ beside the executable or pass --flagfile explicitly.");
            }
        }
        gflags::ParseCommandLineFlags(argc, argv, true);
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace infini_train::training
