#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "gflags/gflags.h"
#include "gtest/gtest.h"

#include "infini_train/include/training/arguments.h"

namespace {
namespace fs = std::filesystem;
using infini_train::training::ParseTrainingFlags;
using infini_train::training::TrainingOptions;
using infini_train::training::TrainingOptionsFromFlags;

class TrainingArgumentsTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto pattern = (fs::temp_directory_path() / "training arguments XXXXXX").string();
        ASSERT_NE(mkdtemp(pattern.data()), nullptr);
        directory_ = pattern;
    }
    void TearDown() override { fs::remove_all(directory_); }

    fs::path Write(const std::string &name, const std::string &contents) {
        const auto path = directory_ / name;
        std::ofstream file(path);
        file << contents;
        return path;
    }

    void Parse(std::vector<std::string> flags, const fs::path &defaults = {}) {
        flags.insert(flags.begin(), "test_training_arguments");
        std::vector<char *> argv;
        for (auto &flag : flags) { argv.push_back(flag.data()); }
        int argc = argv.size();
        argv.push_back(nullptr);
        auto data = argv.data();
        ParseTrainingFlags(&argc, &data, defaults);
    }

    gflags::FlagSaver flag_saver_;
    fs::path directory_;
};

TEST_F(TrainingArgumentsTest, CppAndCliUseTheSameDefaults) {
    const TrainingOptions expected;
    Parse({});
    const auto actual = TrainingOptionsFromFlags();
    EXPECT_EQ(actual.batch_size, expected.batch_size);
    EXPECT_EQ(actual.learning_rate, expected.learning_rate);
    EXPECT_EQ(actual.nthread_per_process, expected.nthread_per_process);
    EXPECT_EQ(actual.lr_decay_iters, expected.lr_decay_iters);
    EXPECT_EQ(actual.overfit_single_batch, expected.overfit_single_batch);
    EXPECT_EQ(actual.lora_target_modules, expected.lora_target_modules);
    EXPECT_EQ(actual.optimizer, expected.optimizer);
    EXPECT_EQ(actual.profile_name, expected.profile_name);
}

TEST_F(TrainingArgumentsTest, FileAndCliOverridesPreserveOrder) {
    auto defaults = Write("default.flags", "--optimizer=sgd\n--learning_rate=0.1\n--save_optimizer_state=false\n");
    auto first = Write("first.flags", "# comment\n\n--learning_rate=0.2\n--profile_name=a profile\n");
    auto nested = Write("nested.flags", "--learning_rate=0.3\n--nosave_optimizer_state\n");
    auto second = Write("second.flags", "--flagfile=" + nested.string() + "\n--learning_rate=0.4\n");
    Parse({"--flagfile=" + first.string(), "--flagfile", second.string(), "--learning_rate=0.5",
           "--save_optimizer_state=true"},
          defaults);
    const auto actual = TrainingOptionsFromFlags();
    EXPECT_EQ(actual.optimizer, "sgd");
    EXPECT_DOUBLE_EQ(actual.learning_rate, 0.5);
    EXPECT_TRUE(actual.save_optimizer_state);
    EXPECT_EQ(actual.profile_name, "a profile");
}

TEST_F(TrainingArgumentsTest, RejectsUnknownFileFlagsEvenWithUndefok) {
    auto file = Write("typo.flags", "# header\n--learning_raet=0.2\n");
    EXPECT_EXIT(Parse({"--undefok=learning_raet", "--flagfile=" + file.string()}), ::testing::ExitedWithCode(1),
                "typo.flags:2: Unknown flag: --learning_raet");
    EXPECT_EXIT(Parse({}, file), ::testing::ExitedWithCode(1), "typo.flags:2: Unknown flag");
}

TEST_F(TrainingArgumentsTest, DefaultFileDoesNotReplayItsNestedInclude) {
    auto base = Write("base.flags", "--learning_rate=0.1\n");
    auto defaults = Write("defaults.flags", "--flagfile=" + base.string() + "\n--learning_rate=0.2\n");
    Parse({}, defaults);
    EXPECT_DOUBLE_EQ(TrainingOptionsFromFlags().learning_rate, 0.2);
}

TEST_F(TrainingArgumentsTest, DefaultFilePathMayContainCommas) {
    auto defaults = Write("default, preset.flags", "--learning_rate=0.2\n");
    Parse({}, defaults);
    EXPECT_DOUBLE_EQ(TrainingOptionsFromFlags().learning_rate, 0.2);
}

TEST_F(TrainingArgumentsTest, RejectsMissingValuesAndUnsupportedSyntax) {
    auto missing = Write("missing.flags", "--learning_rate\n");
    auto section = Write("section.flags", "some_program\n--learning_rate=0.2\n");
    auto type = Write("type.flags", "--batch_size=not-a-number\n");
    EXPECT_EXIT(Parse({"--flagfile=" + missing.string()}), ::testing::ExitedWithCode(1),
                "missing.flags:1: Missing value");
    EXPECT_EXIT(Parse({"--flagfile=" + section.string()}), ::testing::ExitedWithCode(1),
                "section.flags:1: Expected --name=value");
    EXPECT_EXIT(Parse({"--flagfile=" + type.string()}), ::testing::ExitedWithCode(1), "batch_size");
}

TEST_F(TrainingArgumentsTest, ValidatesNestedFilesAndRejectsCycles) {
    auto bad = Write("nested bad.flags", "--does_not_exist=1\n");
    auto outer = Write("outer.flags", "--flagfile=" + bad.string() + "\n");
    EXPECT_EXIT(Parse({"--flagfile=" + outer.string()}), ::testing::ExitedWithCode(1),
                "nested bad.flags:1: Unknown flag");
    auto cycle = Write("cycle.flags", "--flagfile=" + (directory_ / "cycle.flags").string() + "\n");
    EXPECT_EXIT(Parse({"--flagfile=" + cycle.string()}), ::testing::ExitedWithCode(1), "Cyclic.*cycle.flags");
}

TEST_F(TrainingArgumentsTest, ExplicitFileWorksWithoutPackagedDefaults) {
    auto supplied = Write("custom.flags", "--learning_rate=0.25\n");
    Parse({"--flagfile", supplied.string()}, directory_ / "absent-default.flags");
    EXPECT_DOUBLE_EQ(TrainingOptionsFromFlags().learning_rate, 0.25);
    EXPECT_EXIT(Parse({}, directory_ / "absent-default.flags"), ::testing::ExitedWithCode(1),
                "Default training flag file not found");
    EXPECT_EXIT(Parse({"--flagfile=" + (directory_ / "absent.flags").string()}), ::testing::ExitedWithCode(1),
                "Cannot read flag file");
}

} // namespace
