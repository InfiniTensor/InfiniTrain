#include "NoMemberDefinitionsInHeadersCheck.h"
#include "clang-tidy/ClangTidyModule.h"
#include "clang-tidy/ClangTidyModuleRegistry.h"

namespace clang::tidy::infinitrain {
namespace {

class InfiniTrainTidyModule : public ClangTidyModule {
 public:
  void addCheckFactories(ClangTidyCheckFactories &factories) override {
    factories.registerCheck<NoMemberDefinitionsInHeadersCheck>(
        "infinitrain-no-member-definitions-in-headers");
  }
};

static ClangTidyModuleRegistry::Add<InfiniTrainTidyModule> module(
    "infinitrain-module", "Adds InfiniTrain project checks.");

}  // namespace

volatile int InfiniTrainTidyModuleAnchorSource = 0;

}  // namespace clang::tidy::infinitrain
