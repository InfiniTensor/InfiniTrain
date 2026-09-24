#pragma once

#include "clang-tidy/ClangTidyCheck.h"

namespace clang::tidy::infinitrain {

class NoMemberDefinitionsInHeadersCheck : public ClangTidyCheck {
 public:
  NoMemberDefinitionsInHeadersCheck(StringRef name, ClangTidyContext *context);

  void registerMatchers(ast_matchers::MatchFinder *finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &result) override;
  bool isLanguageVersionSupported(const LangOptions &lang_opts) const override;
  void storeOptions(ClangTidyOptions::OptionMap &options) override;

 private:
  const bool ignore_empty_bodies_;
};

}  // namespace clang::tidy::infinitrain
