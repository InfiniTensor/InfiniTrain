#include "NoMemberDefinitionsInHeadersCheck.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace clang::tidy::infinitrain {
namespace {

bool isTemplated(const CXXMethodDecl &method) {
  if (method.getTemplatedKind() != FunctionDecl::TK_NonTemplate) {
    return true;
  }

  for (const DeclContext *context = method.getDeclContext(); context != nullptr;
       context = context->getParent()) {
    const auto *record = dyn_cast<CXXRecordDecl>(context);
    if (record != nullptr && (record->getDescribedClassTemplate() != nullptr ||
                              isa<ClassTemplateSpecializationDecl>(record))) {
      return true;
    }
  }
  return false;
}

}  // namespace

NoMemberDefinitionsInHeadersCheck::NoMemberDefinitionsInHeadersCheck(
    StringRef name, ClangTidyContext *context)
    : ClangTidyCheck(name, context),
      ignore_empty_bodies_(Options.get("IgnoreEmptyBodies", true)) {}

bool NoMemberDefinitionsInHeadersCheck::isLanguageVersionSupported(
    const LangOptions &lang_opts) const {
  return lang_opts.CPlusPlus;
}

void NoMemberDefinitionsInHeadersCheck::storeOptions(
    ClangTidyOptions::OptionMap &options) {
  Options.store(options, "IgnoreEmptyBodies", ignore_empty_bodies_);
}

void NoMemberDefinitionsInHeadersCheck::registerMatchers(
    ast_matchers::MatchFinder *finder) {
  using namespace ast_matchers;
  finder->addMatcher(
      cxxMethodDecl(isDefinition(), unless(isImplicit())).bind("method"), this);
}

void NoMemberDefinitionsInHeadersCheck::check(
    const ast_matchers::MatchFinder::MatchResult &result) {
  const auto *method = result.Nodes.getNodeAs<CXXMethodDecl>("method");
  if (method == nullptr || method->isDefaulted() || method->isDeleted() ||
      method->isConstexpr() || method->getParent()->isLambda() ||
      isTemplated(*method)) {
    return;
  }

  const auto *body = dyn_cast_or_null<CompoundStmt>(method->getBody());
  if (ignore_empty_bodies_ && body != nullptr && body->body_empty()) {
    return;
  }

  SourceLocation location = method->getLocation();
  if (location.isInvalid() || location.isMacroID()) {
    return;
  }
  if (const Stmt *body = method->getBody();
      body != nullptr && body->getBeginLoc().isMacroID()) {
    return;
  }

  const SourceManager &source_manager = *result.SourceManager;
  location = source_manager.getSpellingLoc(location);
  if (source_manager.isInSystemHeader(location) ||
      source_manager.isWrittenInMainFile(location)) {
    return;
  }

  diag(location,
       "member function definition must be moved from the header to an "
       "implementation file");
}

}  // namespace clang::tidy::infinitrain
