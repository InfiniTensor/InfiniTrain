#pragma once

#include <cstdlib>

#include "gflags/gflags.h"

namespace infini_train::example {

inline void ParseModelFlags(int *argc, char ***argv, const char *model_path) {
  gflags::ParseCommandLineNonHelpFlags(argc, argv, true);

  // Keep gflags' --helpshort precedence and allow --helpfull to show all flags.
  if (gflags::GetCommandLineFlagInfoOrDie("help").current_value == "true" &&
      gflags::GetCommandLineFlagInfoOrDie("helpshort").current_value !=
          "true" &&
      gflags::GetCommandLineFlagInfoOrDie("helpfull").current_value != "true") {
    gflags::ShowUsageWithFlagsRestrict((*argv)[0], model_path);
    std::exit(1);  // Match gflags' exit status for help output.
  }

  gflags::HandleCommandLineHelpFlags();
}

}  // namespace infini_train::example
