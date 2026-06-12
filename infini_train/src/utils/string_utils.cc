#include "infini_train/include/utils/string_utils.h"

#include <sstream>

namespace infini_train::utils {
std::string DimsToString(const std::vector<int64_t> &dims) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}
} // namespace infini_train::utils
