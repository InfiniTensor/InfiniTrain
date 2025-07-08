#pragma once

#include <utility>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))
#define LOG_LOC(LEVEL, MSG) LOG(LEVEL) << MSG << " at " << __FILE__ << ":" << __LINE__
#define LOG_UNSUPPORTED_DTYPE(DTYPE, CONTEXT_IDENTIFIER)                                                               \
    LOG_LOC(FATAL, WRAP(CONTEXT_IDENTIFIER << " Unsupported data type: "                                               \
                                                  + kDataTypeToDesc.at(static_cast<infini_train::DataType>(dtype))))
