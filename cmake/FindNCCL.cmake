# FindNCCL.cmake - locate NCCL library and header
#
# This will define:
#   NCCL_FOUND          - true if NCCL found
#   NCCL_INCLUDE_DIRS   - where to find nccl.h
#   NCCL_LIBRARIES      - full path to libnccl.so / .a
#
# You can override automatic search with:
#   -D NCCL_ROOT=/your/nccl/root

# Allow override
if (NCCL_ROOT)
    set(_NCCL_PATHS "${NCCL_ROOT}")
else()
    set(_NCCL_PATHS /usr /usr/local /usr/lib /opt/nccl)
endif()

# Find include dir
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    PATHS ${_NCCL_PATHS}
    PATH_SUFFIXES include include/nccl
)

# Find library
find_library(NCCL_LIBRARY
    NAMES nccl
    PATHS ${_NCCL_PATHS}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)

# Standard result handling
find_package_handle_standard_args(NCCL
    REQUIRED_VARS NCCL_INCLUDE_DIR NCCL_LIBRARY
    VERSION_VAR NCCL_VERSION
)

if (NCCL_FOUND)
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
endif()
