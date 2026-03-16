# Tests CMakeLists.txt
# This file manages the test infrastructure for InfiniTrain

# Add test subdirectories
add_subdirectory(common)

# Tensor tests
add_subdirectory(tensor)

# Optimizer tests  
add_subdirectory(optimizer)

# Autograd operator tests
add_subdirectory(autograd)

# Hook tests
add_subdirectory(hook)

# Slow label tests
add_subdirectory(slow)
