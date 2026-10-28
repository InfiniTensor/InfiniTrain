add_executable(mnist main.cc dataset.cc net.cc)
target_link_libraries(mnist PRIVATE InfiniTrain::infini_train_executable)
