infini_train_add_test(test_model_provider_registry
  SOURCES test_model_provider_registry.cc
  LABELS cpu pretrain
)

infini_train_add_test(test_training_arguments
  SOURCES test_training_arguments.cc
  LINK_LIBRARIES InfiniTrain::training_cli
  LABELS cpu pretrain
)
