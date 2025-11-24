#ifndef MODEL_MANAGER_H_
#define MODEL_MANAGER_H_

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

enum class ModelType {
  SINE_FLOAT32,
  PERSON_DETECTION_INT8,
  // Add more models here as you implement them
};

struct ModelConfig {
  const char* name;
  const unsigned char* model_data;
  int model_data_len;
  const char* quantization;
  int input_size;
  int output_size;
};

class ModelManager {
 public:
  static ModelConfig GetModelConfig(ModelType type);
  static bool SetupOpResolver(ModelType type, 
                              tflite::MicroMutableOpResolver<20>* resolver);
};

#endif  // MODEL_MANAGER_H_