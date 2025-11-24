#include "model_manager.h"
#include "models/sine/model.h"
#include "models/person_detection/model.h"
#include "tensorflow/lite/micro/micro_log.h"

ModelConfig ModelManager::GetModelConfig(ModelType type) {
  switch (type) {
    case ModelType::SINE_FLOAT32:
      return {
        .name = "sine_float32",
        .model_data = g_model,
        .model_data_len = g_model_len,
        .quantization = "float32",
        .input_size = 1,
        .output_size = 1
      };
      
    case ModelType::PERSON_DETECTION_INT8:
      return {
        .name = "person_detection_int8",
        .model_data = g_person_detect_model_data,
        .model_data_len = g_person_detect_model_data_len,
        .quantization = "int8",
        .input_size = 96 * 96,  // 96x96 grayscale image
        .output_size = 2        // person / no-person scores
      };
      
    default:
      MicroPrintf("Unknown model type!");
      return {};
  }
}

bool ModelManager::SetupOpResolver(ModelType type, 
                                   tflite::MicroMutableOpResolver<20>* resolver) {
  switch (type) {
    case ModelType::SINE_FLOAT32:
      // Sine model operations
      if (resolver->AddFullyConnected() != kTfLiteOk) return false;
      if (resolver->AddQuantize() != kTfLiteOk) return false;
      if (resolver->AddDequantize() != kTfLiteOk) return false;
      return true;
      
    case ModelType::PERSON_DETECTION_INT8:
      // Person detection operations
      if (resolver->AddConv2D() != kTfLiteOk) return false;
      if (resolver->AddDepthwiseConv2D() != kTfLiteOk) return false;
      if (resolver->AddAveragePool2D() != kTfLiteOk) return false;
      if (resolver->AddReshape() != kTfLiteOk) return false;
      if (resolver->AddSoftmax() != kTfLiteOk) return false;
      if (resolver->AddFullyConnected() != kTfLiteOk) return false;
      return true;
      
    default:
      MicroPrintf("Unknown model type for op resolver!");
      return false;
  }
}