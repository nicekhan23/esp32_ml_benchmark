/**
 * @file main_functions.cpp
 * @brief Core benchmarking logic with multi-model support
 */

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "constants.h"
#include "output_handler.h"
#include "csv_logger.h"

#include "esp_timer.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "models/sine/model8.h"
#include "models/sine/model32.h"
#include "models/cnn/model8.h"
#include "models/cnn/model32.h"
#include "models/rnn/model8.h"
#include "models/rnn/model32.h"

namespace {
// Tensor arena for model inference
uint8_t tensor_arena[kTensorArenaSize];

// Memory tracking
size_t heap_before_init = 0;
size_t heap_after_init = 0;
size_t min_free_heap = 0;

// Statistics
constexpr int kMaxLatencyHistory = 100;
int64_t latencies[kMaxLatencyHistory];
int latency_index = 0;
bool warmup_done = false;

// Benchmarking variables
int64_t total_inferences = 0;
int64_t total_latency_us = 0;
int64_t min_latency_us = INT64_MAX;
int64_t max_latency_us = 0;

// TFLite Micro objects
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Model-specific variables
const char* current_model_name = "";
const char* current_quantization = "";
float x_val = 0.0f;  // For sine model

// Model configuration
const unsigned char* model_data = nullptr;
int model_data_len = 0;
}

// Calculate standard deviation
float calculate_stddev(int64_t* values, int count, int64_t mean) {
  if (count <= 1) return 0.0f;
  
  int64_t sum_squared_diff = 0;
  for (int i = 0; i < count; i++) {
    int64_t diff = values[i] - mean;
    sum_squared_diff += diff * diff;
  }
  
  return sqrt((float)sum_squared_diff / (count - 1));
}

// Get model info based on CURRENT_MODEL
void select_model() {
  #if CURRENT_MODEL == MODEL_SINE_FLOAT32
    model_data = g_sine_model_float32;
    model_data_len = g_sine_model_float32_len;
    current_model_name = "sine";
    current_quantization = "float32";
  #elif CURRENT_MODEL == MODEL_CNN_FLOAT32
    model_data = g_cnn_model_float32;
    model_data_len = g_cnn_model_float32_len;
    current_model_name = "cnn";
    current_quantization = "float32";
  #elif CURRENT_MODEL == MODEL_CNN_INT8
    model_data = g_cnn_model_int8;
    model_data_len = g_cnn_model_int8_len;
    current_model_name = "cnn";
    current_quantization = "int8";
  #elif CURRENT_MODEL == MODEL_RNN_FLOAT32
    model_data = g_rnn_model_float32;
    model_data_len = g_rnn_model_float32_len;
    current_model_name = "rnn";
    current_quantization = "float32";
  #elif CURRENT_MODEL == MODEL_RNN_INT8
    model_data = g_rnn_model_int8;
    model_data_len = g_rnn_model_int8_len;
    current_model_name = "rnn";
    current_quantization = "int8";
  #else
    #error "Invalid CURRENT_MODEL selection"
  #endif
}

void setup() {
  heap_before_init = esp_get_free_heap_size();
  tflite::InitializeTarget();

  MicroPrintf("=== ESP32 ML Benchmark Framework ===");
  
  // Select model based on configuration
  select_model();
  MicroPrintf("Selected Model: %s (%s)", current_model_name, current_quantization);
  MicroPrintf("Model size: %d bytes", model_data_len);
  
  OutputHandler::PrintSystemInfo();
  
  // Load model
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema mismatch! Expected %d, got %d",
                TFLITE_SCHEMA_VERSION, model->version());
    return;
  }
  MicroPrintf("Model loaded successfully");
  
  // Set up operations resolver
  // Increase size to accommodate all operations
  static tflite::MicroMutableOpResolver<20> resolver;
  
  // Add operations (add more as needed for your models)
  resolver.AddFullyConnected();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddUnidirectionalSequenceLSTM();
  resolver.AddTanh();
  resolver.AddLogistic();
  resolver.AddMul();
  resolver.AddAdd();
  
  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed!");
    return;
  }
  
  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  // Print tensor info
  MicroPrintf("Input tensor: %d bytes, type=%d", 
              input->bytes, input->type);
  MicroPrintf("Output tensor: %d bytes, type=%d", 
              output->bytes, output->type);
  
  heap_after_init = esp_get_free_heap_size();
  min_free_heap = esp_get_minimum_free_heap_size();
  
  size_t memory_used = heap_before_init - heap_after_init;
  
  MicroPrintf("Memory Analysis:");
  MicroPrintf("  Heap before init: %zu bytes", heap_before_init);
  MicroPrintf("  Heap after init: %zu bytes", heap_after_init);
  MicroPrintf("  Memory used by model: %zu bytes", memory_used);
  MicroPrintf("  Arena used: %zu bytes", interpreter->arena_used_bytes());
  MicroPrintf("  Min free heap ever: %zu bytes", min_free_heap);
  MicroPrintf("Starting benchmark...");
  CSVLogger::PrintHeader();
}

// Prepare input based on model type
void prepare_input() {
  #if CURRENT_MODEL == MODEL_SINE_FLOAT32
    // Sine model: single float input
    x_val += 0.1f;
    if (x_val > 2.0f * 3.14159f) x_val = 0.0f;
    input->data.f[0] = x_val;
    
  #elif CURRENT_MODEL == MODEL_CNN_FLOAT32
    // CNN model: 8x8 image, generate random pattern
    for (int i = 0; i < 64; i++) {
      input->data.f[i] = (float)(rand() % 100) / 100.0f;
    }
    
  #elif CURRENT_MODEL == MODEL_CNN_INT8
    // CNN model int8: 8x8 image
    for (int i = 0; i < 64; i++) {
      input->data.int8[i] = (int8_t)(rand() % 256 - 128);
    }
    
  #elif CURRENT_MODEL == MODEL_RNN_FLOAT32
    // RNN model: sequence of 10 floats
    for (int i = 0; i < 10; i++) {
      input->data.f[i] = (float)(rand() % 100) / 10.0f;
    }
    
  #elif CURRENT_MODEL == MODEL_RNN_INT8
    // RNN model int8: sequence of 10 int8
    for (int i = 0; i < 10; i++) {
      input->data.int8[i] = (int8_t)(rand() % 256 - 128);
    }
  #endif
}

void loop() {
  if (interpreter == nullptr) {
    vTaskDelay(pdMS_TO_TICKS(1000));
    return;
  }
  
  #if CURRENT_MODE == DEMO_MODE_WAVEFORM_CLASSIFICATION
    static int demo_iteration = 0;
    
    // Sample from ADC (external signal from ESP32 #1)
    float samples[kWaveformSamples];
    ADCSampler::Sample(samples, kWaveformSamples);
    
    // Copy to model input
    for(int i = 0; i < kWaveformSamples; i++) {
      input->data.f[i] = samples[i];
    }
    
    // Measure inference latency
    int64_t start_time = esp_timer_get_time();
    TfLiteStatus invoke_status = interpreter->Invoke();
    int64_t latency_us = esp_timer_get_time() - start_time;
    
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed!");
      return;
    }
    
    // Get prediction
    int predicted_class = 0;
    float max_score = output->data.f[0];
    for(int i = 1; i < kNumWaveformClasses; i++) {
      if(output->data.f[i] > max_score) {
        max_score = output->data.f[i];
        predicted_class = i;
      }
    }
    
    const char* class_names[] = {"SINE", "SQUARE", "TRIANGLE"};
    
    MicroPrintf("=== Inference %d ===", demo_iteration);
    MicroPrintf("Predicted: %s | Confidence: %.2f%% | Latency: %lld us",
                class_names[predicted_class],
                max_score * 100.0f,
                latency_us);
    
    // Log for benchmarking
    CSVLogger::LogInference(
        demo_iteration, current_model_name, current_quantization,
        latency_us, min_latency_us, max_latency_us, latency_us,
        0.0f, interpreter->arena_used_bytes(), esp_get_free_heap_size()
    );
    
    demo_iteration++;
    vTaskDelay(pdMS_TO_TICKS(500));

  #else
  // Prepare input
  prepare_input();
  
  // Measure inference time
  int64_t start_time = esp_timer_get_time();
  TfLiteStatus invoke_status = interpreter->Invoke();
  int64_t end_time = esp_timer_get_time();
  int64_t latency_us = end_time - start_time;
  
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed!");
    return;
  }
  
  // Update statistics
  total_inferences++;
  total_latency_us += latency_us;
  if (latency_us < min_latency_us) min_latency_us = latency_us;
  if (latency_us > max_latency_us) max_latency_us = latency_us;
  
  // Store latency for stddev
  latencies[latency_index] = latency_us;
  latency_index = (latency_index + 1) % kMaxLatencyHistory;
  
  // Warmup phase
  if (total_inferences == kWarmupInferences) {
    MicroPrintf("Warmup complete, starting measurements...");
    warmup_done = true;
    total_inferences = 0;
    total_latency_us = 0;
    min_latency_us = INT64_MAX;
    max_latency_us = 0;
  }
  
  // Print results every 10 inferences
  if (warmup_done && total_inferences > 0 && total_inferences % 10 == 0) {
    int64_t average_latency = total_latency_us / total_inferences;
    int sample_count = (total_inferences < kMaxLatencyHistory) ? total_inferences : kMaxLatencyHistory;
    float stddev = calculate_stddev(latencies, sample_count, average_latency);
    
    MicroPrintf("=== Iteration %lld ===", total_inferences);
    MicroPrintf("Latency: cur=%lld us, avg=%lld us, min=%lld us, max=%lld us, stddev=%.2f us",
                latency_us, average_latency, min_latency_us, max_latency_us, stddev);

    CSVLogger::LogInference(
        total_inferences, current_model_name, current_quantization,
        latency_us, min_latency_us, max_latency_us, average_latency,
        stddev, interpreter->arena_used_bytes(), esp_get_free_heap_size()
    );
    
    MicroPrintf("");
    #endif
  }
  
  // Print summary every 100 inferences
  if (warmup_done && total_inferences % 100 == 0) {
    int64_t avg_latency = total_latency_us / total_inferences;
    float stddev = calculate_stddev(latencies, kMaxLatencyHistory, avg_latency);
    
    OutputHandler::PrintBenchmarkResult(
        current_model_name,
        avg_latency,
        interpreter->arena_used_bytes()
    );
    
    MicroPrintf("Statistics over last 100 inferences:");
    MicroPrintf("  Min: %lld us, Max: %lld us, StdDev: %.2f us", 
                min_latency_us, max_latency_us, stddev);
  }

  vTaskDelay(pdMS_TO_TICKS(kDelayBetweenTests));
}