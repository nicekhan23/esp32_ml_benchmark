/**
 * @file main_functions.cpp
 * @brief Core benchmarking logic for ESP32 ML inference
 * 
 * This file implements the main setup() and loop() functions that:
 * - Initialize TensorFlow Lite Micro interpreter
 * - Load and run ML models
 * - Measure inference latency and memory usage
 * - Log results in CSV format for analysis
 * 
 * @see docs/ARCHITECTURE.md for system design
 * @see docs/BENCHMARKING.md for usage instructions
 * 
 * @author Darkhan Zhanibekuly
 * @date 2025-11-18
 */

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "model.h"
#include "constants.h"
#include "output_handler.h"
#include "csv_logger.h"

#include "esp_timer.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace {
// Tensor arena for model inference
uint8_t tensor_arena[kTensorArenaSize];

// Memory tracking
size_t heap_before_init = 0;
size_t heap_after_init = 0;
size_t min_free_heap = 0;

// Enhanced statistics
constexpr int kWarmupInferences = 10;
int64_t latencies[100];  // Store last 100 latencies for stddev calculation
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

// For hello_world sine model
float x_val = 0.0f;
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

/**
 * @brief Initialize benchmarking framework and load ML model
 * 
 * Steps performed:
 * 1. Measure baseline memory usage
 * 2. Load TFLite model from g_model array
 * 3. Set up operation resolver with required ops
 * 4. Allocate tensor arena
 * 5. Prepare input/output tensor pointers
 * 6. Print initialization summary
 * 
 * @note Called once at startup by app_main()
 * @see loop() for inference execution
 */
void setup() {
  heap_before_init = esp_get_free_heap_size();
  
  tflite::InitializeTarget();

  
  MicroPrintf("=== ESP32 ML Benchmark Framework ===");
  OutputHandler::PrintSystemInfo();
  
  // Load model
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version %d doesn't match supported version %d",
                model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  MicroPrintf("Model loaded successfully");
  
  // Set up operations resolver
  static tflite::MicroMutableOpResolver<10> resolver;
  
  // Add operations used by hello_world model
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    MicroPrintf("Failed to add FullyConnected op");
    return;
  }
  if (resolver.AddQuantize() != kTfLiteOk) {
    MicroPrintf("Failed to add Quantize op");
    return;
  }
  if (resolver.AddDequantize() != kTfLiteOk) {
    MicroPrintf("Failed to add Dequantize op");
    return;
  }
  
  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }
  
  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);
  
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

/**
 * @brief Execute one model inference and collect metrics
 * 
 * Measurements collected:
 * - Inference latency (microseconds)
 * - Min/max/average latency
 * - Standard deviation
 * - Memory usage
 * 
 * @note Called repeatedly in infinite loop
 * @see CSVLogger for data format
 */
void loop() {
  if (interpreter == nullptr) {
    vTaskDelay(pdMS_TO_TICKS(1000));
    return;
  }
  
  // Prepare input (increment x for sine wave)
  x_val += 0.1f;
  if (x_val > 2.0f * 3.14159f) x_val = 0.0f;
  
  input->data.f[0] = x_val;
  
  // Measure inference time
  int64_t start_time = esp_timer_get_time();
  
  TfLiteStatus invoke_status = interpreter->Invoke();
  
  int64_t end_time = esp_timer_get_time();
  int64_t latency_us = end_time - start_time;
  
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }
  
  // Update statistics
  total_inferences++;
  total_latency_us += latency_us;
  if (latency_us < min_latency_us) min_latency_us = latency_us;
  if (latency_us > max_latency_us) max_latency_us = latency_us;
  
  // Store latency for stddev calculation
  latencies[latency_index] = latency_us;
  latency_index = (latency_index + 1) % 100;
  
  // Skip warmup inferences
  if (total_inferences == kWarmupInferences) {
    MicroPrintf("Warmup complete, starting measurements...");
    warmup_done = true;
    total_inferences = 0;
    total_latency_us = 0;
    min_latency_us = INT64_MAX;
    max_latency_us = 0;
  }
  
  // Print results every 10 inferences (after warmup)
  if (warmup_done && total_inferences > 0 && total_inferences % 10 == 0) {
    int64_t average_latency = total_latency_us / total_inferences;
    float y_val = output->data.f[0];
    
    // Calculate standard deviation
    int sample_count = (total_inferences < 100) ? total_inferences : 100;
    float stddev = calculate_stddev(latencies, sample_count, average_latency);
    
    MicroPrintf("=== Iteration %lld ===", total_inferences);
    MicroPrintf("Inference: x=%.2f -> y=%.4f", x_val, y_val);
    MicroPrintf("Latency: cur=%lld us, avg=%lld us, min=%lld us, max=%lld us, stddev=%.2f us",
                latency_us, average_latency, min_latency_us, max_latency_us, stddev);

    CSVLogger::LogInference(
        total_inferences, "sine_model", "float32",
        latency_us, min_latency_us, max_latency_us, average_latency,
        stddev, interpreter->arena_used_bytes(), esp_get_free_heap_size()
    );
    
    OutputHandler::PrintSystemInfo();
    MicroPrintf("");  // Blank line for readability
  }
  
  // Print final summary every 100 inferences
  if (warmup_done && total_inferences % 100 == 0) {
    int64_t avg_latency = total_latency_us / total_inferences;
    float stddev = calculate_stddev(latencies, 100, avg_latency);
    
    OutputHandler::PrintBenchmarkResult(
        "sine_model_float32",
        avg_latency,
        interpreter->arena_used_bytes()
    );
    
    MicroPrintf("Statistics over last 100 inferences:");
    MicroPrintf("  Min: %lld us, Max: %lld us, StdDev: %.2f us", 
                min_latency_us, max_latency_us, stddev);
  }

  vTaskDelay(pdMS_TO_TICKS(kDelayBetweenTests));
}