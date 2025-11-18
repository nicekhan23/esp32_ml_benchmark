#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "main_functions.h"
#include "model.h"
#include "constants.h"
#include "output_handler.h"

#include "esp_timer.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"  // Add this for FreeRTOS functions
#include "freertos/task.h"      // Add this for vTaskDelay

namespace {
// Tensor arena for model inference
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Benchmarking variables
int64_t total_inferences = 0;
int64_t total_latency_us = 0;
bool benchmark_initialized = false;

// TFLite Micro objects (commented out for now since we're not using them yet)
// tflite::MicroMutableOpResolver<10> resolver;
// tflite::MicroInterpreter* interpreter = nullptr;
}

void setup() {
  // Initialize TFLite Micro system
  tflite::InitializeTarget();
  
  MicroPrintf("=== ESP32 ML Benchmark Framework ===");
  OutputHandler::PrintSystemInfo();
  
  // Note: For now, we'll benchmark without an actual model
  // In the next step, we'll integrate a real model from examples
  MicroPrintf("Benchmark framework initialized");
  MicroPrintf("Ready to integrate TFLite models");
  
  benchmark_initialized = true;
}

void loop() {
  if (!benchmark_initialized) {
    return;
  }

  // Simple latency benchmark without actual model
  int64_t start_time = esp_timer_get_time();
  
  // Simulate inference work (replace with actual model later)
  volatile int dummy_work = 0;
  for (int i = 0; i < 500; i++) {
    dummy_work += i * i;
  }
  
  int64_t end_time = esp_timer_get_time();
  int64_t latency_us = end_time - start_time;
  
  total_inferences++;
  total_latency_us += latency_us;
  
  // Print results every 10 inferences
  if (total_inferences % 10 == 0) {
    int64_t average_latency = total_latency_us / total_inferences;
    MicroPrintf("Iteration: %lld, Current: %lld us, Average: %lld us", 
                total_inferences, latency_us, average_latency);
  }
  
  // Wait before next iteration
  vTaskDelay(pdMS_TO_TICKS(100));
}