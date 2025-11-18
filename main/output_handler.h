#ifndef OUTPUT_HANDLER_H_
#define OUTPUT_HANDLER_H_

#include "tensorflow/lite/micro/micro_log.h"
#include "esp_system.h"  // Add this for heap functions

// Simple output handler for benchmarking results
class OutputHandler {
 public:
  static void PrintBenchmarkResult(const char* model_name, 
                                  int64_t average_latency_us,
                                  size_t memory_usage) {
    MicroPrintf("BENCHMARK: %s", model_name);
    MicroPrintf("  Average latency: %lld us", average_latency_us);
    MicroPrintf("  Memory usage: %zu bytes", memory_usage);
    MicroPrintf("  ---");
  }
  
  static void PrintSystemInfo() {
    MicroPrintf("System Info:");
    MicroPrintf("  Free Heap: %zu bytes", esp_get_free_heap_size());
    MicroPrintf("  Minimum Free Heap: %zu bytes", esp_get_minimum_free_heap_size());
  }
};

#endif  // OUTPUT_HANDLER_H_