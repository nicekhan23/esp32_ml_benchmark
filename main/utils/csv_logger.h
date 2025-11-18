#ifndef CSV_LOGGER_H_
#define CSV_LOGGER_H_

#include "tensorflow/lite/micro/micro_log.h"

class CSVLogger {
 public:
  static void PrintHeader() {
    MicroPrintf("CSV_HEADER,iteration,model_name,quantization,latency_us,min_us,max_us,avg_us,stddev_us,arena_bytes,free_heap");
  }
  
  static void LogInference(int64_t iteration,
                          const char* model_name,
                          const char* quantization,
                          int64_t latency_us,
                          int64_t min_latency,
                          int64_t max_latency,
                          int64_t avg_latency,
                          float stddev,
                          size_t arena_bytes,
                          size_t free_heap) {
    MicroPrintf("CSV_DATA,%lld,%s,%s,%lld,%lld,%lld,%lld,%.2f,%zu,%zu",
                iteration, model_name, quantization,
                latency_us, min_latency, max_latency, avg_latency,
                stddev, arena_bytes, free_heap);
  }
};

#endif  // CSV_LOGGER_H_