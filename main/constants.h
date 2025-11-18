#ifndef CONSTANTS_H_
#define CONSTANTS_H_

// Constants for your benchmarking
constexpr int kTensorArenaSize = 10 * 1024;  // 10KB arena for models

// Benchmarking constants
constexpr int kInferencesPerTest = 100;
constexpr int kDelayBetweenTests = 100;  // ms

#endif  // CONSTANTS_H_