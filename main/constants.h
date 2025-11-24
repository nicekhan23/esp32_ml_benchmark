#ifndef CONSTANTS_H_
#define CONSTANTS_H_

// Model selection defines
#define MODEL_SINE_FLOAT32 1
#define MODEL_SINE_INT8 2
#define MODEL_CNN_FLOAT32 3
#define MODEL_CNN_INT8 4
#define MODEL_RNN_FLOAT32 5
#define MODEL_RNN_INT8 6

// Select which model to benchmark
// Change this to test different models
#define CURRENT_MODEL MODEL_SINE_FLOAT32

// Memory allocation for models
// Adjust based on model size requirements
constexpr int kTensorArenaSize = 20 * 1024;  // 20KB arena (increased for CNN/RNN)

// Benchmarking constants
constexpr int kWarmupInferences = 10;      // Warmup runs before measurement
constexpr int kInferencesPerTest = 100;    // Number of inferences per benchmark
constexpr int kDelayBetweenTests = 100;    // ms between inferences

// Model-specific input sizes
constexpr int kSineInputSize = 1;
constexpr int kCNNInputHeight = 8;
constexpr int kCNNInputWidth = 8;
constexpr int kCNNInputChannels = 1;
constexpr int kRNNSequenceLength = 10;
constexpr int kRNNFeatureSize = 1;

#endif  // CONSTANTS_H_