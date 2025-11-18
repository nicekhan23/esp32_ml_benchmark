# System Architecture

## Overview
```
┌─────────────┐
│   main.cpp  │  Entry point
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ main_functions.cpp  │  Core benchmarking logic
│  - setup()          │
│  - loop()           │
└──────┬──────────────┘
       │
       ├──► TFLite Micro (model inference)
       ├──► Memory Tracker (heap monitoring)
       ├──► CSV Logger (data collection)
       └──► Output Handler (display results)
```

## Data Flow
```
1. Initialization (setup())
   ├─ Load TFLite model from g_model[]
   ├─ Allocate tensor arena
   ├─ Measure initial memory
   └─ Initialize statistics

2. Inference Loop (loop())
   ├─ Prepare input tensor
   ├─ Start timer
   ├─ interpreter->Invoke()
   ├─ Stop timer
   ├─ Update statistics
   ├─ Log to CSV
   └─ Wait for next iteration

3. Output
   ├─ Serial monitor (human readable)
   └─ CSV format (machine parseable)
```

## Key Components

### 1. Model Management
- **Location**: `main/models/`
- **Format**: C arrays (`g_model[]`)
- **Loading**: `tflite::GetModel()`

### 2. Inference Engine
- **Library**: TensorFlow Lite Micro
- **Components**:
  - `MicroInterpreter` - runs inference
  - `MicroMutableOpResolver` - registers ops
  - Tensor arena - working memory

### 3. Measurement Systems

#### Latency
- **Method**: `esp_timer_get_time()` before/after invoke
- **Unit**: microseconds (us)
- **Metrics**: min, max, avg, stddev

#### Memory
- **Methods**:
  - `esp_get_free_heap_size()` - current free heap
  - `interpreter->arena_used_bytes()` - tensor arena
- **Tracked**: before init, after init, during runtime

#### Energy (Future)
- **Hardware**: INA219/INA260 current sensor
- **Calculation**: E = P × t

### 4. Data Collection
- **CSV Logger**: Structured output
- **Format**: `iteration,model,quantization,latency,...`
- **Storage**: Serial output → file

## Memory Layout
```
ESP32 RAM (520KB total)
├─ System (FreeRTOS, WiFi stack, etc.)
├─ Model code (.text)
├─ Global variables
├─ Tensor arena (kTensorArenaSize)
└─ Free heap (monitored)
```

## Configuration

All constants in `constants.h`:
- `kTensorArenaSize` - memory for inference
- `kInferencesPerTest` - iterations per benchmark
- `kDelayBetweenTests` - pause between inferences

## Extension Points

To add a new model:
1. Convert to `.tflite` → C array
2. Create `models/<name>/model.cpp`
3. Update op resolver with required ops
4. Add model switch logic

To add new metrics:
1. Add measurement code in `loop()`
2. Update `CSVLogger` format
3. Update analysis scripts