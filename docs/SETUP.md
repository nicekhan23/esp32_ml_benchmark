# Setup Guide

## Hardware Requirements

- ESP32 development board (ESP32-DevKitC or similar)
- USB cable
- (Optional) INA219 current sensor for energy measurement
- (Optional) Breadboard and jumper wires

## Software Requirements

- ESP-IDF v5.0+ ([installation guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/))
- Python 3.8+
- TensorFlow 2.x (for model generation)

## Installation Steps

### 1. Install ESP-IDF
```bash
# Follow official guide
https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/

# Verify installation
idf.py --version
```

### 2. Clone Project
```bash
git clone https://github.com/nicekhan23/esp32_ml_benchmark
cd esp32-ml-benchmark
```

### 3. Install Python Dependencies
```bash
pip install tensorflow numpy matplotlib pandas
```

### 4. Build and Flash
```bash
# Configure (first time only)
idf.py set-target esp32

# Build
idf.py build

# Flash and monitor
idf.py flash monitor

# To exit monitor: Ctrl+]
```

## Troubleshooting

### Build Errors

**"TFLite headers not found"**
- Ensure ESP-IDF is properly installed
- Check `CMakeLists.txt` includes TFLite component

**"Insufficient memory"**
- Increase `kTensorArenaSize` in `constants.h`
- Reduce model size

### Runtime Errors

**"AllocateTensors() failed"**
- Arena too small, increase `kTensorArenaSize`

**"Invoke failed"**
- Check op resolver has all required ops
- Verify model compatibility

## Hardware Setup (Energy Measurement)
```
ESP32 VIN ──┬── INA219 V+ ──── Power Supply (+5V)
            │
            └── INA219 V-
                  │
ESP32 GND ────────┴────────── Power Supply GND

ESP32 SDA ──────── INA219 SDA
ESP32 SCL ──────── INA219 SCL
```

Configuration in code (future):
```cpp
#define ENABLE_ENERGY_MEASUREMENT 1
```

## Collecting Results
```bash
# Save output to file
idf.py monitor | tee results/raw/run_$(date +%Y%m%d_%H%M%S).txt

# Extract CSV
grep "CSV_" results/raw/run_*.txt > results/processed/data.csv
```