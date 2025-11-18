# ESP32 ML Benchmark Framework

Systematic benchmarking suite for evaluating TinyML models on ESP32 microcontrollers.

## 📋 Project Overview

This thesis project measures and compares:
- Inference latency
- Memory usage (Flash/RAM)
- Energy consumption
- Model types: CNN, RNN, Fully Connected
- Quantization: float32, int8

## 🎯 Objectives

- O1: Benchmarking suite for ESP32 AI inference
- O2: Measure KPIs (latency, memory, energy)
- O3: Compare model types and quantization
- O4: Open dataset and deployment guidelines

## 🚀 Quick Start
```bash
# Clone and build
git clone <repo>
cd esp32-ml-benchmark
idf.py build flash monitor
```

See [docs/SETUP.md](docs/SETUP.md) for detailed instructions.

## 📊 Current Status

- ✅ Framework initialized
- ✅ Sine model (float32) working
- ✅ Latency measurement
- ✅ Memory tracking
- ⏳ CNN model integration
- ⏳ Energy measurement
- ⏳ Int8 quantization

## 📁 Project Structure
```
main/
├── main.cpp              - Entry point
├── main_functions.cpp    - Benchmark core logic
├── models/               - TFLite models
├── utils/                - Helper classes
└── constants.h           - Configuration

docs/                     - Documentation
scripts/                  - Analysis tools
results/                  - Benchmark data
```

## 📖 Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design
- [Setup Guide](docs/SETUP.md) - Hardware/software setup
- [Models](docs/MODELS.md) - Model specifications
- [Benchmarking](docs/BENCHMARKING.md) - How to run tests
- [API Reference](docs/API.md) - Code documentation

## 📈 Results

Latest benchmark results: [docs/RESULTS.md](docs/RESULTS.md)

## 🤝 Contributing

This is a thesis project. For questions: [your.email@university.edu]

## 📝 License

MIT License (or your university's requirement)