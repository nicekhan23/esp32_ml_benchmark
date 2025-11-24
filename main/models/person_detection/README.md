# Person Detection Model

## Overview
- **Task**: Binary classification (person / no person)
- **Input**: 96x96 grayscale image (9216 bytes)
- **Output**: 2 values (person score, no-person score)
- **Quantization**: int8
- **Size**: ~250KB

## Model Architecture
- MobileNetV1 based
- Depthwise separable convolutions
- Optimized for microcontrollers

## Operations Used
- CONV_2D
- DEPTHWISE_CONV_2D
- AVERAGE_POOL_2D
- RESHAPE
- SOFTMAX
- FULLY_CONNECTED

## Expected Performance (estimate)
- Latency: 40-80ms per inference
- Memory: ~100KB arena

## Source
TensorFlow Lite Micro examples:
https://github.com/espressif/esp-tflite-micro/tree/master/examples/person_detection