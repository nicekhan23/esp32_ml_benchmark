import numpy as np
import tensorflow as tf

# Generate training data
X = np.random.uniform(0, 2*np.pi, 1000).astype(np.float32)
y = np.sin(X).astype(np.float32)

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, verbose=0)

print("Model trained successfully!")

# Convert to TFLite (float32)
converter_float32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model_float32 = converter_float32.convert()

with open('sine_model_float32.tflite', 'wb') as f:
    f.write(tflite_model_float32)
print(f"Float32 model saved: {len(tflite_model_float32)} bytes")

# Convert to TFLite (int8)
def representative_dataset():
    for i in range(100):
        yield [X[i:i+1]]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

tflite_model_int8 = converter_int8.convert()

with open('sine_model_int8.tflite', 'wb') as f:
    f.write(tflite_model_int8)
print(f"Int8 model saved: {len(tflite_model_int8)} bytes")

# Convert to C arrays
def convert_to_c_array(tflite_model, var_name, output_file):
    with open(output_file, 'w') as f:
        f.write('#include "model.h"\n\n')
        f.write(f'alignas(8) const unsigned char {var_name}[] = {{\n')
        
        hex_array = [f'0x{byte:02x}' for byte in tflite_model]
        
        for i in range(0, len(hex_array), 12):
            line = ', '.join(hex_array[i:i+12])
            f.write(f'    {line},\n')
        
        f.write('};\n\n')
        f.write(f'const int {var_name}_len = {len(tflite_model)};\n')
    
    print(f"C array saved: {output_file}")

convert_to_c_array(tflite_model_float32, 'g_sine_model_float32', 'sine_model_float32.cpp')
convert_to_c_array(tflite_model_int8, 'g_sine_model_int8', 'sine_model_int8.cpp')