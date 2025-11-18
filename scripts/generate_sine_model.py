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

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the .tflite file
with open('sine_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Model saved: sine_model.tflite ({len(tflite_model)} bytes)")

# Convert to C array
def convert_to_c_array(tflite_model, output_file='model.cpp'):
    with open(output_file, 'w') as f:
        f.write('#include "model.h"\n\n')
        f.write('alignas(8) const unsigned char g_model[] = {\n')
        
        hex_array = [f'0x{byte:02x}' for byte in tflite_model]
        
        # Write 12 bytes per line
        for i in range(0, len(hex_array), 12):
            line = ', '.join(hex_array[i:i+12])
            f.write(f'    {line},\n')
        
        f.write('};\n\n')
        f.write(f'const int g_model_len = {len(tflite_model)};\n')
    
    print(f"C array saved: {output_file}")

convert_to_c_array(tflite_model, 'model.cpp')