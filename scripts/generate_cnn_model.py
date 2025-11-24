import numpy as np
import tensorflow as tf

# Generate simple 2D pattern recognition data
# Patterns: horizontal line, vertical line, diagonal, cross
def generate_pattern_data(num_samples=1000):
    X = []
    y = []
    
    for _ in range(num_samples):
        pattern_type = np.random.randint(0, 4)
        img = np.zeros((8, 8), dtype=np.float32)
        
        if pattern_type == 0:  # Horizontal line
            row = np.random.randint(2, 6)
            img[row, 2:6] = 1.0
        elif pattern_type == 1:  # Vertical line
            col = np.random.randint(2, 6)
            img[2:6, col] = 1.0
        elif pattern_type == 2:  # Diagonal
            for i in range(2, 6):
                img[i, i] = 1.0
        else:  # Cross
            img[4, 2:6] = 1.0
            img[2:6, 4] = 1.0
        
        # Add noise
        noise = np.random.normal(0, 0.1, (8, 8))
        img = np.clip(img + noise, 0, 1)
        
        X.append(img)
        y.append(pattern_type)
    
    X = np.array(X).reshape(-1, 8, 8, 1).astype(np.float32)
    y = tf.keras.utils.to_categorical(y, 4).astype(np.float32)
    
    return X, y

# Generate data
X_train, y_train = generate_pattern_data(2000)
X_test, y_test = generate_pattern_data(200)

# Create CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

print("CNN Model trained successfully!")

# Convert to TFLite (float32)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model_float32 = converter.convert()

with open('cnn_model_float32.tflite', 'wb') as f:
    f.write(tflite_model_float32)
print(f"Float32 model saved: {len(tflite_model_float32)} bytes")

# Convert to TFLite (int8)
def representative_dataset():
    for i in range(100):
        yield [X_train[i:i+1]]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

tflite_model_int8 = converter_int8.convert()

with open('cnn_model_int8.tflite', 'wb') as f:
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

convert_to_c_array(tflite_model_float32, 'g_cnn_model_float32', 'cnn_model_float32.cpp')
convert_to_c_array(tflite_model_int8, 'g_cnn_model_int8', 'cnn_model_int8.cpp')