import numpy as np
import tensorflow as tf

# Generate simple sequence classification data
# Task: Detect if sequence is increasing, decreasing, or random
def generate_sequence_data(num_samples=2000, seq_length=10):
    X = []
    y = []
    
    for _ in range(num_samples):
        seq_type = np.random.randint(0, 3)
        
        if seq_type == 0:  # Increasing sequence
            start = np.random.uniform(0, 5)
            seq = np.linspace(start, start + 5, seq_length)
            seq += np.random.normal(0, 0.2, seq_length)
        elif seq_type == 1:  # Decreasing sequence
            start = np.random.uniform(5, 10)
            seq = np.linspace(start, start - 5, seq_length)
            seq += np.random.normal(0, 0.2, seq_length)
        else:  # Random sequence
            seq = np.random.uniform(0, 10, seq_length)
        
        X.append(seq)
        y.append(seq_type)
    
    X = np.array(X).reshape(-1, seq_length, 1).astype(np.float32)
    y = tf.keras.utils.to_categorical(y, 3).astype(np.float32)
    
    return X, y

# Generate data
X_train, y_train = generate_sequence_data(2000)
X_test, y_test = generate_sequence_data(200)

# Create RNN model with LSTM
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(16, input_shape=(10, 1), return_sequences=False),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)

print("RNN Model trained successfully!")

# Convert to TFLite (float32)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model_float32 = converter.convert()

with open('rnn_model_float32.tflite', 'wb') as f:
    f.write(tflite_model_float32)
print(f"Float32 model saved: {len(tflite_model_float32)} bytes")

# Convert to TFLite (int8)
def representative_dataset():
    for i in range(100):
        yield [X_train[i:i+1]]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

tflite_model_int8 = converter_int8.convert()

with open('rnn_model_int8.tflite', 'wb') as f:
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

convert_to_c_array(tflite_model_float32, 'g_rnn_model_float32', 'rnn_model_float32.cpp')
convert_to_c_array(tflite_model_int8, 'g_rnn_model_int8', 'rnn_model_int8.cpp')