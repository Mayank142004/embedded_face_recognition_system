import os
import tensorflow as tf
from keras_facenet import FaceNet

print("Loading Keras FaceNet...")
embedder = FaceNet()

# FORCE FIXED BATCH SIZE TO PREVENT 32-BIT OVERFLOW BUGS ON RASPBERRY PI
print("Forcing fixed input shape [1, 160, 160, 3]...")
fixed_input = tf.keras.layers.Input(shape=(160, 160, 3), batch_size=1)
fixed_output = embedder.model(fixed_input)
fixed_model = tf.keras.Model(fixed_input, fixed_output)

print("Converting to TFLite (Float16)...")
converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

os.makedirs('facenet_models', exist_ok=True)
out_path = 'facenet_models/facenet.tflite'
with open(out_path, 'wb') as f:
    f.write(tflite_model)

print(f"Successfully saved Fixed-Shape FP16 FaceNet TFLite model to {out_path}")
