import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer

model = TFAutoModelForCausalLM.from_pretrained("gpt2")

# Save model
tf_model_path = "./gpt2_model_direct"
tf.saved_model.save(model, tf_model_path)
print(f"Hugging Face TF model saved as TensorFlow SavedModel at: {tf_model_path}")

# Convert to TF Lite model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('gpt2.tflite', 'wb') as f:
    f.write(tflite_model)

print("GPT-2 TFLite model created as gpt2.tflite")

# Note -> This code works in Kaggle, but doesn't on Local system.