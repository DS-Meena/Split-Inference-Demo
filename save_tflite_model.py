import tensorflow as tf
from transformers import TFGPT2LMHeadModel

# Save the model in a directory ------------
model = TFGPT2LMHeadModel.from_pretrained("gpt2-large")

saved_model_dir = "./saved_model"

tf.saved_model.save(model, saved_model_dir)
print(f"Hugging Face TF model saved as TensorFlow SavedModel at: {saved_model_dir}")

# Convert to TF Lite model ------------------
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open('sahane.tflite', 'wb') as f:
    f.write(tflite_model)
