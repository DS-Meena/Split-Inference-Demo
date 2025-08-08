import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

MODEL_NAME = "gpt2"
full_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Get configuration details from the full model
vocab_size = tokenizer.vocab_size
embedding_dim = full_model.config.n_embd
max_position_embeddings = full_model.config.max_position_embeddings

print(f"Vocab Size: {vocab_size}, Embedding Dim: {embedding_dim}, Max Position Embeddings: {max_position_embeddings}")

# Extract and Save Token Embeddings as NumPy
token_embeddings_weights = full_model.get_input_embeddings().weight.data.numpy()
np.save("client_app/assets/token_embeddings.npy", token_embeddings_weights)
print(f"Token embeddings saved with shape: {token_embeddings_weights.shape}")

# Extract and Save Position Embeddings as NumPy
position_embeddings_weights = full_model.transformer.wpe.weight.data.numpy()
np.save("client_app/assets/position_embeddings.npy", position_embeddings_weights)
print(f"Position embeddings saved with shape: {position_embeddings_weights.shape}")
