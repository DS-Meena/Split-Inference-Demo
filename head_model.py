import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from head_model_arch import CustomGPT2Embedding

# SAVE THE HEAD MODEL

MODEL_NAME = "gpt2"
full_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Get configuration details from the full model
vocab_size = tokenizer.vocab_size
embedding_dim = full_model.config.n_embd
max_position_embeddings = full_model.config.max_position_embeddings

print(f"Vocab Size: {vocab_size}, Embedding Dim: {embedding_dim}, Max Position Embeddings: {max_position_embeddings}")

# instantiate the custom embedding model
custom_embedding_model = CustomGPT2Embedding(vocab_size, embedding_dim, max_position_embeddings)

# Load the weights from the full model's state_dict
custom_embedding_model.token_embeddings.load_state_dict(
    full_model.get_input_embeddings().state_dict()
)
custom_embedding_model.position_embeddings.load_state_dict(
    full_model.transformer.wpe.state_dict()
)

# Save the state dictionary of the custom embedding module (smaller file size)
torch.save(custom_embedding_model.state_dict(), "gpt2_head.pth")

print("Custom GPT-2 embedding model saved successfully.")
