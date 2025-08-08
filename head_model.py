import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class CustomGPT2Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_position_embeddings):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(max_position_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, input_ids):
        # input_ids: shape (batch_size, sequence_length)

        # Generate token embeddings
        token_embeds = self.token_embeddings(input_ids)  # shape (batch_size, sequence_length, embedding_dim)

        # Generate positional embeddings
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_embeds = self.position_embeddings(position_ids)  # shape (sequence_length, embedding_dim)

        # Combine token and positional embeddings
        final_embeddings = token_embeds + position_embeds
        return final_embeddings

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
