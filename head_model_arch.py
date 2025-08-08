import torch
import torch.nn as nn

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
