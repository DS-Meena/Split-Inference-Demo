import numpy as np

class NumPyEmbedding:
    def __init__(self, token_embeddings_path, position_embeddings_path, max_position_embeddings):
        self.token_embeddings_weights = np.load(token_embeddings_path)
        self.position_embeddings_weights = np.load(position_embeddings_path)
        self.max_position_embeddings = max_position_embeddings
        self.embedding_dim = self.token_embeddings_weights.shape[1]

        print(f"NumPy Embedding: Loaded token weights shape: {self.token_embeddings_weights.shape}")
        print(f"NumPy Embedding: Loaded position weights shape: {self.position_embeddings_weights.shape}")

    def __call__(self, input_ids_tensor):
        # input_ids_tensor is expected to be a 2D numpy array of shape (batch_size, sequence_length)
        if input_ids_tensor.ndim != 2:
            raise ValueError(f"Input IDs tensor must be a 2D numpy array, got shape {input_ids_tensor.shape}")

        batch_size, seq_len = input_ids_tensor.shape

        # 1. Token Embeddings Lookup
        # Perform numpy lookup using the input_ids as indices
        token_embeds = self.token_embeddings_weights[input_ids_tensor]

        # 2. Position Embeddings Lookup
        # Generate position_ids (0 to seq_len - 1)
        position_ids = np.arange(0, seq_len, dtype=np.int64)

        if seq_len > self.max_position_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds max position embeddings {self.max_position_embeddings}")
        
        position_embeds = self.position_embeddings_weights[position_ids]

        # 3. Combine Token and Position Embeddings
        final_embeddings = token_embeds + position_embeds

        return final_embeddings

