import json
import regex as re
import numpy as np

class CustomGPT2Tokenizer:
    def __init__(self, vocab_file, merges_file, special_tokens_map_file, tokenizer_config_file):
        # Load vocabulary
        with open(vocab_file, 'r', encoding='utf-8') as f:
            # vocab.json maps tokens to their IDs
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        # Load merges
        self.bpe_merges = self._load_bpe_merges(merges_file)

        # Load special tokens
        with open(special_tokens_map_file, 'r', encoding='utf-8') as f:
            special_tokens = json.load(f)
        self.bos_token = special_tokens.get("bos_token", "<|endoftext|>")
        self.eos_token = special_tokens.get("eos_token", "<|endoftext|>")
        self.unk_token = special_tokens.get("unk_token", "<|endoftext|>")

        # GPT-2 uses eos_token as pad_token
        self.pad_token = self.eos_token

        # Load tokenizer config for paramter like max_length
        with open(tokenizer_config_file, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
        self.model_max_length = tokenizer_config.get("model_max_length", 1024)

        # Byte-level BPE specific regex for GPT-2
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # Initialize byte encoder/decoder for byte-level BPE
        # This mapping is crucial for handling bytes 0-255 as individual "characters"
        # for BPE merging without issues with control characters, etc.
        self.bytes_to_unicode = {i: chr(i) for i in range(256)}
        # The specific GPT-2 mapping
        _bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
        _cs = _bs[:]
        n = 0
        for b in range(256):
            if b not in _bs:
                _bs.append(b)
                _cs.append(256 + n)
                n += 1
        self.byte_encoder = dict(zip(_bs, _cs)) # Maps byte value to remapped Unicode char code
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()} # Maps remapped Unicode char code back to byte value

        # The actual mapping used for the initial 'word' in _get_bpe_tokens:
        # Maps byte values (0-255) to their *remapped* Unicode characters for BPE.
        self.byte_to_char_for_bpe = {b: chr(self.byte_encoder[b]) for b in range(256)}

    def _load_bpe_merges(self, merges_file):
        merges = []
        with open(merges_file, 'r', encoding='utf-8') as f:
            f.readline()   # skip the first line which is 'version: 0.2'
            for line in f:
                merges.append(tuple(line.strip().split()))
        return merges
    
    def _get_bpe_tokens(self, token):
        """Tokenizes text using byte-level BPE."""
        word_bytes = token.encode("utf-8")
        # Use the correct mapping to get the BPE-compatible word
        word = ''.join(self.byte_to_char_for_bpe[b] for b in word_bytes)

        # Split the word into individual characters initially
        word_list = list(word)

        while True:
            # Find the best bigram to merge
            best_bigram = None
            min_merge_rank = float('inf')

            # Iterate through the current word_list to find candidate bigrams
            current_pairs = []
            for i in range(len(word_list) - 1):
                current_pairs.append((word_list[i], word_list[i+1]))

            for pair in current_pairs:
                if pair in self.bpe_merges:
                    merge_rank = self.bpe_merges.index(pair)
                    if merge_rank < min_merge_rank:
                        min_merge_rank = merge_rank
                        best_bigram = pair

            if best_bigram is None: # No more merges possible
                break

            first, second = best_bigram
            merged_token = first + second
            print("Processing bigram:", best_bigram, "->", merged_token) # Debugging

            new_word_list = []
            i = 0
            while i < len(word_list):
                if i + 1 < len(word_list) and word_list[i] == first and word_list[i+1] == second:
                    new_word_list.append(merged_token)
                    i += 2 # Skip the second part of the bigram
                else:
                    new_word_list.append(word_list[i])
                    i += 1
            word_list = new_word_list

            # If the word didn't change (e.g., bigram only appeared once and was merged)
            # or if the word becomes a single token, we can break.
            if len(word_list) == 1:
                break

        # Join the final list of tokens back into a string, separated by spaces
        # GPT-2 typically uses space-separated tokens in the vocabulary for subword units
        return ' '.join(word_list)

    def _get_pairs(self, word):
        """Returns a set of pairs of adjacent characters in the word."""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs

    def encode(self, text):
        """Encodes text into token IDs."""
        bpe_tokens = []
        # Use GPT-2's specific regex to pre-tokenize the text
        tokens = re.findall(self.pat, text)
        print("Pre-tokenized tokens:", tokens)

        for token in tokens:
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self._get_bpe_tokens(token).split(' '))
        
        return bpe_tokens
    
    def tokenize(self, text, padding=True, max_length=None):
        input_ids = self.encode(text)
        print("Tokenized input IDs:", input_ids)

        if max_length is None:
            max_length = self.model_max_length
        
        if padding:
            # Create attention mask
            attention_mask = np.ones(len(input_ids), dtype=np.int64)

            # Pad if shorter than max_length
            if len(input_ids) < max_length:
                padding_length = max_length - len(input_ids)
                input_ids.extend([self.encoder[self.pad_token]] * padding_length)
                attention_mask = np.pad(attention_mask, (0, padding_length), mode='constant', constant_values=0)

            # Truncate if longer than max_length
            elif len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
            
            # Convert to numpy arrays
            input_ids_np = np.array(input_ids, dtype=np.int64)
            return {"input_ids": input_ids_np[np.newaxis, :], "attention_mask": attention_mask[np.newaxis, :]}
        
        else:
            return {"input_ids": np.array(input_ids, dtype=np.int64)[np.newaxis, :], "attention_mask": np.ones(len(input_ids), dtype=np.int64)[np.newaxis, :]}
    
    @property
    def vocab_size(self):
        return len(self.encoder)
