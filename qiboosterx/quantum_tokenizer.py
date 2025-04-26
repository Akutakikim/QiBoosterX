#quantum_tokenizer.py
import os
import re
import json
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from quantum_data_loader import QuantumDataLoader

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]

# ------------------- Helper Functions -------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def split_words(text):
    return text.split()

def prepare_data(file_path, file_type="txt"):
    if file_type == "txt":
        return QuantumDataLoader.load_text_data(file_path)
    elif file_type == "json":
        return QuantumDataLoader.load_json_data(file_path)
    elif file_type == "csv":
        return QuantumDataLoader.load_csv_data(file_path)
    else:
        raise ValueError("Unsupported file type")

# ------------------- Base Tokenizer Class -------------------

class BaseQuantumTokenizer:
    def __init__(self, vocab_size=5000, max_seq_length=512):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.token_to_id = {}
        self.id_to_token = {}
        self.tokenizer_initialized = False

    def build_vocab(self, texts, preprocess=False):
        counter = Counter()
        for text in texts:
            if preprocess:
                text = clean_text(text)
                words = split_words(text)
            else:
                words = text.split()
            counter.update(words)

        vocab_tokens = SPECIAL_TOKENS + [word for word, _ in counter.most_common(self.vocab_size - len(SPECIAL_TOKENS))]
        self.token_to_id = {word: i for i, word in enumerate(vocab_tokens)}
        self.id_to_token = {i: word for word, i in self.token_to_id.items()}
        self.tokenizer_initialized = True
        print(f"[Tokenizer] Vocabulary built with size {len(self.token_to_id)}")

    def encode(self, text):
        raise NotImplementedError

    def decode(self, token_ids):
        return " ".join([self.id_to_token.get(i, "<unk>") for i in token_ids])

    def pad_sequence(self, token_ids, pad_to_length=None):
        pad_id = self.token_to_id.get("<pad>", 0)
        if pad_to_length is None:
            pad_to_length = self.max_seq_length
        return token_ids + [pad_id] * max(0, (pad_to_length - len(token_ids)))

    def save_tokenizer(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "vocab.json"), "w") as f:
            json.dump(self.token_to_id, f)
        print(f"[Tokenizer] Saved to {path}")

    def load_tokenizer(self, path):
        with open(os.path.join(path, "vocab.json"), "r") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.tokenizer_initialized = True
        print(f"[Tokenizer] Loaded from {path}")

    @classmethod
    def from_pretrained(cls, path):
        instance = cls()
        instance.load_tokenizer(path)
        return instance

    def __call__(self, text):
        return self.encode(text)

# ------------------- Tokenizer Lite -------------------

class QuantumTokenizerLite(BaseQuantumTokenizer):
    def encode(self, text):
        if not self.tokenizer_initialized:
            raise ValueError("Tokenizer not initialized. Build vocab first.")
        token_ids = [self.token_to_id.get(word, self.token_to_id["<unk>"]) for word in text.split()]
        return token_ids[:self.max_seq_length]

# ------------------- Tokenizer Pro -------------------

class QuantumTokenizerPro(BaseQuantumTokenizer):
    def __init__(self, vocab_size=5000, max_seq_length=512, noise_level=0.02):
        super().__init__(vocab_size, max_seq_length)
        self.noise_level = noise_level

    def encode(self, text):
        if not self.tokenizer_initialized:
            raise ValueError("Tokenizer not initialized. Build vocab first.")
        words = split_words(clean_text(text))
        token_ids = []
        for word in words:
            token_id = self.token_to_id.get(word, self.token_to_id["<unk>"])
            noise = int(np.random.normal(0, self.noise_level * 10))  # simulate light distortion
            token_ids.append((token_id + noise) % len(self.token_to_id))
        return token_ids[:self.max_seq_length]

# ------------------- Quantum Tokenizer (with randomness) -------------------

class QuantumTokenizer(BaseQuantumTokenizer):
    def __init__(self, vocab_size=5000, max_seq_length=512, noise_level=0.05):
        super().__init__(vocab_size, max_seq_length)
        self.noise_level = noise_level

    def quantum_tokenize(self, text):
        words = text.split()
        token_ids = []
        for word in words:
            token_id = self.token_to_id.get(word, self.token_to_id["<unk>"])
            quantum_shift = np.random.normal(0, self.noise_level)
            adjusted_id = int(token_id + quantum_shift) % len(self.token_to_id)
            token_ids.append(adjusted_id)
        return token_ids

    def encode(self, text):
        if not self.tokenizer_initialized:
            raise ValueError("Tokenizer not initialized. Build vocab first.")
        return self.quantum_tokenize(text)[:self.max_seq_length]
