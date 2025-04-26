# quantum_industry_trainer.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import json

from quantum_tokenizer import QuantumTokenizer, QuantumTokenizerLite, QuantumTokenizerPro
from quantum_data_loader import QuantumDataLoader

# ------------------ Utility Functions -------------------

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, os.path.join(path, f"checkpoint_epoch_{epoch}.pt"))
    print(f"[Checkpoint] Saved at epoch {epoch} to {path}")

# ------------------ Trainer Class -------------------

class QuantumModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.loss_fn = config.loss_fn
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)

        # Tokenizer Setup
        self.tokenizer_type = config.tokenizer_type.lower()
        if self.tokenizer_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)
        elif self.tokenizer_type == "bytebpe":
            from tokenizers import ByteLevelBPETokenizer
            self.tokenizer = ByteLevelBPETokenizer()
            self.tokenizer.train(files=config.bpe_files, vocab_size=config.vocab_size, min_frequency=2)
        elif self.tokenizer_type == "quantum_lite":
            self.tokenizer = QuantumTokenizerLite(vocab_size=config.vocab_size)
        elif self.tokenizer_type == "quantum_pro":
            self.tokenizer = QuantumTokenizerPro(vocab_size=config.vocab_size, noise_level=config.noise_level)
        elif self.tokenizer_type == "quantum":
            self.tokenizer = QuantumTokenizer(vocab_size=config.vocab_size, noise_level=config.noise_level)
        else:
            raise ValueError("Unsupported tokenizer type")
        
        self.tokenizer_initialized = False

    def prepare_data(self, file_path, file_type="txt"):
        if file_type == "txt":
            data = QuantumDataLoader.load_text_data(file_path)
        elif file_type == "json":
            data = QuantumDataLoader.load_json_data(file_path)
        elif file_type == "csv":
            data = QuantumDataLoader.load_csv_data(file_path)
        else:
            raise ValueError("Unsupported file type")
        return data

    def initialize_tokenizer(self, texts):
        if hasattr(self.tokenizer, "build_vocab"):
            self.tokenizer.build_vocab(texts)
            self.tokenizer_initialized = True
            print("[Tokenizer] Vocabulary built and initialized.")

    def encode_batch(self, batch_texts):
        if self.tokenizer_type == "huggingface":
            tokens = self.tokenizer(batch_texts, padding='max_length', truncation=True,
                                    max_length=self.config.max_seq_length, return_tensors='pt')
            return tokens.input_ids
        elif self.tokenizer_type == "bytebpe":
            tokens = [self.tokenizer.encode(text).ids for text in batch_texts]
            tokens = self.pad_batch(tokens)
            return torch.tensor(tokens, dtype=torch.long)
        else:
            tokens = [self.tokenizer.encode(text) for text in batch_texts]
            tokens = self.pad_batch(tokens)
            return torch.tensor(tokens, dtype=torch.long)

    def pad_batch(self, batch):
        max_len = self.config.max_seq_length
        return [seq + [0]*(max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in batch]

    def train(self, train_dataset, val_dataset=None):
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size) if val_dataset else None

        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            running_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"[Epoch {epoch}] Training", leave=False)
            for batch in progress_bar:
                inputs = [item['text'] for item in batch]
                labels = [item['label'] for item in batch]

                if not self.tokenizer_initialized and not self.tokenizer_type == "huggingface":
                    self.initialize_tokenizer(inputs)

                inputs_tensor = self.encode_batch(inputs).to(self.device)
                labels_tensor = torch.tensor(labels).to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs_tensor)

                loss = self.loss_fn(outputs, labels_tensor)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_train_loss = running_loss / len(train_loader)
            print(f"[Epoch {epoch}] Avg Training Loss: {avg_train_loss:.4f}")

            if val_loader:
                self.evaluate(val_loader)

            # Learning rate decay
            self.scheduler.step()

            # Save checkpoint
            save_checkpoint(self.model, self.optimizer, epoch, self.config.checkpoint_dir)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Validation]", leave=False):
                inputs = [item['text'] for item in batch]
                labels = [item['label'] for item in batch]

                inputs_tensor = self.encode_batch(inputs).to(self.device)
                labels_tensor = torch.tensor(labels).to(self.device)

                outputs = self.model(inputs_tensor)
                loss = self.loss_fn(outputs, labels_tensor)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels_tensor).sum().item()
                total += labels_tensor.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        print(f"[Validation] Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")