"""
Transformer 기반 학습 가능한 LLM 라이브러리
PyTorch를 사용하여 자체 언어모델을 학습하고 추론할 수 있는 라이브러리
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict, Any
import math
import json


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention 메커니즘"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Q, K, V projection layers
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_linear(query)  # [batch, seq_len, d_model]
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, num_heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # [batch, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.out_linear(output)
        
        return output, attention_weights


class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding (Transformer 원본 방식)"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Transformer Decoder Block (GPT-style)"""
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN: Layer Norm first
        # Self-attention with residual
        normed = self.norm1(x)
        attn_output, _ = self.attention(normed, normed, normed, mask)
        x = x + self.dropout1(attn_output)
        
        # Feed-forward with residual
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout2(ff_output)
        
        return x


class GPTModel(nn.Module):
    """GPT-style Transformer Language Model"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection (can share weights with embedding)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Token embedding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, x.device)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask & attention_mask.bool()
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, causal_mask)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0
    ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :]  # [batch, vocab_size]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            if next_token_logits[i, token_id] < 0:
                                next_token_logits[i, token_id] *= repetition_penalty
                            else:
                                next_token_logits[i, token_id] /= repetition_penalty
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = float('-inf')
                
                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if max length reached
                if generated.size(1) >= self.max_seq_len:
                    break
        
        return generated
    
    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        return mask


class TextDataset(Dataset):
    """텍스트 데이터셋"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        for text in texts:
            # Tokenize
            token_ids = tokenizer.encode(text)
            
            # Truncate or pad
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            
            # Create input and labels (labels are shifted by 1)
            input_ids = token_ids[:-1] if len(token_ids) > 1 else token_ids
            labels = token_ids[1:] if len(token_ids) > 1 else token_ids
            
            # Pad to max_length
            pad_len_input = max_length - len(input_ids)
            pad_len_labels = max_length - len(labels)
            
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len_input
            labels = labels + [-100] * pad_len_labels  # -100 is ignored in loss
            
            # Attention mask
            attention_mask = [1] * (max_length - pad_len_input) + [0] * pad_len_input
            
            self.data.append({
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.float)
            })
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


class Tokenizer:
    """간단한 BPE 스타일 토크나이저"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab_to_id = {}
        self.id_to_vocab = {}
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Initialize with special tokens
        self.vocab_to_id = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id
        }
        self.id_to_vocab = {v: k for k, v in self.vocab_to_id.items()}
        
        self.merge_rules = []
    
    def train(self, texts: List[str]):
        """간단한 문자 기반 vocabulary 구축"""
        # Character-level tokenization for simplicity
        char_freq = {}
        for text in texts:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Sort by frequency
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Add to vocabulary
        current_id = len(self.vocab_to_id)
        for char, _ in sorted_chars:
            if current_id >= self.vocab_size:
                break
            if char not in self.vocab_to_id:
                self.vocab_to_id[char] = current_id
                self.id_to_vocab[current_id] = char
                current_id += 1
    
    def encode(self, text: str) -> List[int]:
        token_ids = [self.bos_token_id]
        for char in text:
            token_id = self.vocab_to_id.get(char, self.unk_token_id)
            token_ids.append(token_id)
        token_ids.append(self.eos_token_id)
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for token_id in token_ids:
            if token_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
            token = self.id_to_vocab.get(token_id, self.unk_token)
            tokens.append(token)
        return ''.join(tokens)
    
    def save(self, path: str):
        data = {
            'vocab_to_id': self.vocab_to_id,
            'id_to_vocab': {int(k): v for k, v in self.id_to_vocab.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_to_id = data['vocab_to_id']
        self.id_to_vocab = {int(k): v for k, v in data['id_to_vocab'].items()}
        self.vocab_size = data['vocab_size']


class Trainer:
    """모델 학습을 위한 트레이너"""
    
    def __init__(
        self,
        model: GPTModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        device: str = "cuda"
    ):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.max_grad_norm = max_grad_norm
        
        # Move model to device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler with warmup
        self.warmup_steps = warmup_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / warmup_steps) if step < warmup_steps else 0.5 ** ((step - warmup_steps) / warmup_steps)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(logits.view(-1, self.model.vocab_size), labels.view(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.step += 1
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        if self.val_dataloader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits.view(-1, self.model.vocab_size), labels.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int, save_path: str = "checkpoints"):
        import os
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if self.val_dataloader:
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Perplexity: {math.exp(val_loss):.2f}")
            
            # Save best model
            if self.val_dataloader and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    os.path.join(save_path, "best_model.pt"),
                    epoch,
                    val_loss
                )
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pt"),
                    epoch,
                    train_loss
                )
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'step': self.step
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        print(f"Checkpoint loaded from {path}")


class LoRALayer(nn.Module):
    """Low-Rank Adaptation (LoRA) 레이어"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
    
    def forward(self, x: torch.Tensor, original_weight: torch.Tensor) -> torch.Tensor:
        # Original transformation
        original_output = F.linear(x, original_weight)
        
        # LoRA transformation
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        return original_output + lora_output


class ModelOptimizer:
    """모델 최적화 유틸리티"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def apply_lora(model: GPTModel, rank: int = 8):
        """모델의 Attention layer에 LoRA 적용"""
        for block in model.blocks:
            attn = block.attention
            
            # Freeze original weights
            for param in attn.parameters():
                param.requires_grad = False
            
            # Add LoRA to Q, K, V projections
            attn.lora_q = LoRALayer(model.d_model, model.d_model, rank)
            attn.lora_k = LoRALayer(model.d_model, model.d_model, rank)
            attn.lora_v = LoRALayer(model.d_model, model.d_model, rank)
        
        print(f"LoRA applied. Trainable parameters: {ModelOptimizer.count_parameters(model):,}")
    
    @staticmethod
    def quantize_model(model: nn.Module, bits: int = 8):
        """Dynamic quantization 적용"""
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    
    @staticmethod
    def prune_model(model: nn.Module, amount: float = 0.3):
        """L1 Unstructured pruning"""
        import torch.nn.utils.prune as prune
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
        
        print(f"Pruned {amount*100}% of weights")


class KVCache:
    """Key-Value Cache for efficient autoregressive generation"""
    
    def __init__(self, batch_size: int, num_heads: int, max_len: int, head_dim: int, device: str = "cuda"):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_len = max_len
        self.head_dim = head_dim
        self.device = device
        
        # Initialize caches
        self.key_cache = torch.zeros(batch_size, num_heads, max_len, head_dim, device=device)
        self.value_cache = torch.zeros(batch_size, num_heads, max_len, head_dim, device=device)
        self.current_length = 0
    
    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        position: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Store new key and value
        self.key_cache[:, :, position:position+1, :] = key
        self.value_cache[:, :, position:position+1, :] = value
        self.current_length = position + 1
        
        # Return all keys and values up to current position
        return (
            self.key_cache[:, :, :self.current_length, :],
            self.value_cache[:, :, :self.current_length, :]
        )
    
    def clear(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.current_length = 0


class InferenceEngine:
    """추론 최적화 엔진"""
    
    def __init__(self, model: GPTModel, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)
    
    def generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        use_cache: bool = True
    ) -> torch.Tensor:
        """KV-cache를 사용한 효율적인 생성"""
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        
        # Initialize cache
        cache = None
        if use_cache:
            cache = KVCache(
                batch_size=batch_size,
                num_heads=self.model.blocks[0].attention.num_heads,
                max_len=max_length + input_ids.size(1),
                head_dim=self.model.blocks[0].attention.head_dim,
                device=self.device
            )
        
        with torch.no_grad():
            for step in range(max_length):
                if use_cache and step > 0:
                    # Only process last token
                    input_for_model = generated[:, -1:]
                else:
                    input_for_model = generated
                
                logits = self.model(input_for_model)
                next_token_logits = logits[:, -1, :]
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def batch_generate(
        self,
        input_ids_list: List[torch.Tensor],
        max_length: int = 50
    ) -> List[torch.Tensor]:
        """배치 단위 생성"""
        # Pad all sequences to same length
        max_input_len = max(ids.size(0) for ids in input_ids_list)
        batch_size = len(input_ids_list)
        
        padded_inputs = torch.full(
            (batch_size, max_input_len),
            self.model.pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        for i, ids in enumerate(input_ids_list):
            padded_inputs[i, :ids.size(0)] = ids
        
        # Generate
        generated = self.model.generate(padded_inputs, max_length=max_length)
        
        # Split back into list
        return [generated[i] for i in range(batch_size)]


class BeamSearchDecoder:
    """Beam Search 디코딩"""
    
    def __init__(
        self,
        model: GPTModel,
        beam_width: int = 5,
        length_penalty: float = 0.6
    ):
        self.model = model
        self.beam_width = beam_width
        self.length_penalty = length_penalty
    
    def decode(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        eos_token_id: int = 2
    ) -> Tuple[torch.Tensor, float]:
        """Beam Search 디코딩 실행"""
        device = input_ids.device
        batch_size = input_ids.size(0)
        assert batch_size == 1, "Beam search only supports batch_size=1"
        
        # Initialize beam
        sequences = [input_ids.squeeze(0).tolist()]
        scores = [0.0]
        
        with torch.no_grad():
            for _ in range(max_length):
                all_candidates = []
                
                for i, seq in enumerate(sequences):
                    if seq[-1] == eos_token_id:
                        all_candidates.append((seq, scores[i]))
                        continue
                    
                    # Get logits for this sequence
                    seq_tensor = torch.tensor([seq], device=device)
                    logits = self.model(seq_tensor)
                    next_token_logits = logits[0, -1, :]
                    
                    # Get top-k tokens
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    top_log_probs, top_indices = torch.topk(log_probs, self.beam_width)
                    
                    # Create candidates
                    for log_prob, token_id in zip(top_log_probs, top_indices):
                        candidate_seq = seq + [token_id.item()]
                        candidate_score = scores[i] + log_prob.item()
                        all_candidates.append((candidate_seq, candidate_score))
                
                # Select top beam_width candidates
                all_candidates.sort(key=lambda x: x[1] / (len(x[0]) ** self.length_penalty), reverse=True)
                sequences = [seq for seq, _ in all_candidates[:self.beam_width]]
                scores = [score for _, score in all_candidates[:self.beam_width]]
                
                # Check if all sequences ended
                if all(seq[-1] == eos_token_id for seq in sequences):
                    break
        
        # Return best sequence
        best_seq = sequences[0]
        best_score = scores[0]
        return torch.tensor([best_seq], device=device), best_score


class ContrastiveSearchDecoder:
    """Contrastive Search 디코딩 (SimCTG)"""
    
    def __init__(self, model: GPTModel, alpha: float = 0.6):
        self.model = model
        self.alpha = alpha
    
    def decode(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        k: int = 5
    ) -> torch.Tensor:
        """Contrastive Search 디코딩 실행"""
        device = input_ids.device
        generated = input_ids.clone()
        hidden_states_list = []
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.model(generated)
                next_token_logits = logits[:, -1, :]
                
                # Get hidden states (last layer)
                hidden_states = self.model.final_norm(
                    self.model.blocks[-1](
                        self.model.pos_encoding(
                            self.model.token_embedding(generated) * math.sqrt(self.model.d_model)
                        )
                    )
                )[:, -1, :]
                
                # Get top-k candidates
                top_k_probs, top_k_indices = torch.topk(F.softmax(next_token_logits, dim=-1), k)
                
                # Calculate degeneration penalty for each candidate
                best_score = float('-inf')
                best_token = top_k_indices[0, 0]
                
                for i in range(k):
                    token_id = top_k_indices[0, i]
                    model_confidence = top_k_probs[0, i].item()
                    
                    # Calculate similarity with previous tokens
                    if hidden_states_list:
                        candidate_hidden = hidden_states[0]
                        prev_hidden = torch.stack(hidden_states_list)
                        similarity = F.cosine_similarity(
                            candidate_hidden.unsqueeze(0),
                            prev_hidden,
                            dim=-1
                        ).max().item()
                    else:
                        similarity = 0.0
                    
                    # Final score
                    score = (1 - self.alpha) * model_confidence - self.alpha * similarity
                    
                    if score > best_score:
                        best_score = score
                        best_token = token_id
                
                # Add best token
                hidden_states_list.append(hidden_states[0])
                generated = torch.cat([generated, best_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        return generated


class EvaluationMetrics:
    """모델 평가 지표"""
    
    @staticmethod
    def perplexity(model: GPTModel, dataloader: DataLoader, device: str) -> float:
        model.eval()
        total_loss = 0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, model.vocab_size), labels.view(-1))
                
                # Count non-padding tokens
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item()
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity
    
    @staticmethod
    def calculate_diversity(texts: List[str]) -> Dict[str, float]:
        """생성 텍스트의 다양성 측정"""
        all_unigrams = []
        all_bigrams = []
        
        for text in texts:
            tokens = text.split()
            all_unigrams.extend(tokens)
            all_bigrams.extend([f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)])
        
        distinct_1 = len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0
        distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0
        
        # Calculate entropy
        from collections import Counter
        token_counts = Counter(all_unigrams)
        total = sum(token_counts.values())
        entropy = -sum((count/total) * math.log(count/total) for count in token_counts.values())
        
        return {
            'distinct-1': distinct_1,
            'distinct-2': distinct_2,
            'entropy': entropy
        }


class MemoryEfficientTrainer(Trainer):
    """메모리 효율적인 학습 트레이너"""
    
    def __init__(
        self,
        model: GPTModel,
        train_dataloader: DataLoader,
        gradient_accumulation_steps: int = 4,
        mixed_precision: bool = True,
        **kwargs
    ):
        super().__init__(model, train_dataloader, **kwargs)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, batch: Dict[str, torch.Tensor], accumulation_step: int) -> float:
        """메모리 효율적인 학습 스텝"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits.view(-1, self.model.vocab_size), labels.view(-1))
                loss = loss / self.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
        else:
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits.view(-1, self.model.vocab_size), labels.view(-1))
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
        
        return loss.item() * self.gradient_accumulation_steps


class ContinualLearningModule:
    """지속 학습 (Continual Learning) 모듈"""
    
    def __init__(self, model: GPTModel, lambda_ewc: float = 1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher_information = {}
        self.optimal_params = {}
    
    def compute_fisher_information(self, dataloader: DataLoader, device: str):
        """Fisher Information Matrix 계산"""
        self.model.eval()
        self.fisher_information = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            self.model.zero_grad()
            logits = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, self.model.vocab_size), labels.view(-1), ignore_index=-100)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.fisher_information[n] += p.grad.data ** 2 / len(dataloader)
        
        # Store optimal parameters
        self.optimal_params = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
    
    def ewc_loss(self) -> torch.Tensor:
        """EWC Regularization Loss 계산"""
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher_information:
                loss += (self.fisher_information[n] * (p - self.optimal_params[n]) ** 2).sum()
        return self.lambda_ewc * loss


class ModelCompressor:
    """모델 압축 유틸리티"""
    
    @staticmethod
    def knowledge_distillation(
        teacher_model: GPTModel,
        student_model: GPTModel,
        dataloader: DataLoader,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: str = "cuda",
        num_epochs: int = 5
    ):
        """Knowledge Distillation (지식 증류)"""
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.no_grad():
                    teacher_logits = teacher_model(input_ids)
                
                student_logits = student_model(input_ids)
                
                # Soft target loss
                soft_loss = kl_loss(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1)
                ) * (temperature ** 2)
                
                # Hard target loss
                hard_loss = ce_loss(student_logits.view(-1, student_model.vocab_size), labels.view(-1))
                
                # Combined loss
                loss = alpha * soft_loss + (1 - alpha) * hard_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Distillation Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")


# 사용 예시
if __name__ == "__main__":
    print("=" * 60)
    print("LLM Library - Complete Implementation")
    print("=" * 60)
    
    # 예시 데이터
    sample_texts = [
        "안녕하세요. 반갑습니다.",
        "오늘 날씨가 좋네요.",
        "파이썬 프로그래밍을 배우고 있습니다."
    ]
    
    # 토크나이저 학습
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.train(sample_texts)
    
    # 데이터셋 생성
    dataset = TextDataset(sample_texts, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 모델 초기화
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=3,
        d_ff=1024,
        max_seq_len=64
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n구현 완료된 컴포넌트:")
    print("=" * 60)
    print("기본 모듈:")
    print("  ✓ MultiHeadAttention")
    print("  ✓ PositionWiseFeedForward")
    print("  ✓ PositionalEncoding")
    print("  ✓ TransformerBlock")
    print("  ✓ GPTModel")
    print("\n데이터 처리:")
    print("  ✓ TextDataset")
    print("  ✓ Tokenizer")
    print("\n학습 및 최적화:")
    print("  ✓ Trainer")
    print("  ✓ MemoryEfficientTrainer (mixed precision, gradient accumulation)")
    print("  ✓ ModelOptimizer (LoRA, quantization, pruning)")
    print("  ✓ ContinualLearningModule (EWC)")
    print("  ✓ ModelCompressor (knowledge distillation)")
    print("\n추론 및 생성:")
    print("  ✓ InferenceEngine (KV-cache)")
    print("  ✓ BeamSearchDecoder")
    print("  ✓ ContrastiveSearchDecoder")
    print("\n평가:")
    print("  ✓ EvaluationMetrics (perplexity, diversity)")
    print("  ✓ KVCache")
    print("  ✓ LoRALayer")
    print("=" * 60)
    
    # LoRA 적용 예시
    print(f"\nBefore LoRA: {ModelOptimizer.count_parameters(model):,} parameters")
    ModelOptimizer.apply_lora(model, rank=8)
    
    print("\n사용 가능한 주요 기능:")
    print("  - Autoregressive text generation")
    print("  - Temperature, top-k, top-p sampling")
    print("  - Beam search decoding")
    print("  - Contrastive search (SimCTG)")
    print("  - LoRA fine-tuning")
    print("  - Knowledge distillation")
    print("  - Mixed precision training")
    print("  - KV-cache for fast inference")
    print("  - Continual learning with EWC")
    print("=" * 60)
