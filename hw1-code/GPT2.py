import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Moudule):

    def __init__(self, config):
        super().__init__()
        
        # Key Query Value prjections in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1, config.block_size, config.block_size))

    def forward(self, x):
        # Batch size, Seq length, dimension
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k , v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, number of heads, seq length, head size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, number of heads, seq length, head size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, number of heads, seq length, head size)

        # attention
        att = (q @ k.transpose(-2. -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        y = att @ v
        y = y.transpose(1, 2).contiguous.view(B, T, C)
        y = self.c_proj(y)
        return y



class MLP(nn.Module):
    
    def __init__(self, config):
        super.__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))


@dataclass
class GPTConfig:
    block_size : int = 256
    vocab_size : int = 65
    n_layer : int = 6
    n_head : int = 6
    n_embd : int = 384


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Skeleton
        self.transformer = nn.ModuleDict(dict(
            # Token Embedding
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Position Embedding
            wpe = nn.Embedding(config.vocab_size, config.n_embd),
            # Hidden layer
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Layer norm
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # Final classifier
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        