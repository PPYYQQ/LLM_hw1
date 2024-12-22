import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size : int = 1024
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768


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
        