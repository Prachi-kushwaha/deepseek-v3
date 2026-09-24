import torch
import torch.nn as nn

from rope import Rope
from mla_attention import MLAattention
from moe import MOE_layer

class Embedding_layer(nn.Module):
    def __init__(self, hidden_dim, vocab_size = 30000):
        self().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, x):
        return self.embedding(x)

class Transformer_block(nn.Module):
    def __init__(self, hidden_dim, compressed_dim, rope_dim, max_len,num_head, nr, ns, kr):
        super().__init__()
        self.attention_norm = nn.RMSNorm(hidden_dim)
        self.attention = MLAattention(hidden_dim, compressed_dim, rope_dim, num_head)
        self.moe_norm = nn.RMSNorm(hidden_dim)
        self.moe = MOE_layer(nr, ns, kr, hidden_dim)

    def forward(self, x, position_ids=None):
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, positions_ids = position_ids)
        x = residual + x
        x = self.moe_norm(x)
        x = self.moe(x)
        x = residual + x

        return x


class Deepseekv3_model(nn.Module):
    def __init__(self, hidden_dim, compressed_dim, rope_dim, max_len, vocab_size,num_head, nr, ns, kr, num_layer):
        super().__init__()
        self.embedding = Embedding_layer(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([Transformer_block( hidden_dim, compressed_dim, rope_dim, max_len, vocab_size,num_head, nr, ns, kr) for _ in range(num_layer)])

        self.final_norm = nn.RMSNorm(hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, position_ids = None):

        x = input_ids

        x = self.embedding(x)

        for layer in self.layers:
            layer(x, position_ids=position_ids)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits