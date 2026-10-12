import torch
import torch.nn as nn

from rope import Rope
from mla_attention import MLAattention2
from moe import Moe
from mtp import MTP, MTP_block

class Embedding_layer(nn.Module):
    def __init__(self, hidden_dim, vocab_size = 30000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, x):
        return self.embedding(x)

class Transformer_block(nn.Module):
    def __init__(self, hidden_dim, compressed_dim, rope_dim, max_len,num_head, nr,  ns, kr, q_compressed_dim, intermediate_dim, alpha):
        super().__init__()
        self.attention_norm = nn.RMSNorm(hidden_dim)
        self.attention = MLAattention2(compressed_dim, rope_dim, hidden_dim, q_compressed_dim, num_head, max_len)
        self.moe_norm = nn.RMSNorm(hidden_dim)
        self.moe = Moe(hidden_dim, intermediate_dim, nr, ns, kr,alpha)

    def forward(self, x):
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x)
        x = residual + x
        residual = x
        x = self.moe_norm(x)
        moe_output, loss_balance = self.moe(x)
        x = residual + moe_output

        return x, loss_balance

class Deepseekv3_model(nn.Module):
    def __init__(self, hidden_dim, compressed_dim, rope_dim, max_len, vocab_size,num_head, nr, ns, kr, num_layer, mtp_num_layer, q_compressed_dim,intermediate_dim, alpha):
        super().__init__()
        self.embedding = Embedding_layer(hidden_dim, vocab_size)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.layers = nn.ModuleList([Transformer_block( hidden_dim, compressed_dim, rope_dim, max_len,num_head, nr, ns, kr, q_compressed_dim, intermediate_dim, alpha) for _ in range(num_layer)])

        self.mtp_layer =  MTP(hidden_dim, compressed_dim, rope_dim, max_len,num_head, nr, ns, kr, q_compressed_dim, intermediate_dim, alpha, self.lm_head, mtp_num_layer, self.embedding)

        self.final_norm = nn.RMSNorm(hidden_dim)

    def forward(self, input_ids):

        x = input_ids

        model_output = self.embedding(x)

        total_aux_loss = 0.0

        for layer in self.layers:
            model_output, loss_balance = layer(model_output)
            total_aux_loss += loss_balance

        x_model = self.final_norm(model_output)
        model_logits = self.lm_head(x_model)
        mtp_logits, mtp_aux_loss = self.mtp_layer(x_model,input_ids)
        total_aux_loss = total_aux_loss + mtp_aux_loss

        return model_logits, mtp_logits, total_aux_loss


