import torch
import torch.nn as nn

from model import Transformer_block
from model import Embedding_layer

class MTP_block(nn.Module):
    def __init__(self, hidden_dim, compressed_dim, rope_dim, max_len,num_head, nr, ns, kr, q_compressed_dim, intermediate_dim, alpha):
        super().__init__()
        self.mtp_projection = nn.Linear(2*hidden_dim, hidden_dim)
        self.rms_hidden = nn.RMSNorm(hidden_dim)
        self.rms_embed = nn.RMSNorm(hidden_dim)
        self.trm_layer = Transformer_block(hidden_dim, compressed_dim, rope_dim, max_len,num_head, nr, ns, kr, q_compressed_dim, intermediate_dim, alpha)
        # self.output_head= nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, prev_hidden, future_embedding):
        norm_prev_hidden = self.rms_hidden(prev_hidden)
        norm_future_emb = self.rms_embed(future_embedding)
        mtp_hidden_state = self.mtp_projection(torch.cat([norm_prev_hidden, norm_future_emb], dim=-1))
        x, loss_balance = self.trm_layer(mtp_hidden_state)

        return x, loss_balance


class MTP(nn.Module):
    def __init__(self, hidden_dim, compressed_dim, rope_dim, max_len,num_head, nr, ns, kr,q_compressed_dim, intermediate_dim, alpha,output_head, mtp_num_layer, embedding):
        super().__init__()

        self.embedding = embedding
        self.output_head = output_head
        self.mtp_module = nn.ModuleList([MTP_block(hidden_dim, compressed_dim, rope_dim, max_len,num_head, nr, ns, kr, q_compressed_dim, intermediate_dim, alpha) for _ in range(mtp_num_layer)])

    def forward(self, hidden_state, input_ids):

        all_logits = []

        hidden = hidden_state

        for k, mtp_module in enumerate(self.mtp_module, start=1):

            future_ids = input_ids[:, k:]
            future_embedding = self.embedding(future_ids)
            prev_hidden_input = hidden[:, :-1]

            hidden, loss_balance = mtp_module(prev_hidden_input, future_embedding)

            logits = self.output_head(hidden)
            all_logits.append(logits)

        return all_logits, loss_balance





