import torch
import torch.nn as nn

from model import Transformer_block
from model import Embedding_layer

class MTP_block(nn.Module):
    def __init__(self, hidden_dim, compressed_dim, rope_dim, max_len,num_head, nr, ns, kr, vocab_size):
        super().__init__()
        self.mtp_projection = nn.Linear(2*hidden_dim, hidden_dim)
        self.rms_norm = nn.RMSNorm(hidden_dim)
        self.trm_layer = Transformer_block(hidden_dim, compressed_dim, rope_dim, max_len,num_head, nr, ns, kr)
        self.output_head= nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, prev_hidden, future_embedding, position_ids=None):
        norm_prev_hidden = self.rms_norm(prev_hidden)
        norm_future_emb = self.rms_norm(future_embedding)
        mtp_hidden_state = self.mtp_projection(torch.cat(norm_prev_hidden, norm_future_emb))
        x = self.trm_layer(mtp_hidden_state, position_ids=position_ids)

        return x


class MTP(nn.Module):
    def __init__(self, hidden_dim, compressed_dim, rope_dim, max_len,num_head, nr, ns, kr, vocab_size, output_head, mtp_num_layer):
        super().__init__()

        self.embedding = Embedding_layer(vocab_size, hidden_dim)
        self.output_head = output_head
        self.mtp_module = nn.ModuleList([MTP_block(hidden_dim, compressed_dim, rope_dim, max_len,num_head, nr, ns, kr, vocab_size) for _ in range(mtp_num_layer)])

    def forward(self, hidden_state, input_ids, position_ids=None):

        predictions = []

        for k, mtp_module in enumerate(self.mtp_module, start=1):

            all_logits = []

            future_ids = input_ids[:, k:]
            future_embedding = self.embedding[future_ids]
            hidden_input = hidden_state[:, :-k]

            hidden = mtp_module(hidden_input, future_embedding, position_ids[:, :])

            logits = self.output_head(hidden)
            all_logits.append(logits)

            return all_logits



