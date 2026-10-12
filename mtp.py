import torch
import torch.nn as nn

class MTP_block(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        from model import Transformer_block

        self.mtp_projection = nn.Linear(2*model_config.hidden_dim, model_config.hidden_dim)
        self.rms_hidden = nn.RMSNorm(model_config.hidden_dim)
        self.rms_embed = nn.RMSNorm(model_config.hidden_dim)
        self.trm_layer = Transformer_block(model_config)
        # self.output_head= nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, prev_hidden, future_embedding):
        norm_prev_hidden = self.rms_hidden(prev_hidden)
        norm_future_emb = self.rms_embed(future_embedding)
        mtp_hidden_state = self.mtp_projection(torch.cat([norm_prev_hidden, norm_future_emb], dim=-1))
        x, loss_balance = self.trm_layer(mtp_hidden_state)

        return x, loss_balance


class MTP(nn.Module):
    def __init__(self, model_config, output_head, embedding):
        super().__init__()
        if model_config.mtp_num_layer < 1:
            raise ValueError("mtp_num_layer must be at least 1")

        self.embedding = embedding
        self.output_head = output_head
        self.mtp_module = nn.ModuleList([MTP_block(model_config) for _ in range(model_config.mtp_num_layer)])

    def forward(self, hidden_state, input_ids):

        all_logits = []
        total_aux_loss = hidden_state.new_zeros(())


        hidden = hidden_state

        for k, mtp_module in enumerate(self.mtp_module, start=1):

            future_ids = input_ids[:, k:]
            future_embedding = self.embedding(future_ids)
            prev_hidden_input = hidden[:, :-1]

            hidden, loss_balance = mtp_module(prev_hidden_input, future_embedding)

            logits = self.output_head(hidden)
            all_logits.append(logits)
            total_aux_loss = total_aux_loss + loss_balance

        return all_logits, total_aux_loss





