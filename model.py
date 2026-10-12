import torch
import torch.nn as nn

from mla_attention import MLAattention2
from moe import Moe
from mtp import MTP
class Embedding_layer(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.embedding = nn.Embedding(model_config.vocab_size, model_config.hidden_dim)

    def forward(self, x):
        return self.embedding(x)

class Transformer_block(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.attention_norm = nn.RMSNorm(model_config.hidden_dim)
        self.attention = MLAattention2(model_config)
        self.moe_norm = nn.RMSNorm(model_config.hidden_dim)
        self.moe = Moe(model_config)

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
    def __init__(self, model_config):
        super().__init__()
        self.embedding = Embedding_layer(model_config)
        self.lm_head = nn.Linear(model_config.hidden_dim, model_config.vocab_size, bias=False)
        self.layers = nn.ModuleList([Transformer_block( model_config) for _ in range(model_config.num_layer)])

        self.mtp_layer =  MTP(model_config, self.lm_head, self.embedding)

        self.final_norm = nn.RMSNorm(model_config.hidden_dim)

    def forward(self, input_ids):

        x = input_ids

        model_output = self.embedding(x)

        total_aux_loss = model_output.new_zeros(())

        for layer in self.layers:
            model_output, loss_balance = layer(model_output)
            total_aux_loss += loss_balance

        x_model = self.final_norm(model_output)
        model_logits = self.lm_head(x_model)
        mtp_logits, mtp_aux_loss = self.mtp_layer(x_model,input_ids)
        total_aux_loss = total_aux_loss + mtp_aux_loss

        return model_logits, mtp_logits, total_aux_loss


