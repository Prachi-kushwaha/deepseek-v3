import torch
import torch.nn as nn

class Rope(nn.Module):
    def __init__(self, max_seq_len, dim):
        super().__init__()

        freq = 10000 **( -torch.arange(0,dim, 2).unsqueeze(0).float() / dim)
        pos = torch.arange(max_seq_len).unsqueeze(-1).float()
        phi = pos * freq

        self.register_buffer("cos", torch.cos(phi))
        self.register_buffer("sin", torch.sin(phi))

    def forward(self, x):
        _, seq_len, _ = x.shape
        x1 = x[:,:, 0::2]
        x2 = x[:,:, 1::2]

        cos = (self.cos[:seq_len]).unsqueeze(0)
        sin = (self.sin[:seq_len]).unsqueeze(0)

        x1_rot = torch.stack( [x1*cos - x2*sin, x1*sin + x2*cos ], dim=-1).flatten(-2)

        return x1_rot



