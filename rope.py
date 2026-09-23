import torch


def Rope(pos, x):
    b,h,s,d = x.shape

    inv_freq = 1.0/10000**(torch.arange(0, d, 2)/d)
    freq = pos.float().unsqueeze(-1)*inv_freq

    sin = torch.sin(freq).unsqueeze(1)
    cos = torch.cos(freq).unsqueeze(1)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    y1,y2 = [x1*cos - x2*sin, x1*sin + x2*cos]
    rope_val = torch.stack([y1,y2], dim=-1).reshape(b, h, s, d)

    return rope_val



