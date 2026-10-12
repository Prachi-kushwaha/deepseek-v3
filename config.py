from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_dim=256,
    compressed_dim=64,
    rope_dim=32,
    max_len=500,
    vocab_size=12928,
    num_head=4,
    nr=4,
    ns=1,
    kr=2,
    num_layer=2,
    mtp_num_layer=2,
    q_compressed_dim=64,
    intermediate_dim=512,
    alpha=0.001,



@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 2
    lr_rate: float = 0.01
    weight_decay: float = 0.01
    step_size: int = 1
    gamma:float = 0.1
    mtp_lambda:float = 0.001
    betas: tuple[float, float] = (0.9, 0.95)
    eps:float = 1e-8
