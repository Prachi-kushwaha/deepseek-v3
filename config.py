from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_name: str = "roneneldan/TinyStories"
    dataset_split: str = "train"
    num_samples: int = 100000
    validation_ratio: float = 0.1
    split_seed: int = 42

    vocab_size: int = 12928
    min_frequency: int = 1

    min_seq_len: int = 100
    max_seq_len: int = 500
    batch_size: int = 8


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    compressed_dim: int = 64
    rope_dim: int = 32
    max_len: int = 300
    vocab_size: int = 12928

    num_head: int = 4
    nr: int = 4
    ns: int = 1
    kr: int = 2
    num_layer: int = 2
    mtp_num_layer: int = 2

    q_compressed_dim: int = 64
    intermediate_dim: int = 512
    alpha: float = 0.001


@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 4
    lr_rate: float = 0.01
    weight_decay: float = 0.01
    step_size: int = 1
    gamma:float = 0.1
    mtp_lambda:float = 0.001
    betas: tuple[float, float] = (0.9, 0.95)
    eps:float = 1e-8
