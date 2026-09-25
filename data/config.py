from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset_name = "ImagineIt/StoryData"
    dataset_split = "train"
    validation_size = 0.1
    seed = 42
    max_seq_length = 4096
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    tokenize_dir = "data/tokenize"


def get_config():
    return DataConfig()