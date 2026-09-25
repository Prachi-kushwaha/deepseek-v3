from datasets import load_dataset
from config import get_config


def load_dataset(config):
    cfg = get_config()
    dataset = load_dataset(cfg.dataset)
    train_data = dataset['train']

