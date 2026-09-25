
from config import get_config
from load_data import load_raw_dataset, validate_data, clean_example, conversion_hash

config = get_config()
from config import get_config
from load_data import (
    load_raw_dataset,
    validate_data,
    clean_example,
    conversation_hash,
    deduplicate,
    format_conversion
)

from tokenize_data import build_tokenizer, tokenize_with_labels

config = get_config()


def prepare_data(config):

    # 1. Load raw dataset
    dataset = load_raw_dataset(
        config.dataset_name,
        config.dataset_split
    )

    # 2. Validate
    dataset = validate_data(dataset)

    # 3. Clean / normalize
    dataset = dataset.map(clean_example)

    # 4. Add deterministic hash
    dataset = dataset.map(
        lambda example: {
            "conversation_hash": conversation_hash(example)
        }
    )

    # 5. Exact deduplication
    dataset = deduplicate(dataset, key="conversation_hash")

    # 6. Train / validation split
    split_dataset = dataset.train_test_split(
        test_size=config.validation_split,
        seed=config.seed
    )

    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    return train_dataset, val_dataset


def train_data(dataset, config):
    formatted_dataset = dataset.map(format_conversion)
    tokenizer = build_tokenizer(formatted_dataset, config)
    tokenized_dataset = tokenize_with_labels(formatted_dataset,tokenizer)


