
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from config import DataConfig


def load_data(config: DataConfig):
    dataset = load_dataset(
        config.dataset_name,
        split=config.dataset_split
    )

    dataset = dataset.select(
        range(min(config.num_samples, len(dataset)))
    )

    split = dataset.train_test_split(
        test_size=config.validation_ratio,
        seed=config.split_seed
    )

    return split["train"], split["test"]


def get_data(dataset):
    for row in dataset:
        yield row["text"]


def tokenize(train_dataset, config: DataConfig):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        min_frequency=config.min_frequency,
        vocab_size=config.vocab_size
    )

    tokenizer.train_from_iterator(
        get_data(train_dataset),
        trainer=trainer
    )

    return tokenizer


def encode_data(example, tokenizer):
    return {"input_ids": tokenizer.encode(example["text"]).ids}


def collate_fn(batch, config: DataConfig, pad_id: int):
    sequences = []

    for example in batch:
        ids = example["input_ids"][:config.max_seq_len + 1]

        if len(ids) < 2:
            continue

        padding_length = config.max_seq_len + 1 - len(ids)
        ids = ids + [pad_id] * padding_length
        sequences.append(ids)

    if not sequences:
        raise ValueError("Batch contains no valid sequences")

    input_ids = torch.tensor(sequences, dtype=torch.long)

    inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:].clone()

    labels[labels == pad_id] = -100

    return inputs, labels


def create_dataloaders(config: DataConfig):
    train_dataset, val_dataset = load_data(config)

    tokenizer = tokenize(train_dataset, config)

    train_dataset = train_dataset.map(
        lambda example: encode_data(example, tokenizer)
    )
    val_dataset = val_dataset.map(
        lambda example: encode_data(example, tokenizer)
    )

    train_dataset = train_dataset.filter(
        lambda row: config.min_seq_len
        <= len(row["input_ids"])
        <= config.max_seq_len
    )

    val_dataset = val_dataset.filter(
        lambda row: config.min_seq_len
        <= len(row["input_ids"])
        <= config.max_seq_len
    )

    pad_id = tokenizer.token_to_id("[PAD]")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, config, pad_id)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, config, pad_id)
    )

    return train_loader, val_loader, tokenizer
