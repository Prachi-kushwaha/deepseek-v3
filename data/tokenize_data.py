from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


def build_tokenizer(data, config):

    tokenizer_path = Path(config["token_path"])

    if not tokenizer_path.exists():

        tokenizer = Tokenizer(
            WordLevel(unk_token="[UNK]")
        )

        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=[
                "[UNK]",
                "[PAD]",
                "[BOS]",
                "[EOS]",
                "<|system|>",
                "<|user|>",
                "<|assistant|>",
                "<|endoftext|>",
            ],
            min_frequency=1,
            vocab_size=10000,
        )

        tokenizer.train_from_iterator(
            (example["text"] for example in data),
            trainer=trainer,
        )

        tokenizer.save(str(tokenizer_path))

    else:

        tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

    return tokenizer


def tokenize_with_labels(example, tokenizer):

    input_ids = []
    labels = []

    # BOS
    bos = tokenizer.encode("[BOS]").ids
    input_ids.extend(bos)
    labels.extend([-100] * len(bos))

    for message in example["messages"]:

        role = message["role"]
        content = message["content"]

        role_tokens = tokenizer.encode(
            f"<|{role}|>"
        ).ids

        content_tokens = tokenizer.encode(
            content
        ).ids

        input_ids.extend(role_tokens)
        input_ids.extend(content_tokens)

        if role == "assistant":
            labels.extend(role_tokens)
            labels.extend(content_tokens)
        else:
            labels.extend([-100] * len(role_tokens))
            labels.extend([-100] * len(content_tokens))

    eos = tokenizer.encode("[EOS]").ids

    input_ids.extend(eos)
    labels.extend(eos)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids)
    }


class DataCollator:

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):

        max_length = max(
            len(example["input_ids"])
            for example in examples
        )

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for example in examples:

            length = len(example["input_ids"])
            padding = max_length - length

            batch_input_ids.append(
                example["input_ids"] +
                [self.pad_token_id] * padding
            )

            batch_attention_mask.append(
                example["attention_mask"] +
                [0] * padding
            )

            batch_labels.append(
                example["labels"] +
                [-100] * padding
            )

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }