import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import nltk
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import ensure_dir, set_global_seed


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

VOCAB_SIZE = 10_000
DEFAULT_SEQ_LENGTHS = (25, 50, 100)
TRAIN_SIZE = 25_000
TEST_SIZE = 25_000
VAL_SPLIT = 0.1


@dataclass(frozen=True)
class Vocabulary:
    """Bidirectional token <-> id mapping."""

    token_to_id: Dict[str, int]
    id_to_token: List[str]

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, tokens: Sequence[str]) -> List[int]:
        return [self.token_to_id.get(token, UNK_IDX) for token in tokens]

    def decode(self, indices: Sequence[int]) -> List[str]:
        return [self.id_to_token[idx] if idx < self.size else UNK_TOKEN for idx in indices]


class SentimentDataset(Dataset):
    """Torch dataset wrapping token id sequences and binary labels."""

    def __init__(self, inputs: np.ndarray, labels: np.ndarray):
        assert inputs.shape[0] == labels.shape[0], "Inputs and labels must align"
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.labels[idx]


def _ensure_nltk_tokenizer() -> None:
    """Download required NLTK data if not present."""
    required_packages = ["punkt", "punkt_tab"]
    for package in required_packages:
        try:
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass  # If one fails, try the other


def clean_text(text: str) -> str:
    """Lowercase text and strip punctuation / special characters."""
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return nltk.word_tokenize(text)


def build_vocabulary(tokenized_texts: Iterable[List[str]], max_vocab: int = VOCAB_SIZE) -> Vocabulary:
    counter: Counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    most_common = counter.most_common(max_vocab - 2)  # reserve PAD and UNK
    index_to_token = [PAD_TOKEN, UNK_TOKEN] + [token for token, _ in most_common]
    token_to_index = {token: idx for idx, token in enumerate(index_to_token)}

    return Vocabulary(token_to_index, index_to_token)


def pad_or_truncate(sequence: Sequence[int], max_length: int) -> List[int]:
    sequence = list(sequence)
    if len(sequence) >= max_length:
        return sequence[:max_length]
    padding = [PAD_IDX] * (max_length - len(sequence))
    return sequence + padding


def vectorize_samples(
    tokenized_texts: Iterable[List[str]],
    vocabulary: Vocabulary,
    max_length: int,
) -> np.ndarray:
    sequences = [pad_or_truncate(vocabulary.encode(tokens), max_length) for tokens in tokenized_texts]
    return np.array(sequences, dtype=np.int64)


def _load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    required_columns = {"review", "sentiment"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def train_test_split_imdb(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic 25k / 25k split using provided order."""
    if len(df) < TRAIN_SIZE + TEST_SIZE:
        raise ValueError("Dataset must contain at least 50,000 rows.")
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_df = df.iloc[:TRAIN_SIZE].reset_index(drop=True)
    test_df = df.iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE].reset_index(drop=True)
    return train_df, test_df


def preprocess_dataset(
    data_dir: Path,
    cache_dir: Path,
    seq_lengths: Sequence[int] = DEFAULT_SEQ_LENGTHS,
    seed: int = 42,
    force_rebuild: bool = False,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Return dictionary mapping sequence length to processed splits."""
    set_global_seed(seed)
    _ensure_nltk_tokenizer()

    cache_dir = Path(cache_dir)
    ensure_dir(cache_dir)

    cache_index_path = cache_dir / "metadata.json"
    if cache_index_path.exists() and not force_rebuild:
        with open(cache_index_path, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)
        if sorted(int(k) for k in metadata.get("sequence_lengths", [])) == sorted(seq_lengths):
            # Load cached arrays
            processed: Dict[int, Dict[str, np.ndarray]] = {}
            for seq_len in seq_lengths:
                processed[seq_len] = {}
                for split in ("train", "val", "test"):
                    npz_path = cache_dir / f"{split}_seq{seq_len}.npz"
                    with np.load(npz_path) as data:
                        processed[seq_len][split] = data["inputs"]
                        processed[seq_len][f"{split}_labels"] = data["labels"]
            return processed

    raw_path = data_dir / "IMDB Dataset.csv"
    df = _load_raw_dataset(raw_path)
    train_df, test_df = train_test_split_imdb(df, seed=seed)

    train_df["clean_text"] = train_df["review"].astype(str).apply(clean_text)
    test_df["clean_text"] = test_df["review"].astype(str).apply(clean_text)

    train_tokens = train_df["clean_text"].apply(tokenize).tolist()
    test_tokens = test_df["clean_text"].apply(tokenize).tolist()

    vocabulary = build_vocabulary(train_tokens, max_vocab=VOCAB_SIZE)
    vocab_path = cache_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "token_to_id": vocabulary.token_to_id,
                "id_to_token": vocabulary.id_to_token,
                "pad_token": PAD_TOKEN,
                "unk_token": UNK_TOKEN,
                "vocab_size": vocabulary.size,
            },
            fp,
            indent=2,
        )

    # Convert sentiments to binary labels.
    label_map = {"positive": 1, "negative": 0}
    train_labels = train_df["sentiment"].map(label_map).astype(np.int64).values
    test_labels = test_df["sentiment"].map(label_map).astype(np.int64).values

    processed_results: Dict[int, Dict[str, np.ndarray]] = {}
    for seq_len in seq_lengths:
        train_inputs = vectorize_samples(train_tokens, vocabulary, seq_len)
        test_inputs = vectorize_samples(test_tokens, vocabulary, seq_len)

        # Validation split from training data
        val_size = int(len(train_inputs) * VAL_SPLIT)
        train_size = len(train_inputs) - val_size
        indices = np.arange(len(train_inputs))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_split_inputs = train_inputs[train_indices]
        train_split_labels = train_labels[train_indices]
        val_split_inputs = train_inputs[val_indices]
        val_split_labels = train_labels[val_indices]

        processed_results[seq_len] = {
            "train": train_split_inputs,
            "train_labels": train_split_labels,
            "val": val_split_inputs,
            "val_labels": val_split_labels,
            "test": test_inputs,
            "test_labels": test_labels,
        }

        # Cache for reuse
        for split_name, inputs in {
            "train": train_split_inputs,
            "val": val_split_inputs,
            "test": test_inputs,
        }.items():
            labels = processed_results[seq_len][f"{split_name}_labels"]
            np.savez_compressed(
                cache_dir / f"{split_name}_seq{seq_len}.npz",
                inputs=inputs,
                labels=labels,
            )

    stats = {
        "sequence_lengths": list(seq_lengths),
        "train_size": int(len(train_labels) * (1 - VAL_SPLIT)),
        "val_size": int(len(train_labels) * VAL_SPLIT),
        "test_size": int(len(test_labels)),
        "vocab_size": int(vocabulary.size),
        "average_review_length": float(np.mean([len(tokens) for tokens in train_tokens])),
        "median_review_length": float(np.median([len(tokens) for tokens in train_tokens])),
    }
    with open(cache_index_path, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)

    return processed_results


def get_dataloaders(
    seq_length: int,
    batch_size: int,
    data_dir: Path,
    cache_dir: Path,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Return train/val/test dataloaders plus vocab size for given seq length."""
    processed = preprocess_dataset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        seq_lengths=(seq_length,),
        seed=seed,
    )[seq_length]

    train_dataset = SentimentDataset(processed["train"], processed["train_labels"])
    val_dataset = SentimentDataset(processed["val"], processed["val_labels"])
    test_dataset = SentimentDataset(processed["test"], processed["test_labels"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    vocab_path = cache_dir / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError("Vocabulary not found. Run preprocessing first.")
    with open(vocab_path, "r", encoding="utf-8") as fp:
        vocab_meta = json.load(fp)
    vocab_size = int(vocab_meta["vocab_size"])

    return train_loader, val_loader, test_loader, vocab_size

