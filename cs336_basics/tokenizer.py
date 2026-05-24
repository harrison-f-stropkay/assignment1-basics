import os
import pickle
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re

from cs336_basics.train_bpe import get_initial_pseq, refresh_pseq

DATA_DIR = Path(__file__).parents[1] / "data"
PAT = rb"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        # Match special tokens in order from longest to shortest to placate `test_overlapping_special_tokens`
        self.special_tokens = (
            [bytes(special_token, "utf-8") for special_token in sorted(special_tokens, reverse=True)]
            if special_tokens
            else None
        )
        if self.special_tokens:
            for special_token in self.special_tokens:
                if special_token not in self.vocab.values():
                    self.vocab[len(self.vocab)] = special_token

        self.token_to_token_id = {token: id for id, token in self.vocab.items()}

    @classmethod
    def from_file(cls, vocab_merges_filepath: str | os.PathLike, special_tokens: list[str] | None = None):
        with open(vocab_merges_filepath, "rb") as f:
            vocab, merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens)

    def encode_bytes(self, split: bytes) -> list[int]:
        encoding = []
        pretokens = re.findall(PAT, split)
        for pretoken in pretokens:
            pseq = get_initial_pseq(pretoken)
            for merge in self.merges:
                pseq = refresh_pseq(pseq, merge)
                # Stop the loop early if the pseq is just 1 token long
                if len(pseq) == 1:
                    break
            for token in pseq:
                encoding.append(self.token_to_token_id[token])
        return encoding

    def encode(self, text: str) -> list[int]:
        bytestring = bytes(text, "utf-8")

        if self.special_tokens:
            special_tokens_pattern = b"|".join([re.escape(special) for special in self.special_tokens])
            # Add parenthesis to include the special tokens in the list (via capturing group)
            splits = re.split(b"(" + special_tokens_pattern + b")", bytestring)
        else:
            splits = [bytestring]

        encoding = []
        for split in splits:
            if self.special_tokens and split in self.special_tokens:
                encoding.append(self.token_to_token_id[split])
            else:
                encoding += self.encode_bytes(split)
        return encoding

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            yield from self.encode(string)

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab[id] for id in ids]
        bytestring = b"".join(tokens)
        return bytestring.decode("utf-8", errors="replace")
