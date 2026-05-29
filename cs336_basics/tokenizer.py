from typing import Final
from methodtools import lru_cache  # Use `methodtools` so that caches are not shared between class instances
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
        self.special_tokens_pattern: Final[bytes | None] = (
            b"|".join([re.escape(special) for special in self.special_tokens]) if self.special_tokens else None
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

    @lru_cache(maxsize=1_000_000)
    def _encode_pretoken(self, pretoken: bytes) -> list[int]:
        pseq = get_initial_pseq(pretoken)
        for merge in self.merges:
            if len(pseq) == 1:  # Stop the loop early if the pseq is just 1 token long
                break
            pseq = refresh_pseq(pseq, merge)
        return [self.token_to_token_id[token] for token in pseq]

    def _encode_split(self, split: bytes) -> list[int]:
        encoding = []
        pretokens = re.findall(PAT, split)
        for pretoken in pretokens:
            encoding.extend(self._encode_pretoken(pretoken))
        return encoding

    def encode(self, text: str) -> list[int]:
        bytestring = bytes(text, "utf-8")

        if self.special_tokens_pattern:
            # Add parenthesis to include the special tokens in the list (via capturing group)
            splits = re.split(b"(" + self.special_tokens_pattern + b")", bytestring)
        else:
            splits = [bytestring]

        encoding = []
        for split in splits:
            if self.special_tokens and split in self.special_tokens:
                encoding.append(self.token_to_token_id[split])
            else:
                encoding += self._encode_split(split)
        return encoding

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        iterator = iter(iterable)
        buffer = bytes(next(iterator), "utf-8")

        while buffer:
            # Top off the buffer (load at least 1000 code points, if possible)
            while len(buffer) < 1000:
                try:
                    buffer += bytes(next(iterator), "utf-8")
                except StopIteration:
                    break

            # If we find a special token, yield from it and everything before it
            if self.special_tokens_pattern and (match := re.search(self.special_tokens_pattern, buffer)):
                bytes_before_match = buffer[: match.start()]
                buffer = buffer[match.end() :]
                yield from self._encode_split(bytes_before_match)
                yield self.token_to_token_id[match.group()]

            # Otherwise, just yield from the first pretoken from the buffer
            else:
                match = re.match(PAT, buffer)
                assert match
                assert buffer.startswith(match.group())
                buffer = buffer[match.end() :]
                yield from self._encode_pretoken(match.group())

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab[id] for id in ids]
        bytestring = b"".join(tokens)
        return bytestring.decode("utf-8", errors="replace")
