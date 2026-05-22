import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from multiprocessing import Pool

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = rb"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"


def get_pretoken_counts_from_chunk(input_path: str, start: int, end: int, split_pattern: bytes) -> Counter[bytes]:
    # Read the chunk
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    # Split the chunk into subchunks (on `split_pattern`) -> split the subchunks into pretokens (on `PAT`)
    # Note: it's kosher to use `regex` functions with bytes as long as both `pattern` and `string` are bytes (https://docs.python.org/3/library/re.html)
    chunk_counts = Counter()
    for subchunk in re.splititer(split_pattern, chunk):
        for pretoken in re.finditer(PAT, subchunk):
            chunk_counts[pretoken.group()] += 1
    return chunk_counts


def pretokenize(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> Counter[bytes]:
    special_tokens_pattern = bytes("|".join([re.escape(special) for special in special_tokens]), "utf-8")

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 10, b"<|endoftext|>")

    with Pool(os.cpu_count() or 8) as p:
        chunk_pretoken_counters = p.starmap(
            get_pretoken_counts_from_chunk,
            [(input_path, start, end, special_tokens_pattern) for start, end in zip(boundaries[:-1], boundaries[1:])],
        )

    return sum(chunk_pretoken_counters, Counter())


def get_most_common_pair(pair_counter: Counter[tuple[bytes, bytes]]) -> tuple[bytes, bytes]:
    return max(pair_counter, key=lambda pair: (pair_counter[pair], pair))


def get_initial_pseq(pretoken: bytes) -> tuple[bytes, ...]:
    return tuple(pretoken[i : i + 1] for i in range(len(pretoken)))


def get_initial_tokens(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = bytes(special_token, "utf-8")
    return vocab


def refresh_pseq(stale_pseq: tuple[bytes, ...], merge_pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    new_token = b"".join(merge_pair)
    fresh_pseq = []
    i = 0
    while i < len(stale_pseq):
        left = stale_pseq[i]
        right = stale_pseq[i + 1] if i < len(stale_pseq) - 1 else None

        if (left, right) == merge_pair:
            fresh_pseq.append(new_token)
            i += 2
        else:
            fresh_pseq.append(left)
            i += 1
    assert len(fresh_pseq) < len(stale_pseq)
    return tuple(fresh_pseq)


def get_pair_counter(pseq_counter: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    pair_counter: Counter[tuple[bytes, bytes]] = Counter()
    for pseq, pseq_count in pseq_counter.items():
        for pair in zip(pseq[:-1], pseq[1:]):
            pair_counter[pair] += pseq_count
    return pair_counter


def get_pair_to_pseq_counter(
    pseq_counter: Counter[tuple[bytes, ...]],
) -> defaultdict[tuple[bytes, bytes], Counter[tuple[bytes, ...]]]:
    pair_to_pseq_counter: defaultdict[tuple[bytes, bytes], Counter[tuple[bytes, ...]]] = defaultdict(Counter)
    for pseq, pseq_count in pseq_counter.items():
        for pair in zip(pseq[:-1], pseq[1:]):
            pair_to_pseq_counter[pair][pseq] += 1
    return pair_to_pseq_counter


def get_pair_to_pretokens(pseqs: Iterable[tuple[bytes, ...]]) -> defaultdict[tuple[bytes, bytes], set[bytes]]:
    pair_to_pretokens: defaultdict[tuple[bytes, bytes], set] = defaultdict(set)
    for pseq in pseqs:
        for pair in get_pairs(pseq):
            pair_to_pretokens[pair].add(b"".join(pseq))
    return pair_to_pretokens


def get_pairs(pseq: tuple[bytes, ...]) -> tuple[tuple[bytes, bytes]]:
    return tuple(pair for pair in zip(pseq[:-1], pseq[1:]))


def update_caches(
    merge_pair: tuple[bytes, bytes],
    pair_counter: Counter[tuple[bytes, bytes]],
    pair_to_pretokens: defaultdict[tuple[bytes, bytes], set],
    pretoken_counter: Counter[bytes],
    pretoken_to_pseq: dict[bytes, tuple[bytes, ...]],
):
    for affected_pretoken in pair_to_pretokens[merge_pair].copy():
        stale_pseq = pretoken_to_pseq[affected_pretoken]
        for stale_pair in get_pairs(stale_pseq):
            pair_counter[stale_pair] -= pretoken_counter[affected_pretoken]

            # Use `discard` because it won't throw an error if `merge_pair` is in `stale_pseq` multiple times
            pair_to_pretokens[stale_pair].discard(affected_pretoken)

            # If the pair is no longer in the corpus, remove it from the caches
            # Note: use `pair_counter` because `pair_to_pretokens[stale_pair]` could be misleadingly empty (e.g., merging (m, e) and affected_pretoken is "meme", think about the second "me")
            if not pair_counter[stale_pair]:
                assert not pair_to_pretokens[stale_pair]
                pair_counter.pop(stale_pair)
                pair_to_pretokens.pop(stale_pair)

        fresh_pseq = refresh_pseq(pretoken_to_pseq[affected_pretoken], merge_pair)
        pretoken_to_pseq[affected_pretoken] = fresh_pseq

        for fresh_pair in get_pairs(fresh_pseq):
            pair_counter[fresh_pair] += pretoken_counter[affected_pretoken]

            # Use `add` because it won't throw an error if `merge_pair` is in `fresh_pseq` multiple times
            pair_to_pretokens[fresh_pair].add(affected_pretoken)

    assert merge_pair not in pair_to_pretokens
    assert merge_pair not in pair_counter


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Pretokenize
    pretoken_counter = pretokenize(input_path, vocab_size, special_tokens)

    # Initialize our return objects
    vocab = get_initial_tokens(special_tokens)
    merges = []

    # Intitialize our 4 caches
    # Tokenize the pretokens with the initial 256 tokens
    pretoken_to_pseq = {}
    pseq_counter = Counter()
    for pretoken, count in pretoken_counter.items():
        intial_pseq = get_initial_pseq(pretoken)
        pretoken_to_pseq[pretoken] = intial_pseq
        pseq_counter[intial_pseq] = count

    pair_to_pretokens = get_pair_to_pretokens(pseq_counter.keys())
    pair_counter = get_pair_counter(pseq_counter)

    while len(vocab) < vocab_size:
        merge_pair = get_most_common_pair(pair_counter)
        new_token = b"".join(merge_pair)
        merges.append(merge_pair)
        vocab[max(vocab) + 1] = new_token

        update_caches(
            merge_pair,
            pair_counter,
            pair_to_pretokens,
            pretoken_counter,
            pretoken_to_pseq,
        )

    return vocab, merges
