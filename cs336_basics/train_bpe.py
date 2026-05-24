import os
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterable
from multiprocessing import Pool
from pathlib import Path

import regex as re
from tqdm import tqdm

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
    # Note: I tried splitting pair_counter for multiprocessing, but that was far slower
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


def get_pairs(seq: tuple | list) -> tuple:
    return tuple(pair for pair in zip(seq[:-1], seq[1:]))


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

    for i in tqdm(range(vocab_size), desc="Training BPE Tokenenizer"):
        if i < 256 + len(special_tokens):
            continue

        merge_pair = get_most_common_pair(pair_counter)
        new_token = b"".join(merge_pair)
        merges.append(merge_pair)
        vocab[i] = new_token

        update_caches(
            merge_pair,
            pair_counter,
            pair_to_pretokens,
            pretoken_counter,
            pretoken_to_pseq,
        )

    return vocab, merges


def main(corpus_filestem: str, vocab_size: int, special_tokens):
    DATA_DIR = Path(__file__).parents[1] / "data"
    CORPUS_PATH = DATA_DIR / f"{corpus_filestem}.txt"
    VOCAB_MERGES_PATH = DATA_DIR / f"{corpus_filestem}_vocab_merges.pkl"

    vocab, merges = train_bpe(CORPUS_PATH, vocab_size, special_tokens)

    # longest_token = max(vocab.values(), key=len)
    # print(f"Longest token: {longest_token}")
    # Output: b' accomplishment'

    with open((VOCAB_MERGES_PATH), "wb") as f:
        pickle.dump(((vocab, merges)), f)


if __name__ == "__main__":
    CORPUS_FILESTEM = "owt_train.txt"
    CORPUS_FILESTEM = "TinyStoriesV2-GPT4-train"
    VOCAB_SIZE = 10000
    SPECIAL_TOKENS = ["<|endoftext|>"]

    main(CORPUS_FILESTEM, VOCAB_SIZE, SPECIAL_TOKENS)

    # DATA_DIR = Path(__file__).parents[1] / "data"
    # OWT_FILESTEM = "owt_train"
    # TINY_FILESTEM = "TinyStoriesV2-GPT4-train"

    # TINY_VOCAB_MERGES_PATH = DATA_DIR / f"{TINY_FILESTEM}_vocab_merges.pkl"
    # with open((TINY_VOCAB_MERGES_PATH), "rb") as f:
    #     tiny_vocab, tiny_merges = pickle.load(f)

    # OWT_VOCAB_MERGES_PATH = DATA_DIR / f"{OWT_FILESTEM}_vocab_merges.pkl"
    # with open((OWT_VOCAB_MERGES_PATH), "rb") as f:
    #     owt_vocab, owt_merges = pickle.load(f)

    # owt_10k = [owt_vocab[i] for i in range(10000)]
    # tiny_10k = [tiny_vocab[i] for i in range(10000)]

    # print(f"{(len(set(tiny_10k) & set(owt_10k)) / 10000) * 100}% overlap with the first 10k of OWT")
    # # 45.6% overlap with the first 10k of OWT

    # print(f"{(len(set(tiny_10k) & set(owt_vocab.values())) / 10000) * 100}% overlap with all of OWT")
    # # 73.2% overlap with all of OWT

    # print(f"{list(set(tiny_10k) - set(owt_vocab.values()))[:20]}")
    # # [b' Janie', b' tuck', b' lollipop', b' bandages', b' comf', b'Pepper', b' disappoin', b'Ol', b' donkey', b' Flora', b' glided', b' strang', b' tangled', b' scooters', b' Daisy', b' frosting', b' gobbled', b'bbie', b' croaked', b' forgets']

    # print(f"{list(set(owt_vocab.values()) - set(tiny_10k))[:20]}")
    # # [b'NP', b' experts', b' Schwe', b' title', b' fewer', b' Brooks', b' unlimited', b' Download', b' Mold', b'PP', b' neurolog', b' Tam', b'nes', b' WILL', b'ander', b' Bless', b' Assassin', b' dispens', b' rank', b'ci']
