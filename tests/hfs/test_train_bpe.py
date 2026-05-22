from collections import Counter, defaultdict
from pathlib import Path

import pytest

from cs336_basics.train_bpe import (
    get_initial_pseq,
    get_initial_tokens,
    get_most_common_pair,
    get_pair_counter,
    get_pair_to_pretokens,
    get_pair_to_pseq_counter,
    get_pairs,
    pretokenize,
    refresh_pseq,
    update_caches,
)


def test_get_initial_tokens():
    special_tokens = ["<BOS>", "<EOS>"]
    vocab = get_initial_tokens(special_tokens)
    assert len(vocab) == 258
    for i in range(256):
        assert tuple(vocab[i])[0] == i
    assert [vocab[256], vocab[257]] == [bytes(token, "utf-8") for token in special_tokens]


def test_get_most_common_pair():
    pairs_counter: Counter[tuple[bytes, bytes]] = Counter([(b"A", b"B"), (b"A", b"C"), (b"B", b"ZZ"), (b"BA", b"A")])
    assert get_most_common_pair(pairs_counter) == (b"BA", b"A")

    pairs_counter = Counter([(b"A", b"A"), (b"A", b"A"), (b"Z", b"Z")])
    assert get_most_common_pair(pairs_counter) == (b"A", b"A")

    pairs_counter = Counter({(b"h", b"e"): 639606, (b" ", b"t"): 491409})
    assert get_most_common_pair(pairs_counter) == (b"h", b"e")


def test_get_initial_tokenization():
    bytestring = bytes("hey 👋", "utf-8")
    assert len(bytestring) == 8 == len(get_initial_pseq(bytestring))  # 4 ascii chars and 1 4-byte emoji


# def test_refresh_pretoken():
#     stale_pretoken = (b" ", b"t", b"h", b"e")
#     fresh_pretoken = refresh_pretoken(stale_pretoken, [b"t", b"h"])


def test_pretokenize():
    input_path = Path(__file__).parents[2] / "data" / "TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    pretoken_counter = pretokenize(str(input_path), vocab_size, special_tokens)
    three_most_common = pretoken_counter.most_common(3)

    assert three_most_common[0][0] == b"."
    assert three_most_common[1][0] == b","
    assert three_most_common[2][0] == b" the"


def test_refresh_pseq():
    stale_pseq = (b"a", b"b", b"c", b"d")
    merge_pair = (b"a", b"b")
    assert refresh_pseq(stale_pseq, merge_pair) == (b"ab", b"c", b"d")

    with pytest.raises(AssertionError):
        stale_pseq = (b"a", b"b", b"c", b"d")
        merge_pair = (b"a", b"a")
        refresh_pseq(stale_pseq, merge_pair) == stale_pseq

    stale_pseq = (b"a", b"b", b"a", b"b", b"a", b"c")
    merge_pair = (b"a", b"b")
    assert refresh_pseq(stale_pseq, merge_pair) == (b"ab", b"ab", b"a", b"c")


def test_get_pair_counter():
    pseq_counter: Counter[tuple[bytes, ...]] = Counter(
        {
            (b".",): 1,
            (b" t", b"he"): 10,
            (b" he",): 100,
            (b" t", b"he", b"re", b"he", b"re"): 1000,
            (b"ZZ", b"ZZ", b"ZZ"): 10000,
        }
    )
    expected_pair_counter = Counter(
        {
            (b" t", b"he"): 1010,
            (b"he", b"re"): 2000,
            (b"re", b"he"): 1000,
            (b"ZZ", b"ZZ"): 20000,
        }
    )
    assert get_pair_counter(pseq_counter) == expected_pair_counter


def test_get_pair_to_pseq_counter():
    pseq_counter: Counter[tuple[bytes, ...]] = Counter(
        {
            (b".",): 1,
            (b" t", b"he"): 10,
            (b" he",): 100,
            (b" t", b"he", b"re", b"he", b"re"): 1000,
            (b"ZZ", b"ZZ", b"ZZ"): 10000,
        }
    )
    expected_pair_to_pseq_counter = {
        (b" t", b"he"): Counter(
            {
                (b" t", b"he", b"re", b"he", b"re"): 1,
                (b" t", b"he"): 1,
            }
        ),
        (b"he", b"re"): Counter(
            {
                (b" t", b"he", b"re", b"he", b"re"): 2,
            }
        ),
        (b"re", b"he"): Counter(
            {
                (b" t", b"he", b"re", b"he", b"re"): 1,
            }
        ),
        (b"ZZ", b"ZZ"): Counter(
            {
                (b"ZZ", b"ZZ", b"ZZ"): 2,
            }
        ),
    }
    assert get_pair_to_pseq_counter(pseq_counter) == expected_pair_to_pseq_counter


def test_get_pair_to_pretokens():
    pseqs: set[tuple[bytes, ...]] = {
        (b".",),
        (b" t", b"he"),
        (b" he",),
        (b" t", b"he", b"re", b"he", b"re"),
        (b"ZZ", b"ZZ", b"ZZ"),
    }
    expected_pair_to_pretokens: defaultdict[tuple[bytes, bytes], set[bytes]] = defaultdict(
        set,
        {
            (b" t", b"he"): {b" the", b" therehere"},
            (b"he", b"re"): {b" therehere"},
            (b"re", b"he"): {b" therehere"},
            (b"ZZ", b"ZZ"): {b"ZZZZZZ"},
        },
    )
    assert get_pair_to_pretokens(pseqs) == expected_pair_to_pretokens


def test_get_pairs():
    assert get_pairs((b"Here's", b" a", b"me", b"me")) == ((b"Here's", b" a"), (b" a", b"me"), (b"me", b"me"))


def test_update_caches():
    merge_pair = (b"m", b"e")
    pretoken_counter = Counter(
        {
            b"meme": 1,
            b"tie": 10,
            b"time": 100,
        }
    )
    pretoken_to_pseq: dict[bytes, tuple[bytes, ...]] = {
        b"meme": (b"m", b"e", b"m", b"e"),
        b"tie": (b"ti", b"e"),
        b"time": (b"ti", b"m", b"e"),
    }
    pair_counter = Counter(
        {
            (b"m", b"e"): 102,
            (b"e", b"m"): 1,
            (b"ti", b"e"): 10,
            (b"ti", b"m"): 100,
        }
    )
    pair_to_pretokens = defaultdict(
        set,
        {
            (b"m", b"e"): {b"meme", b"time"},
            (b"e", b"m"): {b"meme"},
            (b"ti", b"e"): {b"tie"},
            (b"ti", b"m"): {b"time"},
        },
    )

    expected_pretoken_counter = Counter(
        {
            b"meme": 1,
            b"tie": 10,
            b"time": 100,
        }
    )
    expected_pretoken_to_pseq = {
        b"meme": (b"me", b"me"),
        b"tie": (b"ti", b"e"),
        b"time": (b"ti", b"me"),
    }
    expected_pair_counter = Counter(
        {
            (b"me", b"me"): 1,
            (b"ti", b"e"): 10,
            (b"ti", b"me"): 100,
        }
    )
    expected_pair_to_pretokens = defaultdict(
        set,
        {
            (b"me", b"me"): {b"meme"},
            (b"ti", b"e"): {b"tie"},
            (b"ti", b"me"): {b"time"},
        },
    )

    update_caches(
        merge_pair,
        pair_counter,
        pair_to_pretokens,
        pretoken_counter,
        pretoken_to_pseq,
    )

    assert pair_counter == expected_pair_counter
    assert pair_to_pretokens == expected_pair_to_pretokens
    assert pretoken_counter == expected_pretoken_counter
    assert pretoken_to_pseq == expected_pretoken_to_pseq
