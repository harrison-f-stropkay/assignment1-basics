import time
from pathlib import Path
from cs336_basics.tokenizer import Tokenizer

DATA_DIR = Path(__file__).parents[1] / "data"


def a():
    for vocab_merges_filepath, corpus_path in zip(
        [
            DATA_DIR / "TinyStoriesV2-GPT4-train_vocab_merges.pkl",
            DATA_DIR / "owt_train_vocab_merges.pkl",
        ],
        [
            DATA_DIR / "TinyStoriesV2-GPT4-train.txt",
            DATA_DIR / "owt_train.txt",
        ],
    ):
        with open(corpus_path) as f:
            raw = f.read(10**6)

        sample_10 = raw.split("<|endoftext|>")[:10]
        n_bytes = sum([len(bytes(x, "utf-8")) for x in sample_10])

        tokenizer = Tokenizer.from_file(vocab_merges_filepath)
        n_tokens = sum([len(tokenizer.encode(sample)) for sample in sample_10])

        print(f"Compression ratio (bytes/token): {n_bytes / n_tokens}")

        # Compression ratio (bytes/token): 4.112278761061947
        # Compression ratio (bytes/token): 4.690451363026963


def b():
    with open(DATA_DIR / "owt_train.txt") as f:
        raw = f.read(10**6)

    sample_10 = raw.split("<|endoftext|>")[:10]
    n_bytes = sum([len(bytes(x, "utf-8")) for x in sample_10])

    tokenizer = Tokenizer.from_file(DATA_DIR / "TinyStoriesV2-GPT4-train_vocab_merges.pkl")
    n_tokens = sum([len(tokenizer.encode(sample)) for sample in sample_10])

    print(f"Compression ratio (bytes/token): {n_bytes / n_tokens}")

    # Compression ratio (bytes/token): 3.1892028765319558


def c():
    start = time.time()
    with open(DATA_DIR / "TinyStoriesV2-GPT4-train.txt") as f:
        raw = f.read(10**6)

    sample_100 = raw.split("<|endoftext|>")[:100]
    n_bytes = sum([len(bytes(x, "utf-8")) for x in sample_100])

    tokenizer = Tokenizer.from_file(DATA_DIR / "TinyStoriesV2-GPT4-train_vocab_merges.pkl")
    [(tokenizer.encode(sample)) for sample in sample_100]
    elapsed = time.time() - start

    throughput = n_bytes / elapsed
    print(f"Throughput (bytes/second): {throughput}")
    est_pile = 825 * 10**9 / throughput
    print(f"Estimated time to tokenize the Pile dataset (825GB of text): {est_pile / 3600 / 24} days")

    # Throughput (bytes/second): 78600.61021475201
    # Estimated time to tokenize the Pile dataset (825GB of text): 121.48265878626727 days


# def d():
#     def txt_to_token_ids(
#         txt_path: Path,
#         output_path: Path,
#         vocab_merges_filepath: Path,
#     ):
#         tokenizer = Tokenizer.from_file(vocab_merges_filepath)
#         token_ids = []
#         with open(txt_path) as f:
#             # from the example code:
#             # Get total file size in bytes
#             f.seek(0, os.SEEK_END)
#             file_size = f.tell()
#             f.seek(0)

#             with tqdm(total=file_size) as pbar:
#                 for line in f:
#                     pbar.update(len(bytes(line, "utf-8")))
#                     token_ids.extend(tokenizer.encode(line))

#         token_ids_numpy = np.array(token_ids, dtype=np.uint16)
#         np.save(str(output_path), token_ids_numpy)
#         print(f"Saved token ids to {output_path}")

#     txt_to_token_ids(
#         DATA_DIR / "TinyStoriesV2-GPT4-valid.txt",
#         DATA_DIR / "TinyStoriesV2-GPT4-valid.npy",
#         DATA_DIR / "TinyStoriesV2-GPT4-train_vocab_merges.pkl",
#     )
#     # txt_to_token_ids(
#     #     DATA_DIR / "tester.txt",
#     #     DATA_DIR / "tester.npy",
#     #     DATA_DIR / "TinyStoriesV2-GPT4-train_vocab_merges.pkl",
#     # )


if __name__ == "__main__":
    a()
    b()
    c()
    # d()
