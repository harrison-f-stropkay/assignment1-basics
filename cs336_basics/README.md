## Notes on training a BPE tokenizer

First, some key terminology:

- token (`bytes`): a contigous sequence of bytes; we aim to partition the corpus into pretokens; merging of two tokens is the only permitted form of merging
- pretoken (`bytes`): a contigous sequence of bytes; we intially partition the corpus into pretokens; merging across pretokens is forbidden
- pair (`tuple[bytes, bytes]`): two consecutive tokens
- pseq (`tuple[bytes, ...]`): the current sequence of tokens that represents a pretoken; pseqs change as we merge tokens to create new tokens

The key data structures that we use to accelerate BPE merging (relative to the native algo):

- `pair_counter`: maps a pair P to the number of occurances in the corpus (each occurance within a single pretoken) of P
- `pretoken_to_pseq`: maps a pretoken to its current pseq representation
- `pretoken_counter`: the number of occurances in the corpus of each pretoken
- `pair_to_pretoken_counter`: maps a pair P to a mapping from each pretoken S that contains P to the current number of occurances of P in S (often there is only 1 occurance of P in S)

One merge stpe of the naive algo (loosely):

- Count all of the pairs in the corpus
- Merge the most frequent pair, creating a new token
- Scan the entire corpus, replacing occurances of the pair with the new token

We have to compute the most common pair before each merge. As a first form of acceleration, we pretokenize.
As a second form of accleration, we don't loop over each pretoken: instead, we simply loop over the pretokens that contain the pair that we're merging. To make this possible, we store pair_to_pretokens.

We'll give an example of one merge step of our algo, assuming that right before we merge, the world looks like this:

```
- pretoken_counter = {meme: 1, tie: 10, time: 100}
- pretoken_to_pseq = {meme: (m, e, m, e), tie: (ti, e), time: (ti, m, e)}
- pair_counter = {(m, e): 102, (e, m): 1, (ti, e): 10, (ti, m): 100}
- pair_to_pretokens = {(m, e): {meme, time}, (e, m): {meme}, (ti, e): {tie}, (ti, m): {time}}
```

These are the steps of `update_caches(merge_pair, pair_counter, pair_to_pretokens, pretoken_counter, pretoken_to_pseq)`:

```
- Notice that `(m, e)` is the most common pair in `pair_counter`; add the new token, `me`, to our vocab
- For `affected_pretoken` in `{meme, time}`  #  `{meme, time}` == `pair_to_pretokens[(m, e)]`
    - For `stale_pair` in `pretoken_to_pseq[affected_pretoken]` #  `[(m, e), (e, m), (m, e)]` and `[(ti, m), (m, e)]`
        - Decrement `pair_counter[stale_pair]` by `pretoken_counter[affected_pretoken]`
        - Remove `affected_pretoken` from `pair_to_pretokens[stale_pair]`
    - Refresh the pseq stored at `pretoken_to_pseq[affected_pretoken]` #  (m, e, m, e) -> (me, me) and (ti, m, e) -> (ti, me)
    - For `fresh_pair` in `pretoken_to_pseq[affected_pretoken]` #  `[(me, me)]` and `[(ti, me)]`
        - Increment `pair_counter[fresh_pair]` by `pretoken_counter[affected_pretoken]`
        - Add `affected_pretoken` to `pair_to_pretokens[fresh_pair]`
```

After the above algo, `pair_to_pretokens[(m, e)]` should be empty and `pair_counter[(m, e)]` should be 0, and we should be ready for the next merge.
