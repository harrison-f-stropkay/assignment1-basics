from cs336_basics.transformer_lm import TransformerLM


# (a)
# my math came out to 1,640,452,800
v = 50257
s = 1024
n_layers = 48
d_m = 1600
h = 25
d_feed = 4288
r_theta = 10_000

emb = v * d_m

block_projections = 4 * d_m**2
block_norms = 2 * d_m
block_ffns = 3 * d_m * d_feed
layers = n_layers * (block_projections + block_norms + block_ffns)

final_norm = d_m
unemb = v * d_m
print("hand-computed # of model paramters:", emb + layers + final_norm + unemb)

model = TransformerLM(
    v,
    s,
    n_layers,
    d_m,
    h,
    d_feed,
    r_theta,
)
print("torch-computed # of model paramters:", sum(p.numel() for p in model.parameters()))
# if each parameter is represented using single-precision floating point, we need the following memory to load the model:
# 1640452800 * 4 / 1024 ** 3 ~= 6.11 GiB


# (b, d)
def estimate_flops(num_layers, d_model, num_heads, vocab_size=50257, context_length=1024):
    d_ff = round(8 / 3 * d_model / 64) * 64

    attn_projections = 4 * 2 * context_length * d_model**2
    attn_scores = 2 * context_length**2 * d_model
    att_weighted_sums = 2 * context_length**2 * d_model
    attn = num_layers * (attn_projections + attn_scores + att_weighted_sums)

    ffn = num_layers * (3 * 2 * context_length * d_model * d_ff)

    unembedding = 2 * context_length * vocab_size * d_model

    total = attn + ffn + unembedding
    print("% of FLOPs for unembedding", 100 * unembedding / total)
    print("% of FLOPs for attn", 100 * attn / total)
    print("% of FLOPs for ffn", 100 * ffn / total)
    print("total FLOPs:", total / 10**9, "billion\n")


print("GPT-2 small")
estimate_flops(12, 768, 12)
print("GPT-2 medium")
estimate_flops(24, 1024, 16)
print("GPT-2 large")
estimate_flops(36, 1280, 20)
print("GPT-2 XL")
estimate_flops(48, 1600, 25)

# as model size increases:
# - unembedding FLOPs decrease
# - attn FLOPs decrease
# - ffn FLOPs increase

# (c)
# attn and ffn

# (e)
print("GPT-2 XL with a context length of 16384")
estimate_flops(48, 1600, 25, context_length=16384)
# Total FLOPs went 38x. Sanity check: since in GPT-2 large, ~38% of FLOPs were for attn, ~1/3 of those flops have seq_len^2, and seq_len^2 went 256x, so ~13% of the model went 256x, so ~13% of the model turned into ~3300% of the model.
# Makes sense that % of FLOPs on attn went up: attn is the only part of the model with seq_len^2.

"""
output:

hand-computed # of model paramters: 1640452800
torch-computed # of model paramters: 1640452800
GPT-2 small
% of FLOPs for unembedding 27.103680733450183
% of FLOPs for attn 33.13469057570446
% of FLOPs for ffn 39.761628690845356
total FLOPs: 291.6483072 billion

GPT-2 medium
% of FLOPs for unembedding 12.695746191175097
% of FLOPs for attn 37.24981495843196
% of FLOPs for ffn 50.05443885039295
total FLOPs: 830.172299264 billion

GPT-2 large
% of FLOPs for unembedding 7.449443481792539
% of FLOPs for attn 38.24967649460972
% of FLOPs for ffn 54.300880023597735
total FLOPs: 1768.53090304 billion

GPT-2 XL
% of FLOPs for unembedding 4.682766929455207
% of FLOPs for attn 37.78340770363938
% of FLOPs for ffn 57.53382536690542
total FLOPs: 3516.7698944 billion

GPT-2 XL with a context length of 16384
% of FLOPs for unembedding 1.9725699850812055
% of FLOPs for attn 73.79186613669164
% of FLOPs for ffn 24.235563878227154
total FLOPs: 133577.7296384 billion
"""
