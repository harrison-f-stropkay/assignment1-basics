"""
Let us compute how much memory and compute running AdamW requires. Assume we are using float32 for every tensor.
"""

"""
(a) How much peak memory does running AdamW require? Decompose your answer based on the
memory usage of the parameters, activations, gradients, and optimizer state. Express your
answer in terms of the batch_size and the model hyperparameters (vocab_size,
context_length, num_layers, d_model, num_heads). Assume d_ff = 8 / 3 d_model.
For simplicity, when calculating memory usage of activations, consider only the following components:
- Transformer block
- RMSNorm(s)
- Multi-head self-attention sublayer: QKV projections, QK^T matrix multiply, softmax, weighted sum of values, output projection.
- Position-wise feed-forward (SwiGLU): W_1, W_2, SiLU on the gate branch, element-wise product, W_3
- final RMSNorm
- output embedding
- cross-entropy on logits
"""


def peak_memory(batch_size, vocab_size, context_length, n_layers, d_model, num_heads):
    # parameters
    d_ff = 8 / 3 * d_model
    params_emb = vocab_size * d_model
    params_attn_block_projections = 4 * d_model**2
    params_attn_block_norms = 2 * d_model
    params_attn_block_ffns = 3 * d_model * d_ff
    params_attn_layers = n_layers * (params_attn_block_projections + params_attn_block_norms + params_attn_block_ffns)
    params_final_norm = d_model
    params_unemb = vocab_size * d_model
    n_params = params_emb + params_attn_layers + params_final_norm + params_unemb
    size_params = 4 * n_params

    # activations
    act_attn_block_projections = d_model * 4
    act_attn_block_dot_products = num_heads * context_length
    act_attn_block_softmax = num_heads * context_length
    act_attn_block_weighted_sums = d_model
    act_attn_block_norms = d_model * 2
    act_attn_block_up_projections = d_ff * 2
    act_attn_block_silu = d_ff
    act_attn_block_gated_product = d_ff
    act_attn_block_down_projection = d_model
    act_attn_layers = n_layers * (
        act_attn_block_projections
        + act_attn_block_dot_products
        + act_attn_block_softmax
        + act_attn_block_weighted_sums
        + act_attn_block_norms
        + act_attn_block_up_projections
        + act_attn_block_silu
        + act_attn_block_gated_product
        + act_attn_block_down_projection
    )
    act_final_norm = d_model
    act_unemb = vocab_size
    act_cross_entropy = 1
    size_activations = (
        4 * batch_size * context_length * (act_attn_layers + act_final_norm + act_unemb + act_cross_entropy)
    )

    # gradients
    size_gradients = size_params

    # optimizer state
    size_optimizer = 2 * size_gradients

    # totals
    total_size = size_params + size_activations + size_gradients + size_optimizer
    fixed_cost = size_params + size_gradients + size_optimizer
    marginal_cost = size_activations / batch_size
    return total_size, fixed_cost, marginal_cost


"""
 Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only
depends on the batch_size. What is the maximum batch size you can use and still fit within
80GB memory?
"""


def b():
    batch_size = 1
    while True:
        total, fixed, marginal = peak_memory(
            n_layers=48,
            d_model=1600,
            num_heads=25,
            vocab_size=50257,
            context_length=1024,
            batch_size=batch_size,
        )
        if total < 80 * 10**9:
            batch_size += 1
        else:
            break

    max_batch_size = batch_size - 1
    print("max_batch_size:", max_batch_size)
    total, fixed, marginal = peak_memory(
        n_layers=48,
        d_model=1600,
        num_heads=25,
        vocab_size=50257,
        context_length=1024,
        batch_size=max_batch_size,
    )
    print("total:", total / 10**9)
    print("fixed:", fixed / 10**9)
    print("marginal:", marginal / 10**9)


b()î
