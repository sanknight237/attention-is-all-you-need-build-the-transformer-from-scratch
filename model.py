"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    i = 0
    main_dict = {}
    for word in specials:
        main_dict[word] = i
        i = i+1
    
    for sentence in sentences:
        for chunk in sentence.split():
            if chunk not in main_dict:
                main_dict[chunk] = i
                i = i+1

    return main_dict    
# i ki jagah len(main_dict) ko use karna seekh

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    main_dict = {}
    for key,value in token_to_id.items():
        main_dict[value] = key

    return main_dict

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    embeddings = []
    for word in sentence.split():
        if word not in token_to_id:
            embeddings.append(token_to_id[unk_token])
        else:    
            embeddings.append(token_to_id[word])

    return embeddings

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    output = []

    for i in ids:
        output.append(id_to_token[i])

    return output

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    n = len(ids)
    if n < max_len:
        for i in range(0,max_len - n):
            ids.append(pad_id)
        return ids    
    else:
        output = []
        for i in range(0, max_len):
            output.append(ids[i])
        return output

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    tensor_list = []
    for seq in padded_sequences:
        tensor_list.append(torch.tensor(seq).to(dtype=torch.long))

    return torch.stack(tensor_list)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings*(math.sqrt(d_model))

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    output = []
    for i in range(0,d_model//2):
        output.append(1/(10000**(2*i/(d_model))))

    return torch.tensor(output, dtype=torch.float)

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    output = []
    for i in range(0,max_len):
        output.append(i)

    output_t = torch.tensor(output, dtype = torch.float)    
    return output_t.reshape(max_len,1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    angles = position * div_term # concept of broadcasting important

    pe[:, 0::2] = torch.sin(angles)

    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    angles = position * div_term # concept of broadcasting important

    pe[:, 1::2] = torch.cos(angles)

    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    pe = torch.zeros(max_len,d_model)

    position = build_position_index_column(max_len)
    div_term = compute_positional_div_term(d_model)
    
    pe = fill_even_indices_with_sin(pe, position, div_term)
    pe = fill_odd_indices_with_cos(pe, position, div_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.

    B, L, d_model = embedded_batch.shape
    
    return embedded_batch + positional_encoding[:L]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    B, L = token_ids.shape
    
    mask = (token_ids != pad_id) # important

    return mask.reshape(B, 1, 1 ,L)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)

    mask = torch.tril(mask)

    return mask.reshape(1, 1, seq_len, seq_len)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    mask = (padding_mask & causal_mask) 
    # broadcasting saves the day 

    return mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    return query.matmul(key.transpose(-2, -1))

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores/(math.sqrt(d_k))

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    # ~mask flips True to False, and False to True. 
    # This targets every position where the mask was False and overwrites it with -inf.

    return scores.masked_fill(~mask, float('-inf'))

# Step 20 - softmax_attention_weights
import torch
import torch.nn.functional as F

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    is_all_neg_inf = torch.all(masked_scores == float('-inf'), dim = -1)

    weights = F.softmax(masked_scores, dim = -1)

    weights = weights.masked_fill(is_all_neg_inf.unsqueeze(-1), 0.0)

    return weights

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return attention_weights.matmul(value)

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    raw_scores = compute_raw_attention_scores(query, key)

    d_k = query.shape[-1]
    scaled_scores = scale_attention_scores(raw_scores, d_k)

    if mask == None:
        masked_scores = scaled_scores
    else:    
        masked_scores = mask_attention_scores_with_neg_inf(scaled_scores, mask)
    
    attention_weights = softmax_attention_weights(masked_scores)

    return (apply_attention_weights_to_values(attention_weights, value), attention_weights)

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B, L, d_model = tensor.shape
    return tensor.reshape(B, L, num_heads, d_model//num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).

    return split_tensor.transpose(-3,-2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    B, H, L, d_k = multi_head_tensor.shape

    flip = multi_head_tensor.transpose(-3, -2).contiguous()

    return flip.reshape(B, L, H*d_k)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    if bias is None:
        return x.matmul(weight.transpose(-1,-2))
    else:
        return x.matmul(weight.transpose(-1,-2)) + bias

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    if b_q is None:
        query = x.matmul(w_q.transpose(-1,-2))
    else:
        query = x.matmul(w_q.transpose(-1,-2)) + b_q

    if b_k is None:
        key =   x.matmul(w_k.transpose(-1,-2))
    else:
        key =   x.matmul(w_k.transpose(-1,-2)) + b_k

    if b_v is None:
        value = x.matmul(w_v.transpose(-1,-2))
    else:
        value = x.matmul(w_v.transpose(-1,-2)) + b_v

    return (query, key, value)

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    
    q = split_last_dim_into_heads(q, num_heads)
    q = transpose_heads_before_sequence(q)
    k = split_last_dim_into_heads(k, num_heads)
    k = transpose_heads_before_sequence(k)
    v = split_last_dim_into_heads(v, num_heads)
    v = transpose_heads_before_sequence(v)

    return (q, k, v)

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    
    return apply_linear_projection(merge_heads_back_to_model_dim(context), w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    
    q = query.matmul(w_q.transpose(-1, -2)) 
    k = key.matmul(w_k.transpose(-1, -2)) 
    v = value.matmul(w_v.transpose(-1, -2)) 

    (q, k, v) = split_qkv_into_heads(q, k, v, num_heads)

    (context, weights) = scaled_dot_product_attention(q, k, v, mask)

    return merge_heads_and_project_output(context, w_o, b_o=None)

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    z = x.matmul(w1) + b1
    return torch.relu(z)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return hidden.matmul(w2) + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    return apply_ffn_second_linear(apply_ffn_first_linear_and_relu(x, w1, b1), w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    
    return (torch.mean(x, dim=-1, keepdim=True), torch.var(x, dim=-1, keepdim=True, correction=0))

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean, variance = compute_layer_norm_mean_and_variance(x)

    x_ = (x - mean)/(torch.sqrt(variance + eps))

    return x_*gamma + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    return normalize_and_scale_with_gamma_beta(residual_input + sublayer_output, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.

    numeric_mask = keep_mask.to(dtype = x.dtype)
    
    return (x*numeric_mask)/keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    
    sublayer_output = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)

    return apply_residual_add_and_norm(x, sublayer_output, gamma, beta, eps = 1e-5)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    
    sublayer_output = position_wise_feed_forward_network(x, w1, b1, w2, b2)

    return apply_residual_add_and_norm(x, sublayer_output, gamma, beta, eps = 1e-5)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    w_q = layer_params['w_q']
    w_k = layer_params['w_k']
    w_v = layer_params['w_v']
    w_o = layer_params['w_o']
    attn_gamma = layer_params['attn_gamma']
    attn_beta = layer_params['attn_beta']
    w1 = layer_params['w1']
    b1 = layer_params['b1']
    w2 = layer_params['w2']
    b2 = layer_params['b2']
    ffn_gamma = layer_params['ffn_gamma']
    ffn_beta = layer_params['ffn_beta']

    attn_output = encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, attn_gamma, attn_beta, num_heads, src_mask)
    ffn_output = encoder_layer_feed_forward_sublayer(attn_output, w1, b1, w2, b2, ffn_gamma, ffn_beta)
    return ffn_output

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    output = x
    for param_dict in encoder_layer_params_list:
        output =  assemble_encoder_layer(output,param_dict, num_heads, src_mask)

    return output

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    
    output = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)

    return apply_residual_add_and_norm(y, output, gamma, beta, eps=1e-5)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    
    if src_mask is not None and src_mask.dim() == 2:
        src_mask = src_mask[:, None, None, :]   # (B, 1, 1, Lk)

    sublayer_output = assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, src_mask)

    return apply_residual_add_and_norm(y, sublayer_output, gamma, beta, eps =1e-5)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    
    sublayer_output = position_wise_feed_forward_network(y, w1, b1, w2, b2)

    return apply_residual_add_and_norm(y, sublayer_output, gamma, beta, eps=1e-5)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    # masked self-attention params
    w_q_self = layer_params['w_q_self']
    w_k_self = layer_params['w_k_self']
    w_v_self = layer_params['w_v_self']
    w_o_self = layer_params['w_o_self']
    self_gamma = layer_params['self_gamma']
    self_beta = layer_params['self_beta']

    # cross-attention params
    w_q_cross = layer_params['w_q_cross']
    w_k_cross = layer_params['w_k_cross']
    w_v_cross = layer_params['w_v_cross']
    w_o_cross = layer_params['w_o_cross']
    cross_gamma = layer_params['cross_gamma']
    cross_beta = layer_params['cross_beta']

    # feed-forward params
    w1 = layer_params['w1']
    b1 = layer_params['b1']
    w2 = layer_params['w2']
    b2 = layer_params['b2']
    ffn_gamma = layer_params['ffn_gamma']
    ffn_beta = layer_params['ffn_beta']

    self_attn_output = decoder_layer_masked_self_attention_sublayer(
        y, w_q_self, w_k_self, w_v_self, w_o_self, self_gamma, self_beta, num_heads, tgt_mask)

    cross_attn_output = decoder_layer_cross_attention_sublayer(
        self_attn_output, encoder_output, w_q_cross, w_k_cross, w_v_cross, w_o_cross,
        cross_gamma, cross_beta, num_heads, src_mask)

    return decoder_layer_feed_forward_sublayer(
        cross_attn_output, w1, b1, w2, b2, ffn_gamma, ffn_beta)

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    
    output = y
    for layer_params in decoder_layer_params_list:
        output = assemble_decoder_layer(output, encoder_output, layer_params, num_heads, src_mask, tgt_mask)

    return output

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.transpose(-1,-2)

# Step 50 - apply_log_softmax_over_vocab
import torch.nn.functional as F

def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    
    return F.log_softmax(logits, dim=-1)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    
    # d_model = model_params["d_model"] this doesn't work so here is a little hack
    token_embedding = model_params["token_embedding"]       # (V, d_model)
    d_model = model_params.get("d_model", token_embedding.shape[-1])
    
    B, Ls = src_ids.shape
    _, Lt = tgt_ids.shape

    # Embeddings:
    pe = build_sinusoidal_positional_encoding(max(Ls, Lt), d_model)

    src_embeddings_scaled = scale_embeddings_by_sqrt_d_model(token_embedding[src_ids], d_model)
    src_embeddings_final = add_positional_encoding_to_embeddings(src_embeddings_scaled, pe)

    tgt_embeddings_scaled = scale_embeddings_by_sqrt_d_model(token_embedding[tgt_ids], d_model)
    tgt_embeddings_final = add_positional_encoding_to_embeddings(tgt_embeddings_scaled, pe)

    # Masks:
    src_mask = build_padding_mask(src_ids, pad_id)
    tgt_pad_mask = build_padding_mask(tgt_ids, pad_id)
    causal_mask = build_causal_mask(Lt)
    tgt_mask = combine_padding_and_causal_masks(tgt_pad_mask, causal_mask)

    # Encoder-Decoder:
    encoder_output = stack_encoder_layers(src_embeddings_final, model_params['encoder_layers'], num_heads, src_mask)
    decoder_output = stack_decoder_layers(tgt_embeddings_final, encoder_output, model_params['decoder_layers'], num_heads, src_mask, tgt_mask)

    # Output projection:
    output_projection = model_params["output_projection"]

    if output_projection.shape[-1] != d_model:
        output_projection = output_projection.transpose(-1, -2)

    logits = apply_final_output_projection(decoder_output, output_projection, None)
    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
import torch

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    
    # Standard Xavier spread scale: sqrt(2.0 / (in_features + out_features))
    attn_scale = math.sqrt(2.0 / (d_model + d_model))
    w1_scale = math.sqrt(2.0 / (d_model + d_ff))
    w2_scale = math.sqrt(2.0 / (d_ff + d_model))

    dict_main = {
        # Attention Projections (d_model -> d_model)
        'w_q': torch.randn(d_model, d_model, dtype=torch.float32)*attn_scale,
        'w_k': torch.randn(d_model, d_model, dtype=torch.float32)*attn_scale,
        'w_v': torch.randn(d_model, d_model, dtype=torch.float32)*attn_scale,
        'w_o': torch.randn(d_model, d_model, dtype=torch.float32)*attn_scale,
        
        # Feed-Forward Network Projections (Expansion & Contraction)
        'w1': torch.randn(d_model, d_ff, dtype=torch.float32)*w1_scale,
        'b1': torch.zeros(d_ff, dtype=torch.float32),
        'w2': torch.randn(d_ff, d_model, dtype=torch.float32)*w2_scale,
        'b2': torch.zeros(d_model, dtype=torch.float32),
        
        # Attention Sublayer Normalization Parameters
        'attn_gamma': torch.ones(d_model, dtype=torch.float32),
        'attn_beta': torch.zeros(d_model, dtype=torch.float32),
        
        # Feed-Forward Sublayer Normalization Parameters
        'ffn_gamma': torch.ones(d_model, dtype=torch.float32),
        'ffn_beta': torch.zeros(d_model, dtype=torch.float32)
    }
    for name, tensor in dict_main.items():
        tensor.requires_grad = True
    return dict_main

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    import torch
import math

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    """Initialize all parameters for a single Transformer Decoder Layer
    with a proper random spread (Xavier/Glorot Normal distribution).
    """
    params = {}
    
    # 1. Masked Self-Attention Sublayer Weights
    attn_scale = math.sqrt(2.0 / (d_model + d_model))
    params['w_q_self'] = torch.randn(d_model, d_model) * attn_scale
    params['w_k_self'] = torch.randn(d_model, d_model) * attn_scale
    params['w_v_self'] = torch.randn(d_model, d_model) * attn_scale
    params['w_o_self'] = torch.randn(d_model, d_model) * attn_scale
    
    # 2. Cross-Attention (Encoder-Decoder) Sublayer Weights
    params['w_q_cross'] = torch.randn(d_model, d_model) * attn_scale
    params['w_k_cross'] = torch.randn(d_model, d_model) * attn_scale
    params['w_v_cross'] = torch.randn(d_model, d_model) * attn_scale
    params['w_o_cross'] = torch.randn(d_model, d_model) * attn_scale
    
    # 3. Position-Wise Feed-Forward Network (FFN) Weights & Biases
    # W1 expands: d_model -> d_ff
    w1_scale = math.sqrt(2.0 / (d_model + d_ff))
    params['w1'] = torch.randn(d_model, d_ff) * w1_scale
    params['b1'] = torch.zeros(d_ff)
    
    # W2 contracts: d_ff -> d_model
    w2_scale = math.sqrt(2.0 / (d_ff + d_model))
    params['w2'] = torch.randn(d_ff, d_model) * w2_scale
    params['b2'] = torch.zeros(d_model)
    
    # 4. Layer Normalization Parameters (Gains and Biases)
    # 3 sublayers mean we need 3 sets of LayerNorm parameters
    params['self_gamma'] = torch.ones(d_model)
    params['self_beta'] = torch.zeros(d_model)
    
    params['cross_gamma'] = torch.ones(d_model)
    params['cross_beta'] = torch.zeros(d_model)
    
    params['ffn_gamma'] = torch.ones(d_model)
    params['ffn_beta'] = torch.zeros(d_model)
    
    # Ensure gradients track across backpropagation for all leaf parameters
    for name, tensor in params.items():
        tensor.requires_grad = True
        
    return params

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    params = {}

    params['src_embedding'] = torch.randn(vocab_size, d_model, dtype=torch.float32,requires_grad=True)
    params['tgt_embedding'] = torch.randn(vocab_size, d_model, dtype=torch.float32,requires_grad=True)
    if tie_weights:
        params['output_projection'] = params['tgt_embedding']
    else:
        params['output_projection'] = torch.randn(vocab_size, d_model, dtype=torch.float32, requires_grad = True)    

    return params

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    
    tensors = []
    seen_ids = set()
    
    all_dicts = [encoder_layer_params, decoder_layer_params, [embedding_params]]

    for block in all_dicts:
        for param_dict in block:
            for key, value in param_dict.items():
                if id(value) not in seen_ids:
                    tensors.append(value)
                    seen_ids.add(id(value))

    return tensors

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    output = target_ids
    output = output[:,:-1]
    B,_ = target_ids.shape
    element_to_add = start_token_id
    new_col_having_bos = torch.full((target_ids.shape[0], 1), element_to_add)

    return torch.cat((new_col_having_bos, output), dim=1)

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    if step < 1:
        step = 1

    # warmup terms:
    warmup_term = step*(warmup_steps**-1.5)

    # decay terms:
    decay_term = step**-0.5

    # structural dimemsional scaling factor 
    d_model_factor = d_model**-0.5

    return d_model_factor*min(warmup_term, decay_term)

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    
    return torch.ones(shape, dtype=torch.float32)*(epsilon/(vocab_size-2))

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    output_ids = smoothed_distribution.clone()

    columns_which_are_to_be_filled = gold_token_ids.unsqueeze(-1)

    tensor_having_the_value_to_be_filled_acc_to_indices = torch.full_like(columns_which_are_to_be_filled, confidence, dtype = output_ids.dtype)

    return output_ids.scatter_(dim=-1, index=columns_which_are_to_be_filled, src=tensor_having_the_value_to_be_filled_acc_to_indices)

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
    output_ids = smoothed_distribution.clone()

    gold_token_ids_long = gold_token_ids.to(torch.long)

    output_ids[...,pad_id] = 0.0 # whole pad column set to 0

    is_pad_row = gold_token_ids_long == pad_id # whole pad row to 0 (boolean)
    is_pad_row_3d = is_pad_row.unsqueeze(-1) # making row mask 3d


    return torch.where(is_pad_row_3d, torch.zeros_like(output_ids), output_ids)

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    
    return -torch.sum(smoothed_distribution*log_probabilities) + 0.0

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    
    N = torch.sum(gold_token_ids != pad_id)
    return total_loss/max(N,1)

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only
    
    arg_max_tensor = torch.argmax(log_probabilities, dim=-1)

    N = torch.sum(gold_token_ids != pad_id)

    predictions_compared_with_gold_ids = (arg_max_tensor == gold_token_ids)

    return (torch.sum(torch.where((gold_token_ids == pad_id), torch.zeros_like(gold_token_ids), predictions_compared_with_gold_ids)))/max(N,1)

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    m_list = []
    v_list = []
    return_dict = {}

    for p in parameter_list:
        m_list.append(torch.zeros_like(p))
        v_list.append(torch.zeros_like(p))

    t = 0
    return_dict['t'] = t
    return_dict['m'] = m_list
    return_dict['v'] = v_list
    
    return return_dict

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    
    clean_grad = grad.detach() # detached from the grad engine of pytorch
    return beta1*m_prev + (1-beta1)*clean_grad

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    
    cleaned_grad = grad.detach()
    return beta2*v_prev + (1-beta2)*(cleaned_grad**2)

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    
    return (m_t/(1-(beta1**step)), v_t/(1-(beta2**step)))

# Step 68 - compute_adam_parameter_update
import torch

def compute_adam_parameter_update(m_hat, v_hat, learning_rate, epsilon):
    """Return delta = learning_rate * m_hat / (sqrt(v_hat) + epsilon); the caller subtracts it."""
    # TODO: compute the Adam step from the bias-corrected moments without tracking gradients
    
    with torch.no_grad():
        delta = (learning_rate*m_hat)/(torch.sqrt(v_hat) + epsilon)

    return delta # note : here i already used detach so no need to worry varna use with torch.no_grad():

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    # TODO: increment t, then for each param with a grad update m, v, bias-correct, and subtract delta in place.
    optimizer_state['t'] += 1
    t = optimizer_state['t']

    with torch.no_grad():
        for i, p in enumerate(parameter_list):
            if(p.grad is None):
                continue
        
            p_grad = p.grad
            m_t = update_adam_first_moment(optimizer_state['m'][i], p_grad, beta1)
            v_t = update_adam_second_moment(optimizer_state['v'][i], p_grad, beta2)

            optimizer_state['m'][i] = m_t
            optimizer_state['v'][i] = v_t

            m_hat, v_hat = apply_adam_bias_correction(m_t, v_t, beta1, beta2, t)

            delta = compute_adam_parameter_update(m_hat, v_hat, learning_rate, epsilon)

            p.sub_(delta)

    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    # TODO: clear the accumulated gradient on every parameter tensor in the list
    for p in parameter_list:
        p.grad = None

# Step 71 - compute_batch_training_loss
def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    """One teacher-forced forward pass -> label-smoothed KL loss averaged over non-pad tokens.

    model_params (read only; the same dict run_transformer_forward takes):
      token_embedding, output_projection, encoder_layers, decoder_layers
    config keys:
      pad_id, start_id, vocab_size, smoothing, num_heads
    """
    # TODO: shift targets right, run the forward pass, build smoothed targets, and average the KL loss over non-pad tokens.
    pad_id = config['pad_id']
    start_id = config['start_id']
    vocab_size = config['vocab_size']
    smoothing = config['smoothing']
    num_heads = config['num_heads']

    right_shifted_tgt_input_for_decoder = shift_targets_right_with_start_token(tgt_batch, start_id)
    log_softmax_over_vocab = run_transformer_forward(src_batch, right_shifted_tgt_input_for_decoder, model_params, num_heads, pad_id)

    smoothed_distribution = build_uniform_smoothing_distribution(log_softmax_over_vocab.shape, vocab_size, smoothing)
    confidenced_gold_ids = set_confidence_on_gold_tokens(smoothed_distribution, tgt_batch, 1-smoothing)
    pad_rc_zeroed = zero_pad_column_and_pad_token_rows(confidenced_gold_ids, tgt_batch, pad_id)

    kl_loss = compute_label_smoothed_kl_loss(log_softmax_over_vocab, pad_rc_zeroed)
    return average_loss_over_non_pad_tokens(kl_loss, tgt_batch, pad_id)

# Step 72 - run_training_step_with_backprop
import torch

def run_training_step_with_backprop(src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step_number, config):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.

    Returns the scalar loss value for the step as a Python float.
    """
    # TODO: zero grads, compute loss, backward, look up Noam LR, apply Adam step

    zero_all_parameter_gradients(parameter_list) # zero grads

    average_loss = compute_batch_training_loss(src_batch, tgt_batch, model_params, config)
    average_loss.backward()

    learning_rate = compute_noam_learning_rate(step_number, config['d_model'], config['warmup_steps'])
    apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate)

    return average_loss.item()

# Step 73 - run_training_loop_for_steps
def run_training_loop_for_steps(batches, parameter_list, model_params, optimizer_state, num_steps, config):
    """Run num_steps training iterations, cycling through batches, and return per-step losses."""
    # TODO: iterate for num_steps steps, calling run_training_step_with_backprop each time
    
    losses = []
    for step_number in range(1,num_steps+1):
        src_batch, tgt_batch = batches[(step_number-1)%(len(batches))]
        losses.append(run_training_step_with_backprop(src_batch, tgt_batch, parameter_list, model_params, optimizer_state, step_number, config))

    return losses

# Step 74 - pick_next_token_by_argmax
import torch

def pick_next_token_by_argmax(final_step_logits):
    """Greedy: return argmax token id per batch row.

    final_step_logits: FloatTensor of shape (batch, vocab_size)
    returns: LongTensor of shape (batch,)
    """
    # TODO: pick the next greedy token id by taking the argmax over the vocab axis
    
    return torch.argmax(final_step_logits, dim = -1)

# Step 75 - compute_length_penalty
def compute_length_penalty(sequence_length, alpha):
    # TODO: return the Google NMT length penalty for the given sequence_length and alpha.
    return ((5+sequence_length)/6)**alpha

# Step 76 - compute_candidate_scores
import torch

def compute_candidate_scores(beam_scores, next_token_log_probs):
    # TODO: add each beam's running log-prob to its row of next-token log probs.
    
    return beam_scores.unsqueeze(-1) + next_token_log_probs

# Step 77 - select_top_k_candidates
import torch

def select_top_k_candidates(candidate_scores, k):
    # TODO: pick the top k (beam_index, token_id, score) triples from candidate_scores
    out = {}
    flat = candidate_scores.reshape(-1)
    vocab_size = candidate_scores.shape[-1]

    scores, flat_indices = torch.topk(flat, k=k, largest=True, sorted=True)

    out['scores'] = scores
    out['beam_indices'] = flat_indices//vocab_size
    out['token_ids'] = flat_indices % vocab_size

    return out

# Step 78 - append_tokens_to_beam_sequences
import torch

def append_tokens_to_beam_sequences(beam_sequences, beam_indices, token_ids):
    # TODO: gather parent beam rows and append the new token ids as the last column
    parents = beam_sequences[beam_indices]

    return torch.cat((parents, token_ids.unsqueeze(-1)), dim=1).to(dtype=beam_sequences.dtype)

# Step 79 - mark_finished_beams
import torch

def mark_finished_beams(token_ids, finished_flags, end_token_id):
    # TODO: return updated boolean finished flags for each beam given the new token ids
    
    return ((token_ids == end_token_id) | finished_flags)

# Step 80 - select_best_finished_beam
def select_best_finished_beam(finished_sequences, finished_scores, alpha):
    # TODO: return the finished beam with the highest length-penalized score
    out = {}
    max_score = float('-inf')
    idx_having_max_score = 0
    for i, score in enumerate(finished_scores):

        real_score = score/compute_length_penalty(len(finished_sequences[i]), alpha)

        if real_score > max_score:
            max_score = real_score
            idx_having_max_score = i
            

    out['sequence'] =  finished_sequences[idx_having_max_score]
    out['score'] = max_score

    return out

