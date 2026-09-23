import torch
import torch.nn as nn

import math

from rope import Rope

class MLAattention(nn.Module):
    def __init__(self, hidden_dim, compressed_dim, rope_dim, num_head):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head

        assert hidden_dim % num_head == 0

        self.head_dim = self.hidden_dim // num_head

        self.compressed_dim = compressed_dim
        self.rope_dim = rope_dim

        # Step 1: Compress the input representation for K and V
        # Instead of directly caching full K and V vectors,
        # we project the input into a smaller latent representation.
        # Shape: [compressed_dim, hidden_dim]
        self.down_proj_weight = nn.Parameter(torch.empty(compressed_dim, hidden_dim))

        # Step 2: Compress the input representation for Q
        # Query uses its own down-projection to create a
        # compressed query representation.
        # Shape: [compressed_dim, hidden_dim]
        self.down_proj_query_weight = nn.Parameter(torch.empty(compressed_dim, hidden_dim))

        # Step 3: Up-project the compressed K and V
        # These projections reconstruct the per-head content
        # components of keys and values from the compressed KV.
        # Output dimension: num_head * head_dim = hidden_dim
        self.up_proj_weight_keys = nn.Parameter(torch.empty(self.head_dim * num_head, compressed_dim))
        self.up_proj_weight_values = nn.Parameter(torch.empty(self.head_dim * num_head, compressed_dim))

        # Reconstruct the query content vectors from the
        # compressed query representation.
        # Output dimension: num_head * head_dim
        self.up_proj_query_weight = nn.Parameter(torch.empty(self.head_dim * num_head,compressed_dim))


        # Step 4: Rotary positional embedding projections
        # Project the input into the rotary component of the key.
        # This component carries positional information.
        # Shape: [rope_dim, hidden_dim]
        self.rope_key_weight = nn.Parameter(torch.empty(self.rope_dim, hidden_dim))

        # Project the compressed query into its rotary component.
        # Shape: [num_head * rope_dim, hidden_dim]
        self.rope_query_weight = nn.Parameter(torch.empty(self.rope_dim * num_head, hidden_dim))

        # Step 5: Output projection
        # Projects the concatenated multi-head attention output
        # back into the model's hidden dimension.
        self.output_weight = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim))

        # Initialize all learnable projection weights.
        for p in self.parameters():
            nn.init.xavier_uniform_(p)

    def forward(self, x, cache=None):

        batch, seq_len, _ = x.shape

        # Step 1: Compress K and V into a shared latent
        # [B, S, hidden_dim] -> [B, S, compressed_dim]
        compressed_kv = x @ self.down_proj_weight.T

        # Step 2: Create the rotary component of the key
        # Project the input into the key's rotary dimensions.
        # [B, S, hidden_dim] -> [B, S, rope_dim]
        k_rope_input = x @ self.rope_key_weight.T

        # Add a head dimension so the rotary key can be
        # shared across attention heads.
        # [B, S, rope_dim] -> [B, 1, S, rope_dim]
        k_rope_input = k_rope_input.unsqueeze(1)

        # Step 3: Determine the position offset from the cache
        # If a cache exists, its sequence length tells us
        # how many tokens have already been processed.
        past_len = 0 if cache is None else cache["compressed_kv_cache"].shape[1]

        position = torch.arange(past_len, past_len + seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)

        # Apply RoPE to the key's rotary component.
        # Result: [B, 1, S, rope_dim]
        k_rope = Rope(position, k_rope_input)

        if cache is None:
            # First forward pass: initialize the cache.
            compressed_kv_cache = compressed_kv
            k_rope_cache = k_rope
        else:
            # Append the new rotary key along the sequence axis.
            # The head dimension remains 1 because this key
            # representation is shared across heads.
            past_len = cache["compressed_kv_cache"].shape[1]
            compressed_kv_cache = torch.cat(
                [cache["compressed_kv_cache"], compressed_kv], dim=1
            )
            k_rope_cache = torch.cat(
                [cache["k_rope_cache"], k_rope], dim=2
            )

        # Total number of tokens represented in the cache.
        total_len = compressed_kv_cache.shape[1]

        # Step 5: Compress and reconstruct the query
        # Compress the input for the query pathway.
        # [B, S, hidden_dim] -> [B, S, compressed_dim]
        query_compressed = x @ self.down_proj_query_weight.T

        # Reconstruct the content query vectors for all heads.
        # [B, S, compressed_dim] -> [B, S, hidden_dim]
        query_c = query_compressed @ self.up_proj_query_weight.T

        # Split the hidden dimension into attention heads.
        # [B, S, hidden_dim] -> [B, H, S, head_dim]
        q_head = query_c.view(batch, seq_len, self.num_head, self.head_dim).transpose(1,2)

        # Step 6: Create the rotary component of the query
        # Project the compressed query into rotary dimensions.
        q_rope_input = query_compressed @ self.rope_query_weight.T
        q_rope_input = q_rope_input.unsqueeze(1)

        # Generate positions for the current query tokens.
        position = torch.arange(past_len, past_len+seq_len, device=x.device).unsqueeze(0).expand(batch, seq_len)

        # Apply RoPE to the query's rotary component.
        q_rope = Rope(position,  q_rope_input)

        # Reshape the rotary query into per-head components.
        q_rope_head = q_rope.view(batch, seq_len, self.num_head, self.rope_dim).transpose(1,2)

        # Step 7: Reconstruct K and V from the cached latent
        # Reconstruct content keys for all cached tokens.
        # [B, total_len, compressed_dim] -> [B, H, total_len, head_dim]
        k_content_new = (compressed_kv_cache @ self.up_proj_weight_keys.T).view(batch, total_len, self.num_head, self.head_dim).transpose(1,2)

        # Reconstruct content values for all cached tokens.
        # [B, total_len, compressed_dim] -> [B, H, total_len, head_dim]
        v_content_new = (compressed_kv_cache @ self.up_proj_weight_values.T).view(batch, total_len, self.num_head, self.head_dim).transpose(1,2)

        # Step 8: Combine content and rotary components
        # Expand the shared rotary key across attention heads.
        # expand() does not copy the underlying data.
        k_rope_cache_view = k_rope_cache.expand(-1, self.num_head, -1, -1)

        # Concatenate content and rotary components.
        # K: [B, H, total_len, head_dim + rope_dim]
        k_combine = torch.cat([k_content_new, k_rope_cache_view], dim=-1)
        # Q: [B, H, S, head_dim + rope_dim]
        q_combine = torch.cat([q_head, q_rope_head], dim=-1)

        # Step 9: Calculate attention scores
        scores = q_combine @ k_combine.transpose(-2,-1)


        scores = scores / math.sqrt(self.head_dim + self.rope_dim)

        # Query positions correspond to the new input tokens.
        q_position = torch.arange(past_len, past_len + seq_len, device=x.device)

        # Key positions cover all tokens in the cache.
        k_position = torch.arange(total_len, device=x.device)

        # Step 10: casual mask only when casual mask = True
        causal_mask = (
            k_position[None, :] <= q_position[:, None]
        )

        # Mask future-token scores with negative infinity,
        # so they receive zero probability after softmax.
        scores = scores.masked_fill(
            ~causal_mask[None, None, :, :],
            float("-inf"),
        )

        attention_score = torch.softmax(scores, dim=-1)
        weights = attention_score @ v_content_new

        # Move sequence dimension before heads.
        # [B, H, S, head_dim] -> [B, S, H, head_dim]
        weights = weights.transpose(1, 2).contiguous()

        # Merge all heads into the hidden dimension.
        # [B, S, H, head_dim] -> [B, S, hidden_dim]
        weights = weights.view(batch, seq_len, self.hidden_dim)

        # Final output projection.
        output = weights @ self.output_weight

        new_cache = {
            "compressed_kv_cache" : compressed_kv_cache,
            "k_rope_cache" : k_rope_cache
        }

        return output, new_cache


