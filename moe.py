import torch
import torch.nn as nn

class ffn(nn.Module):
  def __init__(self, intermediate_dim, hidden_dim):
    super().__init__()
    self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias = False)
    self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias = False)
    self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias = False)
  def forward(self, x):
    return self.down_proj(
        F.silu(self.gate_proj(x)) * self.up_proj(x)
    )

class Moe(nn.Module):
  def __init__(self, intermediate_dim, hidden_dim, num_shared_expert, num_routed_expert, top_k, alpha):
    super().__init__()

    # number of routed_expert
    self.n_r = num_routed_expert

    # number of shared_expert
    self.n_s = num_shared_expert

     # Number of routed experts selected per token
    self.k_r = top_k

    # centroid vector: produces one affinity score for each routed expert
    self.centroid_vector = nn.Linear(hidden_dim, self.n_r)

    self.register_buffer("bias",torch.zeros(self.n_r))

    # routed expert
    self.routed_expert = nn.Modulelist(
        ffn(intermediate_dim, hidden_dim) for _ in range(num_routed_expert)
    )

    # shared expert
    self.shared_expert = nn.Modulelist(
        ffn(intermediate_dim, hidden_dim) for _ in range(num_shared_expert)
    )

    self.alpha = alpha

  def forward(self, x):
    b,s,d = x.shape

   # Compute token-to-expert affinity scores
    score = F.sigmoid(self.centroid_vector(x))

    # Select the top-k routed experts for each token
    _, topk_indices = torch.topk(score + self.bias, k = self.k_r, dim=-1)

    # Get ORIGINAL scores of selected experts
    topk_experts_score = score.gather(dim=-1, index=topk_indices)

    # Normalize the selected expert scores to obtain gating weights
    gating_weights = topk_experts_score/topk_experts_score.sum(dim=-1, keepdim=True)

    # output from shared expert path
    shared_output = sum(expert(x) for expert in self.shared_expert)

    # Residual connection + shared expert output
    output = x +  shared_output

    #  calculation of routed expert
    for expert_idx, expert in enumerate(self.routed_expert):

      #  find: batch_index, token_idx and topk position for tokens routed
      # to this expert
      batch_idx, token_idx, topk_slot = torch.where(
          topk_indices == expert_idx
      )

      # gather tokens assigned to current expert
      expert_input = x[batch_idx, token_idx]

      # apply routed_expert for currect expert
      expert_output = expert(expert_input)

      # select the corresponding normalized scores
      selected_weights = gating_weights[batch_idx, token_idx, topk_slot]

      # Weight each expert output by its routing probability
      expert_output = expert_output * selected_weights.unsqueeze(-1)

      # Add the weighted expert output back to its original tokens
      output[batch_idx, token_idx] += expert_output

      # complementary sequence-wise auxiliary loss

      # Normalize affinity scores across all routed experts for each token
      normalized_scores = score/score.sum(dim=-1, keepdim=True)

      # Compute the average normalized affinity score for each expert
      p_i = normalized_scores.mean(dim=1)

      # Initialize the selection count for each routed expert
      f = torch.zeros(b,self.n_r, device=x.device, dtype=x.dtype)

      # Count how many times each expert is selected in Top-K routing
      for batch_idx in range(b):

        f[batch_idx].scatter_add_(
            0,
            topk_indices[batch_idx].reshape(-1),
            torch.ones(
                topk_indices[batch_idx].numel(),
                device=x.device,
                dtype=x.dtype
            )
        )

      # Normalize expert selection counts to obtain f_i
      f = f * self.n_r/(self.k_r * s)

      # complementary sequence-wise auxiliary loss
      loss_balance = self.alpha * torch.sum(f * p_i, dim=-1)

    return output, loss_balance