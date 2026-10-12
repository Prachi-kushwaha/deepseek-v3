import torch
import torch.nn as nn
import torch.nn.functional as F

class ffn(nn.Module):
  def __init__(self, model_config):
    super().__init__()
    self.gate_proj = nn.Linear(model_config.hidden_dim, model_config.intermediate_dim, bias = False)
    self.up_proj = nn.Linear(model_config.hidden_dim, model_config.intermediate_dim, bias = False)
    self.down_proj = nn.Linear(model_config.intermediate_dim, model_config.hidden_dim, bias = False)
  def forward(self, x):
    return self.down_proj(
        F.silu(self.gate_proj(x)) * self.up_proj(x)
    )

class Moe(nn.Module):
  def __init__(self, model_config):
    super().__init__()

    # number of routed_expert
    self.n_r = model_config.nr

    # number of shared_expert
    self.n_s = model_config.ns

     # Number of routed experts selected per token
    self.k_r = model_config.kr

    # centroid vector: produces one affinity score for each routed expert
    self.centroid_vector = nn.Linear(model_config.hidden_dim, self.n_r)

    self.register_buffer("bias",torch.zeros(self.n_r))

    # routed expert
    self.routed_expert = nn.ModuleList(
        ffn(model_config) for _ in range(model_config.nr)
    )

    # shared expert
    self.shared_expert = nn.ModuleList(
        ffn(model_config) for _ in range(model_config.ns)
    )

    self.alpha = model_config.alpha

  def forward(self, x):
    b,s,_ = x.shape

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
    output = shared_output

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

      # Count top-K selections separately for each sequence
      f = torch.zeros(
          b, self.n_r,
          device=x.device,
          dtype=score.dtype
      )

      f.scatter_add_(
          dim=1,
          index=topk_indices.reshape(b, -1),
          src=torch.ones_like(
              topk_indices.reshape(b, -1),
              dtype=score.dtype
          )
      )

      # Normalize expert selection counts to obtain f_i
      f = f * self.n_r/(self.k_r * s)

      # complementary sequence-wise auxiliary loss
      loss_balance = self.alpha * torch.sum(f * p_i, dim=-1).mean()

    return output, loss_balance