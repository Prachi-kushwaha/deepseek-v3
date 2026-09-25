import torch
import torch.nn as nn

class FFn(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ffn_net = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.GELU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
    def forward(self, input):
        return self.ffn_net(input)



class MOE_layer(nn.Module):
    def __init__(self, nr, ns, kr, hidden_dim ):
        super().__init__()

        # number of topk experts
        self.kr = kr

        # number of routed experts
        self.nr = nr

        # number of shared experts
        self.ns = ns

        # learnable centroid vector
        self.centrod_vector = nn.Parameter(torch.empty(hidden_dim, kr))
        nn.init.xavier_uniform(self.centrod_vector)

        # extra bias term to balance biasness of auxialiary load balance
        self.bias = nn.Parameter(torch.empty(self.kr, hidden_dim))
        nn.init.xavier_uniform(self.bias)

        #  shared ffn layer
        self.shared_ffn = nn.ModuleList([
            FFn(hidden_dim) for i in range(ns)
        ])

        #  router ffn layer
        self.routed_ffn = nn.ModuleList([
            FFn(hidden_dim) for i in range(nr)
        ])


    def forward(self, input):
        # applying sigmoid to find out scores for our token (sigmoid gives us value between 0 and 1)
        scores = torch.sigmoid(input.T, self.centrod_vector)

        #  torch.topk gives top k values from scores
        topk_experts_score, topk_indices = torch.topk(scores, self.kr, dim=-1)

        #  gated weights therefore after expert output we can apply how much of expert output we want to use
        gated_weights = topk_experts_score/topk_experts_score.sum(dim=-1, keepdim=True) + self.bias

        #  shared output sum bcs here in deeepseek moe we have two kind of expert shared means all token pass through this and routed from which we select topk and attend them
        shared_output = sum(
            expert(input) for expert in self.shared_ffn
        )

        routed_output = torch.zeros_like(input)

        #  traverse all the experts in routed_ffn and if topt_index match with expert_id we perform our operation
        for expert_id, expert in range(self.routed_ffn):
            token_idx, topk_slot =torch.where(
                topk_indices == expert_id
            )


            expert_input = input[token_idx]
            expert_output = expert(expert_input)

            weights = gated_weights[topk_indices, topk_slot].unsqueeze(-1)

            routed_output.index_add(
                0,
                token_idx,
                expert_output*weights
            )

            output = shared_output + routed_output

        return output

