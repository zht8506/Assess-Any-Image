import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
from copy import deepcopy

class Expert(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Expert, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        
        return x


class Gating(nn.Module):
    def __init__(self, input_dim,
                num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

        # Layers
        self.layer1 = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        x = self.layer1(x)

        return torch.softmax(x, dim=-1)



class MoE_clip(nn.Module):
    def __init__(self, num_experts, input_dim, expert):
        super(MoE_clip, self).__init__()

        self.useful_experts = deepcopy(expert)
        self.experts = nn.ModuleDict({
            f'expert_{i}': deepcopy(expert) for i in range(num_experts)
        })

        self.gating = Gating(input_dim, num_experts)

        self.scale=nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        weights=self.gating(x)

        useful_experts_output = self.useful_experts(x)

        # Calculate the expert outputs
        outputs = torch.stack([self.experts[expert_name](x) for expert_name in self.experts], dim=2)

        weights = weights.unsqueeze(3).expand_as(outputs)
        weighted_experts_output = torch.sum(outputs * weights, dim=2)

        return self.scale * weighted_experts_output + useful_experts_output
