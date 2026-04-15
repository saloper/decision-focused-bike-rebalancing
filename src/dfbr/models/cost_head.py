import torch
import torch.nn as nn
import torch.nn.functional as F

class CostHead(nn.Module):
    def __init__(self, capacities, max_cap, over_cost=1, lost_cost=1):
        super().__init__()
        self.over_cost = over_cost
        self.lost_cost = lost_cost
        self.targets = torch.arange(max_cap + 1, dtype=torch.float32).view(1, 1, -1) #(1, 1, max_cap +1)
        self.capacities = torch.tensor(capacities, dtype=torch.float32).view(1, -1, 1) #(1, S, 1)
    
    def forward(self, predicted_demand):

        d_expanded = predicted_demand.unsqueeze(2) #(B,S,1)
        ending_inv = self.targets - d_expanded #(B, S, max_cap +1)
    
        overflow_cost = self.over_cost * F.relu(ending_inv - self.capacities)
        lost_cost = self.lost_cost * F.relu(-ending_inv,)
        
        total_cost = overflow_cost + lost_cost #(B, S, max_cap +1)
        invalid_mask = self.targets > self.capacities
        total_cost = total_cost.masked_fill(invalid_mask, 1e9)
        
        return total_cost.view(total_cost.shape[0], -1) #(B, S * max_cap + 1)