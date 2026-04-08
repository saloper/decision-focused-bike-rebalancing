import torch
import torch.nn as nn
import torch.nn.functional as F

class CostHead(nn.Module):
    def __init__(self, over_cost, lost_cost):
        super().__init__()
        self.over_cost = over_cost
        self.lost_cost = lost_cost

    def forward(self, predicted_demand, capacities):
        # Ensure capacities match the batch dimension
        batch_size, num_stations = predicted_demand.shape
        max_cap = int(capacities.max().item())
        
        targets = torch.arange(max_cap + 1).view(1, 1, -1) #(1, 1, max_cap +1)
        d_expanded = predicted_demand.unsqueeze(2) #(B,S,1)

        ending_inv = targets + d_expanded #(B, S, max_cap +1)
        
        caps_expanded = capacities.view(1, -1, 1) #(1, S, 1)
        
        overflow_cost = self.over_cost * F.relu(ending_inv - caps_expanded)
        lost_cost = self.lost_cost * F.relu(-ending_inv,)
        
        total_cost = overflow_cost + lost_cost #(B, S, max_cap +1)
        invalid_mask = targets > caps_expanded
        total_cost = total_cost.masked_fill(invalid_mask, 1e9)
        
        return total_cost