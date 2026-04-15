from dfbr.models.cost_head import CostHead
import torch
from torch.utils.data import Dataset
import pandas as pd

class BikeDemandDataset(Dataset):
    def __init__(self, file, start_date, end_date, target_cols, input_scale_cols, input_no_scale_cols, capacities, max_cap, is_train=True, scaling_factor=None):
        
        #Read the file
        df = pd.read_parquet(file, engine='pyarrow')

        #Filter dates
        df = df.loc[start_date : end_date]
        #Separate features and targets
        all_features = input_scale_cols + input_no_scale_cols
        self.X = torch.tensor(df[all_features].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_cols].values, dtype=torch.float32)
        #Transform demand into cost functions 
        costhead = CostHead(capacities, max_cap)
        self.c = costhead(self.y)

        self.dates = df.index.date #Store dates for later indexing
        self.scale_idx = list(range(len(input_scale_cols)))
        
        if len(self.scale_idx) > 0:
            #Special handling for Precipitation (Log Transform)
            if 'precip' in input_scale_cols:
                precip_idx = input_scale_cols.index('precip')
                self.X[:, precip_idx] = torch.log1p(self.X[:, precip_idx])

            #Standardization Logic
            if is_train:
                # Only calculate stats for the weather slice
                self.mean = torch.mean(self.X[:, self.scale_idx], dim=0)
                self.std = torch.std(self.X[:, self.scale_idx], dim=0)
                #Scale targets
                self.y_mean = torch.mean(self.y, dim = 0)
                self.y_std = torch.std(self.y, dim = 0)
            else:
                self.mean = scaling_factor['mean']
                self.std = scaling_factor['std']
                self.y_mean = scaling_factor['y_mean']
                self.y_std = scaling_factor['y_std']

            # Apply the Z-score: (x - mu) / sigma to FEATURES ONLY
            self.X[:, self.scale_idx] = (self.X[:, self.scale_idx] - self.mean) / (self.std + 1e-8)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.c[idx]