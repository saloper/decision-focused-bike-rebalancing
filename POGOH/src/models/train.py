import torch
import torch.nn as nn

def get_loss_func(name, opt_model=None):
    if name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type in config: name")

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
        
    for batch_X, batch_y in dataloader:

        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
            
    return total_loss / len(dataloader)