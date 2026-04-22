import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

def evaluate(pred_model, cost_head, opt_model, dataloader):
    pred_model.eval()
    total_samples = 0
    total_mse = []
    total_cost = 0.0
    optimal_cost = 0.0

    with torch.no_grad():  
        for x, c, w, z, y in dataloader:
            #Get predictions and predicted cost function
            yp = pred_model(x)
            cp = cost_head(yp)
            #Comute mse 
            batch_mse = F.mse_loss(yp, y)
            total_mse.append(batch_mse)
            total_samples += x.shape[0]

            #Compute total cost
            for i in range(x.shape[0]):
                opt_model.setObj(cp[i])
                wp, _ = opt_model.solve()
                total_cost += np.dot(wp,  c[i])
                optimal_cost += z[i].item()

    return (sum(total_mse) / len(total_mse)), (total_cost / total_samples), (optimal_cost / total_samples)