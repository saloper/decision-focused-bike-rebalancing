import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

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

def evaluate(pred_model, cost_head, opt_model, dataloader, split):
    pred_model.eval()
    total_samples = 0
    dates = []
    total_mse = []
    true_demand = []
    true_targets = []
    true_obj = []
    pred_demand = []
    pred_targets = []
    pred_obj = []
    real_obj = []

    with torch.no_grad():  
        for x, c, w, z, y, date in dataloader:
            #Record the ground truth solutions
            dates.extend(date)
            true_demand.append(y)
            true_obj.append(z)
            w = w.view(-1, cost_head.num_stations, cost_head.max_cap +1)
            true_targets.append(torch.argmax(w, axis = 2))

            #Get predictions and predicted cost function
            yp = pred_model(x)
            cp = cost_head(yp)
            #Record predictions
            pred_demand.append(yp)

            #Comute mse 
            batch_mse = F.mse_loss(yp, y)
            total_mse.append(batch_mse.item())
            total_samples += x.shape[0]

            #Compute total cost
            for i in range(x.shape[0]):
                opt_model.setObj(cp[i])
                wp, zp = opt_model.solve()
                real_cost = np.dot(wp,  c[i])
                #Get targets 
                target = torch.tensor(wp).view(cost_head.num_stations, cost_head.max_cap +1)
                #Record the predictions 
                pred_obj.append(zp)
                pred_targets.append(torch.argmax(target, axis = 1).tolist())
                real_obj.append(real_cost)

    #Reshape into dataframe
    df = pd.DataFrame({
        'split' : 'train',
        'date' : dates,
        'true_demand': torch.cat(true_demand, axis=0).tolist(),
        'true_targets': torch.cat(true_targets, axis=0).tolist(),
        'true_obj': torch.cat(true_obj, axis=0).squeeze().tolist(), 
        'pred_demand': torch.cat(pred_demand, axis=0).tolist(),
        'pred_targets': pred_targets,
        'pred_obj': pred_obj, 
        'real_obj': real_obj, 
    })

    return (sum(total_mse) / len(total_mse)), df['real_obj'].mean(), df['true_obj'].mean(), df