import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

#--------------------------------------------------------------------------------------
#Helper Functions
#--------------------------------------------------------------------------------------
def split_x_y(df):
    #Get all the id columns for stations
    station_cols = [col for col in df.columns if str(col).isnumeric()]
    
    #Split off the stations and the features
    y = df[station_cols]
    x = df.drop(columns=station_cols)

    return x,y 



def prepare_dataloaders(train_file_path, test_file_path, batch_size):
    
    #Read in the prepared data
    train = pd.read_csv(train_file_path, index_col=0, parse_dates=True)
    test = pd.read_csv(test_file_path, index_col=0, parse_dates=True)
    
    #Split the data by targets and features 
    x_train, y_train = split_x_y(train)
    x_test, y_test = split_x_y(test)

    #Convert to tensors and datasets
    train_dataset = TensorDataset(torch.tensor(x_train.values, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(x_test.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))

    #Wrap into dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #Calculate input and output sizes to intialize the MLP
    input_size = x_train.shape[1]
    output_size = y_train.shape[1]

    return input_size, output_size, train_loader, test_loader
