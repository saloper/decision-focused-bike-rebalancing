import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(MLP, self).__init__()
        
        #Loop through layers
        layers = []
        curr_input_size = input_size
        for size in hidden_layers:
            layers.append(nn.Linear(curr_input_size, size))
            layers.append(nn.ReLU())
            curr_input_size = size
        
        #Add output layer
        layers.append(nn.Linear(curr_input_size, output_size))

        #Add to sequential
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)