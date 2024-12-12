import torch.nn as nn

def get_mlp(hidden_dims, input_dim, output_dim, activation=nn.ReLU, dropout=0.0):

    layers = []

    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        linear_layer = nn.Linear(prev_dim, hidden_dim)

        nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')

        layers.append(linear_layer) 

        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, output_dim))
    
    return nn.Sequential(*layers)