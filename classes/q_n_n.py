import torch
import torch.nn as nn
import torch.optim as optim


class Q_network(nn.Module):
    def __init__(self, state_size, action_size):
        super(Q_network, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2= nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output_layer(x)  
