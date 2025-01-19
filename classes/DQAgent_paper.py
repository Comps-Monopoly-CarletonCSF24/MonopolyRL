import torch
import torch.nn as nn
import numpy as np
from classes.state import State
from classes.action_paper import Action

state_size = 23
action_size = 1

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # Define layers
        self.layer1 = nn.Linear(state_size + action_size, 150)  # Input layer to hidden layer
        self.activation1 = nn.Sigmoid()  # Sigmoid activation for the hidden layer
        self.layer2 = nn.Linear(150, 1)  # Hidden layer to output layer
        
    def forward(self, state: State, action: Action):
        stacked_input = np.append(state.state, action.action_idx)
        input = torch.tensor(stacked_input, dtype=torch.float32)
        # Forward pass through the network
        output = self.layer1(input)
        output = self.activation1(output)
        output = self.layer2(output)
        return output

class QLambdaAgent:
    def __init__(self):
        # Parameters from the paper
        self.alpha = 0.2       # Learning rate from paper
        self.gamma = 0.95      # Discount factor from paper
        self.lambda_param = 0.85  # Lambda parameter from paper
        # Initialize network and optimizer
        self.model = QNetwork()
        # Initialize eligibility traces
        self.traces = {}
        for name, param in self.model.named_parameters():
            self.e_traces[name] = torch.zeros_like(param.data)


