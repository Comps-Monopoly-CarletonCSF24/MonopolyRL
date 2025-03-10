import os 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from typing import List
from classes.state import State, State_Size, get_initial_state, get_test_state
from classes.DQAgent.action import Action, Action_Size, Total_Actions, Actions
from classes.player_logistics import Player

# in this case, there are only 3: buy, sell, do nothing
model_param_path = "./classes/DQAgent/model_parameters.pth"

win_game_reward = 10
# to prevent the agent from doing nothing all the time, give no penalty for losing
lose_game_reward = 0 

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        # Define layers
        self.input_layer = nn.Linear(State_Size + Action_Size, 150)  # Input layer to hidden layer
        self.activation = nn.Sigmoid()  # Sigmoid activation for the hidden layer
        self.output_layer = nn.Linear(150, 1)  # Hidden layer to output layer
        # Initialize weights and biases randomly
        nn.init.uniform_(self.input_layer.weight, -0.5, 0.5)
        nn.init.uniform_(self.input_layer.bias, -0.5, 0.5)
        nn.init.uniform_(self.output_layer.weight, -0.5, 0.5)
        nn.init.uniform_(self.output_layer.bias, -0.5, 0.5)

    
    def forward(self, input):
        # Forward pass through the network
        output = self.input_layer(input)
        output = self.activation(output)
        output = self.output_layer(output)           
        return output

def create_nn_input(state: State, action: Action):
    stacked_input = np.append(state.state, action.action_index/Total_Actions)
    input = torch.tensor(stacked_input, dtype=torch.float32)
    return input

class Trace:
    def __init__(self, state: State, action: Action, value: float):
        self.state = state
        self.action = action
        self.value = value
    def is_similar_to_state(self, state2: State):
        return self.state.is_similar_to(state2)
    def is_similar_to_action(self, action2: Action):
        return self.action.action_type == action2.action_type
    def __str__(self):
        return f"Trace(state={self.state}, action={self.action}, value={self.value:.2f})"

class Training_Batch:
    def __init__(self):
        self.input = []
        self.q_values = []
    def append_datapoint(self, state, action, q_value):
        self.input.append(create_nn_input(state, action))
        self.q_values.append(q_value)
    def clear(self):
        self.input = []
        self.q_values = []

class QLambdaAgent:
    def __init__(self, is_training = False):
        self.is_training = is_training
        # Parameters from the paper
        self.epsilon = 1 if is_training else 0 # Greedy coeff from paper
        self.alpha = 0.1 # Learning rate
        self.gamma = 0.95      # Discount factor from paper
        self.lambda_param = 0.8  # Lambda parameter from paper
        # Initialize network and optimizer
        self.model = QNetwork()
        if os.path.exists(model_param_path):
            checkpoint = torch.load(model_param_path, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.alpha)
        # Initialize eligibility traces
        self.traces = []
        self.training_batch = Training_Batch()
        self.last_state = get_initial_state()
        q_values_init = self.calculate_all_q_values(self.last_state)
        self.last_action = self.find_action_with_max_value(q_values_init)
        self.survived_last_game = True   
        # For debugging purposes
        self.rewards = []
        self.choices = []
    
    def end_game(self):
        """train the neural network with endgame reward, reset the agent, and decay epsilon"""
        endgame_reward = win_game_reward if self.survived_last_game else lose_game_reward
        self.append_trace_to_training_data(self.last_state, self.last_action, endgame_reward)
        self.train_neural_network()
        self.survived_last_game = True
        self.traces = []
        self.last_state = get_initial_state()
        q_values_init = self.calculate_all_q_values(self.last_state)
        self.last_action = self.find_action_with_max_value(q_values_init)
        # epoislon decay
        self.epsilon = max(self.epsilon * 0.995, 0.1)
        
    def save_nn(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, model_param_path)
        
    def q_learning(self, state, action, reward):
        """Return the updated q-value with the current step taken"""
        last_q_value = self.model(create_nn_input(self.last_state, self.last_action))
        current_q_value = self.model(create_nn_input(state, action))
        q_value = self.alpha * (reward + self.gamma * current_q_value - last_q_value)
        return q_value
            
    def choose_action_helper(self, q_values: List[int]):
        if random.random() < self.epsilon:  # exploration rate\
            if self.is_training:
                self.choices[-1][3] += 1 # for debugging purposes
            return Action(random.choice(Actions))
        else:
            return self.find_action_with_max_value(q_values)
        
    def choose_action(self, state):
        q_values_state = self.calculate_all_q_values(state)
        return self.choose_action_helper(q_values_state)
    
    def find_action_with_max_value(self, q_values):
        q_values_float = [q_values[i].item() for i in range(len(Actions))]
        max_q_values = max(q_values_float)
        max_q_indices = [i for i, x in enumerate(q_values) if math.isclose(x, max_q_values, rel_tol=1e-9)]
        return Action(Actions[random.choice(max_q_indices)])
    
    def calculate_all_q_values(self, state: State):
        """For all possible actions (0-2), generate a list of predicted q-values with the NN"""
        q_values = []
        for action_index in range(Total_Actions):
            action = Action(Actions[action_index])
            q_value = self.model(create_nn_input(state, action))
            q_values.append(q_value)
        return q_values
    
    def update_trace(self, state, action):
        """go through traces and update each one: 
        if the current state already exists: update value to 1
        then, update all traces with the decay function.
        """
        state_action_exists = False
        trace_index_to_pop = None
        if self.traces:
            for i in range(len(self.traces)):
                trace = self.traces[i]
                if trace.is_similar_to_state(state) and trace.is_similar_to_action(action):
                    state_action_exists = True
                    self.traces[i].value = 1
                elif trace.is_similar_to_state(state):
                    trace_index_to_pop = i
                else:
                    self.traces[i].value = trace.value * self.gamma * self.lambda_param
        if trace_index_to_pop: 
            self.traces.pop(trace_index_to_pop)
        if not state_action_exists:
            new_trace = Trace(state, action, 1)
            self.traces.append(new_trace)
        return state_action_exists
    
    def append_training_data(self, input_state: State, input_action: Action, target_q_value: torch.Tensor):
        self.training_batch.append_datapoint(input_state, input_action, target_q_value)
    
    def train_neural_network(self):
        """Train the neural network with the training data from the current turn"""
        if not self.training_batch.input:
            return
        input = torch.stack(self.training_batch.input)
        output = torch.stack(self.training_batch.q_values)
        criterion = nn.MSELoss()
    
        self.optimizer.zero_grad()  # Zero gradients
        outputs = self.model(input)  # Forward pass
        loss = criterion(outputs, output)  # Compute loss
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update weights
        self.training_batch.clear()
        
    def append_trace_to_training_data(self, state, action, reward):
        """append training data for all traces"""
        for trace in self.traces: 
            if trace.is_similar_to_state(state) and trace.is_similar_to_action(action):
                continue
            else:
                q_t = self.model(create_nn_input(trace.state, trace.action))
                q_values_current_state = self.calculate_all_q_values(state)
                predicted_action_current_state = self.find_action_with_max_value(q_values_current_state)
                max_qt = self.model(create_nn_input(state, predicted_action_current_state))
                
                q_values_previous_state = self.calculate_all_q_values(self.last_state)
                predicted_action_previous_state = self.find_action_with_max_value(q_values_previous_state)
                max_q = self.model(create_nn_input(self.last_state, predicted_action_previous_state))

                target_q_value = q_t + self.alpha * trace.value * (reward + self.gamma * max_qt - max_q)
                
                self.append_training_data(trace.state, trace.action, target_q_value)

    def get_reward(self, player: Player, players):
        # Calculate total assets value (v)
        # calculated using cost_base, may need to consider houses
        player_assets = player.net_worth() - player.money
        other_assets = sum((p.net_worth() - p.money) for p in players if p != player)
        v = player_assets - other_assets
        # Calculate money ratio (m)
        total_money = sum(p.money for p in players)
        m = player.money / total_money if total_money > 0 else 0
        
        # Number of players
        p = len(players)
        
        # The smoothing factor that determines the importance of the property value
        c = 2
        
        # Calculate reward using paper's formula
        reward = (v/p * c)/(1 + abs(v/p * c)) + (1/p * m)
        return reward
