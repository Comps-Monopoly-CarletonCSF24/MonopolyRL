import os 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from typing import List
from classes.state import State, State_Size, get_initial_state
from classes.action_paper import Action, Action_Size, Total_Actions, Actions

# in this case, there are only 3: buy, sell, do nothing
all_actions = list(range(Total_Actions))
model_param_path = "./model_parameters.pth"

class QNetwork(nn.Module):
    def __init__(self):

        super(QNetwork, self).__init__()
        # Define layers
        self.input_layer = nn.Linear(State_Size + Action_Size, 150)  # Input layer to hidden layer
        self.activation = nn.Sigmoid()  # Sigmoid activation for the hidden layer
        self.output_layer = nn.Linear(150, 1)  # Hidden layer to output layer

        # Match Java's weight initialization (they had 0.5 as weights)
        nn.init.uniform_(self.input_layer.weight, -0.5, 0.5)
        nn.init.uniform_(self.input_layer.bias, -0.5, 0.5)
        nn.init.uniform_(self.output_layer.weight, -0.5, 0.5)
        nn.init.uniform_(self.output_layer.bias, -0.5, 0.5)
        
        
    def forward(self, state: State, action: Action):
        stacked_input = np.append(state.state, action.action_index)
        input = torch.tensor(stacked_input, dtype=torch.float32)
        # Forward pass through the network
        output = self.input_layer(input)
        output = self.activation(output)
        output = self.output_layer(output)
        return output

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
    
class QLambdaAgent:
    def __init__(self, is_training = False):
        self.is_training = is_training
        # Parameters from the paper
        self.epsilon = 0.5     # Greedy coeff from paper
        self.alpha = 0.2       # Learning rate from paper
        self.gamma = 0.95      # Discount factor from paper
        self.lambda_param = 0.8  # Lambda parameter from paper
        self.current_epoch = 1 

        # Initialize network and optimizer
        self.model = QNetwork()
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.alpha)
        
        if os.path.exists(model_param_path):
            checkpoint = torch.load(model_param_path, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Initialize eligibility traces
        self.traces = []
        self.last_state = get_initial_state()
        self.last_action = Action("do_nothing")
    
    def end_game(self):
        self.traces = []
        self.last_state = get_initial_state()
        self.last_action = Action("do_nothing")
        self.epsilon *= 0.99
        self.alpha *= 0.99
    
    def save_nn(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, model_param_path)
        
    def q_learning(self, state, action, reward):
        last_q_value = self.model(self.last_state, self.last_action)
        current_q_value = self.model(state, action)
        q_value = self.alpha * (reward + self.gamma * current_q_value - last_q_value)
        return q_value
            
    def choose_action_helper(self, q_values: List[int]):
        """_summary_
        Args:
            state (State): _description_
            q_values (List[int]): _description_

        Returns:
            _type_: _description_
        """
        if random.random() < self.epsilon:  # exploration rate
            ## DELETE THIS LINE
            # print("random")
            return Action(random.choice(Actions))
        else:
            ## DELETE
            # valid_q_values = [q_values[i] for i in all_actions]
            # test_str = ""
            # for i in all_actions:
            #     test_str += str(Actions[i]) + ": " + str(valid_q_values[i])
            # print(test_str)
            return self.find_action_with_max_value(q_values)
    
    def choose_action(self, state):
        q_values_state = self.calculate_all_q_values(state)
        return self.choose_action_helper(q_values_state)
    
    def find_action_with_max_value(self, q_values: List[int]):
        valid_q_values = [q_values[i] for i in all_actions]
        max_q_value = max(valid_q_values)
        # Break the tie randomly
        max_q_indices = [i for i, x in enumerate(q_values) if math.isclose(x, max_q_value, rel_tol=1e-9)]
        return Action(Actions[random.choice(max_q_indices)])
    
    def calculate_all_q_values(self, state: State):
        """For all possible actions (0-2), generate a list of predicted q-values with the NN

        Args:
            state (State): the current state
        """
        q_values = []
        for action_index in range(Total_Actions):
            action = Action(Actions[action_index])
            q_value = self.model(state, action)
            q_values.append(q_value)
        return q_values
    
    def update_trace(self, state, action):
        """go through traces and update each one:
        if the current state already exists: update value to 1
        then, update all traces with the decay function.
    
        Args:
            state (_type_): _description_
            action (_type_): _description_
            
        Returns:
            bool: whether the trace was found
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
    
    def train_neural_network(self, input_state: State, input_action: Action, target_q_value: torch.Tensor):
        #tracking epoch like java did 
        self.current_epoch += 1

        criterion = nn.MSELoss()
        output_q_value = self.model(input_state, input_action)
        loss = criterion(output_q_value, target_q_value)
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        # Update weights
        self.optimizer.step()

    def train_nn_with_trace(self, state, action, reward):
        for trace in self.traces: 
            if trace.is_similar_to_state(state) and trace.is_similar_to_action(action):
                continue
            else:
                q_t = self.model(trace.state, trace.action)
                
                q_values_current_state = self.calculate_all_q_values(state)
                predicted_action_current_state = self.find_action_with_max_value(q_values_current_state)
                max_qt = self.model(state, predicted_action_current_state)
                
                q_values_previous_state = self.calculate_all_q_values(self.last_state)
                predicted_action_previous_state = self.find_action_with_max_value(q_values_previous_state)
                max_q = self.model(self.last_state, predicted_action_previous_state)

                target_q_value = q_t + self.alpha * trace.value * (reward + self.gamma * max_qt - max_q)
                
                self.train_neural_network(trace.state, trace.action, target_q_value)

    def get_reward(self, player, players):
        """Implement reward function from paper"""
        # Calculate total assets value (v)
        # calculated using cost_base, may need to consider houses
        player_assets = sum(property.cost_base for property in player.owned)
        other_assets = sum(sum(property.cost_base for property in p.owned) for p in players if p != player)
        
        v = player_assets - other_assets
        
        # Calculate money ratio (m)
        total_money = sum(p.money for p in players)
        m = player.money / total_money if total_money > 0 else 0
        
        # Number of players
        p = len(players)
        
        # Smoothing factor (can be tuned)
        c = 0.5
        
        # Calculate reward using paper's formula
        reward = (v/p * c)/(1 + abs(v/p * c)) + (1/p * m)
        return reward
    

