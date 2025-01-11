from q_n_n import QNetwork
import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQAgent:
    def __init__(self, actions, state_size, action_size, learning_rate=0.001, gamma=0.95, epsilon=0.1):
        self.actions = actions
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = learning_rate
        
        # Neural network models
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

        # Replay buffer
        self.replay_buffer = deque(maxlen=10000)

    def choose_action(self, state):
        if random.random() < self.epsilon:  # Exploration
            return random.choice(self.actions)
        else:  # Exploitation
            with torch.no_grad():
                q_values = self.model(torch.tensor(state, dtype=torch.float32))
                return torch.argmax(q_values).item()

    def update_model(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return  # Not enough data to train

        # Sample a batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Get current Q-values
        q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
            targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute loss
        loss = self.criterion(q_values, targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def take_turn(self, action_obj, player, board, state):
        "Same thing as original take turn function but now it's adding in NN integration"

        action_idx = self.choose_action(state)
        property_idx, action_type = action_obj.map_action_index(action_idx)
        action_obj.execute_action(player, board, property_idx, action_type)

        reward = self.get_reward(player)
        next_state_instance = State(player, board.players)
        next_state = next_state_instance.state

        # Add to replay buffer
        done = False  # Adjust based on game logic
        self.replay_buffer.append((state, action_idx, reward, next_state, done))

        # Train the model
        self.update_model()

        return next_state
    



# simpler approach matching paper

# class QNetwork(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(QNetwork, self).__init__()
#         # Paper uses a simple 3-layer network
#         self.layer1 = nn.Linear(state_size, 64)  
#         self.layer2 = nn.Linear(64, 32)
#         self.output_layer = nn.Linear(32, action_size)
        
#     def forward(self, x):
#         x = torch.relu(self.layer1(x))
#         x = torch.relu(self.layer2(x))
#         return self.output_layer(x)

# class QLambdaAgent:
#     def __init__(self, actions, state_size, action_size):
#         # Parameters from the paper
#         self.actions = actions
#         self.state_size = state_size
#         self.action_size = action_size
#         self.alpha = 0.2       # Learning rate from paper
#         self.gamma = 0.95      # Discount factor from paper
#         self.lambda_param = 0.85  # Lambda parameter from paper
        
#         # Initialize network and optimizer
#         self.model = QNetwork(state_size, action_size)
#         self.optimizer = optim.SGD(self.model.parameters(), lr=self.alpha)
        
#         # Initialize eligibility traces
#         self.e_traces = {}
#         for name, param in self.model.named_parameters():
#             self.e_traces[name] = torch.zeros_like(param.data)
            
#     def reset_traces(self):
#         """Reset eligibility traces at the end of each episode"""
#         for name in self.e_traces:
#             self.e_traces[name].zero_()
            
#     def choose_action(self, state):
#         """Select action using ε-greedy policy"""
#         if random.random() < 0.1:  # 10% exploration rate
#             return random.choice(self.actions)
#         else:
#             with torch.no_grad():
#                 state_tensor = torch.tensor(state, dtype=torch.float32)
#                 q_values = self.model(state_tensor)
#                 return self.actions[torch.argmax(q_values).item()]
                
#     def are_states_similar(self, s1, s2):
#         """Implement state similarity metric from paper"""
#         # Extract area, finance, and position from states
#         area1, finance1, pos1 = s1[:20], s1[22:], s1[21]
#         area2, finance2, pos2 = s2[:20], s2[22:], s2[21]
        
#         # Check conditions from paper
#         area_diff = sum(abs(a1 - a2) for a1, a2 in zip(area1, area2)) <= 0.1
#         finance_diff = abs(finance1 - finance2) <= 0.1
#         position_same = pos1 == pos2
        
#         return area_diff and finance_diff and position_same
        
#     def get_reward(self, player, all_players):
#         """Implement reward function from paper"""
#         # Calculate total assets value (v)
#         player_assets = sum(prop.value for prop in player.properties)
#         other_assets = sum(sum(prop.value for prop in p.properties) 
#                           for p in all_players if p != player)
#         v = player_assets - other_assets
        
#         # Calculate money ratio (m)
#         total_money = sum(p.money for p in all_players)
#         m = player.money / total_money if total_money > 0 else 0
        
#         # Number of players
#         p = len(all_players)
        
#         # Smoothing factor (can be tuned)
#         c = 0.5
        
#         # Calculate reward using paper's formula
#         reward = (v/p * c)/(1 + abs(v/p * c)) + (1/p * m)
#         return reward
        
#     def update(self, state, action, reward, next_state):
#         """Update Q-values using Q(λ) learning"""
#         # Convert states to tensors
#         state_tensor = torch.tensor(state, dtype=torch.float32)
#         next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        
#         # Get current Q-values
#         current_q = self.model(state_tensor)
#         action_idx = self.actions.index(action)
        
#         # Get next state Q-values
#         with torch.no_grad():
#             next_q = self.model(next_state_tensor)
#             next_max_q = torch.max(next_q)
        
#         # Calculate TD error
#         td_error = (reward + self.gamma * next_max_q - 
#                    current_q[action_idx])
        
#         # Backward pass to get gradients
#         current_q[action_idx].backward()
        
#         # Update eligibility traces and parameters
#         with torch.no_grad():
#             for name, param in self.model.named_parameters():
#                 # Update eligibility traces
#                 self.e_traces[name] = (self.gamma * self.lambda_param * 
#                                      self.e_traces[name] + param.grad)
                
#                 # Update parameters using eligibility traces
#                 param.data += self.alpha * td_error * self.e_traces[name]
                
#                 # Zero gradients
#                 param.grad.zero_()
                
#     def take_turn(self, action_obj, player, board, state):
#         """Execute a turn in the game"""
#         # Choose and execute action
#         action = self.choose_action(state)
#         property_idx, action_type = action_obj.map_action_index(
#             self.actions.index(action))
#         action_obj.execute_action(player, board, property_idx, action_type)
        
#         # Get new state and reward
#         next_state_instance = State(player, board.players)
#         next_state = next_state_instance.state
#         reward = self.get_reward(player, board.players)
        
#         # Update Q-values
#         self.update(state, action, reward, next_state)
        
#         return next_state
        
#     def end_episode(self):
#         """Clean up at the end of an episode"""
#         self.reset_traces()