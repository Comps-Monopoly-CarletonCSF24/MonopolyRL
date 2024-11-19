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