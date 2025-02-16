import numpy as np
import random
import random

from classes.player import Player

class ApproxQLearningAgent(Player):
    def __init__(self, name, settings, alpha=0.1, gamma=0.9, epsilon=0.1, feature_size=30, num_actions=3):
        """
        Initializes the Q-learning agent.

        Args:
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate for epsilon-greedy policy.
            feature_size (int): Size of the feature vector.
            num_actions (int): Number of possible actions.
        """
        super().__init__(name,settings)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.feature_size = feature_size
        self.num_actions = num_actions
        
        # Initialize weights for the Q-function approximator
        self.weights = np.random.randn(feature_size, num_actions) * 0.01

    def extract_features(self, state, action):
        """
        Extracts features from the state and action.

        Args:
            state (np.ndarray): Raw state vector (size 23).
            action (int): Action index (0, 1, or 2).

        Returns:
            np.ndarray: Feature vector of size 33.
        """
        # state = state.state
        state_features = state.copy()  # Size: 23
        action_features = np.zeros(self.num_actions)  # Size: 3
        action_features[action] = 1
        padding = np.zeros(self.feature_size - len(state_features) - len(action_features))  # Size: 7
        return np.concatenate((state_features, action_features, padding))  # Total size: 33



    def get_q_values(self, state):
        """
        Computes Q-values for all actions given a state.

        Args:
            state (np.ndarray): Current state vector.

        Returns:
            np.ndarray: Q-values for all actions.
        """
        # state = state.state
        q_values = []
        for action in range(self.num_actions):
            features = self.extract_features(state, action)
            q_values.append(np.dot(features, self.weights[:, action]))
        return np.array(q_values)

    def select_action(self, state):
        """
        Selects an action using epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state vector.

        Returns:
            int: Selected action index.
        """
        state = state.state
        if random.random() < self.epsilon:
            # Explore: choose a random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: choose the action with the highest Q-value
            q_values = self.get_q_values(state)
            return np.argmax(q_values)

    def simulate_action(self, state, action):
        """
        Simulates the effect of an action on the state.

        Args:
            state (np.ndarray): Current state vector.
            action (int): Action to simulate.

        Returns:
            np.ndarray: Next state vector after the action.
        """
        # This function should simulate the game's response to the action.
        # For now, we assume the state changes slightly depending on the action.
        state = state.state
        next_state = state.copy()
        if action == 0:  # Sell
            next_state[1] = max(0, next_state[1] - 0.1)  # Example change
        elif action == 1:  # Buy
            next_state[1] = min(1, next_state[1] + 0.1)  # Example change
        # No change for "do nothing"
        return next_state

    def update(self, state, action, reward, next_state):
        """
        Updates the weights based on the Q-learning update rule.

        Args:
            state (np.ndarray): Current state vector.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state vector.
        """
        # Compute Q-value for the current state-action pair
        state = state.state
        features = self.extract_features(state, action)
        q_value = np.dot(features, self.weights[:, action])

        # Compute the target
        next_q_values = self.get_q_values(next_state)
        max_next_q_value = np.max(next_q_values)
        target = reward + self.gamma * max_next_q_value

        # Compute the temporal difference error
        td_error = target - q_value

        # Update weights for the taken action
        self.weights[:, action] += self.alpha * td_error * features
