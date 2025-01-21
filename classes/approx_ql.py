import numpy as np
import random
import random

from classes.player import Player
from classes.action import Action
from classes.board import Board
import copy
class ApproxQLearningAgent(Player):
    def __init__(self, name, settings, alpha=0.1, gamma=0.9, epsilon=0.1, feature_size=200):
        super().__init__(name, settings)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.feature_size = feature_size

        # Initialize the Action class
        self.action_handler = Action()
        self.total_actions = self.action_handler.total_actions

        # Initialize weights for the Q-function approximator
        self.weights = np.random.randn(feature_size, self.total_actions) * 0.01

    def extract_features(self, state, action_index):
        """
        Extracts features from the state and flattened action index.

        Args:
            state (np.ndarray): Raw state vector (size 23).
            action_index (int): Flattened action index (0 to total_actions-1).

        Returns:
            np.ndarray: Feature vector of size 33.
        """
        if  not isinstance(state, np.ndarray):
            state = state.state
        state_features = state.copy()  
        action_features = np.zeros(self.total_actions)  # One-hot encode the action
        action_features[action_index] = 1
        padding = np.zeros(self.feature_size - len(state_features) - len(action_features))  # Adjust padding size
        return np.concatenate((state_features, action_features, padding))  # Total size: feature_size


    def get_q_values(self, state):
        """
        Computes Q-values for all actions given a state.

        Args:
            state (np.ndarray): Current state vector.

        Returns:
            np.ndarray: Q-values for all actions in the flattened action space.
        """
        q_values = []
        for action_index in range(self.total_actions):  # Iterate over all flattened actions
            features = self.extract_features(state, action_index)
            q_values.append(np.dot(features, self.weights[:, action_index]))
        return np.array(q_values)



    def select_action(self, state):
        """
        Selects an action using epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state vector.

        Returns:
            tuple: Selected flattened action index and property index.
        """
        if random.random() < self.epsilon:
            # Explore: choose a random action
            action_index = random.randint(0, self.total_actions - 1)
        else:
            # Exploit: choose the action with the highest Q-value
            q_values = self.get_q_values(state)
            action_index = np.argmax(q_values)
        
        # Map action_index to property index
        property_index, _ = self.action_handler.map_action_index(action_index)
        
        return action_index, property_index


    def simulate_action(self, board, state, player, players, action_index):
        """
        Simulates the effect of an action on the state.
        
        Args:
            state (np.ndarray): Current state vector.
            action_index (int): Action index in the flattened action space.

        Returns:
            np.ndarray: Next state vector after the action.
        """
        next_state = copy.deepcopy(state)
        
        # Map action_index to property_idx and action_type
        property_idx, action_type = self.action_handler.map_action_index(action_index)
         
        the_property = board.get_property(property_idx)
        property_price = the_property.cost_base

        # Handle the action based on action_type
        if action_type == 'buy':
            buying_status = next_state.update_after_purchase(player, players, the_property, property_price)
            if isinstance(buying_status, int):
                print("could not buy property")
                next_state = state
            else:
                next_state = buying_status
        elif action_type == 'sell':
            sayles_price = 0
            selling_status = next_state.update_after_sale (player, players, the_property, sayles_price)
            if isinstance(selling_status, int):
                print ("could not sell property")
                next_state = state
            else:
                next_state = selling_status
        elif action_type == 'trade':
            pass
            '''TODO: come up with the logic for trading and how it works'''
           

        elif action_type == 'do_nothing':
            # No changes to the state
            pass

        else:
            print(f"Unknown action type: {action_type}")

        return next_state

    def update(self, state, action_index, reward, next_state):
        """
        Updates the weights based on the Q-learning update rule.

        Args:
            state (np.ndarray): Current state vector.
            action_index (int): Flattened action index.
            reward (float): Reward received.
            next_state (np.ndarray): Next state vector.
        """
        features = self.extract_features(state, action_index)
        q_value = np.dot(features, self.weights[:, action_index])

        # Compute the target
        next_q_values = self.get_q_values(next_state)
        max_next_q_value = np.max(next_q_values)
        target = reward + self.gamma * max_next_q_value

        # Compute the temporal difference error
        td_error = target - q_value

        # Update weights for the taken action
        self.weights[:, action_index] += self.alpha * td_error * features

