import numpy as np
import random
import matplotlib.pyplot as plt

from classes.player_logistics import Player
from classes.action import Action
from classes.board import Property
from classes.state import State
import classes.simulate_actions as simulation
import pickle
import os

class ApproxQLearningAgent(Player):
    def __init__(self, name, settings, alpha=0.05, gamma=0.9, epsilon=0.05, feature_size=200):
        super().__init__(name, settings)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.feature_size = feature_size
        self.action_handler = Action()
        self.total_actions = self.action_handler.total_actions
        self.name = name
        self.q_value_log = []

        self.weights = np.random.randn(feature_size, self.total_actions)/np.sqrt(feature_size)

    def extract_features(self, state, action_index):
        if isinstance(state, State):
            state = state.state
        state_features = state.copy()  # 1*23 vector (area, position, finance) this is a copy
        action_features = np.zeros(self.total_actions) # 1*84 vector: these are zeros
        action_features[action_index] = 1 # make the action at the specified index 1
        padding = np.zeros(self.feature_size - len(state_features) - len(action_features)) # add zeros to remaining space
        return np.concatenate((state_features, action_features, padding))  # add it all together

    def get_q_values(self, state):
        q_values = []
        for action_index in range(self.total_actions):  
            features = self.extract_features(state, action_index)
            q_values.append(np.dot(features, self.weights[:, action_index])) # find the q_values by doing the dot product between features and  and weights
        return np.array(q_values)

    def select_action(self, state):
        
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.total_actions - 1)
        else:
            q_values = self.get_q_values(state)
            action_index = np.argmax(q_values)
        _,action_type = self.action_handler.map_action_index(action_index)
        actual_actions = self.action_handler.actions
        action_index_in_smaller_list = actual_actions.index(action_type)
        action_index_in_bigger_list = action_index
        return action_index_in_smaller_list, action_index_in_bigger_list

    def select_next_best_q_value(self, state, current_index):
        """
        Select the next best action, excluding the action that failed.
        
        Args:
            state (np.ndarray): Current state vector.
            excluded_action_index (int): Action index to exclude.

        Returns:
            int: Next best action index.
        """
        q_values = np.sort(self.get_q_values(state))
        
        # Select the next best action
        return q_values[current_index]

    def simulate_action(self, board, state, player, players, action_index, group_idx, max_attempts = 1):
        """
        Simulates the effect of an action on the state. If the intended action fails,
        attempts the next-best action based on Q-values.
        
        Args:
            board: Game board
            state: Current state vector
            player: Current player
            players: List of all players
            action_index: Initial action index in the flattened action space
            max_attempts: Maximum number of attempts to find a valid action

        Returns:
            np.ndarray: Next state vector after a valid action.
        """
        # return state

        _, action_type = self.action_handler.map_action_index(action_index)
        q_values = self.get_q_values(state)
        # print (q_values )
        if action_type == 'buy':
            the_property = board.cells[player.position]
            updgraded_buying_state =  simulation.update_state_after_spending(group_idx, board, player, the_property, players)
            if isinstance(updgraded_buying_state, int):
                if max_attempts < len(q_values):
                    buying_action_index = np.where(q_values == self.select_next_best_q_value(state, max_attempts))[0][0]
                    return self.simulate_action(board, state, player, players, buying_action_index,  group_idx, max_attempts+1)
                return state
            return updgraded_buying_state
    
        elif action_type == 'sell':
            updgraded_state = simulation.update_state_after_selling(group_idx, board, player, players)
            if isinstance(updgraded_state, int):
                if max_attempts < len(q_values):
                    selling_action_index = np.where(q_values == self.select_next_best_q_value(state, max_attempts))[0][0]
                    return self.simulate_action(board, state, player, players, selling_action_index,  group_idx, max_attempts+1)
                # return state
            return updgraded_state
        
        elif action_type == "do_nothing":
            return state 

    def update(self, state, action_index, reward, next_state):
        features = self.extract_features(state, action_index)
        q_value = np.dot(features, self.weights[:, action_index])

        # Compute the target
        next_q_values = self.get_q_values(next_state)
        max_next_q_value = np.max(next_q_values)
        target = reward + self.gamma * max_next_q_value

        td_error = target - q_value
        self.weights[:, action_index] += self.alpha * td_error * features
        self.q_value_log.append(q_value)

    def save_q_values(self):
        """Append new Q-values to the existing file."""
        if os.path.exists("q_values.pkl"):
            with open("q_values.pkl", "rb") as f:
                existing_q_values = pickle.load(f)
        else:
            existing_q_values = []

        # Append new Q-values and save
        updated_q_values = existing_q_values + self.q_value_log
        with open("q_values.pkl", "wb") as f:
            pickle.dump(updated_q_values, f)

        self.q_value_log = []
