import numpy as np
import random

from classes.player_logistics import Player
from classes.act import ActionHandler
from classes.state import State
import classes.simulate_actions as simulation
import pickle
import os

class ApproxQLearningAgent(Player):
    def __init__(self, name, settings, alpha=0.5, gamma=0.9, epsilon=1, feature_size=200, decay_rate=0.001):
        super().__init__(name, settings)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.feature_size = feature_size
        self.action_handler = ActionHandler()
        self.total_actions = self.action_handler.total_actions
        self.name = name
        self.decay_rate = decay_rate 
        self.epsilon_min = 0.1
        self.new_epsilon = epsilon
        self.new_alpha = alpha
        self.weights = np.random.randn(feature_size, self.total_actions)/np.sqrt(feature_size)

        self.load_model()
    def load_model(self, filename="q_learning_model.pkl"):
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                
                self.weights = data["weights"]
                self.epsilon = data["epsilon"]
                self.alpha = data["alpha"]
                self.gamma = data["gamma"]
                
                print(f"Model loaded safely from {filename}")
            except (pickle.UnpicklingError, EOFError, ValueError) as e:
                print(f"Error loading model: {e}")
                print("The file may be corrupted. Starting fresh.")
        else:
            print("No saved model found. Starting fresh.")

    def get_alpha(self, state, action, episode):
        """Adaptive learning rate that decays over time or by visit count"""
        return max(0.01, self.alpha / (1 + self.decay_rate * episode/100))  # Time-based decay

    def get_gamma(self, episode):
        """Increase gamma over time to favor long-term rewards"""
        return min(1.0, self.gamma + (0.1 * episode / 1000))
    
    def get_epsilon(self, episode):
        """Exponential decay"""
        return max(self.epsilon_min, self.epsilon * np.exp(-self.decay_rate * episode/10))
    
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

    def select_action(self, state, episode):
        epsilon_t = self.get_epsilon(episode)
        self.new_epsilon = epsilon_t
        if random.random() < epsilon_t:
            print (f"E_X_P_L_O_R_I_N_G with Espsilon {epsilon_t} and alpha {self.new_alpha}")
            action_index = random.randint(0, self.total_actions - 1)
        else:
            print (f"EXPLOITING with Espsilon {epsilon_t} and alpha {self.new_alpha}")
            q_values = self.get_q_values(state)
            action_index = np.argmax(q_values)
        _,action_type = self.action_handler.map_action_index(action_index)
        actual_actions = self.action_handler.actions
        action_index_in_smaller_list = actual_actions.index(action_type)
        action_index_in_bigger_list = action_index
        return action_index_in_smaller_list, action_index_in_bigger_list

    def select_next_best_q_value(self, state, current_index):
        q_values = np.sort(self.get_q_values(state))

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

        _, action_type = self.action_handler.map_action_index(action_index)
        q_values = self.get_q_values(state)
        if action_type == 'buy':
            the_property = board.cells[player.position]
            updgraded_buying_state =  simulation.update_state_after_spending(group_idx, board, player, the_property, players)
            if isinstance(updgraded_buying_state, int):
                if max_attempts < 6:
                    indices = np.where(q_values == self.select_next_best_q_value(state, max_attempts))[0]
                    if len(indices) > 0:
                        buying_action_index = indices[0]
                        return self.simulate_action(board, state, player, players, buying_action_index,  group_idx, max_attempts+1)
                    else:
                        return state
                return state
            return updgraded_buying_state
    
        elif action_type == 'sell':
            updgraded_state = simulation.update_state_after_selling(group_idx, board, player, players)
            if isinstance(updgraded_state, int):
                if max_attempts < 6:
                    indices = np.where(q_values == self.select_next_best_q_value(state, max_attempts))[0]
                    if len(indices) > 0:
                        selling_action_index = indices[0]
                        return self.simulate_action(board, state, player, players, selling_action_index,  group_idx, max_attempts+1)
                    else:
                        return state
                return state
            return updgraded_state
        
        elif action_type == "do_nothing":
            return state 

    def update(self, state, action_index, reward, next_state, episode, action):
        alpha_t = self.get_alpha(state, action, episode)
        self.new_alpha = alpha_t
        features = self.extract_features(state, action_index)
        q_value = np.dot(features, self.weights[:, action_index])
        next_q_values = self.get_q_values(next_state)
        max_next_q_value = np.max(next_q_values)
        target = reward + self.gamma * max_next_q_value

        td_error = target - q_value
        self.weights[:, action_index] += alpha_t * td_error * features
        