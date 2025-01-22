import numpy as np
import random
import random

from classes.player_logistics import Player
from classes.action import Action
from classes.board import Board, Property
import copy
class ApproxQLearningAgent(Player):
    def __init__(self, name, settings, alpha=0.1, gamma=0.9, epsilon=0.1, feature_size=200):
        super().__init__(name, settings)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.feature_size = feature_size
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
        action_features = np.zeros(self.total_actions)
        action_features[action_index] = 1
        padding = np.zeros(self.feature_size - len(state_features) - len(action_features)) 
        return np.concatenate((state_features, action_features, padding))  


    def get_q_values(self, state):
        """
        Computes Q-values for all actions given a state.

        Args:
            state (np.ndarray): Current state vector.

        Returns:
            np.ndarray: Q-values for all actions in the flattened action space.
        """
        q_values = []
        for action_index in range(self.total_actions):  
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
            action_index = random.randint(0, self.total_actions - 1)
        else:
            q_values = self.get_q_values(state)
            action_index = np.argmax(q_values)
        property_index, action_type = self.action_handler.map_action_index(action_index)
        actual_actions = self.action_handler.actions
        return actual_actions.index(action_type), action_index


    def simulate_action(self, board, state, player, players, action_index):
        """
        Simulates the effect of an action on the state. If the intended action fails,
        attempts the next-best action based on Q-values.
        
        Args:
            state (np.ndarray): Current state vector.
            action_index (int): Action index in the flattened action space.

        Returns:
            np.ndarray: Next state vector after the action.
        """
        original_action_index = action_index
        attempted_actions = set()
        
        while True:
            property_idx, action_type = self.action_handler.map_action_index(action_index)
            
            # Simulate the action
            if action_type == 'buy':
                the_property = board.cells[self.position]
                if isinstance(the_property, Property):
                    property_price = the_property.cost_base
                    buying_status = state.update_after_purchase(player, players, the_property, property_price)
                else:
                    print (f"Cannot buy a {type(the_property)}")
                    buying_status = 0
                if isinstance(buying_status, int):
                    print("Could not buy property")
                else:
                    print("Action taken: Baught")
                    return buying_status  # Action succeeded
            
            elif action_type == 'sell':
                the_property = board.get_property(property_idx)
                sale_price = the_property.cost_base // 2
                selling_status = state.update_after_sale(player, players, the_property, sale_price)
                if isinstance(selling_status, int):
                    print("Could not sell property")
                else:
                    print("Action taken: Sold")
                    return selling_status  # Action succeeded
            
            elif action_type == 'do_nothing':
                print("Was not meant to do anything")
                return state  # No state change required
            
            else:
                print(f"Unknown action type: {action_type}")
            
            # Record the failed action
            attempted_actions.add(action_index)
            
            # Get Q-values and find the next best action that hasn't been tried
            q_values = self.get_q_values(state)
            sorted_indices = np.argsort(-q_values)  # Sort in descending order of Q-values
            next_action_found = False
            for next_index in sorted_indices:
                if next_index not in attempted_actions:
                    action_index = next_index
                    next_action_found = True
                    break
            
            if not next_action_found:
                print("No viable action found. Remaining in current state.")
                return state  # No more actions to try, return the current state


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

        td_error = target - q_value
        self.weights[:, action_index] += self.alpha * td_error * features

