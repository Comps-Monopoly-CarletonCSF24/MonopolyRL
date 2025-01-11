from classes.player import Player
from classes.board import Board
from classes.log import Log
import numpy as np

# class Action:
#     ''' 
#         To represent the action space in MDP we will have a 28x3 matrix. 28
#         because there are 28 properties and for each property there are three possible
#         actions: buy, sell, or do nothing.
#     '''
#     def __init__(self):
#         self.num_properties = 28 #Total number of properties in Monopoly
#         self.actions = ['buy','sell','do_nothing'] #The 3 possible actions

#         self.action_matrix = np.zeros((self.num_properties, len(self.actions)),dtype=1)

#     def get_action_vector(self, property_idx :int, action :str) -> np.ndarray:
#         '''
#             Generate a 1-hot encoded vector for a given action on a specific property.
            
#             Parameters:
#                 - property_idx (int): The index of the property (0 to 27).
#                 - action (str): The action to be taken ('buy', 'sell', 'do_nothing').
            
#             Returns:
#                 - np.ndarray: A 1-hot encoded vector for the given action on the property.
#         '''
#         if action not in self.actions:
#             raise ValueError(f"Invalid action. Choose from {self.actions}.")
        
#         action_idx = self.actions.index(action)  # Find the index of the action
#         action_vector = np.zeros(len(self.actions), dtype=int)
#         action_vector[action_idx] = 1  # Set the corresponding action to 1
#         return action_vector
        
        
        
#     def set_action(self, property_idx :int, action :str):
#         '''
#         Set the action for a given property. Only one action can be active at a time for each property.
        
#         Parameters:
#             - property_idx (int): The index of the property (0 to 27).
#             - action (str): The action to be taken ('buy', 'sell', or 'do_nothing').
#         '''
#         if action not in self.actions:
#             raise ValueError(f"Invalid action. Choose from {self.actions}.")
        
#         action_idx = self.actions.index(action)
        
#         # Set the corresponding action for the given property
#         self.action_matrix[property_idx] = np.zeros(len(self.actions), dtype=int)
#         self.action_matrix[property_idx][action_idx] = 1

#     def consolidate_action_matrix(self) -> np.ndarray:
#         '''
#         Return the current action matrix, which represents all properties' possible actions.
        
#         Returns:
#             - np.ndarray: The action matrix (28 properties x 3 actions).
#         '''
#         self.action_matrix = self.action_matrix.flatten()

# Making a new file here so that if this doesn't work, we have the original action.py file to go back too. 
# Switching it to a flattened 1x84 vector for function calls

class Action:
    def __init__(self):
        self.properties = list(range(28))  # Property indices from 0 to 27
        self.actions = ['buy', 'sell', 'do_nothing']  # Available actions for each property
        self.total_actions = len(self.properties) * len(self.actions)  # 1x84 action space

    def map_action_index(self, action_index):
        """
        Maps a flattened action index to a specific property and action type.

        Parameters:
            - action_index (int): The index of the action in the flattened action space.
        
        Returns:
            - tuple: A tuple containing the property index and the action type.
        """
        if not isinstance(action_index, (int, np.integer)):
            try:
                action_index = int(action_index)
            except ValueError:
                raise ValueError(
                    f"action_index must be an integer, got {type(action_index)}."
                )
    
        if action_index < 0 or action_index >= self.total_actions:
            raise ValueError(
                f"The action index must be an integer between 0 and {self.total_actions - 1}."
            )
        property_idx = action_index // len(self.actions)
        action_type = self.actions[action_index % len(self.actions)]
        return property_idx, action_type

    def execute_action(self, player, board, property_idx, action_type):
        """
        Executes the action on the given property for the specified player.
        """
        if action_type == 'buy':
            if not board[property_idx].is_owned() and player.can_afford(board[property_idx].price):
                player.buy_property(board[property_idx])
        elif action_type == 'sell':
            if board[property_idx].owner == player:
                player.sell_property(board[property_idx])
        elif action_type == 'do_nothing':
            pass  # No action is taken